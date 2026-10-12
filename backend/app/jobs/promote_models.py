"""
promote_models.py — Daily Safe Model Promotion & Versioned Archival Job for AgroIntel.

Safely promotes candidate models approved by validate_models.py:
  1. Backs up current production model to models/archive/{crop}/{timestamp}/.
  2. Promotes candidate .pkl models from models/candidates/ to models/ and models/production/.
  3. Updates models/model_registry.json with new model versions, training timestamps, and metrics.
  4. Preserves rollback capabilities to revert to any previous model version.
"""

import json
import logging
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.jobs.build_dataset import SUPPORTED_CROPS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("app.jobs.promote_models")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "app" / "data"
MODELS_DIR = BASE_DIR / "models"
CANDIDATES_DIR = MODELS_DIR / "candidates"
PRODUCTION_DIR = MODELS_DIR / "production"
ARCHIVE_DIR = MODELS_DIR / "archive"

PRODUCTION_DIR.mkdir(parents=True, exist_ok=True)
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

EXP_DIR = DATA_DIR / "experimental"
PROMOTION_AUDIT_PATH = EXP_DIR / "model_promotion_audit.json"
REGISTRY_PATH = MODELS_DIR / "model_registry.json"


class ModelPromotionJob:
    """Production Safe Model Promotion & Archival Job."""

    @staticmethod
    def archive_current_model(crop: str, timestamp_str: str) -> Optional[Path]:
        """Save current active production model to models/archive/{crop}/{timestamp}/."""
        crop_archive_dir = ARCHIVE_DIR / crop / timestamp_str
        crop_archive_dir.mkdir(parents=True, exist_ok=True)

        files_to_backup = [
            MODELS_DIR / f"xgboost_state_{crop}.pkl",
            MODELS_DIR / f"state_encoder_{crop}.pkl",
            MODELS_DIR / f"data_tail_state_{crop}.pkl",
            MODELS_DIR / f"metrics_state_{crop}.json",
        ]

        backed_up = 0
        for f in files_to_backup:
            if f.exists():
                shutil.copy2(f, crop_archive_dir / f.name)
                backed_up += 1

        if backed_up > 0:
            logger.info(f"[{crop.upper()}] Archived {backed_up} current production files to {crop_archive_dir}")
            return crop_archive_dir
        return None

    @staticmethod
    def promote_candidate_files(crop: str) -> bool:
        """Copy candidate files to models/ and models/production/."""
        files_to_promote = [
            f"xgboost_state_{crop}.pkl",
            f"state_encoder_{crop}.pkl",
            f"data_tail_state_{crop}.pkl",
            f"metrics_state_{crop}.json",
        ]

        for fname in files_to_promote:
            cand_src = CANDIDATES_DIR / fname
            if not cand_src.exists():
                logger.error(f"Candidate file missing: {cand_src}")
                return False

            # Copy to active models/ root
            shutil.copy2(cand_src, MODELS_DIR / fname)
            # Copy to models/production/
            shutil.copy2(cand_src, PRODUCTION_DIR / fname)

        return True

    @staticmethod
    def update_model_registry(promoted_crops: List[str], timestamp_iso: str) -> None:
        """Update models/model_registry.json with new versions and metrics."""
        registry = {
            "system_version": "4.1.0",
            "last_updated": timestamp_iso,
            "registry": {}
        }

        if REGISTRY_PATH.exists():
            try:
                with open(REGISTRY_PATH, "r") as f:
                    existing = json.load(f)
                    if "registry" in existing:
                        registry = existing
                        registry["last_updated"] = timestamp_iso
            except Exception:
                pass

        for crop in SUPPORTED_CROPS:
            metrics_path = MODELS_DIR / f"metrics_state_{crop}.json"
            metrics_data = {}
            if metrics_path.exists():
                try:
                    with open(metrics_path, "r") as f:
                        metrics_data = json.load(f)
                except Exception:
                    pass

            existing_crop_entry = registry["registry"].get(crop, {})
            crop_entry = {
                **existing_crop_entry,
                "crop": crop,
                "production_model": "state_aware_xgboost",
                "model_version": f"v_{timestamp_iso[:10].replace('-', '')}",
                "training_timestamp": metrics_data.get("trained_at", timestamp_iso),
                "holdout_mae": metrics_data.get("holdout_mae", 0.0),
                "holdout_rmse": metrics_data.get("holdout_rmse", 0.0),
                "holdout_r2": metrics_data.get("holdout_r2", 0.0),
                "forecast_horizon_days": 30,
                "state_aware": True,
                "multi_horizon_metrics": metrics_data.get("multi_horizon_metrics", {}),
                "model_file_path": f"models/xgboost_state_{crop}.pkl",
            }
            registry["registry"][crop] = crop_entry

        with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2)
        logger.info(f"Updated authoritative model registry at {REGISTRY_PATH}")

    @staticmethod
    def rollback(crop: str, archive_timestamp: str) -> bool:
        """Rollback a crop model to a specific archived timestamp."""
        source_dir = ARCHIVE_DIR / crop / archive_timestamp
        if not source_dir.exists():
            logger.error(f"Archive directory not found: {source_dir}")
            return False

        for f in source_dir.glob("*.*"):
            shutil.copy2(f, MODELS_DIR / f.name)
            shutil.copy2(f, PRODUCTION_DIR / f.name)

        logger.info(f"Successfully rolled back [{crop.upper()}] to version {archive_timestamp}")
        return True

    def run(self) -> Dict[str, Any]:
        """Execute safe model promotion for all validated crops."""
        t_start = time.time()
        logger.info("Starting Safe Model Promotion & Registry Update Job...")

        if not PROMOTION_AUDIT_PATH.exists():
            logger.warning(f"No validation audit found at {PROMOTION_AUDIT_PATH}. Run validate_models first.")
            return {"status": "NO_VALIDATION_AUDIT_FOUND"}

        with open(PROMOTION_AUDIT_PATH, "r") as f:
            audit = json.load(f)

        promoted_crops = audit.get("promoted_crops", [])
        rejected_crops = audit.get("rejected_crops", [])

        now_utc = datetime.now(timezone.utc)
        ts_dir = now_utc.strftime("%Y%m%d_%H%M%S")
        ts_iso = now_utc.isoformat()

        promoted_successful = []

        for crop in promoted_crops:
            logger.info(f"Promoting candidate model for [{crop.upper()}]...")
            # 1. Archive current production model
            self.archive_current_model(crop, ts_dir)
            # 2. Promote candidate files
            ok = self.promote_candidate_files(crop)
            if ok:
                promoted_successful.append(crop)

        # Update registry
        self.update_model_registry(promoted_successful, ts_iso)

        elapsed = round(time.time() - t_start, 2)

        summary = {
            "job_name": "promote_models",
            "timestamp": ts_iso,
            "execution_time_seconds": elapsed,
            "models_promoted_count": len(promoted_successful),
            "models_promoted": promoted_successful,
            "models_retained_old": rejected_crops,
            "archive_version_tag": ts_dir,
            "registry_updated": True,
        }

        logger.info(f"Model Promotion Complete in {elapsed}s: {len(promoted_successful)} models safely deployed to production.")
        return summary


if __name__ == "__main__":
    job = ModelPromotionJob()
    res = job.run()
    print(json.dumps(res, indent=2))
