from pathlib import Path
from typing import Literal, Optional

import os
from pydantic import SecretStr, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MCPhasesSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Required for this project ---
    DATA_PATH: Path = Field(..., description="Path to folder containing mcPHASES CSV files")
    HOPSWORKS_API_KEY: Optional[SecretStr] = None
    HOPSWORKS_PROJECT: Optional[str] = None
    HOPSWORKS_HOST: Optional[str] = None

    # --- Experiment knobs (optional defaults) ---
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.2

    # Two stakeholder modes
    MODE: Literal["wearables_only", "wearables_plus_lags"] = "wearables_only"
    LAGS: tuple[int, ...] = (1, 2, 3)

    # Targets
    MOOD_TARGET: str = "y_mood"
    ENERGY_TARGET: str = "y_energy"

    # Naming conventions in Hopsworks
    FEATURE_GROUP_NAME: str = "mcphases_daily_fg"
    FEATURE_VIEW_NAME: str = "mcphases_daily_fv"

    def model_post_init(self, __context):
        # Set env vars for hopsworks.login() if not already set
        if os.getenv("HOPSWORKS_API_KEY") is None and self.HOPSWORKS_API_KEY is not None:
            os.environ["HOPSWORKS_API_KEY"] = self.HOPSWORKS_API_KEY.get_secret_value()
        if os.getenv("HOPSWORKS_PROJECT") is None and self.HOPSWORKS_PROJECT is not None:
            os.environ["HOPSWORKS_PROJECT"] = self.HOPSWORKS_PROJECT
        if os.getenv("HOPSWORKS_HOST") is None and self.HOPSWORKS_HOST is not None:
            os.environ["HOPSWORKS_HOST"] = self.HOPSWORKS_HOST

        # Validate only what mcPHASES actually needs
        missing = []
        if not (self.HOPSWORKS_API_KEY or os.getenv("HOPSWORKS_API_KEY")):
            missing.append("HOPSWORKS_API_KEY")
        if self.DATA_PATH is None:
            missing.append("DATA_PATH")

        if missing:
            raise ValueError(
                "Missing required settings:\n  " + "\n  ".join(missing)
            )


settings = MCPhasesSettings()
