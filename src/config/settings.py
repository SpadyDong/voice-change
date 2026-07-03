import json
import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


DEFAULT_CONFIG_PATH = Path.home() / ".voice-change" / "config.json"


class AudioSettings(BaseModel):
    sample_rate: int = Field(default=48000, ge=16000, le=192000)
    block_size: int = Field(default=1024, ge=256, le=8192)
    input_device: Optional[int] = None
    output_device: Optional[int] = None


class RvcSettings(BaseModel):
    model_dir: str = Field(default="models/rvc")
    hubert_path: str = Field(default="models/hubert/hubert_base.pt")
    pitch_shift: int = Field(default=0, ge=-24, le=24)
    index_rate: float = Field(default=0.75, ge=0.0, le=1.0)
    protect: float = Field(default=0.33, ge=0.0, le=0.5)
    filter_radius: int = Field(default=3, ge=0, le=10)
    f0_method: str = Field(default="rmvpe")


class AppSettings(BaseModel):
    window_width: int = 960
    window_height: int = 640
    audio: AudioSettings = Field(default_factory=AudioSettings)
    rvc: RvcSettings = Field(default_factory=RvcSettings)


class ConfigManager:
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self._settings = AppSettings()

    def load(self) -> AppSettings:
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._settings = AppSettings(**data)
            except Exception:
                self._settings = AppSettings()
        return self._settings

    def save(self) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self._settings.model_dump(), f, indent=2, ensure_ascii=False)

    @property
    def settings(self) -> AppSettings:
        return self._settings
