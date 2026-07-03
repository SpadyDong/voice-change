import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class ModelInfo:
    name: str
    pth_path: str
    index_path: Optional[str]
    version: str = "v2"
    sample_rate: int = 48000


class ModelManager:
    def __init__(self, model_dir: str = "models/rvc"):
        self.model_dir = Path(model_dir)
        self.models: List[ModelInfo] = []
        self.current_model: Optional[ModelInfo] = None

    def scan_models(self) -> List[ModelInfo]:
        self.models = []
        if not self.model_dir.exists():
            return self.models

        pth_files = sorted(self.model_dir.glob("*.pth"))
        for pth in pth_files:
            name = pth.stem
            index_candidates = list(self.model_dir.glob(f"{name}*.index"))
            index_path = str(index_candidates[0]) if index_candidates else None

            # Try to infer version / sample rate from a sidecar JSON if present
            info_path = pth.with_suffix(".json")
            version = "v2"
            sr = 48000
            if info_path.exists():
                try:
                    with open(info_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    version = meta.get("version", version)
                    sr = meta.get("sample_rate", sr)
                except Exception:
                    pass
            else:
                # Heuristic from filename
                lname = name.lower()
                if "32k" in lname:
                    sr = 32000
                elif "40k" in lname:
                    sr = 40000
                elif "48k" in lname:
                    sr = 48000
                if "v1" in lname or "v1-" in lname:
                    version = "v1"
                elif "v2" in lname or "v2-" in lname:
                    version = "v2"

            self.models.append(
                ModelInfo(
                    name=name,
                    pth_path=str(pth),
                    index_path=index_path,
                    version=version,
                    sample_rate=sr,
                )
            )

        return self.models

    def get_model_names(self) -> List[str]:
        return [m.name for m in self.models]

    def get_model_by_name(self, name: str) -> Optional[ModelInfo]:
        for m in self.models:
            if m.name == name:
                return m
        return None

    def set_current_model(self, name: str) -> Optional[ModelInfo]:
        model = self.get_model_by_name(name)
        self.current_model = model
        return model
