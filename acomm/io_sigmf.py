"""Minimal SigMF writer/reader for real float32 audio captures."""
import json
from pathlib import Path
import numpy as np


def save(path_stem, samples, meta):
    path_stem = Path(path_stem)
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    samples.astype("<f4").tofile(str(path_stem) + ".sigmf-data")
    doc = {
        "global": {
            "core:datatype": "rf32_le",
            "core:sample_rate": meta["fs"],
            "core:version": "1.0.0",
            "core:description": "Acoustic constellation capture (known payload)",
            **{"acomm:" + k: v for k, v in meta.items()},
        },
        "captures": [{"core:sample_start": 0, "core:frequency": meta["fc"]}],
        "annotations": [],
    }
    with open(str(path_stem) + ".sigmf-meta", "w") as f:
        json.dump(doc, f, indent=2)


def load(meta_path):
    meta_path = Path(meta_path)
    with open(meta_path) as f:
        g = json.load(f)["global"]
    meta = {k.split(":", 1)[1]: v for k, v in g.items() if k.startswith("acomm:")}
    data = np.fromfile(str(meta_path).replace(".sigmf-meta", ".sigmf-data"), dtype="<f4")
    return data, meta
