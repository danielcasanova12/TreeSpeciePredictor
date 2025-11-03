#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import subprocess
from pathlib import Path

# Caminho base onde ficam os datasets
BASE = Path("/home/agrilab/Documentos/Casanova/Arvores/Dataset/dados")

# Lista dos datasets a treinar (muda só o final do --data)
DATASETS = [
    "madeira_completo",
    "madeira_completo_dureza",
    "madeira_tangencial",
    "madeira_tangencial_dureza",
    "madeira_transversal",
    "madeira_transversal_dureza",
]

# Modelos a treinar (YOLOv8 e YOLOv11 classificação)
MODELS = [
    "yolov8m-cls.pt",
    "yolo11m-cls.pt",
]

# Onde está o train.py (ajuste se necessário)
TRAIN_PY = Path(__file__).parent / "train.py"

# Épocas padrão (pode sobrescrever passando --epochs nas opções extras)
DEFAULT_EPOCHS = os.getenv("EPOCHS", "150")

def stem_no_ext(path_or_name: str) -> str:
    return Path(path_or_name).stem

def run_one(model_name: str, ds_name: str, extra_args):
    data_path = BASE / ds_name
    if not data_path.exists():
        print(f"⚠️  Dataset não encontrado: {data_path} — pulando.")
        return

    model_id = stem_no_ext(model_name)  # ex.: 'yolov8m-cls'
    # --project = model_id  |  --name = dataset
    cmd = [
        sys.executable, str(TRAIN_PY),
        "--data", str(data_path),
        "--epochs", str(DEFAULT_EPOCHS),
        "--model", model_name,
        "--project", model_id,
        "--name", ds_name,               # pasta/execução por dataset
        "--wandb-project", "wood_classification",
        *extra_args,
    ]

    print("\n" + "#" * 70)
    print(f"# TREINANDO: {ds_name}  |  {model_id}")
    print("#" * 70)
    print(" ".join(cmd))

    t0 = time.time()
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Falha em {ds_name} ({model_id}) (exit {e.returncode})")
    else:
        dt = time.time() - t0
        print(f"✅ Concluído {ds_name} ({model_id}) em {dt/60:.2f} min")

def main():
    # Tudo que você passar após o script vai para o train.py (ex.: --imgsz 224 --batch 32)
    extra_args = sys.argv[1:]

    for model_name in MODELS:
        for ds in DATASETS:
            run_one(model_name, ds, extra_args)

if __name__ == "__main__":
    main()
