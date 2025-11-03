#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cria subconjuntos 'Tangencial2' e 'Transversal2' com apenas as primeiras N imagens por classe.
Seleção determinística: ordenação natural pelo nome do arquivo.

Uso:
    python subset_first_120.py --root "/caminho/para/dados" --n 120 --suffix 2
Exemplo:
    python subset_first_120.py --root "~/Documentos/Casanova/Arvores/Dataset/dados" --n 120 --suffix 2
"""

import argparse
import os
import re
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def natural_key(s: str):
    # Ordenação "natural" para nomes com números (ex: img2 < img10)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def list_images(folder: Path):
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    files.sort(key=lambda p: natural_key(p.name))
    return files

def copy_first_n(src_class_dir: Path, dst_class_dir: Path, n: int):
    imgs = list_images(src_class_dir)
    dst_class_dir.mkdir(parents=True, exist_ok=True)
    selected = imgs[:n] if len(imgs) > n else imgs
    for p in selected:
        shutil.copy2(p, dst_class_dir / p.name)
    return len(selected), len(imgs)

def process_dataset(root: Path, dataset_name: str, suffix: str, n: int):
    src = root / dataset_name
    dst = root / f"{dataset_name}{suffix}"
    if not src.is_dir():
        print(f"⚠️  Dataset não encontrado: {src}")
        return

    dst.mkdir(parents=True, exist_ok=True)
    print(f"\n▶ Processando '{dataset_name}' → '{dst.name}' (N={n})")

    total_copied = 0
    for entry in sorted(src.iterdir(), key=lambda p: natural_key(p.name)):
        if entry.is_dir():
            dst_class = dst / entry.name
            copied, total = copy_first_n(entry, dst_class, n)
            total_copied += copied
            print(f" - {entry.name}: {copied}/{min(n, total)} (de {total})")

    print(f"✅ Concluído: {dataset_name} → {dst.name}. Total de imagens copiadas: {total_copied}")

def main():
    ap = argparse.ArgumentParser(description="Cria subsets com as primeiras N imagens por classe.")
    ap.add_argument("--root", required=True, help="Pasta raiz que contém 'Tangencial' e 'Transversal'")
    ap.add_argument("--n", type=int, default=120, help="Quantidade de imagens por classe (default: 120)")
    ap.add_argument("--suffix", default="2", help="Sufixo para os novos datasets (default: '2')")
    args = ap.parse_args()

    root = Path(os.path.expanduser(args.root)).resolve()
    if not root.is_dir():
        raise SystemExit(f"Raiz inválida: {root}")

    for ds in ["Tangencial", "Transversal"]:
        process_dataset(root, ds, args.suffix, args.n)

if __name__ == "__main__":
    main()
