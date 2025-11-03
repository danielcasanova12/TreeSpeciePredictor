#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import time
import argparse
import yaml
from pathlib import Path
import sys

import torch
from ultralytics import YOLO
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import random, numpy as np, torch.backends.cudnn as cudnn

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Aviso: wandb não instalado. Instale com: pip install wandb")

# =============================================================================
# UTILS
# =============================================================================
import re, unicodedata

def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9_-]+", "-", text).strip("-").lower()
    return text or "run"

def model_id_from_weights(weights_name: str) -> str:
    return Path(weights_name).stem.lower()

# =============================================================================
# CALLBACK DE MÉTRICAS
# =============================================================================
def create_metrics_callback(csv_path, use_wandb=True):
    def on_fit_epoch_end(trainer):
        # ... (código do callback mantido, pois é interno ao treino)
        val_loader = getattr(trainer.validator, "dataloader", None)
        if val_loader is None:
            print(f"\n[Época {trainer.epoch+1}] Aviso: sem dataloader de validação\n")
            return

        model_torch = trainer.model
        was_training = model_torch.training
        device = next(model_torch.parameters()).device

        preds, gts = [], []
        model_torch.eval()
        with torch.no_grad():
            for batch in val_loader:
                imgs, labels = batch["img"], batch["cls"]
                imgs = imgs.to(device, non_blocking=True)
                logits = model_torch(imgs)
                if isinstance(logits, (list, tuple)): logits = logits[0]
                preds.extend(logits.argmax(dim=1).cpu().tolist())
                gts.extend(labels.view(-1).cpu().tolist())
        model_torch.train(was_training)

        acc = accuracy_score(gts, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(gts, preds, average="macro", zero_division=0)
        epoch = trainer.epoch + 1
        train_loss = trainer.metrics.get('train/loss', 0.0)
        val_loss = trainer.metrics.get('val/loss', 0.0)

        # Salvar em CSV
        try:
            write_header = not os.path.exists(csv_path)
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(["epoch", "train_loss", "val_loss", "accuracy", "precision_macro", "recall_macro", "f1_macro"])
                writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{acc:.6f}", f"{prec:.6f}", f"{rec:.6f}", f"{f1:.6f}"])
        except Exception as e:
            print(f"Aviso: erro ao salvar CSV: {e}")

        # Logar no W&B
        if use_wandb and WANDB_AVAILABLE and wandb.run is not None:
            wandb.log({
                "epoch": epoch, "metrics/accuracy": acc, "metrics/precision_macro": prec,
                "metrics/recall_macro": rec, "metrics/f1_macro": f1,
                "train/loss": train_loss, "val/loss": val_loss
            }, step=epoch)
    return on_fit_epoch_end

def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

# =============================================================================
# TREINO
# =============================================================================
def train_model(args, config):
    set_global_seed(42)
    
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {data_path}")

    model = YOLO(args.model)
    model_id = model_id_from_weights(args.model)
    dataset_name = data_path.name

    # O diretório do projeto e da run são definidos para o Ultralytics
    # Salvará em: <output_dir>/<model_id>/<dataset_name>
    output_dir = Path(config.get("output_dir", "results"))
    project_name = args.project or model_id
    run_name = args.name or dataset_name
    
    run_output_dir = output_dir / project_name / run_name
    csv_path = run_output_dir / "metrics_per_epoch.csv"

    print(f"\n{'#'*70}\n# YOLO CLASSIFIER TRAINING\n{'#'*70}")
    print(f"Dataset:       {data_path}")
    print(f"Modelo:        {args.model} (id: {model_id})")
    print(f"Saída:         {run_output_dir}")
    print(f"{'#'*70}\n")

    model.add_callback("on_fit_epoch_end", create_metrics_callback(csv_path, args.wandb))

    model.train(
        project=str(output_dir / project_name),
        name=run_name,
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        patience=args.patience,
        device=args.device,
        # ... outros parâmetros ...
    )

# =============================================================================
# CLI
# =============================================================================
def parse_args(config={}, argv=None):
    parser = argparse.ArgumentParser(description="Treinar classificador YOLO com base em config.")
    
    cfg_train = config.get("training_defaults", {})
    data_path = Path(config.get("data_base_dir", "data")) / cfg_train.get("dataset_name", "")
    
    parser.add_argument('--config', type=str, help='Caminho para um arquivo de configuração YAML específico.')
    parser.add_argument("--data", type=str, default=str(data_path), help="Caminho do dataset.")
    parser.add_argument("--model", type=str, default=cfg_train.get("model_name"), help="Nome do modelo/arquivo .pt.")
    parser.add_argument("--epochs", type=int, default=cfg_train.get("epochs"), help="Épocas.")
    parser.add_argument("--batch", type=int, default=cfg_train.get("batch_size"), help="Batch size.")
    parser.add_argument("--lr0", type=float, default=cfg_train.get("learning_rate"), help="Learning rate.")
    parser.add_argument("--imgsz", type=int, default=cfg_train.get("img_size"), help="Tamanho da imagem.")
    parser.add_argument("--patience", type=int, default=50, help="Paciência para early stopping.")
    parser.add_argument("--project", type=str, help="Override do nome do projeto (default: model_id).")
    parser.add_argument("--name", type=str, help="Override do nome da run (default: dataset_name).")
    parser.add_argument("--device", type=str, default="", help="Device (padrão: auto).")
    parser.add_argument("--wandb", action="store_true", default=False, help="Ativar W&B.")

    return parser.parse_args(argv)

def main():
    # Primeiro, parseia apenas o argumento --config para saber qual arquivo ler
    conf_parser = argparse.ArgumentParser(add_help=False)
    conf_parser.add_argument('--config', type=str, help='Caminho para o arquivo de configuração YAML.')
    args, remaining_argv = conf_parser.parse_known_args()

    config = {}
    config_path = None

    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Arquivo de configuração especificado não encontrado: {config_path}")
    else:
        # Fallback para o config padrão se nenhum for especificado
        config_path = Path(__file__).resolve().parent.parent / 'config' / 'config.yaml'
        if not config_path.exists():
            print("Nenhum --config especificado e config/config.yaml padrão não encontrado.")
            # Ainda assim, tenta rodar com os defaults do parser
            args = parse_args({}, remaining_argv)
            try:
                train_model(args, {})
            except Exception as e:
                print(f"\n\n❌ Erro durante o treinamento: {e}")
                return 1
            return 0

    print(f"📖 Carregando configuração de: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Segundo, parseia todos os args, usando o config para os defaults
    args = parse_args(config, remaining_argv)
    
    try:
        train_model(args, config)
    except Exception as e:
        print(f"\n\n❌ Erro durante o treinamento: {e}")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
