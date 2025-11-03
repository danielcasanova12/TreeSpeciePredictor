#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import time
import argparse
from pathlib import Path

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

def infer_size_letter_from_model_hint(model_hint: str) -> str:
    if not model_hint:
        return "m"
    m = re.search(r"(?<![a-z])[nsmlx](?=[-_]?cls\.pt|\.pt|$)", model_hint, flags=re.I)
    return (m.group(0).lower() if m else "m")

def try_load_yolo(weights: str):
    try:
        model = YOLO(weights)
        return model, None
    except Exception as e:
        return None, e

def resolve_yolo_weights_and_load(args):
    """
    Regras:
    1) Se args.model for um arquivo local, usa direto.
    2) Se args.model mencionar explicitamente a versão (v8, v11, v12, v13), tenta exatamente o nome informado.
       Sem fallback para outras versões.
    3) Caso args.model esteja vazio ou seja genérico, aplica fallback 13 -> 12 -> 11 usando a letra do tamanho.
    """
    model_hint = args.model or ""

    # 1) Arquivo local?
    if model_hint and Path(model_hint).exists():
        print(f"Usando arquivo local de pesos: {model_hint}")
        model, err = try_load_yolo(model_hint)
        if model:
            return model, model_hint
        raise RuntimeError(f"Falha ao carregar pesos locais '{model_hint}': {err}")

    # Detecta se o usuário pediu explicitamente uma versão (yolov8..., yolov11..., etc.)
    import re
    explicit_version = re.match(r"^yolo(v?\d+)[nsmlx]-cls\.pt$", Path(model_hint).name, flags=re.I)

    if model_hint and explicit_version:
        # 2) Pedido explícito: tentar exatamente o que foi passado
        print(f"Tentando carregar explicitamente o modelo solicitado: '{model_hint}'")
        model, err = try_load_yolo(model_hint)
        if model:
            print(f"OK: carregado '{model_hint}'.")
            return model, model_hint
        raise RuntimeError(f"Falha ao carregar o modelo solicitado '{model_hint}': {err}")

    # 3) Genérico → fallback 13→12→11
    size = infer_size_letter_from_model_hint(model_hint)
    candidates = [
        f"yolo13{size}-cls.pt",   f"yolov13{size}-cls.pt",
        f"yolo12{size}-cls.pt",   f"yolov12{size}-cls.pt",
        f"yolo11{size}-cls.pt",   f"yolov11{size}-cls.pt",
    ]
    tried_msgs = []

    print(f"Tentando carregar YOLO de classificação nesta ordem (tamanho='{size}'): 13 → 12 → 11")
    for w in candidates:
        print(f" - Tentando {w} ...")
        model, err = try_load_yolo(w)
        if model:
            print(f"OK: carregado '{w}'.")
            return model, w
        tried_msgs.append(f"{w}: {err}")

    msg = "Falha ao carregar modelos de classificação (tentativas):\n  - " + "\n  - ".join(tried_msgs)
    raise RuntimeError(msg)

def model_id_from_weights(weights_name: str) -> str:
    """
    Extrai um identificador curto e limpo do peso escolhido.
    Ex.: 'yolov8m-cls.pt' -> 'yolov8m-cls'
         '/path/to/yolo11l-cls.pt' -> 'yolo11l-cls'
    """
    return Path(weights_name).stem.lower()

# =============================================================================
# CALLBACK DE MÉTRICAS
# =============================================================================
def create_metrics_callback(csv_path, use_wandb=True):
    def on_fit_epoch_end(trainer):
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
                if isinstance(batch, dict):
                    imgs = batch.get("img", batch.get("imgs"))
                    labels = batch.get("cls", batch.get("label", batch.get("labels")))
                else:
                    imgs, labels = batch[0], batch[1]

                imgs = imgs.to(device, non_blocking=True)
                logits = model_torch(imgs)
                if isinstance(logits, (list, tuple)):
                    logits = logits[0]
                pred_idx = logits.argmax(dim=1)
                preds.extend(pred_idx.cpu().tolist())

                if isinstance(labels, torch.Tensor):
                    labels = labels.view(-1).cpu().tolist()
                else:
                    labels = list(labels)
                gts.extend(labels)

        model_torch.train(was_training)

        acc = accuracy_score(gts, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(
            gts, preds, average="macro", zero_division=0
        )
        epoch = trainer.epoch + 1
        train_loss = getattr(trainer, 'loss', None) or trainer.metrics.get('train/loss', 0.0)
        val_loss = getattr(trainer.validator, 'loss', None) or trainer.metrics.get('val/loss', 0.0)

        print(f"\n{'='*70}")
        print(f"[Época {epoch}]")
        print(f"  Train Loss: {train_loss:.4f}" if train_loss else "  Train Loss: N/A")
        print(f"  Val Loss:   {val_loss:.4f}" if val_loss else "  Val Loss: N/A")
        print(f"  Accuracy:   {acc:.4f}")
        print(f"  Precision:  {prec:.4f}")
        print(f"  Recall:     {rec:.4f}")
        print(f"  F1-Score:   {f1:.4f}")
        print(f"{'='*70}\n")

        try:
            write_header = not os.path.exists(csv_path)
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow([
                        "epoch", "train_loss", "val_loss", "accuracy",
                        "precision_macro", "recall_macro", "f1_macro"
                    ])
                writer.writerow([
                    epoch,
                    f"{float(train_loss):.6f}" if train_loss else "N/A",
                    f"{float(val_loss):.6f}" if val_loss else "N/A",
                    f"{acc:.6f}",
                    f"{prec:.6f}",
                    f"{rec:.6f}",
                    f"{f1:.6f}"
                ])
        except Exception as e:
            print(f"Aviso: erro ao salvar CSV: {e}")

        if use_wandb and WANDB_AVAILABLE and wandb.run is not None:
            try:
                log_dict = {
                    "epoch": epoch,
                    "metrics/accuracy": acc,
                    "metrics/precision_macro": prec,
                    "metrics/recall_macro": rec,
                    "metrics/f1_macro": f1,
                }
                if train_loss is not None:
                    log_dict["train/loss"] = float(train_loss)
                if val_loss is not None:
                    log_dict["val/loss"] = float(val_loss)
                wandb.log(log_dict, step=epoch)
            except Exception as e:
                print(f"Aviso: erro ao logar no W&B: {e}")
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
def train_model(args):
    set_global_seed(42)
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {data_path}")

    # Carrega modelo e descobre o identificador curto (ex.: yolov8m-cls / yolov11m-cls)
    model, chosen_weights = resolve_yolo_weights_and_load(args)
    model_id = model_id_from_weights(chosen_weights)

    dataset_name = slugify(data_path.name)
    # projeto = pasta imediatamente sob runs/, nomeada pelo modelo
    project_name = args.project or model_id
    # nome da run = dataset + opcional nome custom
    base_run = slugify(args.name) if args.name and args.name.strip() else dataset_name
    run_name = f"{dataset_name}__{base_run}" if base_run != dataset_name else dataset_name

    # Diretório de saída para CSV fora do diretório interno do Ultralytics (opcional)
    output_dir = Path(args.output) if args.output else Path.cwd() / "runs" / project_name / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "metrics_per_epoch.csv"

    # W&B
    wandb_run = None
    if args.wandb and WANDB_AVAILABLE:
        try:
            wandb_run = wandb.init(
                project=args.wandb_project or "wood_classification",
                name=f"{model_id}__{run_name}",
                group=model_id,
                tags=[model_id, dataset_name, f"img{args.imgsz}", f"ep{args.epochs}"],
                config={
                    "resolved_weights": chosen_weights,
                    "model_id": model_id,
                    "epochs": args.epochs,
                    "imgsz": args.imgsz,
                    "batch": args.batch,
                    "lr0": args.lr0,
                    "dropout": args.dropout,
                    "mixup": args.mixup,
                    "optimizer": args.optimizer,
                    "patience": args.patience,
                    "dataset": str(data_path),
                },
                reinit=True
            )
            print("W&B iniciado.\n")
        except Exception as e:
            print(f"Aviso: erro ao iniciar W&B: {e}")
            args.wandb = False
    elif args.wandb and not WANDB_AVAILABLE:
        print("Aviso: W&B solicitado mas não está instalado. Continuando sem W&B.\n")
        args.wandb = False

    print(f"\n{'#'*70}")
    print(f"# YOLO CLASSIFIER TRAINING")
    print(f"{'#'*70}")
    print(f"Dataset:       {data_path}")
    print(f"Pesos:         {chosen_weights} (model_id={model_id})")
    print(f"Épocas:        {args.epochs}")
    print(f"Img Size:      {args.imgsz}")
    print(f"Batch:         {args.batch}")
    print(f"Projeto:       {project_name}   (runs/{project_name}/{run_name})")
    print(f"Run:           {run_name}")
    print(f"CSV:           {csv_path}")
    print(f"W&B:           {'Ativado' if args.wandb else 'Desativado'}")
    print(f"{'#'*70}\n")

    model.info()
    model.add_callback("on_fit_epoch_end", create_metrics_callback(csv_path, args.wandb))

    start_time = time.time()
    results = model.train(
        project=project_name,      # <-- salva em runs/<model_id>/
        name=run_name,             # <-- .../<dataset>/
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        verbose=args.verbose,
        dropout=args.dropout,
        val=True,
        mixup=args.mixup,
        mosaic=0.0,
        optimizer=args.optimizer,
        lr0=args.lr0,
        patience=args.patience,
        save=args.save,
        device=args.device,
        workers=0,
    )
    elapsed_time = time.time() - start_time

    print(f"\n{'#'*70}")
    print(f"# Treinamento concluído")
    print(f"{'#'*70}")
    print(f"Tempo total: {elapsed_time:.2f}s ({elapsed_time/60:.2f}min)")
    print(f"Métricas:    {csv_path}")
    print(f"Resultados:  runs/{project_name}/{run_name}/")
    if args.wandb and wandb_run:
        print(f"W&B:         {wandb.run.url}")
    print(f"{'#'*70}\n")

    if args.wandb and wandb_run:
        try:
            wandb.finish()
        except Exception as e:
            print(f"Aviso: erro ao finalizar W&B: {e}")

    return results

# =============================================================================
# CLI
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Treinar classificador YOLO com métricas customizadas (fallback 13→12→11)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="..."
    )
    parser.add_argument("--data", type=str, required=True,
                        help="Caminho do dataset (deve conter train/val/test)")
    parser.add_argument("--model", type=str, default="yolov13m-cls.pt",
                        help="Nome do modelo/arquivo. Se não for caminho local, tenta 13→12→11.")
    parser.add_argument("--imgsz", type=int, default=224, help="Tamanho da imagem (padrão: 224)")
    parser.add_argument("--epochs", type=int, default=150, help="Épocas (padrão: 150)")
    parser.add_argument("--batch", type=int, default=-1, help="Batch size (padrão: -1 = auto)")
    parser.add_argument("--lr0", type=float, default=0.01, help="Learning rate inicial (padrão: 0.01)")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout (padrão: 0.1)")
    parser.add_argument("--mixup", type=float, default=0.1, help="Mixup (padrão: 0.1)")
    parser.add_argument("--optimizer", type=str, default="auto",
                        choices=["SGD", "Adam", "AdamW", "auto"], help="Otimizador")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience")

    parser.add_argument("--project", type=str, default=None,
                        help="Nome da pasta do projeto (default: <model_id>)")
    parser.add_argument("--name", type=str, default=None,
                        help="Nome base da run (default: <dataset>)")
    parser.add_argument("--output", type=str, default=None,
                        help="Diretório p/ CSV (default: runs/<project>/<name>)")

    parser.add_argument("--wandb", action="store_true", default=True, help="Usar W&B (default: True)")
    parser.add_argument("--no-wandb", dest="wandb", action="store_false", help="Desativar W&B")
    parser.add_argument("--wandb-project", type=str, default="wood_classification",
                        help="Projeto no W&B (default: wood_classification)")

    parser.add_argument("--device", type=str, default="", help="Device (padrão: auto)")
    parser.add_argument("--verbose", action="store_true", help="Verbose")
    parser.add_argument("--save", action="store_true", default=True, help="Salvar checkpoints")
    return parser.parse_args()

def main():
    args = parse_args()
    try:
        train_model(args)
    except KeyboardInterrupt:
        print("\n\nTreinamento interrompido pelo usuário")
    except Exception as e:
        print(f"\n\nErro durante o treinamento: {e}")
        import traceback
        traceback.print_exc()
        return 1
    return 0

if __name__ == "__main__":
    exit(main())
