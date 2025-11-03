#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO Classifier Training Script
================================
Script para treinar classificadores YOLO com métricas customizadas e logging.

Uso:
    python train.py --data /path/to/dataset --epochs 150 --imgsz 224
    python train.py --data dataset_tangencial --model yolov8l-cls.pt --batch 32
"""

import os
import csv
import time
import argparse
from pathlib import Path

import torch
from ultralytics import YOLO
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  wandb não instalado. Install com: pip install wandb")


# =============================================================================
# CALLBACK DE MÉTRICAS
# =============================================================================
import re, unicodedata

def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9_-]+", "-", text).strip("-").lower()
    return text or "run"


def create_metrics_callback(csv_path, use_wandb=True):
    """
    Cria callback para calcular métricas ao final de cada época.
    
    Args:
        csv_path: Caminho para salvar o CSV com as métricas
        use_wandb: Se True, loga métricas no W&B
    """
    def on_fit_epoch_end(trainer):
        val_loader = getattr(trainer.validator, "dataloader", None)
        if val_loader is None:
            print(f"\n[Época {trainer.epoch+1}] ⚠️  Sem dataloader de validação\n")
            return

        model_torch = trainer.model
        was_training = model_torch.training
        device = next(model_torch.parameters()).device

        preds, gts = [], []
        model_torch.eval()
        
        with torch.no_grad():
            for batch in val_loader:
                # Extrair imagens e labels
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

        # Calcular métricas
        acc = accuracy_score(gts, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(
            gts, preds, average="macro", zero_division=0
        )
        
        epoch = trainer.epoch + 1
        
        # Obter losses do trainer
        train_loss = getattr(trainer, 'loss', None)
        val_loss = getattr(trainer.validator, 'loss', None)
        
        # Se não conseguir do trainer, tentar dos metrics
        if train_loss is None:
            train_loss = trainer.metrics.get('train/loss', 0.0)
        if val_loss is None:
            val_loss = trainer.metrics.get('val/loss', 0.0)
        
        print(f"\n{'='*70}")
        print(f"[Época {epoch}]")
        print(f"  Train Loss: {train_loss:.4f}" if train_loss else "  Train Loss: N/A")
        print(f"  Val Loss:   {val_loss:.4f}" if val_loss else "  Val Loss: N/A")
        print(f"  Accuracy:   {acc:.4f}")
        print(f"  Precision:  {prec:.4f}")
        print(f"  Recall:     {rec:.4f}")
        print(f"  F1-Score:   {f1:.4f}")
        print(f"{'='*70}\n")

        # Salvar em CSV
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
                    f"{train_loss:.6f}" if train_loss else "N/A",
                    f"{val_loss:.6f}" if val_loss else "N/A",
                    f"{acc:.6f}", 
                    f"{prec:.6f}", 
                    f"{rec:.6f}", 
                    f"{f1:.6f}"
                ])
        except Exception as e:
            print(f"⚠️  Erro ao salvar CSV: {e}")
        
        # Log no W&B
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
                print(f"⚠️  Erro ao logar no W&B: {e}")

    return on_fit_epoch_end


# =============================================================================
# FUNÇÃO PRINCIPAL DE TREINO
# =============================================================================

def train_model(args):
    """Executa o treinamento do modelo YOLO."""
    
    # Validar dataset
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {data_path}")
    
    # Nome do projeto baseado no dataset se não fornecido
    project_name = args.project
    if project_name is None:
        project_name = f"train_{data_path.name}"
    
    # Nome da run
    run_name = slugify(args.name) if args.name and args.name.strip() else slugify(data_path.name)
    
    # Caminho do CSV de métricas
    output_dir = Path(args.output) if args.output else Path.cwd() / "runs" / project_name / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "metrics_per_epoch.csv"
    
    # Configurar W&B
    wandb_run = None
    if args.wandb and WANDB_AVAILABLE:
        try:
            wandb_run = wandb.init(
                project=args.wandb_project or project_name,
                name=run_name,
                config={
                    "model": args.model,
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
            print(f"✅ W&B iniciado: {wandb.run.url}\n")
        except Exception as e:
            print(f"⚠️  Erro ao iniciar W&B: {e}")
            args.wandb = False
    elif args.wandb and not WANDB_AVAILABLE:
        print("⚠️  W&B solicitado mas não está instalado. Continuando sem W&B.\n")
        args.wandb = False
    
    print(f"\n{'#'*70}")
    print(f"# YOLO CLASSIFIER TRAINING")
    print(f"{'#'*70}")
    print(f"Dataset:    {data_path}")
    print(f"Modelo:     {args.model}")
    print(f"Épocas:     {args.epochs}")
    print(f"Img Size:   {args.imgsz}")
    print(f"Batch:      {args.batch}")
    print(f"Projeto:    {project_name}")
    print(f"Run:        {run_name}")
    print(f"CSV:        {csv_path}")
    print(f"W&B:        {'Ativado' if args.wandb else 'Desativado'}")
    print(f"{'#'*70}\n")
    
    # Carregar modelo
    model = YOLO(args.model)
    model.info()
    
    # Adicionar callback de métricas
    model.add_callback("on_fit_epoch_end", create_metrics_callback(csv_path, args.wandb))
    
    # Iniciar treino
    start_time = time.time()
    
    results = model.train(
        project=project_name,
        name=run_name,
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
    )
    
    elapsed_time = time.time() - start_time
    
    print(f"\n{'#'*70}")
    print(f"# ✅ TREINAMENTO CONCLUÍDO")
    print(f"{'#'*70}")
    print(f"Tempo total: {elapsed_time:.2f}s ({elapsed_time/60:.2f}min)")
    print(f"Métricas:    {csv_path}")
    print(f"Resultados:  runs/{project_name}/{run_name}/")
    if args.wandb and wandb_run:
        print(f"W&B:         {wandb.run.url}")
    print(f"{'#'*70}\n")
    
    # Finalizar W&B
    if args.wandb and wandb_run:
        try:
            wandb.finish()
        except Exception as e:
            print(f"⚠️  Erro ao finalizar W&B: {e}")
    
    return results


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    """Parse argumentos da linha de comando."""
    parser = argparse.ArgumentParser(
        description="Treinar classificador YOLO com métricas customizadas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:

  # Treino básico (com W&B por padrão)
  python train.py --data dataset_tangencial --epochs 150
  
  # Treino sem W&B
  python train.py --data dataset_tangencial --epochs 150 --no-wandb
  
  # Treino customizado com W&B
  python train.py --data dataset_completo_dureza \\
                  --model yolov8l-cls.pt \\
                  --epochs 200 \\
                  --imgsz 320 \\
                  --batch 16 \\
                  --name experimento_01 \\
                  --wandb-project wood_classification
  
  # Treino com early stopping
  python train.py --data dataset_transversal \\
                  --epochs 300 \\
                  --patience 50

Métricas logadas no W&B:
  - epoch
  - metrics/accuracy
  - metrics/precision_macro
  - metrics/recall_macro
  - metrics/f1_macro
  - train/loss
  - val/loss
        """
    )
    
    # Argumentos obrigatórios
    parser.add_argument("--data", type=str, required=True,
                       help="Caminho do dataset (deve conter train/val/test)")
    
    # Modelo e arquitetura
    parser.add_argument("--model", type=str, default="yolov8m-cls.pt",
                       help="Modelo YOLO (padrão: yolov8m-cls.pt)")
    parser.add_argument("--imgsz", type=int, default=224,
                       help="Tamanho da imagem (padrão: 224)")
    
    # Hiperparâmetros
    parser.add_argument("--epochs", type=int, default=150,
                       help="Número de épocas (padrão: 150)")
    parser.add_argument("--batch", type=int, default=-1,
                       help="Batch size (padrão: -1 = auto)")
    parser.add_argument("--lr0", type=float, default=0.01,
                       help="Learning rate inicial (padrão: 0.01)")
    parser.add_argument("--dropout", type=float, default=0.1,
                       help="Dropout rate (padrão: 0.1)")
    parser.add_argument("--mixup", type=float, default=0.1,
                       help="Mixup augmentation (padrão: 0.1)")
    parser.add_argument("--optimizer", type=str, default="auto",
                       choices=["SGD", "Adam", "AdamW", "auto"],
                       help="Otimizador (padrão: auto)")
    parser.add_argument("--patience", type=int, default=50,
                       help="Early stopping patience (padrão: 50)")
    
    # Organização
    parser.add_argument("--project", type=str, default=None,
                       help="Nome do projeto (padrão: train_<dataset_name>)")
    parser.add_argument("--name", type=str, default=None,
                       help="Nome da run (padrão: nome do dataset)")
    parser.add_argument("--output", type=str, default=None,
                       help="Diretório de saída para CSV (padrão: runs/<project>/<name>)")
    
    # W&B (Weights & Biases)
    parser.add_argument("--wandb", action="store_true", default=True,
                       help="Usar W&B para logging (padrão: True)")
    parser.add_argument("--no-wandb", dest="wandb", action="store_false",
                       help="Desativar W&B")
    parser.add_argument("--wandb-project", type=str, default="wood_classification",
                       help="Nome do projeto no W&B (padrão: usa --project)")
    
    # Outros
    parser.add_argument("--device", type=str, default="",
                       help="Device (padrão: auto)")
    parser.add_argument("--verbose", action="store_true",
                       help="Modo verbose")
    parser.add_argument("--save", action="store_true", default=True,
                       help="Salvar checkpoints")
    
    return parser.parse_args()


def main():
    """Função principal."""
    args = parse_args()
    
    try:
        train_model(args)
    except KeyboardInterrupt:
        print("\n\n⚠️  Treinamento interrompido pelo usuário")
    except Exception as e:
        print(f"\n\n❌ Erro durante o treinamento: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())