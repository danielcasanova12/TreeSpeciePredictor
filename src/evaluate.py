#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO Classifier Evaluation Script
==================================
Script para avaliar classificadores YOLO treinados no conjunto de teste.

Uso:
    python evaluate.py --model /path/to/weights/best.pt --data /path/to/dataset --batch 64 --imgsz 224
    python evaluate.py --model-dir /path/to/runs --data-dir /path/to/datasets --batch 64 --imgsz 224
"""

import os
import csv
import json
from pathlib import Path

import torch
import numpy as np
from ultralytics import YOLO
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  wandb não instalado. Instale com: pip install wandb")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False
    print("⚠️  matplotlib/seaborn não instalados. Plots desabilitados.")


# =============================================================================
# FUNÇÕES DE AVALIAÇÃO
# =============================================================================

def evaluate_model(model_path, data_path, output_dir, device="", verbose=False,
                   batch=-1, imgsz=224):
    """
    Avalia um modelo YOLO no conjunto de teste.

    Args:
        model_path: Caminho para o modelo (.pt)
        data_path: Caminho para o dataset
        output_dir: Diretório para salvar resultados
        device: Device para inferência (ex.: '0' ou 'cpu')
        verbose: Modo verbose
        batch: batch size para validação
        imgsz: tamanho da imagem na validação

    Returns:
        dict: Dicionário com todas as métricas
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Avaliando: {Path(model_path).name}")
    print(f"Dataset:   {data_path}")
    print(f"{'='*70}\n")

    # Carregar modelo
    model = YOLO(str(model_path))

    # Validar no conjunto de teste
    print("🔍 Executando validação no conjunto de teste...")
    results = model.val(
        data=str(data_path),
        split="test",
        batch=batch,
        workers=0,        # evita EOFError do pin_memory
        device=device or 0,
        imgsz=imgsz,
        verbose=verbose
    )

    # Métricas básicas do YOLO
    metrics = {
        "model_path": str(model_path),
        "dataset": str(data_path),
        "top1_accuracy": float(getattr(results, 'top1', 0.0)),
        "top5_accuracy": float(getattr(results, 'top5', 0.0)),
    }

    # Métricas detalhadas manuais
    print("📊 Calculando métricas detalhadas...")
    preds, gts, probs = get_predictions(model, data_path, device, imgsz)

    if len(preds) == 0:
        print("⚠️  Nenhuma predição obtida!")
        return metrics

    accuracy = accuracy_score(gts, preds)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        gts, preds, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        gts, preds, average="weighted", zero_division=0
    )

    metrics.update({
        "accuracy": float(accuracy),
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(prec_weighted),
        "recall_weighted": float(rec_weighted),
        "f1_weighted": float(f1_weighted),
        "num_samples": len(gts),
    })

    # Nomes das classes
    class_names = get_class_names(data_path)

    # Métricas por classe
    prec_pc, rec_pc, f1_pc, support = precision_recall_fscore_support(
        gts, preds, average=None, zero_division=0
    )
    metrics["per_class"] = {}
    for i, name in enumerate(class_names):
        metrics["per_class"][name] = {
            "precision": float(prec_pc[i]) if i < len(prec_pc) else 0.0,
            "recall": float(rec_pc[i]) if i < len(rec_pc) else 0.0,
            "f1": float(f1_pc[i]) if i < len(f1_pc) else 0.0,
            "support": int(support[i]) if i < len(support) else 0,
        }

    # Matriz de confusão
    cm = confusion_matrix(gts, preds)
    metrics["confusion_matrix"] = cm.tolist()

    # Print resumo
    print(f"\n{'='*70}")
    print("RESULTADOS DA AVALIAÇÃO")
    print(f"{'='*70}")
    print(f"Amostras:           {len(gts)}")
    print(f"Accuracy:           {accuracy:.4f}")
    print(f"Precision (macro):  {prec_macro:.4f}")
    print(f"Recall (macro):     {rec_macro:.4f}")
    print(f"F1-Score (macro):   {f1_macro:.4f}")
    print(f"{'='*70}\n")

    # Salvar métricas em JSON
    json_path = output_dir / "metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Métricas salvas em: {json_path}")

    # Classification report
    report_path = output_dir / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(classification_report(gts, preds, target_names=class_names, zero_division=0))
    print(f"✅ Relatório salvo em: {report_path}")

    # Matriz de confusão em CSV
    cm_path = output_dir / "confusion_matrix.csv"
    with open(cm_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + class_names)
        for i, row in enumerate(cm):
            writer.writerow([class_names[i]] + row.tolist())
    print(f"✅ Matriz de confusão salva em: {cm_path}")

    # Plot da matriz de confusão
    if PLOT_AVAILABLE:
        plot_confusion_matrix(cm, class_names, output_dir / "confusion_matrix.png")
        print(f"✅ Plot salvo em: {output_dir / 'confusion_matrix.png'}")

    return metrics


def get_predictions(model, data_path, device="", imgsz=224):
    """
    Obtém predições do modelo no conjunto de teste.
    Returns: (predictions, ground_truths, probabilities)
    """
    data_path = Path(data_path)
    test_dir = data_path / "test"

    if not test_dir.exists():
        print(f"⚠️  Diretório de teste não encontrado: {test_dir}")
        return [], [], []

    # Coletar imagens
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    image_paths, labels = [], []

    for class_dir in sorted(test_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() in exts:
                image_paths.append(img_path)
                labels.append(class_name)

    if not image_paths:
        print("⚠️  Nenhuma imagem encontrada no conjunto de teste!")
        return [], [], []

    class_names = get_class_names(data_path)
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    preds, gts, probs = [], [], []
    model_torch = model.model
    model_torch.eval()

    with torch.no_grad():
        for img_path, label in zip(image_paths, labels):
            try:
                results = model.predict(str(img_path), verbose=False, device=device or 0, imgsz=imgsz)
                if len(results) > 0:
                    result = results[0]
                    pred_idx = int(result.probs.top1)
                    pred_probs = result.probs.data.cpu().numpy()
                    preds.append(pred_idx)
                    probs.append(pred_probs)
                    gts.append(class_to_idx[label])
            except Exception as e:
                print(f"⚠️  Erro ao processar {img_path}: {e}")
                continue

    return preds, gts, probs


def get_class_names(data_path):
    """Obtém os nomes das classes do dataset."""
    data_path = Path(data_path)
    train_dir = data_path / "train"
    if not train_dir.exists():
        return []
    return sorted([d.name for d in train_dir.iterdir() if d.is_dir()])


def plot_confusion_matrix(cm, class_names, save_path):
    """Plota e salva a matriz de confusão."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        cbar_kws={"label": "Count"}
    )
    plt.title("Confusion Matrix", fontsize=16, pad=20)
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# =============================================================================
# AVALIAÇÃO EM LOTE
# =============================================================================

def find_best_models(base_dir):
    """Encontra todos os modelos 'best.pt' em um diretório."""
    base_dir = Path(base_dir)
    models = list(base_dir.rglob("weights/best.pt"))
    return sorted(models)


def evaluate_batch(models, data_dir, output_base, device="",
                   use_wandb=False, wandb_project="wood_evaluation",
                   batch=-1, imgsz=224):
    """
    Avalia múltiplos modelos em seus respectivos datasets.
    """
    output_base = Path(output_base)
    output_base.mkdir(parents=True, exist_ok=True)

    summary_csv = output_base / "evaluation_summary.csv"
    all_results = []

    for i, model_path in enumerate(models, 1):
        print(f"\n{'#'*70}")
        print(f"# MODELO {i}/{len(models)}")
        print(f"{'#'*70}")

        # Inferir dataset pelo caminho (ajuste conforme sua convenção)
        # Ex.: .../yolo11m-cls/madeira_completo/weights/best.pt -> "madeira_completo"
        parts = model_path.parts
        dataset_name = None
        # tenta pegar o diretório imediatamente acima de "weights"
        try:
            idx = parts.index("weights")
            dataset_name = parts[idx-1]
        except ValueError:
            pass

        if dataset_name is None:
            print(f"⚠️  Não foi possível identificar o dataset de {model_path}")
            continue

        data_path = Path(data_dir) / dataset_name
        if not data_path.exists():
            print(f"⚠️  Dataset não encontrado: {data_path}")
            continue

        # Diretório de saída específico
        output_dir = output_base / dataset_name / model_path.parent.parent.name

        # Avaliar modelo
        try:
            metrics = evaluate_model(
                model_path, data_path, output_dir,
                device=device, verbose=False,
                batch=batch, imgsz=imgsz
            )
            all_results.append(metrics)

            if use_wandb and WANDB_AVAILABLE:
                try:
                    wandb.init(
                        project=wandb_project,
                        name=f"{dataset_name}_{model_path.parent.parent.name}",
                        config={"model": str(model_path), "dataset": dataset_name},
                        reinit=True
                    )
                    if "accuracy" in metrics:
                        wandb.log({
                            "test/accuracy": metrics["accuracy"],
                            "test/precision_macro": metrics["precision_macro"],
                            "test/recall_macro": metrics["recall_macro"],
                            "test/f1_macro": metrics["f1_macro"],
                            "test/num_samples": metrics["num_samples"],
                        })
                    cm_path = output_dir / "confusion_matrix.png"
                    if PLOT_AVAILABLE and cm_path.exists():
                        wandb.log({"confusion_matrix": wandb.Image(str(cm_path))})
                    wandb.finish()
                except Exception as e:
                    print(f"⚠️  Erro ao logar no W&B: {e}")

        except Exception as e:
            print(f"❌ Erro ao avaliar {model_path}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Salvar resumo consolidado
    if all_results:
        with open(summary_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "dataset", "model", "accuracy", "precision_macro", "recall_macro",
                "f1_macro", "precision_weighted", "recall_weighted", "f1_weighted",
                "num_samples"
            ])
            for result in all_results:
                dataset = Path(result["dataset"]).name
                model_name = Path(result["model_path"]).parent.parent.name
                writer.writerow([
                    dataset,
                    model_name,
                    f"{result.get('accuracy', 0):.6f}",
                    f"{result.get('precision_macro', 0):.6f}",
                    f"{result.get('recall_macro', 0):.6f}",
                    f"{result.get('f1_macro', 0):.6f}",
                    f"{result.get('precision_weighted', 0):.6f}",
                    f"{result.get('recall_weighted', 0):.6f}",
                    f"{result.get('f1_weighted', 0):.6f}",
                    result.get('num_samples', 0),
                ])

        print(f"\n{'#'*70}")
        print(f"# ✅ AVALIAÇÃO CONCLUÍDA")
        print(f"{'#'*70}")
        print(f"Total de modelos: {len(all_results)}")
        print(f"Resumo:           {summary_csv}")
        print(f"Resultados:       {output_base}/")
        print(f"{'#'*70}\n")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    """Parse argumentos da linha de comando."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Avaliar classificadores YOLO no conjunto de teste",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:

  # Avaliar um modelo específico
  python evaluate.py --model /path/to/weights/best.pt \
                     --data /path/to/dataset --batch 64 --imgsz 224

  # Avaliar todos os modelos em um diretório
  python evaluate.py --model-dir /path/to/runs \
                     --data-dir /path/to/datasets \
                     --output results --batch 64 --imgsz 224
  
  # Avaliar com W&B
  python evaluate.py --model-dir /path/to/runs \
                     --data-dir /path/to/datasets \
                     --wandb --wandb-project wood_evaluation
        """
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", type=str,
                       help="Caminho para um modelo específico (.pt)")
    group.add_argument("--model-dir", type=str,
                       help="Diretório contendo múltiplos modelos (busca best.pt)")

    parser.add_argument("--data", type=str,
                        help="Caminho do dataset (para --model)")
    parser.add_argument("--data-dir", type=str,
                        help="Diretório contendo datasets (para --model-dir)")

    parser.add_argument("--output", type=str, default="evaluation_results",
                        help="Diretório de saída (padrão: evaluation_results)")

    parser.add_argument("--wandb", action="store_true",
                        help="Usar W&B para logging")
    parser.add_argument("--wandb-project", type=str, default="wood_evaluation",
                        help="Nome do projeto no W&B (padrão: wood_evaluation)")

    parser.add_argument("--device", type=str, default="",
                        help="Device (padrão: auto)")
    parser.add_argument("--verbose", action="store_true",
                        help="Modo verbose")

    # parâmetros de validação
    parser.add_argument("--batch", type=int, default=-1,
                        help="Batch size na validação (padrão: -1=auto)")
    parser.add_argument("--imgsz", type=int, default=224,
                        help="Tamanho da imagem na validação (padrão: 224)")

    return parser.parse_args()


def main():
    args = parse_args()

    try:
        if args.model:
            # Modo single
            if not args.data:
                print("❌ --data é obrigatório quando usar --model")
                return 1

            model_path = Path(args.model)
            if not model_path.exists():
                print(f"❌ Modelo não encontrado: {model_path}")
                return 1

            data_path = Path(args.data)
            if not data_path.exists():
                print(f"❌ Dataset não encontrado: {data_path}")
                return 1

            output_dir = Path(args.output)
            evaluate_model(
                model_path, data_path, output_dir,
                device=args.device, verbose=args.verbose,
                batch=args.batch, imgsz=args.imgsz
            )

        else:
            # Modo batch
            if not args.data_dir:
                print("❌ --data-dir é obrigatório quando usar --model-dir")
                return 1

            model_dir = Path(args.model_dir)
            if not model_dir.exists():
                print(f"❌ Diretório de modelos não encontrado: {model_dir}")
                return 1

            data_dir = Path(args.data_dir)
            if not data_dir.exists():
                print(f"❌ Diretório de datasets não encontrado: {data_dir}")
                return 1

            print("🔍 Procurando modelos best.pt...")
            models = find_best_models(model_dir)
            if len(models) == 0:
                print(f"⚠️  Nenhum modelo best.pt encontrado em {model_dir}")
                return 1

            print(f"✅ Encontrados {len(models)} modelos\n")
            for model in models:
                print(f"  - {model}")

            evaluate_batch(
                models, data_dir, args.output,
                device=args.device,
                use_wandb=args.wandb,
                wandb_project=args.wandb_project,
                batch=args.batch,
                imgsz=args.imgsz
            )

    except KeyboardInterrupt:
        print("\n\n⚠️  Avaliação interrompida pelo usuário")
        return 1
    except Exception as e:
        print(f"\n\n❌ Erro durante a avaliação: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
