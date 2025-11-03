#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import json
import yaml
import argparse
from pathlib import Path
from ultralytics import YOLO
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False

# =============================================================================
# FUNÇÕES PRINCIPAIS
# =============================================================================

def evaluate_batch(model_search_dir, data_base_dir, eval_output_dir, args):
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🔍 Procurando modelos 'best.pt' em: {model_search_dir}")
    models = sorted(list(model_search_dir.rglob("**/weights/best.pt")))
    if not models:
        raise FileNotFoundError(f"Nenhum modelo 'best.pt' encontrado em {model_search_dir}")

    print(f"✅ Encontrados {len(models)} modelos para avaliação.")
    all_results = []

    for i, model_path in enumerate(models, 1):
        print(f"\n{'#'*30} AVALIANDO MODELO {i}/{len(models)} {'#'*30}")
        try:
            # Infere o nome do dataset e do modelo a partir da estrutura de pastas
            dataset_name = model_path.parent.parent.name
            model_name = model_path.parent.parent.parent.name
            data_path = data_base_dir / dataset_name
            if not data_path.exists():
                print(f"⚠️  Dataset inferido não encontrado: {data_path}, pulando.")
                continue

            # Define o diretório de saída para este resultado específico
            current_eval_dir = eval_output_dir / model_name / dataset_name
            metrics = evaluate_single_model(model_path, data_path, current_eval_dir, args)
            if metrics:
                all_results.append(metrics)

        except Exception as e:
            print(f"❌ Erro inesperado ao avaliar {model_path.name}: {e}")
            continue

    if all_results:
        summary_csv = eval_output_dir / "evaluation_summary_all.csv"
        save_summary_csv(all_results, summary_csv)
        print(f"\n{'#'*80}\n# ✅ AVALIAÇÃO EM LOTE CONCLUÍDA\n# Resumo consolidado salvo em: {summary_csv}\n# Resultados detalhados em:    {eval_output_dir}/\n{'#'*80}\n")

def evaluate_single_model(model_path, data_path, output_dir, args):
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Avaliando: {model_path.name} | Dataset: {data_path.name}")

    model = YOLO(str(model_path))
    preds, gts = get_predictions(model, data_path, args.device, args.imgsz, args.batch)
    if not gts:
        print("⚠️  Nenhuma amostra encontrada no conjunto de teste.")
        return None

    class_names = get_class_names(data_path)
    metrics = calculate_metrics(preds, gts, class_names, model_path, data_path)
    save_evaluation_artifacts(preds, gts, class_names, metrics, output_dir)
    return metrics

# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def get_predictions(model, data_path, device, imgsz, batch_size):
    test_dir = data_path / "test"
    image_paths = sorted(list(test_dir.rglob("*.jpg")) + list(test_dir.rglob("*.png")))
    class_names = get_class_names(data_path)
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    valid_image_paths = []
    gts = []
    for img_path in image_paths:
        label_name = img_path.parent.name
        if label_name in class_to_idx:
            valid_image_paths.append(str(img_path))
            gts.append(class_to_idx[label_name])

    if not valid_image_paths:
        return [], []

    preds = []
    # Executa a predição em lotes para maior eficiência
    results_generator = model.predict(valid_image_paths, verbose=False, device=device or 0, imgsz=imgsz, batch=batch_size)
    
    for result in results_generator:
        if result.probs:
            preds.append(int(result.probs.top1))
        else:
            preds.append(-1)

    # Filtrar predições onde o modelo pode não ter retornado um resultado
    valid_preds_gts = [(p, g) for p, g in zip(preds, gts) if p != -1]
    if not valid_preds_gts:
        return [], []
        
    final_preds, final_gts = zip(*valid_preds_gts)
    return list(final_preds), list(final_gts)

def calculate_metrics(preds, gts, class_names, model_path, data_path):
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(gts, preds, average="macro", zero_division=0)
    return {
        "model_path": str(model_path),
        "dataset": str(data_path),
        "accuracy": accuracy_score(gts, preds),
        "f1_macro": f1_macro,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "num_samples": len(gts),
    }

def save_evaluation_artifacts(preds, gts, class_names, metrics, output_dir):
    with open(output_dir / "evaluation_metrics.json", "w") as f: json.dump(metrics, f, indent=2)
    with open(output_dir / "classification_report.txt", "w") as f:
        f.write(classification_report(gts, preds, target_names=class_names, zero_division=0))
    cm = confusion_matrix(gts, preds)
    if PLOT_AVAILABLE: plot_confusion_matrix(cm, class_names, output_dir / "confusion_matrix.png")

def get_class_names(data_path):
    train_dir = data_path / "train"
    if not train_dir.exists(): return []
    return sorted([d.name for d in train_dir.iterdir() if d.is_dir()])

def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(max(12, len(class_names)//2), max(10, len(class_names)//2.5)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix"), plt.ylabel("True Label"), plt.xlabel("Predicted Label")
    plt.tight_layout(), plt.savefig(save_path, dpi=300), plt.close()

def save_summary_csv(all_results, csv_path):
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["model", "dataset", "accuracy", "f1_macro", "precision_macro", "recall_macro", "num_samples"]
        writer.writerow(header)
        for r in all_results:
            writer.writerow([
                Path(r["model_path"]).parent.parent.parent.parent.name,
                Path(r["dataset"]).name,
                f"{r.get('accuracy', 0):.6f}", f"{r.get('f1_macro', 0):.6f}",
                f"{r.get('precision_macro', 0):.6f}", f"{r.get('recall_macro', 0):.6f}",
                r.get('num_samples', 0)
            ])

# =============================================================================
# CLI
# =============================================================================

def parse_args(config={}):
    parser = argparse.ArgumentParser(description="Avaliar classificadores YOLO.")
    parser.add_argument("--model", type=str, help="Avaliar um modelo .pt específico.")
    parser.add_argument("--data", type=str, help="Dataset a ser usado com --model.")
    parser.add_argument("--output", type=str, help="Diretório para salvar os resultados da avaliação.")
    parser.add_argument("--imgsz", type=int, default=config.get("training_defaults", {}).get("img_size", 224))
    parser.add_argument("--batch", type=int, default=64, help="Tamanho do lote para predição.")
    parser.add_argument("--device", type=str, default="")
    return parser.parse_args()

def main():
    config_path = Path(__file__).resolve().parent.parent / 'config' / 'config.yaml'
    if not config_path.exists():
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    args = parse_args(config)
    
    output_dir = Path(config.get("output_dir", "results"))
    data_base_dir = Path(config.get("data_base_dir", "data"))

    try:
        if args.model:
            # Modo de avaliação única
            if not args.data: raise ValueError("--data é obrigatório com --model")
            eval_dir = Path(args.output) if args.output else output_dir / "evaluation_single" / Path(args.model).stem
            evaluate_single_model(Path(args.model), Path(args.data), eval_dir, args)
        else:
            # Modo de avaliação em lote (padrão)
            eval_summary_dir = output_dir / "evaluation_summary"
            evaluate_batch(output_dir, data_base_dir, eval_summary_dir, args)
            
    except Exception as e:
        print(f"\n\n❌ Erro durante a avaliação: {e}")
        import traceback; traceback.print_exc()
        return 1
    return 0

if __name__ == "__main__":
    exit(main())