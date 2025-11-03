#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate All Models Script
===========================
Script para avaliar todos os modelos YOLOv8 e YOLOv11 treinados.
Salva os resultados em pastas organizadas por versão do modelo.
"""

import sys
import subprocess
from pathlib import Path

# Base paths
ROOT_DIR = Path("/home/agrilab/Documentos/Casanova/Arvores/Dataset")
DATA_DIR = ROOT_DIR / "data"
RUNS_DIR = ROOT_DIR / "results" 
BASE_OUTPUT_DIR = ROOT_DIR / "results" / "evaluation_results"

# Versões dos modelos a avaliar
MODEL_VERSIONS = ["yolov8m-cls", "yolo11m-cls"]

# Mapeamento de nomes de dataset para nomes de diretório
DATASET_MAP = {
    "tangential_transverse": "madeira_completo",
    "tangential_transverse_hardness": "madeira_completo_dureza",
    "tangential": "madeira_tangencial",
    "tangential_hardness": "madeira_tangencial_dureza",
    "transverse": "madeira_transversal",
    "transverse_hardness": "madeira_transversal_dureza",
}


def build_model_list():
    """Constrói a lista completa de modelos a avaliar."""
    models = []
    
    print(f"🔍 Procurando modelos em {RUNS_DIR}...")
    
    for version in MODEL_VERSIONS:
        version_dir = RUNS_DIR / version
        if not version_dir.is_dir():
            continue

        for model_run_dir in version_dir.iterdir():
            dataset_folder_name = model_run_dir.name
            
            # Ignorar se não for um dataset conhecido
            if dataset_folder_name not in DATASET_MAP:
                continue

            dataset_name_alias = DATASET_MAP[dataset_folder_name]
            model_path = model_run_dir / "weights" / "best.pt"
            data_path = DATA_DIR / dataset_folder_name
            
            if model_path.exists():
                models.append({
                    "model": model_path,
                    "data": data_path,
                    "name": dataset_name_alias, # Usar o nome amigável
                    "version": version,
                })

    print(f"✅ Encontrados {len(models)} modelos.")
    return models


def evaluate_one(model_info, extra_args, evaluate_py):
    """Avalia um modelo específico."""
    model_path = model_info["model"]
    data_path = model_info["data"]
    name = model_info["name"]
    version = model_info["version"]
    
    # Verificar se o modelo existe
    if not model_path.exists():
        print(f"⚠️  Modelo não encontrado: {model_path}")
        print(f"   Pulando {version}/{name}...")
        return False
    
    # Verificar se o dataset existe
    if not data_path.exists():
        print(f"⚠️  Dataset não encontrado: {data_path}")
        print(f"   Pulando {version}/{name}...")
        return False
    
    # Diretório de saída organizado por versão
    output_dir = BASE_OUTPUT_DIR / version / name
    
    # Montar comando
    cmd = [
        sys.executable, str(evaluate_py),
        "--model", str(model_path),
        "--data", str(data_path),
        "--output", str(output_dir),
        "--batch", "64",          # << opcional: defina aqui
        "--imgsz", "224",         # << opcional: defina aqui
        *extra_args,
    ]
    
    print("\n" + "#" * 70)
    print(f"# AVALIANDO: {version} - {name}")
    print("#" * 70)
    print(f"Versão:  {version}")
    print(f"Modelo:  {model_path}")
    print(f"Dataset: {data_path}")
    print(f"Output:  {output_dir}")
    print("#" * 70)
    
    # Executar
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Concluído: {version}/{name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Falhou: {version}/{name} (exit {e.returncode})")
        return False


def create_summary(results):
    """Cria um resumo consolidado dos resultados."""
    import csv
    import json
    
    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Criar resumo geral
    general_summary_csv = BASE_OUTPUT_DIR / "evaluation_summary_all.csv"
    
    all_metrics = []
    
    # Coletar métricas de todos os modelos
    for model_info in results:
        if not model_info["success"]:
            continue
            
        version = model_info["version"]
        name = model_info["name"]
        metrics_file = BASE_OUTPUT_DIR / version / name / "evaluation_metrics.json"
        
        if metrics_file.exists():
            try:
                with open(metrics_file, "r") as f:
                    metrics = json.load(f)
                    all_metrics.append({
                        "version": version,
                        "dataset": name,
                        "metrics": metrics
                    })
            except Exception as e:
                print(f"⚠️  Erro ao ler métricas de {version}/{name}: {e}")
    
    # Salvar resumo geral consolidado
    if all_metrics:
        with open(general_summary_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "model_version", "dataset", "accuracy", "precision_macro", 
                "recall_macro", "f1_macro", "precision_weighted", 
                "recall_weighted", "f1_weighted", "num_samples"
            ])
            
            for item in all_metrics:
                version = item["version"]
                name = item["dataset"]
                m = item["metrics"]
                
                writer.writerow([
                    version,
                    name,
                    f"{m.get('accuracy', 0):.6f}",
                    f"{m.get('precision_macro', 0):.6f}",
                    f"{m.get('recall_macro', 0):.6f}",
                    f"{m.get('f1_macro', 0):.6f}",
                    f"{m.get('precision_weighted', 0):.6f}",
                    f"{m.get('recall_weighted', 0):.6f}",
                    f"{m.get('f1_weighted', 0):.6f}",
                    m.get('num_samples', 0),
                ])
        
        print(f"\n✅ Resumo geral salvo em: {general_summary_csv}")
    
    # Criar resumos por versão
    for version in MODEL_VERSIONS:
        version_metrics = [m for m in all_metrics if m["version"] == version]
        
        if not version_metrics:
            continue
        
        version_summary_csv = BASE_OUTPUT_DIR / version / "evaluation_summary.csv"
        version_summary_csv.parent.mkdir(parents=True, exist_ok=True)
        
        with open(version_summary_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "dataset", "accuracy", "precision_macro", "recall_macro",
                "f1_macro", "precision_weighted", "recall_weighted", 
                "f1_weighted", "num_samples"
            ])
            
            for item in version_metrics:
                name = item["dataset"]
                m = item["metrics"]
                
                writer.writerow([
                    name,
                    f"{m.get('accuracy', 0):.6f}",
                    f"{m.get('precision_macro', 0):.6f}",
                    f"{m.get('recall_macro', 0):.6f}",
                    f"{m.get('f1_macro', 0):.6f}",
                    f"{m.get('precision_weighted', 0):.6f}",
                    f"{m.get('recall_weighted', 0):.6f}",
                    f"{m.get('f1_weighted', 0):.6f}",
                    m.get('num_samples', 0),
                ])
        
        print(f"✅ Resumo {version} salvo em: {version_summary_csv}")
    
    # Imprimir tabela comparativa no terminal
    print("\n" + "=" * 120)
    print("RESUMO COMPARATIVO DOS RESULTADOS")
    print("=" * 120)
    print(f"{'Versão':<15} {'Dataset':<35} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'Samples':>8}")
    print("-" * 120)
    
    for item in sorted(all_metrics, key=lambda x: (x["version"], x["dataset"])):
        version = item["version"]
        name = item["dataset"]
        m = item["metrics"]
        print(f"{version:<15} {name:<35} "
              f"{m.get('accuracy', 0):>8.4f} "
              f"{m.get('precision_macro', 0):>8.4f} "
              f"{m.get('recall_macro', 0):>8.4f} "
              f"{m.get('f1_macro', 0):>8.4f} "
              f"{m.get('num_samples', 0):>8}")
    print("=" * 120)
    
    # Comparação entre versões
    print("\n" + "=" * 120)
    print("COMPARAÇÃO ENTRE VERSÕES (Accuracy média por dataset)")
    print("=" * 120)
    print(f"{'Dataset':<35} ", end="")
    for version in MODEL_VERSIONS:
        print(f"{version:>15} ", end="")
    print("Diferença")
    print("-" * 120)
    
    for dataset in sorted(DATASET_MAP.values()):
        accuracies = {}
        for item in all_metrics:
            if item["dataset"] == dataset:
                accuracies[item["version"]] = item["metrics"].get("accuracy", 0)
        
        if len(accuracies) == len(MODEL_VERSIONS):
            print(f"{dataset:<35} ", end="")
            acc_values = []
            for version in MODEL_VERSIONS:
                acc = accuracies.get(version, 0)
                acc_values.append(acc)
                print(f"{acc:>15.4f} ", end="")
            
            # Calcular diferença (v11 - v8)
            if len(acc_values) == 2:
                diff = acc_values[1] - acc_values[0]
                sign = "+" if diff >= 0 else ""
                print(f"{sign}{diff:>8.4f}")
            else:
                print()
    print("=" * 120)


def main():
    """Função principal."""
    # Verificar se evaluate.py existe
    evaluate_py = ROOT_DIR / "src" / "evaluate.py"
    if not evaluate_py.exists():
        print(f"❌ Script evaluate.py não encontrado: {evaluate_py}")
        print("   Certifique-se de que evaluate.py está no mesmo diretório!")
        return 1
    
    # Argumentos extras do usuário
    extra_args = sys.argv[1:]
    
    # Construir lista de modelos
    models = build_model_list()
    
    print("#" * 70)
    print("# AVALIANDO TODOS OS MODELOS YOLO")
    print("#" * 70)
    print(f"Versões: {', '.join(MODEL_VERSIONS)}")
    print(f"Datasets: {len(DATASET_MAP)}")
    print(f"Total de modelos: {len(models)}")
    print(f"Diretório de saída: {BASE_OUTPUT_DIR}")
    print("#" * 70)
    
    # Listar modelos encontrados
    print("\n📋 Modelos a avaliar:")
    for model_info in models:
        version = model_info["version"]
        name = model_info["name"]
        exists = "✓" if model_info["model"].exists() else "✗"
        print(f"  {exists} {version:<15} {name}")
    
    
    
    # Avaliar cada modelo
    results = []
    for model_info in models:
        success = evaluate_one(model_info, extra_args, evaluate_py)
        results.append({
            "version": model_info["version"],
            "name": model_info["name"],
            "success": success
        })
    
    # Criar resumo consolidado
    print("\n" + "#" * 70)
    print("# CRIANDO RESUMOS CONSOLIDADOS")
    print("#" * 70)
    create_summary(results)
    
    # Estatísticas finais
    total = len(results)
    success_count = sum(1 for r in results if r["success"])
    failed_count = total - success_count
    
    print("\n" + "#" * 70)
    print("# ✅ AVALIAÇÃO CONCLUÍDA")
    print("#" * 70)
    print(f"Total:      {total} modelos")
    print(f"Sucesso:    {success_count} modelos")
    print(f"Falhas:     {failed_count} modelos")
    print(f"\nResultados salvos em:")
    print(f"  📁 {BASE_OUTPUT_DIR}/")
    for version in MODEL_VERSIONS:
        if (BASE_OUTPUT_DIR / version).exists():
            print(f"    📁 {version}/")
            for dataset in sorted(DATASET_MAP.values()):
                if (BASE_OUTPUT_DIR / version / dataset).exists():
                    print(f"      📁 {dataset}/")
    print(f"\nResumos:")
    print(f"  📊 {BASE_OUTPUT_DIR / 'evaluation_summary_all.csv'} (geral)")
    for version in MODEL_VERSIONS:
        summary_file = BASE_OUTPUT_DIR / version / "evaluation_summary.csv"
        if summary_file.exists():
            print(f"  📊 {summary_file} ({version})")
    print("#" * 70)
    
    if failed_count > 0:
        print("\n⚠️  Modelos que falharam:")
        for r in results:
            if not r["success"]:
                print(f"  - {r['version']}/{r['name']}")
    
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrompido pelo usuário")
        exit(1)
    except Exception as e:
        print(f"\n\n❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        exit(1)