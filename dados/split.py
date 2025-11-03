#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wood Dataset Splitter
=====================
Organiza imagens de madeira em datasets train/test/val, com opção de adicionar
barras de dureza coloridas APENAS nos datasets *_dureza.

Autor: [Seu Nome]
Repositório: [URL do GitHub]
"""

import os
import re
import shutil
import random
import argparse
from pathlib import Path
from PIL import Image, ImageDraw


# =============================================================================
# CONFIGURAÇÕES
# =============================================================================

# Inclui grafias com e sem acento conforme suas pastas (ls) e sinônimos comuns
SPECIES_HARDNESS = {
    'Angelim-vermelho': 'Dura',
    'Castanha-de-Macaco': 'Macia',
    'Catingueira': 'Dura',
    'Cedro': 'Macia',
    'Cerejeira': 'Média',
    'Cupiuba': 'Dura',
    'Garapa': 'Dura',
    'Imbuia': 'Dura',
    'Jacareúba': 'Dura',
    'Jacareuba': 'Dura',
    'Jatobá': 'Dura',
    'Jatoba': 'Dura',
    'Maçaranduba': 'Dura',
    'Macaranduba': 'Dura',
    'Mogno': 'Macia',
    'Muiracatiara': 'Dura',
    'Pariri': 'Dura',
    'Roxinho': 'Dura',
    'Sucupira-preta': 'Dura',
    'Tauari': 'Macia',
    'Timbaúva': 'Macia',
    'Timbauba': 'Macia',
}

HARDNESS_COLORS = {
    'Dura': (255, 0, 0),    # Vermelho
    'Média': (0, 0, 255),   # Azul
    'Macia': (0, 255, 0),   # Verde
}

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}


# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def _natsort_key(s: str):
    """Chave para ordenação natural (1, 2, 10...)."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def normalize_species_name(name):
    """Normaliza o nome da espécie removendo acentos e compatibilizando com o dicionário."""
    replacements = {
        'ú': 'u', 'í': 'i', 'á': 'a', 'à': 'a', 'â': 'a', 'ã': 'a',
        'ê': 'e', 'é': 'e',
        'ó': 'o', 'ô': 'o', 'õ': 'o',
        'ç': 'c'
    }
    normalized = name.lower()
    for old, new in replacements.items():
        normalized = normalized.replace(old, new)

    for species in SPECIES_HARDNESS.keys():
        species_norm = species.lower()
        for old, new in replacements.items():
            species_norm = species_norm.replace(old, new)
        if species_norm == normalized:
            return species
    return None

def add_hardness_bar_below_to_image(img: Image.Image, hardness: str, bar_height=40, bg=(255, 255, 255)) -> Image.Image:
    """
    Concatena uma tarja colorida (sem transparência) abaixo da imagem,
    aumentando a altura total em +bar_height.
    """
    img = img.convert('RGB')
    width, height = img.size
    pane_h = max(1, min(bar_height, 10_000))  # sanidade

    base = HARDNESS_COLORS.get(hardness, (128, 128, 128))

    out = Image.new('RGB', (width, height + pane_h), color=bg)
    out.paste(img, (0, 0))

    draw = ImageDraw.Draw(out)
    draw.rectangle([(0, height), (width, height + pane_h)], fill=base, outline=base)

    return out

def add_hardness_bar_below_file(image_path, output_path, hardness, bar_height=40):
    """Wrapper: abre a imagem, concatena a tarja abaixo e salva."""
    try:
        img = Image.open(image_path).convert('RGB')
        img = add_hardness_bar_below_to_image(img, hardness, bar_height=bar_height)
        img.save(output_path, quality=95, optimize=False)
        return True
    except Exception as e:
        print(f"      ⚠️  Erro ao aplicar tarja: {e}")
        return False

def _combine_side_by_side(img_left: Image.Image, img_right: Image.Image) -> Image.Image:
    """Redimensiona as imagens para a MESMA ALTURA (a menor delas) e concatena lado a lado."""
    h_target = min(img_left.height, img_right.height)

    def _resize_keep_aspect(im: Image.Image, h: int) -> Image.Image:
        w = int(round(im.width * (h / im.height)))
        return im.resize((w, h), Image.Resampling.LANCZOS)

    L = _resize_keep_aspect(img_left, h_target)
    R = _resize_keep_aspect(img_right, h_target)

    out = Image.new('RGB', (L.width + R.width, h_target), (255, 255, 255))
    out.paste(L, (0, 0))
    out.paste(R, (L.width, 0))
    return out

def get_image_files(directory):
    """Retorna lista de arquivos de imagem em um diretório."""
    return [
        f for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f)) and
        Path(f).suffix.lower() in IMAGE_EXTENSIONS
    ]

def _compute_split_sizes(total: int, train_ratio: float, test_ratio: float, val_ratio: float):
    """
    Calcula tamanhos de splits (train/test/val) que SOMAM exatamente 'total',
    com arredondamento estável.
    """
    exact = [total * train_ratio, total * test_ratio, total * val_ratio]
    sizes = [int(x) for x in exact]
    remainder = total - sum(sizes)
    # distribui o restante para os maiores restos fracionários
    fracs = [ex - sz for ex, sz in zip(exact, sizes)]
    order = sorted(range(3), key=lambda i: fracs[i], reverse=True)
    for i in range(remainder):
        sizes[order[i % 3]] += 1
    # retorna na ordem train, test, val
    return sizes[0], sizes[1], sizes[2]


# =============================================================================
# CRIAÇÃO DE DATASETS
# =============================================================================

def create_dataset(
    source_dir,
    output_dir,
    train_ratio=0.7,
    test_ratio=0.2,
    val_ratio=0.1,
    seed=42,
    add_bar=False,
    bar_height=40,
    max_per_class=120,
):
    """
    Cria dataset organizado em train/test/val com TOTAL por classe = max_per_class.
    GARANTE que não há imagens duplicadas entre train/test/val.

    Args:
        source_dir: Diretório com as imagens organizadas por categoria
        output_dir: Diretório de saída
        train_ratio: Proporção de treino
        test_ratio: Proporção de teste
        val_ratio: Proporção de validação
        seed: Seed para reprodutibilidade
        add_bar: Adicionar tarja de dureza abaixo (APENAS use True nas variantes *_dureza)
        bar_height: Altura da tarja em pixels
        max_per_class: TOTAL de imagens por categoria (somando todos os splits)
    """
    # Validar proporções
    if abs(train_ratio + test_ratio + val_ratio - 1.0) > 1e-6:
        raise ValueError("A soma das proporções deve ser 1.0")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    random.seed(seed)

    # Obter categorias
    categories = [
        d for d in os.listdir(source_dir)
        if os.path.isdir(os.path.join(source_dir, d))
    ]

    print(f"\n{'='*60}")
    print(f"📁 Processando: {Path(source_dir).name}  →  {Path(output_dir).name}")
    print(f"🎨 {'Com' if add_bar else 'Sem'} tarja de dureza (concatenada abaixo)")
    print(f"🔢 Total por classe: {max_per_class}")
    print(f"✅ Garantia: SEM duplicatas entre train/test/val")
    print(f"{'='*60}")
    print(f"Categorias: {len(categories)}\n")

    stats = {'train': 0, 'test': 0, 'val': 0}

    for category in sorted(categories):
        print(f"📂 {category}")

        # Determinar dureza
        species_name = normalize_species_name(category)
        hardness = SPECIES_HARDNESS.get(species_name)

        if hardness:
            emoji = {'Dura': '🔴', 'Média': '🔵', 'Macia': '🟢'}[hardness]
            print(f"   {emoji} {hardness}")
        else:
            print(f"   ⚠️  Dureza desconhecida")

        # Obter imagens
        category_path = os.path.join(source_dir, category)
        images = get_image_files(category_path)

        if not images:
            print(f"   ⚠️  Sem imagens")
            continue

        # Ordenar e embaralhar de forma reprodutível
        images.sort(key=_natsort_key)
        random.shuffle(images)

        # Selecionar exatamente max_per_class (ou todas, se houver menos)
        original_count = len(images)
        if original_count >= max_per_class:
            selected = images[:max_per_class]
        else:
            selected = images
            print(f"   ⚠️  Classe com apenas {original_count} imagens (< {max_per_class}). Usando todas as disponíveis.")

        # Calcula quotas por split somando exatamente ao total selecionado
        n_train, n_test, n_val = _compute_split_sizes(len(selected), train_ratio, test_ratio, val_ratio)

        # GARANTE separação exclusiva (sem overlap)
        splits = {
            'train': selected[:n_train],
            'test' : selected[n_train:n_train + n_test],
            'val'  : selected[n_train + n_test: n_train + n_test + n_val],
        }

        # Verificação de duplicatas (sanidade)
        all_files = splits['train'] + splits['test'] + splits['val']
        if len(all_files) != len(set(all_files)):
            print(f"   ⚠️  AVISO: Duplicatas detectadas em {category}!")
        
        # Processar cada split
        for split_name, split_images in splits.items():
            split_dir = Path(output_dir) / split_name / category
            split_dir.mkdir(parents=True, exist_ok=True)

            for img_name in split_images:
                src = os.path.join(category_path, img_name)
                dst = split_dir / img_name

                if add_bar and hardness:
                    add_hardness_bar_below_file(src, dst, hardness, bar_height)
                else:
                    shutil.copy2(src, dst)

            stats[split_name] += len(split_images)

        print(
            f"   Total usado: {len(selected)} "
            f"| Train: {len(splits['train'])} | Test: {len(splits['test'])} | Val: {len(splits['val'])}"
        )

    # Contagem detalhada por classe
    print(f"\n{'='*60}")
    print(f"📊 RESUMO DO DATASET: {Path(output_dir).name}")
    print(f"{'='*60}")
    print(f"✅ Total: {sum(stats.values())} imagens")
    print(f"   Train: {stats['train']} | Test: {stats['test']} | Val: {stats['val']}")
    print(f"\n📋 Imagens por classe:")
    
    # Coletar contagens por classe
    class_counts = {}
    for split in ['train', 'test', 'val']:
        split_dir = Path(output_dir) / split
        if split_dir.exists():
            for category_dir in sorted(split_dir.iterdir()):
                if category_dir.is_dir():
                    category = category_dir.name
                    count = len(get_image_files(str(category_dir)))
                    if category not in class_counts:
                        class_counts[category] = {'train': 0, 'test': 0, 'val': 0, 'total': 0}
                    class_counts[category][split] = count
                    class_counts[category]['total'] += count
    
    # Mostrar tabela
    for category in sorted(class_counts.keys()):
        counts = class_counts[category]
        print(f"   {category:25} | Total: {counts['total']:3} "
              f"(Train: {counts['train']:3} | Test: {counts['test']:3} | Val: {counts['val']:3})")
    
    print(f"{'='*60}\n")


def create_paired_composite_dataset(
    tangential_ds_dir: Path,
    transversal_ds_dir: Path,
    output_dir: Path,
    add_hardness_bar: bool = False,
    bar_height: int = 40,
):
    """
    Cria dataset pareado juntando tangencial i com transversal i lado a lado.
    Assume que ambos os datasets base foram criados com o MESMO total por classe
    e com as MESMAS proporções de split, garantindo ~120 pares por classe (default).
    GARANTE que não há pares duplicados entre train/test/val.
    """
    dataset_name = output_dir.name
    bar_status = "COM tarja de dureza" if add_hardness_bar else "SEM tarja"
    
    print(f"\n{'='*60}")
    print(f"🧩 Gerando dataset pareado {bar_status}: {dataset_name}")
    print(f"   Tangencial base: {tangential_ds_dir}")
    print(f"   Transversal base: {transversal_ds_dir}")
    print(f"   Saída: {output_dir}")
    print(f"✅ Garantia: SEM duplicatas entre train/test/val")
    print(f"{'='*60}\n")

    output_dir.mkdir(parents=True, exist_ok=True)
    stats = {'train': 0, 'test': 0, 'val': 0}

    for split in ['train', 'test', 'val']:
        tang_split = tangential_ds_dir / split
        trans_split = transversal_ds_dir / split
        if not tang_split.exists() or not trans_split.exists():
            print(f"⚠️  Split '{split}' indisponível em um dos datasets base. Pulando.")
            continue

        # categorias presentes em ambos
        tang_cats = {d.name for d in tang_split.iterdir() if d.is_dir()}
        trans_cats = {d.name for d in trans_split.iterdir() if d.is_dir()}
        common_cats = sorted(tang_cats & trans_cats)

        print(f"📊 {split.capitalize()} | categorias pareadas: {len(common_cats)}")

        for category in common_cats:
            out_cat = output_dir / split / category
            out_cat.mkdir(parents=True, exist_ok=True)

            # dureza pela categoria (espécie)
            species_name = normalize_species_name(category)
            hardness = SPECIES_HARDNESS.get(species_name)

            tang_files = [f for f in os.listdir(tang_split / category)
                          if Path(f).suffix.lower() in IMAGE_EXTENSIONS]
            trans_files = [f for f in os.listdir(trans_split / category)
                           if Path(f).suffix.lower() in IMAGE_EXTENSIONS]

            tang_files.sort(key=_natsort_key)
            trans_files.sort(key=_natsort_key)

            # emparelha até o mínimo (se ambos foram criados com 120 totais/mesmas proporções,
            # teremos o número esperado de pares em cada split e 120 no total)
            # Como os datasets base já garantem separação exclusiva, os pares também serão exclusivos
            n_pairs = min(len(tang_files), len(trans_files))

            if n_pairs == 0:
                print(f"   ⚠️  {category}: sem pares válidos, pulando.")
                continue

            for i in range(n_pairs):
                tang_fp = tang_split / category / tang_files[i]
                trans_fp = trans_split / category / trans_files[i]

                try:
                    img_t = Image.open(tang_fp).convert('RGB')
                    img_r = Image.open(trans_fp).convert('RGB')

                    composite = _combine_side_by_side(img_t, img_r)

                    # aplica tarja (concatenada abaixo) no composto SE solicitado
                    if add_hardness_bar and hardness:
                        composite = add_hardness_bar_below_to_image(composite, hardness, bar_height=bar_height)

                    # nome estável e curto
                    out_name = f"pair_{i+1:05d}.jpg"
                    composite.save(out_cat / out_name, quality=95, optimize=False)

                    stats[split] += 1

                except Exception as e:
                    print(f"      ⚠️  Erro processando par {i+1} em '{category}': {e}")

    # Contagem detalhada por classe
    print(f"\n{'='*60}")
    print(f"📊 RESUMO DO DATASET: {output_dir.name}")
    print(f"{'='*60}")
    total = sum(stats.values())
    print(f"✅ Total de pares: {total}")
    print(f"   Train: {stats['train']} | Test: {stats['test']} | Val: {stats['val']}")
    
    print(f"{'='*60}\n")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Organizador de datasets de madeira com classificação de dureza',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  
  # Criar apenas Tangencial2 (sem dureza) com 120 por classe:
  python split.py --tipo tangencial2 --train 0.5 --test 0.4 --val 0.1 --max-por-classe 120
  
  # Criar apenas Transversal2 (sem dureza) com 120 por classe:
  python split.py --tipo transversal2 --train 0.5 --test 0.4 --val 0.1 --max-por-classe 120
  
  # Criar os dois simples (SEM dureza) com 120 por classe:
  python split.py --tipo ambos --train 0.5 --test 0.4 --val 0.1 --max-por-classe 120
  
  # Criar Tangencial2 com dureza (COM tarja)
  python split.py --tipo tangencial_dureza --train 0.5 --test 0.4 --val 0.1 --max-por-classe 120
  
  # Criar Transversal2 com dureza (COM tarja)
  python split.py --tipo transversal_dureza --train 0.5 --test 0.4 --val 0.1 --max-por-classe 120
  
  # Criar completo (pareado SEM tarja) após criar os dois simples:
  python split.py --tipo completo

  # Criar completo com tarja (pareado COM tarja) após criar os dois simples:
  python split.py --tipo completo_dureza

  # Criar o pacote completo (6 datasets):
  python split.py --tipo all --train 0.5 --test 0.4 --val 0.1 --max-por-classe 120
        """
    )

    parser.add_argument('--tipo', required=True,
                        choices=[
                            'tangencial2', 'transversal2',
                            'tangencial_dureza', 'transversal_dureza',
                            'ambos', 'completo', 'completo_dureza', 'all'
                        ],
                        help='Tipo de dataset a criar')
    parser.add_argument('--train', type=float, default=0.5,
                        help='Proporção treino (padrão: 0.5)')
    parser.add_argument('--test', type=float, default=0.4,
                        help='Proporção teste (padrão: 0.4)')
    parser.add_argument('--val', type=float, default=0.1,
                        help='Proporção validação (padrão: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed (padrão: 42)')
    parser.add_argument('--sem-barra', action='store_true',
                        help='(Ignorado nos tipos *_dureza) Não adicionar barra de dureza')
    parser.add_argument('--altura-barra', type=int, default=30,
                        help='Altura da barra em pixels (padrão: 30)')
    parser.add_argument('--pasta-dados', type=str, default='.',
                        help='Pasta com dados (padrão: .)')
    parser.add_argument('--max-por-classe', type=int, default=120,
                        help='TOTAL por classe (somando train/test/val) (padrão: 120)')

    args = parser.parse_args()

    # Validar
    if abs(args.train + args.test + args.val - 1.0) > 1e-6:
        print(f"❌ Erro: Proporções devem somar 1.0")
        return

    print(f"\n{'#'*60}")
    print(f"# WOOD DATASET SPLITTER")
    print(f"{'#'*60}")
    print(f"Config: {args.train*100:.0f}% train / {args.test*100:.0f}% test / {args.val*100:.0f}% val")
    print(f"Seed: {args.seed}")
    print(f"Total por classe: {args.max_por_classe}")

    # Caminhos base
    base_dir = Path(args.pasta_dados).resolve()
    tang_src = base_dir / 'Tangencial2'
    trans_src = base_dir / 'Transversal2'

    datasets = {
        'tang': base_dir / 'madeira_tangencial',
        'trans': base_dir / 'madeira_transversal',
        'tang_dur': base_dir / 'madeira_tangencial_dureza',
        'trans_dur': base_dir / 'madeira_transversal_dureza',
        'complete': base_dir / 'madeira_completo',
        'complete_dur': base_dir / 'madeira_completo_dureza',
    }

    try:
        if args.tipo == 'tangencial2':
            create_dataset(
                tang_src, datasets['tang'],
                args.train, args.test, args.val,
                args.seed, add_bar=False, bar_height=args.altura_barra,
                max_per_class=args.max_por_classe
            )

        elif args.tipo == 'transversal2':
            create_dataset(
                trans_src, datasets['trans'],
                args.train, args.test, args.val,
                args.seed, add_bar=False, bar_height=args.altura_barra,
                max_per_class=args.max_por_classe
            )

        elif args.tipo == 'tangencial_dureza':
            create_dataset(
                tang_src, datasets['tang_dur'],
                args.train, args.test, args.val,
                args.seed, add_bar=True, bar_height=args.altura_barra,
                max_per_class=args.max_por_classe
            )

        elif args.tipo == 'transversal_dureza':
            create_dataset(
                trans_src, datasets['trans_dur'],
                args.train, args.test, args.val,
                args.seed, add_bar=True, bar_height=args.altura_barra,
                max_per_class=args.max_por_classe
            )

        elif args.tipo == 'ambos':
            # Sempre SEM tarja
            create_dataset(
                tang_src, datasets['tang'],
                args.train, args.test, args.val,
                args.seed, add_bar=False, bar_height=args.altura_barra,
                max_per_class=args.max_por_classe
            )
            create_dataset(
                trans_src, datasets['trans'],
                args.train, args.test, args.val,
                args.seed, add_bar=False, bar_height=args.altura_barra,
                max_per_class=args.max_por_classe
            )

        elif args.tipo == 'completo':
            # Gera PAREADO SEM tarja
            if not datasets['tang'].exists() or not datasets['trans'].exists():
                print("❌ Erro: 'madeira_tangencial' ou 'madeira_transversal' não encontrados.")
                print("   Rode primeiro: --tipo ambos  (para criar os bases SEM dureza).")
                return
            create_paired_composite_dataset(
                datasets['tang'],
                datasets['trans'],
                datasets['complete'],
                add_hardness_bar=False,
                bar_height=args.altura_barra
            )

        elif args.tipo == 'completo_dureza':
            # Gera PAREADO COM tarja
            if not datasets['tang'].exists() or not datasets['trans'].exists():
                print("❌ Erro: 'madeira_tangencial' ou 'madeira_transversal' não encontrados.")
                print("   Rode primeiro: --tipo ambos  (para criar os bases SEM dureza).")
                return
            create_paired_composite_dataset(
                datasets['tang'],
                datasets['trans'],
                datasets['complete_dur'],
                add_hardness_bar=True,
                bar_height=args.altura_barra
            )

        elif args.tipo == 'all':
            print(f"\n{'='*60}")
            print(f"MODO ALL - 6 datasets")
            print(f"{'='*60}\n")

            steps = [
                ("1/6", "Tangencial2 (sem dureza)", tang_src, datasets['tang'], False),
                ("2/6", "Transversal2 (sem dureza)", trans_src, datasets['trans'], False),
                ("3/6", "Tangencial2 + dureza (tarja abaixo)", tang_src, datasets['tang_dur'], True),
                ("4/6", "Transversal2 + dureza (tarja abaixo)", trans_src, datasets['trans_dur'], True),
            ]

            for step, name, src, dst, bar in steps:
                print(f"\n🔹 [{step}] {name}")
                create_dataset(
                    src, dst,
                    args.train, args.test, args.val,
                    args.seed, add_bar=bar, bar_height=args.altura_barra,
                    max_per_class=args.max_por_classe
                )

            print(f"\n🔹 [5/6] Completo (pareado SEM dureza)  ← tangencial i + transversal i")
            create_paired_composite_dataset(
                datasets['tang'],
                datasets['trans'],
                datasets['complete'],
                add_hardness_bar=False,
                bar_height=args.altura_barra
            )

            print(f"\n🔹 [6/6] Completo (pareado COM tarja)  ← tangencial i + transversal i + tarja")
            create_paired_composite_dataset(
                datasets['tang'],
                datasets['trans'],
                datasets['complete_dur'],
                add_hardness_bar=True,
                bar_height=args.altura_barra  # CORRIGIDO: era args_altura_barra
            )

            print(f"\n{'#'*60}")
            print(f"# ✅ 6 DATASETS CRIADOS COM SUCESSO!")
            print(f"{'#'*60}\n")
            for i, key in enumerate(['tang', 'trans', 'tang_dur', 'trans_dur', 'complete', 'complete_dur'], 1):
                print(f"  {i}. {datasets[key].name}/")
            print()

    except Exception as e:
        print(f"\n❌ Erro: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()