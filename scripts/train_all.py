#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import sys
from pathlib import Path

def main():
    """Encontra e executa todos os configs .yaml nas subpastas de 'config'"""
    
    # O diretório 'config' está um nível acima, no mesmo nível que 'scripts'
    project_root = Path(__file__).resolve().parent.parent
    config_root = project_root / 'config'
    script_path = project_root / 'src' / 'train.py'

    if not script_path.exists():
        raise FileNotFoundError(f"Script de treino não encontrado em: {script_path}")

    # Encontra todos os arquivos .yaml em subdiretórios de 'config'
    config_files = sorted(list(config_root.glob('*/*.yaml')))

    if not config_files:
        print(f"Nenhum arquivo de configuração .yaml encontrado em subpastas de '{config_root}'.")
        return

    print(f"✅ Encontrados {len(config_files)} arquivos de configuração para processar.")

    for i, config_file in enumerate(config_files, 1):
        print(f"\n{'-'*25} EXECUTANDO {i}/{len(config_files)} {'-'*25}")
        print(f"Config: {config_file.relative_to(project_root)}")
        
        command = [
            sys.executable,
            str(script_path),
            '--config',
            str(config_file)
        ]
        
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ FALHA ao executar com a configuração: {config_file.name}. Código: {e.returncode}")
            print("Continuando para o próximo...")
        except KeyboardInterrupt:
            print("\n🛑 Processo interrompido pelo usuário.")
            sys.exit(1)

    print(f"\n🎉 Todos os {len(config_files)} treinos foram concluídos.")

if __name__ == "__main__":
    main()
