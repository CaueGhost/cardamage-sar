"""
main.py — Menu principal do Projeto SAR
Execute este arquivo para acessar todas as etapas do projeto.
Autor: Cauê Menezes — UNIFESP Laboratório 3
"""

import os
import subprocess
import sys

DIR = os.path.dirname(os.path.abspath(__file__))

ETAPAS = {
    "1": ("Exploração e visualização do dataset",  "datareading.py"),
    "2": ("Treinar YOLOv8",                        "yolotraining.py"),
    "3": ("Treinar Faster R-CNN",                  "faster_r_cnntraining.py"),
    "4": ("Treinar Mask R-CNN",                    "mask_r_cnntraining.py"),
    "5": ("Comparar os 3 modelos (lado a lado)",   "comparacao.py"),
}

def menu():
    print("\n" + "=" * 50)
    print("   Projeto SAR — Detecção de Danos em Carros")
    print("   Cauê Menezes — UNIFESP Laboratório 3")
    print("=" * 50)
    for key, (nome, _) in ETAPAS.items():
        print(f"  [{key}] {nome}")
    print("  [0] Sair")
    print("=" * 50)

def rodar(script):
    path = os.path.join(DIR, script)
    if not os.path.exists(path):
        print(f"\nArquivo não encontrado: {path}")
        print(f"Certifique-se de que '{script}' está na mesma pasta que main.py")
        return
    subprocess.run([sys.executable, path], cwd=DIR)

if __name__ == "__main__":
    while True:
        menu()
        escolha = input("\nEscolha uma opção: ").strip()
        if escolha == "0":
            print("\nEncerrando o projeto SAR. Até mais!\n")
            break
        elif escolha in ETAPAS:
            nome, script = ETAPAS[escolha]
            print(f"\nIniciando: {nome}...\n")
            rodar(script)
            input("\nPressione Enter para voltar ao menu...")
        else:
            print("\nOpção inválida. Tente novamente.")