"""Ponto de entrada: ``python main.py`` abre a interface gráfica; veja ``python main.py --help``."""
import sys

from biocofre.cli import principal

if __name__ == "__main__":
    sys.exit(principal())
