# MSM-semester-work-1

## Požadavky

- Python 3.13
- knihovny `numpy`, `sympy`, `matplotlib`

## Instalace

```bash
python3.13 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Alternativa přes [uv](https://docs.astral.sh/uv/) (verze Pythonu i knihoven se vezmou z `.python-version` a `uv.lock`):

```bash
uv sync
```

## Spuštění

```bash
python main.py      # nebo: uv run main.py
```

Skript vypočítá konstantu `c` sdružené hustoty, marginální hustoty a distribuční funkce včetně jejich charakteristik a zobrazí grafy (okna `matplotlib` je potřeba zavřít, aby program pokračoval).
