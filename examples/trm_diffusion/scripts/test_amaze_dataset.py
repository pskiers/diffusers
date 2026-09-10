"""Self-consistency test for AMAZE maze and queens pass metrics across multiple files.
Verify metric's consistency against GT for test datasets
"""

import argparse
import base64
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

# Dynamiczne dodanie katalogu głównego projektu do ścieżki systemowej
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval.amaze_eval import AmazeMetrics


def decode(raw):
    """Dekoduje obraz z formatu zapisanego w Parquet do obiektu PIL.Image."""
    if raw is None or isinstance(raw, float):
        return None
    if isinstance(raw, (bytes, bytearray)):
        return Image.open(io.BytesIO(bytes(raw))).convert("RGB")
    if isinstance(raw, str):
        if raw.startswith("data:"):
            raw = raw.split(",", 1)[1]
        return Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGB")
    return None


def build_meta(row):
    """Zwraca słownik metadanych niezbędny do ewaluacji pojedynczej próbki."""
    return {
        "id": row.get("id"),
        "metadata": row.get("metadata"),
        "sample_json": row.get("sample_json"),
        "original_img": decode(row.get("original_img")),
        "m_original_img": decode(row.get("m_original_img")),
        "sol_img": decode(row.get("sol_img")),
        "mask_img": decode(row.get("mask_img")),
        "cell_map": decode(row.get("cell_map")),
    }


def pil_to_chw01(img):
    """Konwertuje obraz PIL do tensora PyTorch w formacie [C, H, W] i zakresie [0, 1]."""
    arr = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1))


def process_file(path: Path, scorers: dict, n: int) -> dict:
    """
    Przetwarza pojedynczy plik Parquet, oblicza metryki dla zadanej liczby próbek
    i zwraca słownik z uśrednionymi wynikami.
    """
    df = pd.read_parquet(path)
    total_rows = len(df)

    # Rozpoznanie zadania na podstawie ścieżki (jeśli w ścieżce jest 'queen', to queens, inaczej maze)
    task = "queens" if "queen" in str(path).lower() else "maze"
    scorer = scorers[task]

    # Określenie indeksów do przetestowania
    if n <= 0 or n >= total_rows:
        idxs = np.arange(total_rows)
    else:
        idxs = np.linspace(0, total_rows - 1, n).astype(int)

    passes, covs, viols = [], [], []

    for i in idxs:
        row = df.iloc[int(i)]
        meta = build_meta(row)
        gen = pil_to_chw01(meta["sol_img"])  # Ground Truth traktujemy jako predykcję modelu

        # Obliczenie metryk na podstawie zidentyfikowanego zadania
        if task == "maze":
            res = scorer._compute_maze_metrics(gen, meta)
        else:
            res = scorer._compute_queen_metrics(gen, meta)

        passes.append(res["pass"])
        covs.append(res["gt_cell_coverage"])
        viols.append(res["background_violation"])

    return {
        "File": str(path),
        "Task": task,
        "Total_Rows": total_rows,
        "Tested": len(idxs),
        "Mean_Pass": np.mean(passes),
        "Mean_Cov": np.mean(covs),
        "Mean_Viol": np.mean(viols),
    }


def main():
    parser = argparse.ArgumentParser(description="Test metryk GT na wielu plikach Parquet.")
    parser.add_argument("--root", type=str, default="data/amaze", help="Katalog główny z danymi.")
    parser.add_argument("--n", type=int, default=12, help="Liczba próbek z każdego pliku do testu. Wartość 0 przetwarza wszystkie wiersze.")
    args = parser.parse_args()

    root_dir = Path(args.root)
    if not root_dir.exists():
        print(f"Błąd: Katalog {root_dir} nie istnieje.")
        sys.exit(1)

    # Inicjalizacja instancji metryk tylko raz dla optymalizacji pamięci i czasu
    scorers = {
        "maze": AmazeMetrics(task="maze"),
        "queens": AmazeMetrics(task="queens")
    }

    # Wyszukiwanie plików docelowych: zawierających 'all_test' w nazwie lub będących plikiem 'maze_dataset_test.parquet'
    target_files = []
    for p in root_dir.rglob("*.parquet"):
        if "all_test" in p.name or p.name == "maze_dataset_test.parquet":
            target_files.append(p)

    if not target_files:
        print(f"Nie znaleziono plików odpowiadających wzorcom w katalogu {root_dir}.")
        sys.exit(0)

    print(f"Rozpoczęto analizę {len(target_files)} plików (Próbek na plik: {'wszystkie' if args.n == 0 else args.n})...\n")

    results = []
    for idx, path in enumerate(sorted(target_files), start=1):
        print(f"[{idx}/{len(target_files)}] Przetwarzanie: {path.name}")
        try:
            file_stats = process_file(path, scorers, args.n)
            results.append(file_stats)
        except Exception as e:
            print(f"Błąd podczas przetwarzania {path}: {e}")

    # Generowanie raportu sumarycznego
    if results:
        df_report = pd.DataFrame(results)

        # Formatowanie kolumn zmiennoprzecinkowych dla czytelności
        df_report["Mean_Pass"] = df_report["Mean_Pass"].map("{:.3f}".format)
        df_report["Mean_Cov"] = df_report["Mean_Cov"].map("{:.3f}".format)
        df_report["Mean_Viol"] = df_report["Mean_Viol"].map("{:.3f}".format)

        print("\n" + "="*110)
        print("RAPORT SUMARYCZNY METRYK GROUND TRUTH")
        print("="*110)
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 150, 'display.max_colwidth', 80):
            print(df_report.to_string(index=False))
        print("="*110)
