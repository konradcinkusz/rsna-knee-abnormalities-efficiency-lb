"""
RSNA Knee Abnormality Detection — dataset / DICOM loading
============================================================================

Zgodne z OFICJALNYM schematem danych (zakładka Data, zweryfikowane):
  train.csv         : StudyInstanceUID, PatientSex, Report, + 12 etykiet 0/1
  train_series.csv   : StudyInstanceUID, SeriesInstanceUID, Fluid_Sensitive,
                        Fat_Suppression, Anatomical_Plane
  train_series/<StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm

DICOM notes (oficjalnie potwierdzone):
  - Mieszane transfer syntaxes: uncompressed Explicit VR LE, JPEG Lossless,
    JPEG 2000, Implicit VR LE -> wymaga pydicom + backend dekompresji
    (pylibjpeg lub gdcm) zainstalowanego w środowisku.
  - Każdy plik okrojony do allowlisty 86 tagów metadanych -> NIE zakładaj
    obecności pełnego standardowego zestawu tagów; kod poniżej sprawdza
    obecność tagu przed użyciem i ma fallback.
  - Series: zwykle 20-45 slice'ów (mediana 30), długi ogon do kilkuset.

Ten moduł jest przeznaczony do URUCHAMIANIA NA KAGGLE (dane zamontowane pod
/kaggle/input/...), ale ścieżki są parametryzowane, więc działa też lokalnie
na małej próbce do debugowania.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False


TARGET_COLUMNS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]
NUM_TARGETS = len(TARGET_COLUMNS)

# Ile sąsiednich slice'ów tworzy jedno 2.5D "pseudo-RGB" okno.
WINDOW_SIZE = 3
# Docelowy rozmiar wejścia do DINOv2 (ViT-S/14, patch 14 -> 224 = 16*14).
IMG_SIZE = 224
# Maksymalna liczba 2.5D okien branych z JEDNEJ serii (dla kontroli pamięci/czasu).
MAX_WINDOWS_PER_SERIES = 8
# Maksymalna liczba serii branych z JEDNEGO studium.
MAX_SERIES_PER_STUDY = 6


@dataclass
class KneeStudyConfig:
    """Ścieżki i parametry — dopasuj root_dir do środowiska (Kaggle vs lokalnie)."""

    root_dir: Path
    train_csv: str = "train.csv"
    train_series_csv: str = "train_series.csv"
    series_dir: str = "train_series"
    img_size: int = IMG_SIZE
    window_size: int = WINDOW_SIZE
    max_windows_per_series: int = MAX_WINDOWS_PER_SERIES
    max_series_per_study: int = MAX_SERIES_PER_STUDY

    @classmethod
    def kaggle_default(cls, split: str = "train") -> "KneeStudyConfig":
        """
        Typowa ścieżka montowania danych konkursu na Kaggle:
        /kaggle/input/rsna-knee-abnormality-detection/
        Podmień 'rsna-knee-abnormality-detection' jeśli slug datasetu się różni
        (sprawdź w prawym panelu 'Add Input' na Kaggle).
        """
        base = Path("/kaggle/input/rsna-knee-abnormality-detection")
        if split == "train":
            return cls(root_dir=base, train_csv="train.csv",
                        train_series_csv="train_series.csv", series_dir="train_series")
        return cls(root_dir=base, train_csv="test.csv",
                    train_series_csv="test_series.csv", series_dir="test_series")


# ---------------------------------------------------------------------------
# 1. Wczytywanie i łączenie metadanych (CSV)
# ---------------------------------------------------------------------------

def load_metadata(cfg: KneeStudyConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Wczytuje train.csv i train_series.csv (lub test odpowiedniki).
    Zwraca (studies_df, series_df) bez modyfikacji — łączenie robimy
    świadomie w build_study_index, żeby zachować kontrolę nad brakami danych.
    """
    studies_df = pd.read_csv(cfg.root_dir / cfg.train_csv)
    series_df = pd.read_csv(cfg.root_dir / cfg.train_series_csv)
    return studies_df, series_df


def label_coverage_report(studies_df: pd.DataFrame) -> dict:
    """
    Sprawdza, ile studiów ma faktycznie wypełnione (nie-NaN) etykiety —
    to jest TWOJA weryfikacja hipotezy "small subset gold labels" z README,
    sekcja 3. Uruchom to jako pierwszy krok EDA, zanim zaplanujesz strategię
    treningu.
    """
    present_cols = [c for c in TARGET_COLUMNS if c in studies_df.columns]
    missing_cols = [c for c in TARGET_COLUMNS if c not in studies_df.columns]

    if not present_cols:
        return {"error": "Żadna z 12 oczekiwanych kolumn etykiet nie znaleziona w train.csv",
                "columns_found": list(studies_df.columns)}

    # Studium liczymy jako "gold" jeśli WSZYSTKIE 12 etykiet jest wypełnione (nie-NaN).
    has_all_labels = studies_df[present_cols].notna().all(axis=1)
    has_any_label = studies_df[present_cols].notna().any(axis=1)

    return {
        "total_studies": len(studies_df),
        "missing_label_columns": missing_cols,
        "studies_with_all_12_labels": int(has_all_labels.sum()),
        "studies_with_any_label": int(has_any_label.sum()),
        "studies_with_no_labels": int((~has_any_label).sum()),
        "per_label_positive_count": {
            col: int(studies_df[col].sum(skipna=True)) for col in present_cols
        },
        "per_label_missing_count": {
            col: int(studies_df[col].isna().sum()) for col in present_cols
        },
    }


# ---------------------------------------------------------------------------
# 2. DICOM loading — pojedyncza seria
# ---------------------------------------------------------------------------

def _sort_key_for_slice(dcm) -> float:
    """
    Kolejność warstw NIE jest gwarantowana wprost (patrz README, sekcja 4).
    Preferujemy ImagePositionPatient (rzut na oś normalną płaszczyzny) jeśli
    dostępny (może nie przetrwać filtrowania do 86 dozwolonych tagów),
    z fallbackiem na InstanceNumber, a w ostateczności na nazwę pliku.
    """
    if hasattr(dcm, "ImagePositionPatient") and dcm.ImagePositionPatient is not None:
        try:
            # Suma współrzędnych jako prosty, deterministyczny proxy porządku
            # przestrzennego — wystarczające dla ułożenia w stos, niekoniecznie
            # idealnie fizycznie poprawne dla każdej orientacji.
            return float(sum(dcm.ImagePositionPatient))
        except (TypeError, ValueError):
            pass
    if hasattr(dcm, "InstanceNumber") and dcm.InstanceNumber is not None:
        try:
            return float(dcm.InstanceNumber)
        except (TypeError, ValueError):
            pass
    return 0.0


def load_series_volume(series_path: Path, img_size: int = IMG_SIZE) -> np.ndarray:
    """
    Wczytuje wszystkie slice'y jednej serii DICOM, sortuje, normalizuje
    intensywności i przeskalowuje do (N, img_size, img_size).

    Obsługuje mieszane transfer syntaxes dzięki pydicom + zainstalowanemu
    backendowi dekompresji (pylibjpeg / gdcm — patrz requirements.txt).

    Zwraca: np.ndarray float32, kształt (N, img_size, img_size), wartości 0..1.
    """
    if not PYDICOM_AVAILABLE:
        raise ImportError(
            "pydicom nie jest zainstalowany. Na Kaggle: "
            "!pip install pydicom pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg --quiet"
        )

    dcm_files = sorted(series_path.glob("*.dcm"))
    if not dcm_files:
        return np.zeros((0, img_size, img_size), dtype=np.float32)

    loaded = []
    for fp in dcm_files:
        try:
            dcm = pydicom.dcmread(fp)
            loaded.append(dcm)
        except Exception as e:
            # Pojedynczy uszkodzony plik nie powinien wywrócić całej serii —
            # loguj i pomiń. W diagnostics.py zliczamy takie przypadki.
            print(f"[WARN] nie udało się wczytać {fp}: {e}")
            continue

    if not loaded:
        return np.zeros((0, img_size, img_size), dtype=np.float32)

    loaded.sort(key=_sort_key_for_slice)

    slices = []
    for dcm in loaded:
        try:
            arr = dcm.pixel_array.astype(np.float32)
        except Exception as e:
            print(f"[WARN] nie udało się zdekodować pixel_array: {e}")
            continue

        # VOI LUT / windowing jeśli dostępne w metadanych (poprawia kontrast
        # dla wielu sekwencji MRI); fallback na min-max jeśli tag nieobecny.
        try:
            arr = apply_voi_lut(arr, dcm).astype(np.float32)
        except Exception:
            pass

        # Normalizacja do 0..1 per-slice (proste i odporne na braki metadanych
        # WindowCenter/WindowWidth po filtrowaniu do 86 tagów).
        lo, hi = np.percentile(arr, [1, 99])
        if hi > lo:
            arr = np.clip((arr - lo) / (hi - lo), 0, 1)
        else:
            arr = np.zeros_like(arr)

        # Resize do wspólnego rozmiaru — bez zewnętrznej zależności od cv2,
        # używamy prostego resamplingu przez torch (dokładniejszy resize
        # w dataset.py produkcyjnym można podmienić na cv2.resize/INTER_AREA).
        t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        t = torch.nn.functional.interpolate(
            t, size=(img_size, img_size), mode="bilinear", align_corners=False
        )
        slices.append(t.squeeze(0).squeeze(0).numpy())

    if not slices:
        return np.zeros((0, img_size, img_size), dtype=np.float32)

    return np.stack(slices, axis=0).astype(np.float32)  # (N, H, W)


def build_2p5d_windows(volume: np.ndarray, window_size: int = WINDOW_SIZE,
                        max_windows: int = MAX_WINDOWS_PER_SERIES) -> np.ndarray:
    """
    Z wolumenu (N, H, W) buduje 2.5D okna (window_size sąsiednich slice'ów
    jako pseudo-RGB kanały): wynik (M, window_size, H, W), M <= max_windows.

    Strategia próbkowania: równomiernie rozłożone środki okien wzdłuż
    głębokości serii, żeby objąć całą anatomiczną rozciągłość bez brania
    WSZYSTKICH nakładających się okien (za drogie przy 20-45+ slice'ach
    i limicie czasowym 9h na cały test set).
    """
    n = volume.shape[0]
    half = window_size // 2

    if n == 0:
        return np.zeros((0, window_size, volume.shape[-2] if volume.ndim == 3 else IMG_SIZE,
                          volume.shape[-1] if volume.ndim == 3 else IMG_SIZE), dtype=np.float32)

    if n <= window_size:
        # Za mało slice'ów na pełne okno — powiel brzegowe, żeby dostać window_size.
        padded = np.pad(volume, ((0, max(0, window_size - n)), (0, 0), (0, 0)), mode="edge")
        return padded[np.newaxis, :window_size]

    valid_centers = np.arange(half, n - half)
    if len(valid_centers) <= max_windows:
        centers = valid_centers
    else:
        centers = np.linspace(half, n - 1 - half, max_windows).astype(int)

    windows = [volume[c - half: c + half + 1] for c in centers]
    return np.stack(windows, axis=0).astype(np.float32)  # (M, window_size, H, W)


# ---------------------------------------------------------------------------
# 3. PyTorch Dataset — poziom studium
# ---------------------------------------------------------------------------

class KneeStudyDataset(Dataset):
    """
    Jeden __getitem__ = jedno studium (StudyInstanceUID), agregujące
    wszystkie jego serie jako 2.5D okna + maskę + (opcjonalnie) etykiety
    i embedding raportu (tylko trening — patrz README sekcja 5: Report
    nieobecny w test.csv).
    """

    def __init__(
        self,
        cfg: KneeStudyConfig,
        studies_df: pd.DataFrame,
        series_df: pd.DataFrame,
        is_train: bool = True,
        report_embedder=None,
    ):
        self.cfg = cfg
        self.studies_df = studies_df.reset_index(drop=True)
        self.series_df = series_df
        self.is_train = is_train
        self.report_embedder = report_embedder  # callable(str) -> np.ndarray, patrz text branch

        # Indeks: StudyInstanceUID -> lista wierszy series_df (max_series_per_study)
        self._series_by_study = {
            uid: grp.head(cfg.max_series_per_study)
            for uid, grp in series_df.groupby("StudyInstanceUID")
        }

    def __len__(self) -> int:
        return len(self.studies_df)

    def __getitem__(self, idx: int) -> dict:
        row = self.studies_df.iloc[idx]
        study_uid = row["StudyInstanceUID"]
        series_rows = self._series_by_study.get(study_uid, pd.DataFrame())

        all_windows = []
        plane_ids = []  # 0=Sagittal, 1=Coronal, 2=Axial, wykorzystywane jako side-channel
        for _, srow in series_rows.iterrows():
            series_path = self.cfg.root_dir / self.cfg.series_dir / study_uid / srow["SeriesInstanceUID"]
            volume = load_series_volume(series_path, img_size=self.cfg.img_size)
            windows = build_2p5d_windows(volume, self.cfg.window_size, self.cfg.max_windows_per_series)
            if windows.shape[0] == 0:
                continue
            all_windows.append(windows)
            plane = str(srow.get("Anatomical_Plane", "")).lower()
            plane_id = {"sagittal": 0, "coronal": 1, "axial": 2}.get(plane, -1)
            plane_ids.extend([plane_id] * windows.shape[0])

        if all_windows:
            images = np.concatenate(all_windows, axis=0)  # (N_total, window, H, W)
        else:
            images = np.zeros((0, self.cfg.window_size, self.cfg.img_size, self.cfg.img_size),
                               dtype=np.float32)

        item = {
            "study_uid": study_uid,
            "images": torch.from_numpy(images),          # (N, window_size, H, W)
            "plane_ids": torch.tensor(plane_ids, dtype=torch.long),
            "num_windows": images.shape[0],
        }

        if self.is_train:
            labels = row[TARGET_COLUMNS].astype(float).values
            # NaN dla etykiet report-derived/brakujących -> zamieniamy na -1
            # jako sentinel "brak informacji", obsługiwany w loss (train.py)
            # przez maskę, NIE przez zamianę na 0 (to zafałszowałoby klasę
            # negatywną tam, gdzie po prostu brak adnotacji).
            labels = np.nan_to_num(labels, nan=-1.0)
            item["labels"] = torch.tensor(labels, dtype=torch.float32)

            if self.report_embedder is not None and pd.notna(row.get("Report")):
                item["report_embedding"] = torch.tensor(
                    self.report_embedder(row["Report"]), dtype=torch.float32
                )

        return item


def collate_knee_batch(batch: list[dict]) -> dict:
    """
    Custom collate: pada liczbę 2.5D okien (N) do maksimum w batchu i buduje
    maskę — zgodnie z tym, czego oczekuje AttentionPool w model.py.
    """
    max_n = max(item["num_windows"] for item in batch)
    max_n = max(max_n, 1)  # unikaj zerowego wymiaru przy pustym batchu

    B = len(batch)
    window_size = batch[0]["images"].shape[1] if batch[0]["num_windows"] > 0 else WINDOW_SIZE
    img_size = batch[0]["images"].shape[-1] if batch[0]["num_windows"] > 0 else IMG_SIZE

    images = torch.zeros(B, max_n, window_size, img_size, img_size)
    mask = torch.zeros(B, max_n, dtype=torch.bool)

    for i, item in enumerate(batch):
        n = item["num_windows"]
        if n > 0:
            images[i, :n] = item["images"]
            mask[i, :n] = True

    out = {
        "study_uid": [item["study_uid"] for item in batch],
        "images": images,   # (B, N, window_size, H, W) — model.py oczekuje 3 kanały;
                             # jeśli window_size != 3, dopasuj SliceEncoder wejście lub window_size=3
        "mask": mask,
    }

    if "labels" in batch[0]:
        out["labels"] = torch.stack([item["labels"] for item in batch])

    if "report_embedding" in batch[0]:
        out["report_embedding"] = torch.stack([
            item.get("report_embedding", torch.zeros(384)) for item in batch
        ])

    return out


if __name__ == "__main__":
    # Smoke test na syntetycznych danych (bez prawdziwych plików DICOM) —
    # weryfikuje logikę windowingu i collate, nie sam DICOM I/O.
    print("Test build_2p5d_windows...")
    vol = np.random.rand(30, 224, 224).astype(np.float32)
    windows = build_2p5d_windows(vol, window_size=3, max_windows=8)
    print("windows shape:", windows.shape)
    assert windows.shape == (8, 3, 224, 224)

    print("Test build_2p5d_windows z bardzo krótką serią (n < window_size)...")
    short_vol = np.random.rand(2, 224, 224).astype(np.float32)
    short_windows = build_2p5d_windows(short_vol, window_size=3, max_windows=8)
    print("short windows shape:", short_windows.shape)
    assert short_windows.shape == (1, 3, 224, 224)

    print("Test collate_knee_batch...")
    fake_batch = [
        {
            "study_uid": "study_A",
            "images": torch.rand(5, 3, 224, 224),
            "plane_ids": torch.zeros(5, dtype=torch.long),
            "num_windows": 5,
            "labels": torch.rand(NUM_TARGETS),
        },
        {
            "study_uid": "study_B",
            "images": torch.rand(2, 3, 224, 224),
            "plane_ids": torch.zeros(2, dtype=torch.long),
            "num_windows": 2,
            "labels": torch.rand(NUM_TARGETS),
        },
    ]
    collated = collate_knee_batch(fake_batch)
    print("collated images shape:", collated["images"].shape)
    print("collated mask shape:", collated["mask"].shape)
    assert collated["images"].shape == (2, 5, 3, 224, 224)
    assert collated["mask"].shape == (2, 5)
    assert collated["mask"][1].sum().item() == 2  # study_B miał tylko 2 realne okna

    print("Test label_coverage_report na syntetycznym train.csv...")
    fake_df = pd.DataFrame({
        "StudyInstanceUID": ["s1", "s2", "s3"],
        "ACL": [1, np.nan, 0],
        "MCL": [0, np.nan, 1],
        "Medial Meniscus": [1, np.nan, 0],
        "Lateral Meniscus": [0, np.nan, 1],
        "Medial OA": [1, np.nan, 0],
        "Lateral OA": [0, np.nan, 1],
        "PF OA": [1, np.nan, 0],
        "Effusion": [0, np.nan, 1],
        "Synovitis": [1, np.nan, 0],
        "Baker's": [0, np.nan, 1],
        "Contusion": [1, np.nan, 0],
        "Fracture": [0, np.nan, 1],
    })
    report = label_coverage_report(fake_df)
    print(report)
    assert report["studies_with_all_12_labels"] == 2
    assert report["studies_with_no_labels"] == 1

    print("\nOK — wszystkie smoke testy przeszły")
