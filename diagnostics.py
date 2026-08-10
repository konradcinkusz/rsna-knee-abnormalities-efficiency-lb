"""
RSNA Knee Abnormality Detection — moduł diagnostyczny
============================================================================

Cel: po KAŻDYM uruchomieniu notebooka (EDA / trening / inference) wygenerować
jeden ustrukturyzowany raport (JSON + markdown), który można w całości
wkleić do rozmowy z Claude Code do analizy/optymalizacji — zamiast
przeklejać porozrzucane printy z logów.

Trzy główne funkcje wywoływane z notebooka:
  1. data_diagnostics_report(...)   -> po EDA / przed treningiem
  2. training_diagnostics_report(...) -> po każdym foldzie/epoce treningu
  3. inference_diagnostics_report(...) -> po wygenerowaniu submission.csv

Każda zwraca dict (JSON-serializable) i opcjonalnie zapisuje pliki do
/kaggle/working/diagnostics/ (albo innej ścieżki podanej jako out_dir).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import roc_auc_score, roc_curve
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")  # bez GUI — bezpieczne w headless Kaggle
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


TARGET_COLUMNS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]


def _ensure_dir(out_dir: Optional[Path]) -> Optional[Path]:
    if out_dir is None:
        return None
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _save_json(data: dict, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# 1. DATA DIAGNOSTICS — uruchamiane raz, po wczytaniu CSV, przed treningiem
# ---------------------------------------------------------------------------

def data_diagnostics_report(
    studies_df: pd.DataFrame,
    series_df: pd.DataFrame,
    out_dir: Optional[Path] = None,
) -> dict:
    """
    Kompletny obraz danych treningowych: pokrycie etykiet, class imbalance,
    rozkład sekwencji, długość serii, języki raportów (przybliżone).

    To jest PIERWSZA rzecz, którą powinieneś uruchomić i wkleić do Claude
    Code — zanim zaczniesz cokolwiek trenować, żeby wspólnie zaplanować
    strategię pod class imbalance i weak supervision (patrz README, sekcja 3).
    """
    out_dir = _ensure_dir(out_dir)
    report = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "section": "data_diagnostics"}

    # --- Pokrycie etykiet ---
    present_cols = [c for c in TARGET_COLUMNS if c in studies_df.columns]
    if present_cols:
        has_all = studies_df[present_cols].notna().all(axis=1)
        has_any = studies_df[present_cols].notna().any(axis=1)
        report["label_coverage"] = {
            "total_studies": len(studies_df),
            "studies_with_all_12_labels": int(has_all.sum()),
            "studies_with_any_label": int(has_any.sum()),
            "studies_with_no_labels": int((~has_any).sum()),
            "fraction_gold": round(float(has_all.sum()) / max(len(studies_df), 1), 4),
        }
        # Class imbalance per etykieta — KLUCZOWE pod pos_weight w compute_loss (model.py)
        pos_counts, neg_counts, pos_weights = {}, {}, {}
        for col in present_cols:
            valid = studies_df[col].dropna()
            pos = int((valid == 1).sum())
            neg = int((valid == 0).sum())
            pos_counts[col] = pos
            neg_counts[col] = neg
            pos_weights[col] = round(neg / pos, 2) if pos > 0 else None
        report["label_balance"] = {
            "positive_count": pos_counts,
            "negative_count": neg_counts,
            "recommended_pos_weight": pos_weights,  # -> torch.tensor(...) do compute_loss()
        }
    else:
        report["label_coverage"] = {"error": "brak oczekiwanych kolumn etykiet w studies_df"}

    # --- PatientSex ---
    if "PatientSex" in studies_df.columns:
        report["patient_sex_distribution"] = studies_df["PatientSex"].value_counts(dropna=False).to_dict()

    # --- Raporty tekstowe: długość, braki (języki wymagają osobnego wykrywacza) ---
    if "Report" in studies_df.columns:
        report_lengths = studies_df["Report"].dropna().str.len()
        report["report_text_stats"] = {
            "studies_with_report": int(studies_df["Report"].notna().sum()),
            "studies_without_report": int(studies_df["Report"].isna().sum()),
            "char_length_mean": round(float(report_lengths.mean()), 1) if len(report_lengths) else None,
            "char_length_p50": round(float(report_lengths.median()), 1) if len(report_lengths) else None,
            "char_length_p95": round(float(report_lengths.quantile(0.95)), 1) if len(report_lengths) else None,
            "note": "Liczba języków NIE jest znana z góry (patrz README sekcja 8) — "
                     "uruchom osobno wykrywacz języka (langdetect/fasttext) na tej kolumnie.",
        }

    # --- Serie / sekwencje ---
    if series_df is not None and len(series_df) > 0:
        series_per_study = series_df.groupby("StudyInstanceUID").size()
        report["series_stats"] = {
            "total_series": len(series_df),
            "series_per_study_mean": round(float(series_per_study.mean()), 2),
            "series_per_study_min": int(series_per_study.min()),
            "series_per_study_max": int(series_per_study.max()),
        }
        for col in ["Fluid_Sensitive", "Fat_Suppression", "Anatomical_Plane"]:
            if col in series_df.columns:
                report["series_stats"][f"{col}_distribution"] = (
                    series_df[col].value_counts(dropna=False).to_dict()
                )

    if out_dir:
        _save_json(report, out_dir / "data_diagnostics.json")
        _write_markdown_summary(report, out_dir / "data_diagnostics.md", title="Data Diagnostics")
        if MATPLOTLIB_AVAILABLE and "label_balance" in report:
            _plot_label_balance(report["label_balance"], out_dir / "label_balance.png")

    return report


def _plot_label_balance(label_balance: dict, path: Path) -> None:
    labels = list(label_balance["positive_count"].keys())
    pos = [label_balance["positive_count"][l] for l in labels]
    neg = [label_balance["negative_count"][l] for l in labels]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    ax.bar(x, neg, label="negative (0)", color="#4C72B0")
    ax.bar(x, pos, bottom=neg, label="positive (1)", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Liczba studiów")
    ax.set_title("Rozkład klas per etykieta (tylko studia z wypełnioną etykietą)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. TRAINING DIAGNOSTICS — uruchamiane po każdym foldzie / na końcu treningu
# ---------------------------------------------------------------------------

def training_diagnostics_report(
    fold: int,
    epoch_history: list[dict],
    oof_predictions: np.ndarray,
    oof_targets: np.ndarray,
    oof_study_uids: list[str],
    target_columns: list[str] = TARGET_COLUMNS,
    out_dir: Optional[Path] = None,
    extra_info: Optional[dict] = None,
) -> dict:
    """
    epoch_history: lista dictów [{"epoch": 0, "train_loss": .., "val_loss": ..}, ...]
    oof_predictions: (N, 12) array sigmoidowanych prawdopodobieństw na out-of-fold
    oof_targets:     (N, 12) array etykiet 0/1 (użyj tylko wierszy z gold labels —
                      przefiltruj -1/NaN sentinel PRZED wywołaniem tej funkcji)
    oof_study_uids:  lista StudyInstanceUID odpowiadająca wierszom powyżej

    To jest raport, który chcesz wkleić do Claude Code po każdym foldzie —
    pokazuje dokładnie które z 12 etykiet są słabe (nie tylko macro AUC),
    rozkład predykcji (wykrywa collapse modelu) i krzywą uczenia.
    """
    out_dir = _ensure_dir(out_dir)
    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "section": "training_diagnostics",
        "fold": fold,
    }

    # --- Krzywa uczenia ---
    report["epoch_history"] = epoch_history
    if epoch_history:
        losses = [e.get("val_loss") for e in epoch_history if e.get("val_loss") is not None]
        report["learning_curve_summary"] = {
            "num_epochs": len(epoch_history),
            "best_val_loss": round(min(losses), 5) if losses else None,
            "best_val_loss_epoch": int(np.argmin(losses)) if losses else None,
            "final_val_loss": round(losses[-1], 5) if losses else None,
            "val_loss_improving": (losses[-1] < losses[0]) if len(losses) >= 2 else None,
        }

    # --- AUC per etykieta (nie tylko macro) ---
    per_label_auc = {}
    per_label_pred_stats = {}
    if SKLEARN_AVAILABLE:
        for i, col in enumerate(target_columns):
            y_true = oof_targets[:, i]
            y_pred = oof_predictions[:, i]
            valid = ~np.isnan(y_true)
            y_true_valid, y_pred_valid = y_true[valid], y_pred[valid]

            per_label_pred_stats[col] = {
                "n_valid": int(valid.sum()),
                "n_positive": int((y_true_valid == 1).sum()),
                "pred_mean": round(float(y_pred_valid.mean()), 4) if len(y_pred_valid) else None,
                "pred_std": round(float(y_pred_valid.std()), 4) if len(y_pred_valid) else None,
            }

            if len(np.unique(y_true_valid)) < 2:
                per_label_auc[col] = None  # AUC niezdefiniowane przy jednej klasie w OOF
                continue
            try:
                per_label_auc[col] = round(float(roc_auc_score(y_true_valid, y_pred_valid)), 4)
            except ValueError:
                per_label_auc[col] = None

    valid_aucs = [v for v in per_label_auc.values() if v is not None]
    report["per_label_auc"] = per_label_auc
    report["per_label_prediction_stats"] = per_label_pred_stats
    report["macro_auc"] = round(float(np.mean(valid_aucs)), 4) if valid_aucs else None
    report["worst_labels"] = sorted(
        [(k, v) for k, v in per_label_auc.items() if v is not None],
        key=lambda kv: kv[1]
    )[:3]
    report["best_labels"] = sorted(
        [(k, v) for k, v in per_label_auc.items() if v is not None],
        key=lambda kv: -kv[1]
    )[:3]

    # --- Collapse detection: czy model przewiduje prawie zawsze ~0.5? ---
    overall_pred_std = float(np.nanstd(oof_predictions))
    report["collapse_warning"] = overall_pred_std < 0.05
    report["overall_prediction_std"] = round(overall_pred_std, 4)
    if report["collapse_warning"]:
        report["collapse_note"] = (
            "Odchylenie std predykcji < 0.05 dla całego OOF — silny sygnał "
            "prediction collapse (model przewiduje prawie stałą wartość ~0.5 "
            "niezależnie od wejścia). Publiczny baseline V02 miał ten dokładny "
            "problem — patrz README sekcja 3/9, warto sprawdzić LR, "
            "inicjalizację head'a i wagi pos_weight."
        )

    if extra_info:
        report["extra_info"] = extra_info

    if out_dir:
        _save_json(report, out_dir / f"training_diagnostics_fold{fold}.json")
        _write_markdown_summary(report, out_dir / f"training_diagnostics_fold{fold}.md",
                                 title=f"Training Diagnostics — Fold {fold}")
        if MATPLOTLIB_AVAILABLE:
            _plot_learning_curve(epoch_history, out_dir / f"learning_curve_fold{fold}.png")
            _plot_per_label_auc(per_label_auc, out_dir / f"per_label_auc_fold{fold}.png")
            _plot_prediction_histograms(oof_predictions, target_columns,
                                         out_dir / f"pred_histograms_fold{fold}.png")

    return report


def _plot_learning_curve(epoch_history: list[dict], path: Path) -> None:
    if not epoch_history:
        return
    epochs = [e.get("epoch", i) for i, e in enumerate(epoch_history)]
    train_loss = [e.get("train_loss") for e in epoch_history]
    val_loss = [e.get("val_loss") for e in epoch_history]

    fig, ax = plt.subplots(figsize=(7, 4))
    if any(v is not None for v in train_loss):
        ax.plot(epochs, train_loss, label="train_loss", marker="o")
    if any(v is not None for v in val_loss):
        ax.plot(epochs, val_loss, label="val_loss", marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (BCE)")
    ax.set_title("Krzywa uczenia")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_per_label_auc(per_label_auc: dict, path: Path) -> None:
    labels = [k for k, v in per_label_auc.items() if v is not None]
    values = [per_label_auc[k] for k in labels]
    if not labels:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#C44E52" if v < 0.6 else "#DD8452" if v < 0.75 else "#55A868" for v in values]
    ax.bar(labels, values, color=colors)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="random (0.5)")
    ax.set_ylim(0, 1)
    ax.set_ylabel("ROC AUC")
    ax.set_title("AUC per etykieta (OOF) — czerwony <0.6, pomarańczowy <0.75, zielony ≥0.75")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_prediction_histograms(oof_predictions: np.ndarray, target_columns: list[str], path: Path) -> None:
    n = len(target_columns)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.5))
    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(target_columns):
        ax = axes[i]
        preds = oof_predictions[:, i]
        preds = preds[~np.isnan(preds)]
        ax.hist(preds, bins=20, range=(0, 1), color="#4C72B0")
        ax.set_title(col, fontsize=9)
        ax.set_xlim(0, 1)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Rozkład predykcji per etykieta (wykrywanie collapse)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. INFERENCE DIAGNOSTICS — po wygenerowaniu submission.csv
# ---------------------------------------------------------------------------

def inference_diagnostics_report(
    submission_df: pd.DataFrame,
    sample_submission_df: pd.DataFrame,
    total_inference_seconds: float,
    num_test_studies: int,
    target_columns: list[str] = TARGET_COLUMNS,
    time_budget_seconds: float = 9 * 3600,
    out_dir: Optional[Path] = None,
) -> dict:
    """
    Weryfikacja PRZED faktycznym submitem: format zgodny z sample_submission,
    brak NaN, wartości w [0,1], oraz ekstrapolacja czasu na pełny ~1300-studyjny
    zbiór testowy pod limit 9h (patrz README sekcja 6).
    """
    out_dir = _ensure_dir(out_dir)
    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "section": "inference_diagnostics",
    }

    # --- Zgodność formatu ---
    expected_cols = set(sample_submission_df.columns)
    actual_cols = set(submission_df.columns)
    report["format_check"] = {
        "columns_match": expected_cols == actual_cols,
        "missing_columns": sorted(expected_cols - actual_cols),
        "unexpected_columns": sorted(actual_cols - expected_cols),
        "row_count": len(submission_df),
    }

    # --- Braki i zakres wartości ---
    value_issues = {}
    for col in target_columns:
        if col not in submission_df.columns:
            continue
        series = submission_df[col]
        value_issues[col] = {
            "n_nan": int(series.isna().sum()),
            "min": round(float(series.min()), 4) if series.notna().any() else None,
            "max": round(float(series.max()), 4) if series.notna().any() else None,
            "out_of_range_count": int(((series < 0) | (series > 1)).sum()),
        }
    report["value_checks"] = value_issues
    report["has_any_nan"] = any(v["n_nan"] > 0 for v in value_issues.values())
    report["has_any_out_of_range"] = any(v["out_of_range_count"] > 0 for v in value_issues.values())

    # --- Budżet czasowy (KRYTYCZNE pod limit 9h code competition) ---
    seconds_per_study = total_inference_seconds / max(num_test_studies, 1)
    full_test_estimate = seconds_per_study * 1300  # oficjalnie potwierdzone ~1300 studiów
    report["timing"] = {
        "measured_studies": num_test_studies,
        "measured_total_seconds": round(total_inference_seconds, 1),
        "seconds_per_study": round(seconds_per_study, 3),
        "estimated_full_test_seconds": round(full_test_estimate, 1),
        "estimated_full_test_hours": round(full_test_estimate / 3600, 2),
        "time_budget_hours": round(time_budget_seconds / 3600, 1),
        "within_budget": full_test_estimate < time_budget_seconds * 0.8,  # 20% margines bezpieczeństwa
        "safety_margin_note": (
            "within_budget uwzględnia 20% margines bezpieczeństwa względem "
            "twardego limitu 9h — nie licz na dotarcie do samej granicy."
        ),
    }

    if out_dir:
        _save_json(report, out_dir / "inference_diagnostics.json")
        _write_markdown_summary(report, out_dir / "inference_diagnostics.md",
                                 title="Inference Diagnostics")

    return report


# ---------------------------------------------------------------------------
# 4. Markdown summary — jeden plik gotowy do wklejenia w rozmowę z Claude Code
# ---------------------------------------------------------------------------

def _write_markdown_summary(report: dict, path: Path, title: str) -> None:
    lines = [f"# {title}", "", f"_Wygenerowano: {report.get('generated_at', '?')}_", ""]
    lines.append("```json")
    lines.append(json.dumps(report, indent=2, default=str))
    lines.append("```")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def combine_reports_for_claude_code(out_dir: Path, output_path: Optional[Path] = None) -> str:
    """
    Skanuje out_dir w poszukiwaniu wszystkich *_diagnostics*.json wygenerowanych
    w trakcie runa i skleja je w JEDEN tekst — dokładnie to, co chcesz wkleić
    do Claude Code po zakończeniu 'Run All' na Kaggle.
    """
    out_dir = Path(out_dir)
    json_files = sorted(out_dir.glob("*diagnostics*.json"))

    combined = ["# Skonsolidowany raport diagnostyczny — RSNA Knee", ""]
    for fp in json_files:
        with open(fp) as f:
            data = json.load(f)
        combined.append(f"## {fp.name}")
        combined.append("```json")
        combined.append(json.dumps(data, indent=2, default=str))
        combined.append("```")
        combined.append("")

    text = "\n".join(combined)
    if output_path:
        with open(output_path, "w") as f:
            f.write(text)
    return text


if __name__ == "__main__":
    # Smoke test wszystkich trzech raportów na syntetycznych danych.
    print("Test data_diagnostics_report...")
    fake_studies = pd.DataFrame({
        "StudyInstanceUID": [f"s{i}" for i in range(10)],
        "PatientSex": ["Male", "Female"] * 5,
        "Report": ["some report text " * (i + 1) for i in range(10)],
        **{col: [1, 0] * 5 for col in TARGET_COLUMNS},
    })
    fake_series = pd.DataFrame({
        "StudyInstanceUID": [f"s{i}" for i in range(10) for _ in range(3)],
        "SeriesInstanceUID": [f"ser{i}_{j}" for i in range(10) for j in range(3)],
        "Fluid_Sensitive": [1, 0, 1] * 10,
        "Fat_Suppression": [0, 1, 0] * 10,
        "Anatomical_Plane": ["Sagittal", "Coronal", "Axial"] * 10,
    })
    dr = data_diagnostics_report(fake_studies, fake_series, out_dir=Path("/tmp/diag_test"))
    assert dr["label_coverage"]["total_studies"] == 10
    print("  OK, macro fields:", list(dr.keys()))

    print("Test training_diagnostics_report...")
    n_oof = 50
    rng = np.random.default_rng(0)
    fake_preds = rng.uniform(0, 1, size=(n_oof, len(TARGET_COLUMNS)))
    fake_targets = rng.integers(0, 2, size=(n_oof, len(TARGET_COLUMNS))).astype(float)
    fake_history = [{"epoch": e, "train_loss": 0.7 - 0.05 * e, "val_loss": 0.72 - 0.04 * e} for e in range(5)]
    tr = training_diagnostics_report(
        fold=0,
        epoch_history=fake_history,
        oof_predictions=fake_preds,
        oof_targets=fake_targets,
        oof_study_uids=[f"s{i}" for i in range(n_oof)],
        out_dir=Path("/tmp/diag_test"),
    )
    assert "macro_auc" in tr
    print("  OK, macro_auc:", tr["macro_auc"], "collapse_warning:", tr["collapse_warning"])

    print("Test inference_diagnostics_report...")
    fake_sub = pd.DataFrame({"StudyInstanceUID": [f"t{i}" for i in range(20)],
                              **{col: rng.uniform(0, 1, 20) for col in TARGET_COLUMNS}})
    fake_sample_sub = fake_sub.copy()
    ir = inference_diagnostics_report(
        submission_df=fake_sub,
        sample_submission_df=fake_sample_sub,
        total_inference_seconds=120.0,
        num_test_studies=20,
        out_dir=Path("/tmp/diag_test"),
    )
    assert ir["format_check"]["columns_match"] is True
    print("  OK, timing estimate (h):", ir["timing"]["estimated_full_test_hours"],
          "within_budget:", ir["timing"]["within_budget"])

    print("Test combine_reports_for_claude_code...")
    combined_text = combine_reports_for_claude_code(Path("/tmp/diag_test"))
    assert "data_diagnostics" in combined_text
    assert "training_diagnostics" in combined_text
    print("  OK, combined length:", len(combined_text), "znaków")

    print("\nOK — wszystkie smoke testy diagnostics.py przeszły")
