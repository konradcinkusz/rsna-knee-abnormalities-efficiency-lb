"""
RSNA Knee Abnormality Detection — pętla treningowa
============================================================================

Uruchamiane W KAGGLE NOTEBOOKU TRENINGOWYM (internet ON, długi czas życia).
NIE jest to notebook submisyjny — patrz README, sekcja 6, o rozdziale na
dwa osobne notebooki (treningowy vs submisyjny).

Spina: dataset.py (dane) + model.py (architektura) + diagnostics.py (raporty).

Kluczowe decyzje zaimplementowane tutaj:
  - GroupKFold PO STUDIUM (StudyInstanceUID), nigdy per-slice (README sekcja 11)
    -> tu każdy wiersz w train.csv to już jedno studium, więc zwykły KFold
       na poziomie wierszy train.csv jest bezpieczny (nie ma leakage
       pomiędzy foldami, bo nie dzielimy pojedynczego studium).
  - Maska -1 dla brakujących etykiet (report-derived / brak adnotacji) —
    NIE liczymy loss na tych pozycjach (patrz masked_bce_loss poniżej).
  - Po każdym foldzie generowany jest pełny raport diagnostyczny
    (diagnostics.py) gotowy do wklejenia w Claude Code.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from dataset import (
    KneeStudyConfig, KneeStudyDataset, collate_knee_batch,
    load_metadata, label_coverage_report, TARGET_COLUMNS, NUM_TARGETS,
)
from model import KneeMultimodalModel
from diagnostics import data_diagnostics_report, training_diagnostics_report


# ---------------------------------------------------------------------------
# Konfiguracja — dopasuj przed uruchomieniem na Kaggle
# ---------------------------------------------------------------------------

class TrainConfig:
    n_folds: int = 5
    folds_to_run: tuple[int, ...] = (0,)  # zacznij od jednego folda; rozszerz po weryfikacji pipeline'u
    batch_size: int = 4          # 2.5D + attention pooling na wielu oknach jest kosztowne pamięciowo
    num_epochs: int = 10
    lr_backbone: float = 1e-5    # niższy LR dla pretrenowanego DINOv2
    lr_head: float = 1e-3        # wyższy LR dla nowej fuzji + klasyfikatora
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir: Path = Path("/kaggle/working/diagnostics")
    checkpoint_dir: Path = Path("/kaggle/working/checkpoints")
    seed: int = 42


# ---------------------------------------------------------------------------
# Loss z maskowaniem brakujących etykiet (-1 sentinel z dataset.py)
# ---------------------------------------------------------------------------

def masked_bce_loss(logits: torch.Tensor, targets: torch.Tensor,
                     pos_weight: torch.Tensor = None) -> torch.Tensor:
    """
    targets zawiera -1 tam, gdzie etykieta jest nieznana (brak adnotacji /
    tylko report-derived bez ekstrakcji). Te pozycje są WYKLUCZONE z loss,
    zamiast być fałszywie traktowane jako klasa negatywna.
    """
    mask = (targets >= 0).float()
    safe_targets = targets.clamp(min=0)  # -1 -> 0, ale i tak zamaskowane
    per_element_loss = nn.functional.binary_cross_entropy_with_logits(
        logits, safe_targets, pos_weight=pos_weight, reduction="none"
    )
    masked_loss = per_element_loss * mask
    denom = mask.sum().clamp(min=1.0)
    return masked_loss.sum() / denom


# ---------------------------------------------------------------------------
# Jedna epoka
# ---------------------------------------------------------------------------

def run_epoch(model, loader, optimizer, device, pos_weight, is_train: bool) -> float:
    model.train(is_train)
    total_loss, total_count = 0.0, 0

    with torch.set_grad_enabled(is_train):
        for batch in loader:
            images = batch["images"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["labels"].to(device)
            report_emb = batch.get("report_embedding")
            if report_emb is not None:
                report_emb = report_emb.to(device)

            logits = model(images, mask, report_emb)
            loss = masked_bce_loss(logits, labels, pos_weight=pos_weight)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            total_count += images.size(0)

    return total_loss / max(total_count, 1)


@torch.no_grad()
def run_inference_for_oof(model, loader, device) -> tuple[np.ndarray, np.ndarray, list[str]]:
    model.eval()
    all_preds, all_targets, all_uids = [], [], []

    for batch in loader:
        images = batch["images"].to(device)
        mask = batch["mask"].to(device)
        report_emb = batch.get("report_embedding")
        if report_emb is not None:
            report_emb = report_emb.to(device)

        logits = model(images, mask, report_emb)
        preds = torch.sigmoid(logits).cpu().numpy()

        all_preds.append(preds)
        all_targets.append(batch["labels"].numpy())
        all_uids.extend(batch["study_uid"])

    return np.concatenate(all_preds), np.concatenate(all_targets), all_uids


# ---------------------------------------------------------------------------
# Główna pętla: EDA -> foldy -> raporty diagnostyczne
# ---------------------------------------------------------------------------

def main(cfg: TrainConfig):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    study_cfg = KneeStudyConfig.kaggle_default(split="train")
    studies_df, series_df = load_metadata(study_cfg)

    # --- Krok 1: EDA / data diagnostics — ZAWSZE najpierw ---
    print("=== Generuję data diagnostics report ===")
    data_report = data_diagnostics_report(studies_df, series_df, out_dir=cfg.out_dir)
    print(f"Gold-label coverage: {data_report['label_coverage']}")

    # pos_weight per etykieta z realnego rozkładu klas (patrz diagnostics.py)
    pos_weight_dict = data_report.get("label_balance", {}).get("recommended_pos_weight", {})
    pos_weight = torch.tensor(
        [pos_weight_dict.get(col) or 1.0 for col in TARGET_COLUMNS],
        dtype=torch.float32, device=cfg.device,
    )
    print(f"pos_weight per etykieta: {pos_weight.tolist()}")

    # --- Krok 2: split po studium (GroupKFold na poziomie wierszy = poziom studium) ---
    kf = KFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    fold_indices = list(kf.split(studies_df))

    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for fold in cfg.folds_to_run:
        print(f"\n=== FOLD {fold} ===")
        train_idx, val_idx = fold_indices[fold]
        train_df = studies_df.iloc[train_idx].reset_index(drop=True)
        val_df = studies_df.iloc[val_idx].reset_index(drop=True)

        train_ds = KneeStudyDataset(study_cfg, train_df, series_df, is_train=True)
        val_ds = KneeStudyDataset(study_cfg, val_df, series_df, is_train=True)

        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                                   collate_fn=collate_knee_batch, num_workers=2)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                                 collate_fn=collate_knee_batch, num_workers=2)

        model = KneeMultimodalModel(use_text_branch=False).to(cfg.device)
        # use_text_branch=False na starcie: najpierw zweryfikuj pipeline obrazowy
        # end-to-end (patrz README, Etap 4 planu działania), text branch dołóż
        # w kolejnej iteracji zgodnie z Etapem 5.

        optimizer = torch.optim.AdamW([
            {"params": model.vision.slice_encoder.parameters(), "lr": cfg.lr_backbone},
            {"params": model.vision.pool.parameters(), "lr": cfg.lr_head},
            {"params": model.fusion.parameters(), "lr": cfg.lr_head},
            {"params": model.classifier.parameters(), "lr": cfg.lr_head},
        ])

        epoch_history = []
        best_val_loss = float("inf")

        for epoch in range(cfg.num_epochs):
            t0 = time.time()
            train_loss = run_epoch(model, train_loader, optimizer, cfg.device, pos_weight, is_train=True)
            val_loss = run_epoch(model, val_loader, optimizer, cfg.device, pos_weight, is_train=False)
            dt = time.time() - t0

            epoch_history.append({
                "epoch": epoch, "train_loss": round(train_loss, 5),
                "val_loss": round(val_loss, 5), "seconds": round(dt, 1),
            })
            print(f"  epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} ({dt:.1f}s)")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), cfg.checkpoint_dir / f"fold{fold}_best.pt")

        # --- Krok 3: OOF predictions + pełny raport diagnostyczny per fold ---
        model.load_state_dict(torch.load(cfg.checkpoint_dir / f"fold{fold}_best.pt"))
        oof_preds, oof_targets, oof_uids = run_inference_for_oof(model, val_loader, cfg.device)

        # Sentinel -1 -> NaN, żeby training_diagnostics_report poprawnie
        # wykluczył brakujące etykiety z liczenia AUC (patrz diagnostics.py)
        oof_targets_masked = np.where(oof_targets < 0, np.nan, oof_targets)

        print(f"=== Generuję training diagnostics report dla folda {fold} ===")
        training_diagnostics_report(
            fold=fold,
            epoch_history=epoch_history,
            oof_predictions=oof_preds,
            oof_targets=oof_targets_masked,
            oof_study_uids=oof_uids,
            out_dir=cfg.out_dir,
            extra_info={
                "batch_size": cfg.batch_size,
                "lr_backbone": cfg.lr_backbone,
                "lr_head": cfg.lr_head,
                "device": cfg.device,
                "checkpoint_path": str(cfg.checkpoint_dir / f"fold{fold}_best.pt"),
            },
        )

    print(f"\nGotowe. Wszystkie raporty diagnostyczne w: {cfg.out_dir}")
    print("Skopiuj zawartość plików *_diagnostics*.md (albo użyj "
          "diagnostics.combine_reports_for_claude_code) i wklej do Claude Code.")


if __name__ == "__main__":
    main(TrainConfig())
