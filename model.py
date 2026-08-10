"""
RSNA Knee Abnormality Detection — model startowy (2.5D vision + multilingual text branch)
============================================================================

UWAGA: to jest SZKIELET pod trening/fine-tuning, uruchamiany W OSOBNYM notebooku
treningowym (z internetem, na własnym GPU lub Kaggle GPU sesji treningowej).
Finalny submission notebook (offline, <=9h) ładuje już wytrenowane wagi i tylko
robi inference — patrz inference.py.

Zgodne z OFICJALNYM schematem danych (zakładka Data, zweryfikowane):
  train.csv         : StudyInstanceUID, PatientSex, Report, + 12 etykiet 0/1
  train_series.csv   : StudyInstanceUID, SeriesInstanceUID, Fluid_Sensitive,
                        Fat_Suppression, Anatomical_Plane (Sagittal/Coronal/Axial)
  train_series/<StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm

POTWIERDZONE OFICJALNIE: kolumna `Report` istnieje TYLKO w train.csv, NIE w
test.csv. Text branch jest więc użyteczny wyłącznie w treningu (dodatkowy
sygnał / destylacja dla studiów bez gold labels) — na inference model MUSI
działać wyłącznie na obrazie (use_text_branch bez report_embeddings, patrz
forward()).

Best public score per backbone (zakładka Models, realne wyniki społeczności):
  DINOv2-small: 0.906  |  DINOv2-base: 0.861  |  DINOv2-large: 0.5 (przeuczony)
  -> stąd domyślny wybór dinov2_vits14 (small) poniżej.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# 12 celów z opisu konkursu — kolejność MUSI się zgadzać z submission.csv
TARGET_COLUMNS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]
NUM_TARGETS = len(TARGET_COLUMNS)


# ---------------------------------------------------------------------------
# 1. VISION BRANCH — 2.5D per-sekwencja encoder z DINOv2 + attention pooling
# ---------------------------------------------------------------------------

class SliceEncoder(nn.Module):
    """
    Koduje pojedynczy 2.5D "obrazek" (kilka sąsiednich slice'ów jako pseudo-RGB)
    przy pomocy DINOv2-Small. Wagi pretrenowane, fine-tunowane end-to-end
    (ewentualnie z niższym LR niż head, patrz optimizer w train.py).
    """

    def __init__(self, backbone_name: str = "dinov2_vits14", freeze_backbone: bool = False):
        super().__init__()
        # DINOv2 przez torch.hub — w środowisku treningowym z internetem.
        # W notebooku offline: wagi wcześniej pobrane i zapisane jako Kaggle Dataset,
        # wczytywane lokalnie (torch.hub.load(..., source="local") albo state_dict).
        self.backbone = torch.hub.load("facebookresearch/dinov2", backbone_name)
        self.embed_dim = self.backbone.embed_dim  # 384 dla ViT-S/14

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, 224, 224) — 3 sąsiednie slice'y jako pseudo-RGB kanały
        feats = self.backbone(x)  # (B, embed_dim) — CLS token / pooled output
        return feats


class AttentionPool(nn.Module):
    """
    Agreguje embeddingi wielu slice'ów/sekwencji w jeden wektor per badanie,
    ucząc się wag istotności zamiast prostego mean-poolingu.
    Osobna projekcja attention per target pozwala modelowi "patrzeć" na inne
    slice'y przy szukaniu ACL niż przy szukaniu wysięku.
    """

    def __init__(self, embed_dim: int, num_targets: int = NUM_TARGETS):
        super().__init__()
        self.num_targets = num_targets
        # (embed_dim -> num_targets) skalarnych wag uwagi per element sekwencji
        self.attn = nn.Linear(embed_dim, num_targets)

    def forward(self, feats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        feats: (B, N, embed_dim)  — N = liczba slice'ów/sekwencji w badaniu (padded)
        mask:  (B, N) bool        — True = realny element, False = padding
        return: (B, num_targets, embed_dim) — osobny zagregowany wektor per target
        """
        scores = self.attn(feats)  # (B, N, num_targets)
        scores = scores.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        weights = F.softmax(scores, dim=1)  # (B, N, num_targets)
        # (B, N, num_targets, 1) * (B, N, 1, embed_dim) -> sum over N
        pooled = torch.einsum("bnt,bnd->btd", weights, feats)
        return pooled  # (B, num_targets, embed_dim)


class VisionBranch(nn.Module):
    """
    Pełna gałąź wizyjna: koduje wszystkie 2.5D "obrazki" ze wszystkich sekwencji
    jednego badania, potem agreguje attention-poolingiem per target.

    Rozszerzenie do rozważenia (na bazie oficjalnych metadanych w
    train_series.csv): dodać embedding pozycyjny/kategorialny per element
    sekwencji na bazie Anatomical_Plane (Sagittal/Coronal/Axial) i
    Fluid_Sensitive/Fat_Suppression, dodawany do feats przed AttentionPool.
    To da modelowi jawny sygnał "jaki to typ sekwencji", zamiast wymuszać
    naukę tego wyłącznie z pikseli. Pominięte w tym szkielecie dla
    czytelności — TODO przy rozbudowie w dataset.py + tu w forward().
    """

    def __init__(self, backbone_name: str = "dinov2_vits14"):
        super().__init__()
        self.slice_encoder = SliceEncoder(backbone_name)
        self.pool = AttentionPool(self.slice_encoder.embed_dim, NUM_TARGETS)
        self.out_dim = self.slice_encoder.embed_dim

    def forward(self, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        images: (B, N, 3, 224, 224) — N = maks. liczba slice'ów w batchu (padded)
        mask:   (B, N) bool
        return: (B, num_targets, embed_dim)
        """
        B, N, C, H, W = images.shape
        flat = images.view(B * N, C, H, W)
        feats = self.slice_encoder(flat)          # (B*N, embed_dim)
        feats = feats.view(B, N, -1)               # (B, N, embed_dim)
        pooled = self.pool(feats, mask)             # (B, num_targets, embed_dim)
        return pooled


# ---------------------------------------------------------------------------
# 2. TEXT BRANCH — mały multilingual sentence encoder (NIE duży LLM!)
# ---------------------------------------------------------------------------

class TextBranch(nn.Module):
    """
    Koduje raport radiologiczny (12 języków) przy pomocy małego, zamrożonego
    (lub lekko fine-tunowanego) multilingual encodera zdań.
    Rekomendowany checkpoint: 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    (~470MB, mieści się łatwo jako Kaggle Dataset, szybki inference).

    WAŻNE: cały model musi działać LOKALNIE w notebooku (wagi wgrane jako
    Kaggle Dataset). Zgodnie z regulaminem konkursu tekst raportu NIE MOŻE
    być wysyłany do żadnego hostowanego API LLM (reguła Data Security).

    Jeśli raporty nie są dostępne w zbiorze testowym (bardzo prawdopodobne —
    to odzwierciedlałoby realny scenariusz kliniczny), ta gałąź jest używana
    TYLKO w treningu, np. do:
      a) dodatkowego sygnału multimodalnego przy uczeniu,
      b) destylacji: model tekstowy generuje "miękkie" etykiety pomocnicze,
         które trenują lepiej sam vision branch.
    """

    def __init__(self, hidden_dim: int = 384, freeze: bool = True):
        super().__init__()
        # Placeholder — w treningowym środowisku:
        # from transformers import AutoModel, AutoTokenizer
        # self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        # self.encoder = AutoModel.from_pretrained(MODEL_PATH)
        self.hidden_dim = hidden_dim
        self.encoder = nn.Identity()  # TODO: podmień na realny multilingual encoder
        self.freeze = freeze

    def forward(self, report_embeddings: torch.Tensor) -> torch.Tensor:
        # Zakładamy, że tokenizacja/embedding zrobione wcześniej (cache'owane —
        # patrz plan działania, etap 3: "Reading once, training many times")
        # report_embeddings: (B, hidden_dim)
        return report_embeddings


# ---------------------------------------------------------------------------
# 3. FUZJA + GŁOWY KLASYFIKACYJNE
# ---------------------------------------------------------------------------

class KneeMultimodalModel(nn.Module):
    """
    Pełny model: vision branch (zawsze) + opcjonalny text branch (tylko trening,
    jeśli raporty niedostępne w teście) -> fuzja -> 12 niezależnych head'ów sigmoid.
    """

    def __init__(
        self,
        backbone_name: str = "dinov2_vits14",
        text_hidden_dim: int = 384,
        use_text_branch: bool = True,
        fusion_hidden: int = 256,
    ):
        super().__init__()
        self.vision = VisionBranch(backbone_name)
        self.use_text_branch = use_text_branch

        vision_dim = self.vision.out_dim
        fusion_in = vision_dim + (text_hidden_dim if use_text_branch else 0)

        if use_text_branch:
            self.text = TextBranch(hidden_dim=text_hidden_dim)

        # Osobna mała MLP-head per target, na bazie per-target pooled vision feats
        # (+ współdzielony tekst, jeśli obecny)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, fusion_hidden),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.classifier = nn.Linear(fusion_hidden, 1)  # aplikowany per target niezależnie

    def forward(
        self,
        images: torch.Tensor,          # (B, N, 3, 224, 224)
        image_mask: torch.Tensor,      # (B, N) bool
        report_embeddings: torch.Tensor = None,  # (B, text_hidden_dim) lub None
    ) -> torch.Tensor:
        vision_feats = self.vision(images, image_mask)  # (B, num_targets, vision_dim)
        B, T, _ = vision_feats.shape

        if self.use_text_branch and report_embeddings is not None:
            text_feats = self.text(report_embeddings)          # (B, text_hidden_dim)
            text_feats = text_feats.unsqueeze(1).expand(-1, T, -1)  # broadcast per target
            fused = torch.cat([vision_feats, text_feats], dim=-1)
        else:
            fused = vision_feats

        h = self.fusion(fused)              # (B, num_targets, fusion_hidden)
        logits = self.classifier(h).squeeze(-1)  # (B, num_targets)
        return logits  # surowe logity — sigmoid aplikowany w loss / na inference


# ---------------------------------------------------------------------------
# 4. LOSS
# ---------------------------------------------------------------------------

def compute_loss(logits: torch.Tensor, targets: torch.Tensor, pos_weight: torch.Tensor = None):
    """
    logits:  (B, num_targets) — surowe logity z modelu
    targets: (B, num_targets) — etykiety 0/1 (lub soft labels 0..1 przy destylacji)
    pos_weight: (num_targets,) — wagi dla rzadkich klas (np. Fracture, Baker's),
                policzone z train setu jako neg_count / pos_count per target.
    """
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)


if __name__ == "__main__":
    # Szybki smoke test kształtów tensorów (bez realnego DINOv2 — tu tylko sanity check
    # logiki attention poolingu i fuzji na losowych danych).
    torch.manual_seed(0)

    B, N, embed_dim, text_dim = 2, 5, 384, 384

    class DummyVision(nn.Module):
        def __init__(self):
            super().__init__()
            self.out_dim = embed_dim
            self.pool = AttentionPool(embed_dim, NUM_TARGETS)

        def forward(self, images, mask):
            feats = torch.randn(images.shape[0], images.shape[1], embed_dim)
            return self.pool(feats, mask)

    # Budujemy model "ręcznie" z dummy vision branch, żeby test nie próbował
    # ściągać prawdziwych wag DINOv2 z internetu (w tym środowisku sandbox brak
    # dostępu do github.com dla torch.hub; w Twoim notebooku treningowym z
    # internetem torch.hub.load zadziała normalnie).
    model = KneeMultimodalModel.__new__(KneeMultimodalModel)
    nn.Module.__init__(model)
    model.vision = DummyVision()
    model.use_text_branch = True
    model.text = TextBranch(hidden_dim=text_dim)
    model.fusion = nn.Sequential(
        nn.Linear(embed_dim + text_dim, 256), nn.GELU(), nn.Dropout(0.2)
    )
    model.classifier = nn.Linear(256, 1)

    images = torch.randn(B, N, 3, 224, 224)
    mask = torch.ones(B, N, dtype=torch.bool)
    mask[0, 3:] = False  # pierwsze badanie ma tylko 3 realne slice'y, reszta padding
    report_emb = torch.randn(B, text_dim)

    logits = model(images, mask, report_emb)
    print("logits shape:", logits.shape)  # oczekiwane: (2, 12)
    assert logits.shape == (B, NUM_TARGETS)

    targets = torch.randint(0, 2, (B, NUM_TARGETS)).float()
    loss = compute_loss(logits, targets)
    print("loss:", loss.item())
    print("OK — smoke test przeszedł")
