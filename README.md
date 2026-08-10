# RSNA Knee Abnormality Detection — dokumentacja projektu

> Towarzyszy plikowi `model.py` (szkielet architektury 2.5D + multilingual text branch).
> Stan wiedzy: 10 sierpnia 2026. Sekcje 1–2, 4–7, 9–12 oparte na **oficjalnej
> zakładce Data i Rules** konkursu (dostęp potwierdzony przez użytkownika).
> Sekcja 3 (weak supervision) częściowo nadal opiera się na niezależnym audycie
> społeczności — oficjalny opis potwierdza sam fakt ("only a small subset of
> training studies carry per-condition labels"), ale nie podaje dokładnej
> liczby 58/4407 wprost, więc tę liczbę traktuj jako punkt wyjścia do
> weryfikacji, nie pewnik.

---

## 1. Czym jest ten konkurs — jednym akapitem

RSNA (Radiological Society of North America) organizuje konkurs ML, w którym
trzeba zbudować model wykrywający **12 klinicznie istotnych nieprawidłowości
kolana** na podstawie **badania MRI + towarzyszącego mu oryginalnego raportu
radiologicznego**. To pierwszy konkurs RSNA łączący obraz i tekst raportu
zarówno w treningu, jak i ewaluacji, i pierwszy dotyczący MRI układu
mięśniowo-szkieletowego. Pula nagród: **77 000 USD**, w tym pierwsza w historii
tych konkursów osobna pula za **efektywność obliczeniową** modelu.

**Dlaczego to ważne klinicznie**: osteoartroza dotyka ~654 mln ludzi na
świecie, ostre urazy kolana to 15–40% wszystkich urazów sportowych, a dostęp
do radiologów specjalizujących się w układzie mięśniowo-szkieletowym jest
ograniczony — stąd potrzeba narzędzi wspomagających diagnozę.

---

## 2. Definicja zadania ML

### Typ problemu
**Multi-label binary classification** na poziomie całego badania (study-level),
nie pojedynczego obrazu. Kolano może mieć jednocześnie kilka problemów
(np. zerwanie ACL + wysięk + OA), więc każda z 12 etykiet jest przewidywana
**niezależnie** (sigmoid), nie jako jedna klasa spośród wielu (softmax).

### 12 celów (targets)
```
ACL, MCL, Medial Meniscus, Lateral Meniscus,
Medial OA, Lateral OA, PF OA, Effusion,
Synovitis, Baker's, Contusion, Fracture
```

| Skrót | Co oznacza klinicznie |
|---|---|
| ACL / MCL | zerwanie/uszkodzenie więzadła krzyżowego przedniego / pobocznego piszczelowego |
| Medial / Lateral Meniscus | uszkodzenie łąkotki przyśrodkowej / bocznej |
| Medial / Lateral / PF OA | zwyrodnienie (osteoarthritis) przedziału przyśrodkowego / bocznego / rzepkowo-udowego |
| Effusion | wysięk stawowy |
| Synovitis | zapalenie błony maziowej |
| Baker's | torbiel Bakera (podkolanowa) |
| Contusion | stłuczenie kości (bone bruise / bone marrow lesion) |
| Fracture | złamanie |

### Wejście
- **Obraz**: badanie MRI kolana w formacie **DICOM**, złożone z kilku
  **sekwencji** (np. sagittal PD, coronal T1, axial T2 fat-sat), z których
  każda to stos kilkunastu–kilkudziesięciu 2D warstw (slice'ów).
- **Tekst**: oryginalny raport radiologiczny towarzyszący badaniu, w wielu
  językach (patrz sekcja 8 — liczba języków jest niespójna między źródłami:
  8/9/12 w zależności od materiału).

### Wyjście
Dla każdego `StudyInstanceUID` w zbiorze testowym — 12 liczb (confidence
score 0–1), po jednej na etykietę:

```csv
StudyInstanceUID,ACL,MCL,Medial Meniscus,Lateral Meniscus,Medial OA,Lateral OA,PF OA,Effusion,Synovitis,Baker's,Contusion,Fracture
<uid_1>,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5
```

### Metryka
**Macro-averaged ROC AUC** — AUC liczone osobno dla każdej z 12 etykiet,
potem zwykła średnia arytmetyczna z tych 12 wartości. Konsekwencja: etykiety
rzadkie (np. Fracture) mają **taką samą wagę** w metryce jak etykiety częste
(np. Effusion), mimo że mają dużo mniej przykładów pozytywnych — to wprost
premiuje dobre radzenie sobie z class imbalance.

---

## 3. To NIE jest zwykła klasyfikacja nadzorowana — problem etykiet

To najważniejsza rzecz do zrozumienia w tym konkursie, bo zmienia całe
podejście do treningu.

Według niezależnego audytu danych przeprowadzonego przez jednego z
uczestników (repozytorium `homeshwarnelakurthi/RSNA-Knee-Abnormality-Detection`,
stan na 8 sierpnia 2026):

- Zbiór treningowy: **4407 badań**.
- **Tylko 58 z nich ma prawdziwe etykiety eksperckie** ("gold labels") —
  ustalone przez dwóch radiologów MSK (musculoskeletal) + adjudykatora,
  na podstawie **jawnie progowanych kryteriów nasilenia** (severity-thresholded
  criteria), z zasadą, że przypadki "na granicy" oceniano jako negatywne.
- **Pozostałe 4349 badań ma tylko raport tekstowy**, bez etykiet eksperckich
  wyprowadzonych z obrazu.
- Etykiety wyprowadzone bezpośrednio z raportu (report-derived) zgadzają się
  z etykietami gold (image-derived) tylko w **~82%** przypadków, i ta
  rozbieżność jest **systematyczna, nie losowa** — czyli nie da się jej
  wygładzić przez uśrednianie, trzeba ją modelować świadomie.

### Konsekwencja: to jest problem weak supervision, nie standardowa klasyfikacja

Skoro tylko 58 studiów ma twarde etykiety, a reszta ma jedynie tekst raportu
jako "słabe" źródło sygnału, praktyczne podejście wymaga jednego z (lub
kombinacji) poniższych:

1. **Report-to-label mapping** — zbudowanie (regułowego, słownikowego, albo
   za pomocą małego, lokalnie hostowanego modelu NLP) ekstraktora, który z
   tekstu raportu wyciąga słabe etykiety dla 12 celów, ze świadomością że
   będą one obciążone błędem ~18%.
2. **Trening na 58 gold + walidacja przeciw report-derived jako proxy** —
   z pełną świadomością, że lokalny wynik walidacyjny na report-derived
   labels będzie **optymistycznie obciążony** względem prawdziwego AUC na
   ukrytym, image-derived zbiorze testowym.
3. **Kalibracja pewności (confidence calibration)** — jeden z publicznych
   baseline'ów (`JunhaoLiXD/RSNA_Knee_Abnormality_Detection`, V03) stosuje
   m.in.: hierarchiczne, kalibrowane priorytety soft-label per stan kliniczny,
   ograniczenia porządku i marginesu między etykietami, oraz **15% floor
   pewności** dla klas o zerowym support — to sugeruje, że "surowe" etykiety
   report-derived bez takiej kalibracji dają gorsze wyniki.
4. **Semi-supervised / self-training** — trenowanie na 58 gold, użycie
   modelu do wygenerowania pseudo-etykiet na resztę, iteracyjne dotrenowanie.

### Co to oznacza dla Twojej strategii
- Nie licz na to, że masz "zwykłe" 4407 przykładów treningowych z twardymi
  etykietami — realnie masz **58 pewnych + 4349 niepewnych**.
- Walidacja lokalna musi to odzwierciedlać: raportowany wynik na 58 gold
  (`OOF AUC`) jest bardziej wiarygodnym sygnałem niż wynik na całości danych
  z etykietami report-derived.
- To tłumaczy, dlaczego jeden z publicznych baseline'ów raportuje wprost
  osobno "Kaggle public score" (0.664, na pełnym zbiorze z report-derived
  labels) i "58-gold OOF AUC" (0.632) — te dwie liczby mierzą coś innego i
  nie należy ich mylić.

---

## 4. Charakterystyka danych obrazowych (OFICJALNIE POTWIERDZONE)

### Struktura plików i folderów

```
train.csv                                              # 1 wiersz = 1 studium
train_series.csv                                       # 1 wiersz = 1 seria/sekwencja
train_series/<StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm
test.csv                                                # przykładowe (3 studia) — realny test podmieniany przy scoringu
test_series.csv                                         # jw.
test_series/<StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm
sample_submission.csv                                   # wszystkie etykiety = 0.5
```

### `train.csv` — kolumny
| Kolumna | Opis |
|---|---|
| `StudyInstanceUID` | unikalny identyfikator studium; = nazwa folderu w `train_series/` |
| `PatientSex` | Male / Female / puste |
| `Report` | wolny tekst raportu radiologicznego, wielojęzyczny (język zależy od ośrodka) |
| 12× etykieta 0/1 | `ACL`, `MCL`, `Medial Meniscus`, `Lateral Meniscus`, `Medial OA`, `Lateral OA`, `PF OA`, `Effusion`, `Synovitis`, `Baker's`, `Contusion`, `Fracture` |

Etykiety są więc **wprost binarne w `train.csv`** dla tych studiów, które je
mają — nie trzeba ich samodzielnie wyprowadzać z raportu dla podzbioru z
etykietami. Zgodnie z oficjalnym opisem: **"Only a small subset of training
studies carry per-condition labels. We also provide the original text of the
radiology report from which you may wish to derive the labels for the
remaining studies."** — czyli host wprost sugeruje wyprowadzanie etykiet z
raportu dla reszty (patrz sekcja 3).

### `train_series.csv` — kolumny
| Kolumna | Opis |
|---|---|
| `StudyInstanceUID` | studium, do którego należy seria |
| `SeriesInstanceUID` | unikalny identyfikator serii; = nazwa folderu w `train_series/<StudyInstanceUID>/` |
| `Fluid_Sensitive` | 1 jeśli sekwencja eksponuje sygnał płynu (T2, PD, STIR i podobne), inaczej 0 |
| `Fat_Suppression` | 1 jeśli zastosowano saturację tłuszczu, inaczej 0 |
| `Anatomical_Plane` | płaszczyzna: `Sagittal`, `Coronal`, `Axial` |

To bardzo praktyczne — **nie trzeba zgadywać typu sekwencji z surowych
nagłówków DICOM**, host dostarcza to wprost jako metadane. Warto to
wykorzystać przy projektowaniu wejścia modelu (np. osobne embeddingi
pozycyjne per `Anatomical_Plane`, albo filtrowanie do sekwencji
`Fluid_Sensitive=1` przy szukaniu wysięku/zapalenia).

### DICOM — szczegóły techniczne
- **819 640 plików, 569.76 GB łącznie** (potwierdzone oficjalnie — wcześniejsze
  szacunki community rzędu 15.9GB dotyczyły innego, mniejszego podzbioru).
- Każda seria to zwykle **20–45 warstw (mediana 30)**, z długim ogonem do
  kilkuset warstw w pojedynczych przypadkach — pipeline musi obsłużyć dużą
  wariancję długości sekwencji (padding/maskowanie, jak w `model.py`).
- **Mieszane transfer syntaxes**: uncompressed Explicit VR Little Endian,
  JPEG Lossless, JPEG 2000, Implicit VR Little Endian — loader DICOM musi
  poprawnie dekodować wszystkie cztery warianty (np. `pydicom` +
  `pylibjpeg`/`gdcm` jako backend dekompresji).
- **Każdy plik DICOM okrojony do allowlisty 86 tagów metadanych** — nie
  zakładaj obecności pełnego standardowego zestawu tagów DICOM; zweryfikuj
  które z potrzebnych Ci pól (`ImagePositionPatient`, `PixelSpacing`,
  `InstanceNumber` itp.) faktycznie przetrwały tę anonimizację/filtrowanie.
- **Laterality**: nie ma osobnej kolumny lewe/prawe w dostarczonych CSV —
  jeśli chcesz normalizować lateralność (odbicie lustrzane), trzeba to
  wyprowadzić z metadanych DICOM (jeśli dostępne po filtrowaniu do 86 tagów)
  albo estymować z samego obrazu.
- **Rozkład etykiet nie jest gwarantowany identyczny** między train/public
  leaderboard/private eval ("Dataset Distribution Notice") — nie zakładaj,
  że lokalny rozkład klas 1:1 odzwierciedla test.

### Test set
- **~1300 studiów** w rzeczywistym zbiorze testowym (liczba podana wprost w
  oficjalnym opisie).
- `test.csv`/`test_series.csv`/`test_series/` widoczne przed submitem to
  tylko **3 przykładowe studia** — realne dane podstawiane dopiero przy
  scoringu (standard code competition, bez internetu).

---

## 5. Charakterystyka danych tekstowych

- Oryginalne raporty radiologiczne w kolumnie `Report` pliku `train.csv`,
  wolny tekst.
- **Wielojęzyczność** potwierdzona oficjalnie ("may be in any of several
  languages, depending on the reporting institution"), ale **dokładna liczba
  języków nie jest podana wprost** w oficjalnym opisie danych. Zewnętrzne
  źródła podają sprzeczne liczby (9 vs 12) — zweryfikuj samodzielnie przez
  wyliczenie unikalnych języków w kolumnie `Report` po pobraniu danych
  (np. wykrywaczem języka typu `langdetect`/`fasttext`).
- **`Report` jest dostępny TYLKO w `train.csv`** — `test.csv` zawiera
  wyłącznie `StudyInstanceUID`, **bez kolumny raportu**. To rozstrzyga
  wcześniejszą niepewność: **text branch może być używany wyłącznie w
  treningu** (jako słabe źródło etykiet / sygnał do destylacji), a finalny
  model do inferencji na teście musi działać **tylko na obrazie**. To
  bezpośrednio potwierdza architekturę zaproponowaną w `model.py`
  (`use_text_branch=True` przy treningu, text branch pomijany na inference).
- Gotowy multilingual encoder dostępny jako publiczny Kaggle Model:
  **`paraphrase-multilingual-MiniLM-L12-v2`** (patrz sekcja 9b) — dokładnie
  ten checkpoint, który był już zarekomendowany w `model.py`.

---

## 6. Ograniczenia formatu konkursu (Code Competition)

To konkurs typu **Code Competition** — nie wysyłasz pliku wyników, tylko
**notebook**, który generuje wyniki na serwerach Kaggle.

| Ograniczenie | Wartość |
|---|---|
| Typ submisji | Kaggle Notebook, commit z aktywnym przyciskiem "Submit" |
| Limit czasu (CPU) | ≤ 9 godzin |
| Limit czasu (GPU) | ≤ 9 godzin |
| Dostęp do internetu podczas submitu | **Wyłączony** |
| Dane zewnętrzne / modele pretrenowane | Dozwolone, jeśli **darmowo i publicznie dostępne** |
| Nazwa pliku wynikowego | dokładnie `submission.csv` |

### Konsekwencje praktyczne
- Duże modele (wagi rzędu dziesiątek-setek GB, np. duże LLM typu DeepSeek w
  pełnej wersji) trzeba **wcześniej pobrać i wgrać jako Kaggle Dataset**,
  potem ładować lokalnie w notebooku offline — samo "wywołanie API" nie
  zadziała (brak internetu) i jest niewskazane z innych powodów (patrz niżej).
- 9h to budżet na **przejście całego zbioru testowego**, nie na jedną
  predykcję. Jeden z publicznych pipeline'ów raportuje: budowa cache ~1h,
  pełny etap treningowo-inferencyjny mieści się wygodnie w limicie przy
  dobrze zoptymalizowanym 2.5D podejściu.
- **Uwaga sprzętowa** z publicznego repo: na Kaggle **nie wybierać GPU P100**
  — obecna wersja PyTorch na Kaggle nie zawiera kerneli dla architektury
  Pascal, więc sesja wywala się przy pierwszej konwolucji. Bezpieczny wybór
  to **T4**.

---

## 7. Ograniczenia prawne — regulamin oficjalny (POTWIERDZONE)

### 7.1 Data Security (Rule 2.4.b) — tekst dosłowny z regulaminu

> "You agree to use reasonable and suitable measures to prevent persons who
> have not formally agreed to these Rules from gaining access to the
> Competition Data. You agree not to transmit, duplicate, publish,
> redistribute or otherwise provide or make available the Competition Data
> to any party not participating in the Competition."

To **potwierdza** wcześniejszą ostrożność społeczności wobec wysyłania
tekstu raportu (`Report`) do zewnętrznych, hostowanych API LLM — takie API
(OpenAI, Anthropic, DeepSeek przez chmurę itd.) to z definicji "party not
participating in the Competition" w rozumieniu tej reguły, więc transmisja
danych konkursu (w tym treści raportów, danych pacjentów) do nich jest
**prawdopodobnie naruszeniem tej reguły**. Nie jest to jednoznacznie
rozstrzygnięte wprost dla przypadku "API modelu AI", ale ostrożna
interpretacja jest zasadna.

**Praktyczna rekomendacja**: przetwarzaj tekst raportu wyłącznie **lokalnie
lub wewnątrz samego Kaggle Notebooka** — open-weights modele NLP pobrane i
uruchamiane offline (np. `paraphrase-multilingual-MiniLM-L12-v2`, dostępny
jako publiczny Kaggle Model — patrz sekcja 9), nigdy przez zewnętrzne API.

### 7.2 External Data and Tools — "Reasonableness Standard" (Rule 2.6.b)

Regulamin dopuszcza użycie zewnętrznych danych/modeli/narzędzi, o ile są
**"reasonably accessible to all" i "of minimal cost"**. Host ocenia to wg
**Reasonableness Standard**: mały koszt subskrypcji (przykład z regulaminu:
dodatkowe funkcje Gemini Advanced) jest akceptowalny; zakup drogiej,
proprietary licencji przewyższającej wartość nagrody — nie.

**Konsekwencja dla wyboru modeli**: darmowe, open-weights modele (DINOv2,
MiniLM itd.) są bezpiecznym wyborem bez wątpliwości. Płatne API dużych LLM
(nawet przy niskim koszcie) wprowadzają podwójne ryzyko: (a) możliwy konflikt
z Data Security (7.1) przy przesyłaniu tekstu raportu, (b) ocena
"Reasonableness" leży w gestii hosta i nie jest z góry gwarantowana.

### 7.3 Zakaz prywatnego dzielenia się kodem (Rule 3.6.a)

Nie wolno prywatnie dzielić się kodem konkursowym z osobami spoza własnego
Team (poza scenariuszem mergera). Można publicznie dzielić kod przez
forum/Notebooks Kaggle — ale wtedy automatycznie licencjonuje się go na
zasadach Open Source (bez ograniczeń użycia komercyjnego). Ma to znaczenie,
jeśli planujesz współpracować z kimś spoza swojego zgłoszonego zespołu.

### 7.4 Licencja zwycięzcy (Rule 1.6 / 2.5)
Jeśli wygrasz, finalny kod i wagi modelu muszą zostać udostępnione na
licencji **CC-BY-NC 4.0** (open source, ale non-commercial) — inne niż to,
co sugerował wcześniejszy nieoficjalny research (ogólne "open source" bez
sprecyzowania wariantu licencji).

---

## 8. Otwarte pytania — stan po weryfikacji oficjalnych danych

Dzięki oficjalnej zakładce Data/Rules większość wcześniejszych niepewności
została rozstrzygnięta. Zestawienie:

| Pytanie | Status | Odpowiedź |
|---|---|---|
| Czy raport dostępny w teście? | ✅ ROZSTRZYGNIĘTE | Nie — `test.csv` ma tylko `StudyInstanceUID`, brak kolumny `Report` |
| Nazwy plików/kolumn | ✅ ROZSTRZYGNIĘTE | Patrz sekcja 4 — pełny, oficjalny schemat |
| Rozmiar zbioru testowego | ✅ ROZSTRZYGNIĘTE | ~1300 studiów |
| Rozmiar całości danych | ✅ ROZSTRZYGNIĘTE | 569.76 GB, 819 640 plików (obala wcześniejsze szacunki community rzędu 15.9GB) |
| Metadane per sekwencja | ✅ ROZSTRZYGNIĘTE | `train_series.csv`: Fluid_Sensitive, Fat_Suppression, Anatomical_Plane |
| Liczba języków raportów | ❌ NADAL OTWARTE | Regulamin potwierdza wielojęzyczność, ale nie podaje liczby — policz sam z kolumny `Report` |
| Dokładna liczba studiów z gold labels | ❌ NADAL OTWARTE | Oficjalnie potwierdzone "small subset" bez liczby; nieoficjalny audyt community sugerował 58/4407 — zweryfikuj sam licząc niepuste etykiety w `train.csv` |
| Kolumna laterality (lewe/prawe kolano) | ❌ NADAL OTWARTE | Brak wprost w CSV; sprawdź czy przetrwała w 86 dozwolonych tagach DICOM |
| Kolejność warstw w serii | ❌ DO SPRAWDZENIA | Nie opisana wprost — zweryfikuj przez `InstanceNumber`/`ImagePositionPatient` w dostępnych tagach |

---

## 9. Publiczne modele dostępne w zakładce Models (potwierdzone)

Kaggle udostępnia dla tego konkursu zakładkę **Models** z gotowymi, publicznie
dostępnymi checkpointami wraz z **realnym "Best public score"** osiągniętym
przez społeczność przy ich użyciu:

| Model | Właściciel | Best public score | Uwagi |
|---|---|---|---|
| DINOv2 · **small** | Meta | **0.906** | najlepszy wynik ze wszystkich; 40 użytkowników |
| DINOv2 · base | Meta | 0.861 | 7 użytkowników |
| DINOv2 · large | Meta | **0.5** | tylko 1 użytkownik — wynik = losowe zgadywanie, sugeruje overfitting/problem z fine-tuningiem dużego modelu na małym zbiorze gold labels |
| EfficientNet-B3 | społeczność | 0.701 | |
| dinov2-with-registers-base | społeczność | 0.696 | wariant DINOv2 z tokenami rejestrowymi |
| **paraphrase-multilingual-MiniLM-L12-v2** | społeczność | 0.656 | dokładnie ten model tekstowy zarekomendowany w `model.py` |
| EfficientNet-B0 | społeczność | 0.622 | |

### Wniosek praktyczny — potwierdza wcześniejszą rekomendację

**DINOv2-small bije zarówno base, jak i (drastycznie) large.** To silny,
empiryczny argument za tym, żeby **nie** iść w stronę większych/cięższych
backbone'ów w nadziei na lepszy wynik — przy tak małym zbiorze gold labels
(sekcja 3) większy model prawdopodobnie **przeucza się** albo źle się
fine-tunuje przy dostępnym budżecie danych/czasu. Architektura w `model.py`
domyślnie używa `dinov2_vits14` (small) — zgodnie z tym, co realnie najlepiej
punktuje w tym konkursie.

Warto też potraktować te "Best public score" jako **przybliżony sufit
pojedynczego, niezespolonego modelu** — wyniki >0.9 raczej wymagają dobrego
wykorzystania metadanych `train_series.csv` (Fluid_Sensitive, Anatomical_Plane)
i sensownej strategii dla brakujących gold labels (sekcja 3), nie samego
wyboru backbone'u.

---

## 10. Warianty architektury — przegląd opcji

### A. Podejście czysto obrazowe (baseline, punkt wyjścia)
- Tylko vision branch, bez tekstu.
- 2.5D CNN/ViT per sekwencja, agregacja (mean lub attention pooling).
- Zaleta: prostsze, szybciej wdrożyć end-to-end pipeline, dobry pierwszy submit.
- Wada: ignoruje sygnał tekstowy dostępny w treningu (4349 raportów) — traci
  potencjalnie dużo informacji do słabego nadzoru.

### B. Podejście multimodalne z destylacją (rekomendowane)
- Trening: vision + text branch połączone, tekst dostarcza dodatkowego
  sygnału / słabych etykiet dla 4349 badań bez gold labels.
- Inference: jeśli raport niedostępny w teście — używany tylko vision branch
  (text branch "uczy" vision branch podczas treningu, potem jest odrzucany
  lub używany tylko gdy dostępny).
- To jest architektura zaproponowana w `model.py` towarzyszącym temu
  dokumentowi.

### C. Podejście ensemble
- Kilka niezależnych modeli (różne backbone'y: DINOv2, ConvNeXt, EfficientNet;
  różne foldy walidacyjne) łączone przez uśrednianie/wagowanie predykcji.
- Zaleta: zwykle poprawia macro AUC.
- Wada: kosztowne czasowo — ryzyko przekroczenia limitu 9h i słaby wynik w
  Efficiency Track. Wymaga starannego budżetowania czasu inferencji.

### D. Podejście z 3D CNN
- Pełny wolumetryczny model 3D zamiast 2.5D.
- Zaleta: teoretycznie najlepiej wykorzystuje strukturę przestrzenną danych.
- Wada: drogie obliczeniowo, ryzykowne pod limit czasu i Efficiency Track;
  brak potwierdzonych publicznych wyników pokazujących przewagę nad 2.5D w
  tym konkretnym konkursie.

### Rekomendacja
Start od **B** (multimodalna destylacja, 2.5D + DINOv2 + mały multilingual
text encoder), zgodnie z tym, co pokazują najlepiej punktujące publiczne
baseline'y (Public Score 0.809 dla podejścia opartego na DINOv2 z
attention poolingiem i normalizacją lateralności). Ensembling (**C**) jako
etap końcowy, po ustabilizowaniu pojedynczego modelu.

---

## 11. Plan działania — checklist

- [ ] **Etap 0 — Setup**: zaakceptować regulamin, pobrać dane, zweryfikować
      realną strukturę (`train.csv`, foldery DICOM, czy raport jest w teście).
- [ ] **Etap 1 — EDA**: rozkład 12 etykiet (spodziewany silny class
      imbalance), liczba sekwencji/warstw per badanie, języki raportów,
      rozkład ośrodków/site'ów, braki danych.
- [ ] **Etap 2 — Strategia etykiet**: zaprojektować sposób wykorzystania
      58 gold + 4349 report-derived (mapping regułowy / mały model NLP
      lokalny / kalibracja pewności — patrz sekcja 3).
- [ ] **Etap 3 — Walidacja**: **GroupKFold po pacjencie/studium** (nigdy per
      slice — inaczej data leakage i sztucznie zawyżony wynik), osobne
      śledzenie metryki na 58 gold vs. na całości.
- [ ] **Etap 4 — Baseline**: prosty 2.5D, tylko obraz, jedna sekwencja,
      żeby zweryfikować cały pipeline end-to-end (łącznie z poprawnym
      formatem `submission.csv` i czasem działania w warunkach zbliżonych
      do code competition).
- [ ] **Etap 5 — Rozbudowa**: wszystkie sekwencje + attention pooling +
      text branch/destylacja.
- [ ] **Etap 6 — Iteracja**: ensembling, kalibracja progów, pomiar czasu
      inferencji pod kątem limitu 9h i Efficiency Track.
- [ ] **Etap 7 — Finalizacja**: osobny, czysty inference-only notebook —
      ładuje gotowe wagi z Kaggle Dataset, robi predykcję offline, zapisuje
      `submission.csv`. Wybór GPU: **T4**, nie P100.

---

## 12. Harmonogram konkursu

| Data | Wydarzenie |
|---|---|
| 30 lipca 2026 | Start konkursu |
| 15 października 2026 | Deadline zgłoszeń (akceptacja regulaminu) i mergerów zespołów |
| 22 października 2026 | Ostateczny deadline submisji |
| 5 listopada 2026 | Deadline obowiązków zwycięzców (kod, wideo, opis metody) |
| 29 listopada – 3 grudnia 2026 | RSNA 2026 (Chicago) — ogłoszenie i nagrodzenie zwycięzców |

---

## 13. Nagrody

**Main Leaderboard** (top 10): od 5 000 do 9 000 USD za miejsce, łącznie
w puli głównej.

**Efficiency Track** (top 3): 7 000 / 6 000 / 5 000 USD, oceniane wg
specjalnego wskaźnika efficiency score łączącego jakość (macro AUC względem
baseline'u i najlepszego wyniku) z czasem ewaluacji w sekundach — cel to
**minimalizacja** tego wskaźnika. Kwalifikują się tylko submity, które i tak
zostały wybrane do głównego leaderboardu (albo automatycznie zakwalifikowane
wg zasad z zakładki "My Submissions"), i które biją benchmark
`sample_submission.csv` na Private Leaderboard.

**Dodatkowe obowiązki zwycięzców** (poza standardowymi Kaggle Winners'
Obligations): nagranie krótkiego wideo prezentującego podejście,
publikacja linku do open-source kodu i wag na forum konkursu, udostępnienie
finalnej wersji modelu publicznie do dystrybucji i walidacji.

---

## 14. Struktura repozytorium i przepływ pracy z Claude Code

### Pliki kodu (ten pakiet)

```
rsna-knee/
├── model.py          # architektura: 2.5D vision branch (DINOv2) + text branch + fuzja
├── dataset.py         # DICOM loading, 2.5D windowing, wykorzystanie train_series.csv
├── diagnostics.py      # raporty diagnostyczne (data/training/inference) — JSON+markdown+wykresy
├── train.py            # pętla treningowa spinająca powyższe, z KFold i checkpointingiem
├── requirements.txt
├── README.md            # ten dokument
└── download_data.md      # instrukcja pobierania metadanych przez Kaggle CLI
```

Każdy moduł ma na końcu blok `if __name__ == "__main__":` z lekkim smoke
testem na syntetycznych danych — uruchamialny lokalnie bez GPU/danych
Kaggle, do szybkiej weryfikacji że logika (kształty tensorów, maskowanie,
metryki) działa poprawnie, zanim uruchomisz coś kosztownego na Kaggle.

### Rekomendowany przepływ: GitHub + Kaggle + Claude Code

```
GitHub repo (źródło prawdy dla kodu, .py)
        │  git pull / Kaggle GitHub integration
        ▼
Kaggle Notebook — trening (internet ON)
        │  Save & Run All
        ▼
Output: checkpointy (.pt) + diagnostics/*.json + *.md + wykresy .png
        │  ty pobierasz / kopiujesz zawartość *_diagnostics*.md
        ▼
Claude Code — analiza raportu, propozycje zmian w .py
        │  commit + push
        ▼
powrót do Kaggle Notebooka, git pull najnowszej wersji, kolejny run
```

**Dwa osobne notebooki** (patrz sekcja 6) — trzymaj to w repo jako dwa pliki
`.ipynb` lub, lepiej, jako cienkie notebooki importujące `train.py`/osobny
`infer.py`:
1. **Notebook treningowy** — internet ON, długi czas życia, uruchamia
   `train.py`, zapisuje checkpointy jako Kaggle Dataset.
2. **Notebook submisyjny** — internet OFF, ≤9h, ładuje checkpointy,
   generuje `submission.csv` (moduł `infer.py` — do zbudowania w kolejnym
   kroku, poza zakresem tego pakietu).

### Jak korzystać z `diagnostics.py` w praktyce

Po `Save & Run All` na Kaggle masz katalog `/kaggle/working/diagnostics/`
pełen plików `*.json`/`*.md`/`*.png`. Zamiast kopiować je pojedynczo:

```python
from diagnostics import combine_reports_for_claude_code
text = combine_reports_for_claude_code(
    Path("/kaggle/working/diagnostics"),
    output_path=Path("/kaggle/working/diagnostics/COMBINED.md"),
)
```

`COMBINED.md` zawiera wszystkie wygenerowane raporty (data + training per
fold + inference) w jednym pliku — to jest dokładnie to, co warto wkleić
w całości do rozmowy z Claude Code po każdym runie. Wykresy (`.png`) są
osobnymi plikami — jeśli chcesz, żeby Claude Code je "widział", dołącz je
jako obrazy, nie tylko liczby z JSON-a.

### Co raporty diagnostyczne wykrywają automatycznie

- **Class imbalance per etykieta** + gotowy `pos_weight` do wstawienia w
  `compute_loss`/`masked_bce_loss`.
- **Collapse detection** — ostrzeżenie, gdy odchylenie std predykcji na OOF
  spada poniżej progu (dokładnie problem opisany w publicznym V02 baseline,
  patrz sekcja 3/9 — model przewidujący stale ~0.5).
- **Najsłabsze/najlepsze etykiety** (per-label AUC, nie tylko macro) — od
  razu wiadomo, które z 12 klas wymagają uwagi (np. rzadkie jak Fracture).
- **Budżet czasowy** — ekstrapolacja zmierzonego czasu inferencji na pełne
  ~1300 studiów testowych, z 20% marginesem bezpieczeństwa względem
  twardego limitu 9h.

## 15. Źródła

### Oficjalne (najwyższa wiarygodność)
- **Zakładka Data** — `kaggle.com/competitions/rsna-knee-abnormality-detection/data` (pełny opis plików, schemat kolumn, rozmiar 569.76GB/819640 plików — dostęp potwierdzony przez użytkownika)
- **Zakładka Rules** — pełny regulamin (sekcje 1–3 tego dokumentu, dane, licencje, obowiązki zwycięzców — dostęp potwierdzony przez użytkownika)
- **Zakładka Models** — `kaggle.com/competitions/rsna-knee-abnormality-detection/models` (realne "Best public score" per backbone — sekcja 9)
- Ogłoszenie RSNA — `rsna.org/news/2026/august/ai-challenge-knee-mri`

### Nieoficjalne / community (niższa wiarygodność, oznaczone w tekście)
- RuntimeWire, *"RSNA opens $77,000 challenge for AI that reads knee MRI and reports"*, 6 sierpnia 2026
- AuntMinnie, *"RSNA launches 2026 Knee Abnormality Detection AI Challenge"*
- Publiczne repozytorium: `github.com/JunhaoLiXD/RSNA_Knee_Abnormality_Detection` (baseline 2.5D, V01–V03)
- Publiczne repozytorium: `github.com/homeshwarnelakurthi/RSNA-Knee-Abnormality-Detection` (audyt danych, strategia — źródło szacunku 58/4407 gold labels)
- Publiczny notebook Kaggle: `kaggle.com/code/pilkwang/rsna-knee-baseline-v1` (DINOv2, Public Score 0.809)

**Zastrzeżenie**: sekcja 3 (dokładna liczba gold labels: 58/4407) i część
sekcji 8 nadal opierają się na niezależnych, nieoficjalnych analizach
społeczności Kaggle, nie na oficjalnej dokumentacji RSNA/Kaggle wprost.
Wszystko inne w tym dokumencie (sekcje 1, 2, 4, 5, 6, 7, 9, 12, 13) zostało
zweryfikowane względem oficjalnej treści zakładek Data/Rules/Models.
