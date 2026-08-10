# Pobieranie danych — RSNA Knee Abnormality Detection

## 1. Warunek wstępny
Musisz mieć konto Kaggle i **zaakceptować regulamin konkursu** na stronie:
https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules
(bez tego API zwróci błąd 403 nawet z poprawnym tokenem).

## 2. Autoryzacja Kaggle CLI

```bash
pip install kaggle --break-system-packages

# Wygeneruj token: kaggle.com -> Settings -> API -> "Create New Token"
# Pobiera się plik kaggle.json — umieść go tutaj:
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

## 3. POTWIERDZONY oficjalnie rozmiar danych — WAŻNE

```
Rozmiar całości:  569.76 GB
Liczba plików:    819 640
```

To **nie** jest literówka — to prawie 570 GB, głównie plików `.dcm`. Wcześniejsze
szacunki z community (rzędu 15.9 GB) dotyczyły innego, dużo mniejszego
podzbioru (prawdopodobnie samych nagłówków/metadanych) i były mylące.

**Realistyczna rekomendacja: NIE pobieraj całości na własny komputer.**
Trenuj bezpośrednio w Kaggle Notebooks — tam dane są zamontowane lokalnie
na infrastrukturze Kaggle bez konieczności pobierania czegokolwiek. Poniższe
kroki (lokalne pobieranie) mają sens tylko dla plików CSV (metadane, małe)
oraz ewentualnie pojedynczych przykładowych studiów DICOM do debugowania.

```bash
kaggle competitions files -c rsna-knee-abnormality-detection
```

## 4. Pliki metadanych — bezpieczne do pobrania lokalnie (małe)

Oficjalna, potwierdzona lista plików w zakładce Data:

```bash
mkdir -p data
kaggle competitions download -c rsna-knee-abnormality-detection -f train.csv -p data/
kaggle competitions download -c rsna-knee-abnormality-detection -f train_series.csv -p data/
kaggle competitions download -c rsna-knee-abnormality-detection -f test.csv -p data/
kaggle competitions download -c rsna-knee-abnormality-detection -f test_series.csv -p data/
kaggle competitions download -c rsna-knee-abnormality-detection -f sample_submission.csv -p data/
```

`sample_submission.csv` to tylko 470 B — praktycznie darmowe do pobrania i
warto to zrobić od razu, żeby mieć dokładny wzorzec formatu wyjścia.

## 5. Dane obrazowe (DICOM) — NIE pobieraj całości lokalnie

Foldery `train_series/` i `test_series/` (łącznie ~570 GB) zawierają pliki
`.dcm` w strukturze:
```
train_series/<StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm
```
`test_series/` w widoku przed submitem to tylko **3 przykładowe studia** —
realne ~1300 studiów testowych podstawiane dopiero przy scoringu na
serwerach Kaggle (offline).

Jeśli chcesz zobaczyć strukturę na pojedynczym przykładzie bez ściągania
całości: zrób to przez **Kaggle Notebook** (dane zamontowane automatycznie
pod `/kaggle/input/rsna-knee-abnormality-detection/`), nie przez pobieranie
lokalne — to i tak jest docelowe środowisko treningowe/submisyjne.

## 6. .gitignore — żeby nie wrzucić 16GB do repozytorium

```gitignore
# Surowe dane konkursu — pobierane przez Kaggle CLI, nie commitowane
data/
artifacts/
*.dcm
*.npz
*.pt
*.ckpt

# Sekrety
kaggle.json
.env
```

## 7. Zalecana kolejność pierwszego przejścia przez dane

1. `sample_submission.csv` — zobacz dokładny oczekiwany format wyjścia.
2. `train.csv` (lub odpowiednik) — sprawdź kolumny: czy jest tam wprost 12
   etykiet, czy trzeba je wyprowadzić z tekstu raportu; sprawdź czy jest
   kolumna wskazująca gold vs report-derived label (to rozróżnienie jest
   kluczowe — patrz README.md, sekcja 3).
3. Metadane serii DICOM (jeśli plik istnieje osobno) — struktura sekwencji
   per studium.
4. Dopiero na końcu — pełne obrazy DICOM, i to najlepiej zacznij od
   pojedynczego studium, żeby zweryfikować parsowanie (spacing, kolejność
   warstw, laterality) zanim uruchomisz coś na całości.
