# Klasyfikacja sentymentu recenzji Steam przy użyciu modeli liniowych

Niniejsze repozytorium zawiera kod źródłowy oraz analizy powiązane z pracą licencjacką poświęconą automatycznej klasyfikacji sentymentu opinii użytkowników platformy Steam. Celem badania jest porównanie klasycznych reprezentacji tekstowych oraz ocena stabilności modeli za pomocą walidacji krzyżowej.

## Struktura projektu

* data/ - Pliki arkuszy Excel zawierające surowe recenzje gier wideo pobrane za pomocą Steam API.
* data_preparation/ - Skrypty Python służące do ekstrakcji i filtrowania opinii bezpośrednio z API platformy Steam.
* notebooks/ - Interaktywne notatniki Jupyter przedstawiające przebieg eksperymentów, eksplorację danych oraz wnioski z analizy błędów.
* results/ - Wykresy rozkładu klas, histogramy długości recenzji, chmury słów oraz macierze pomyłek wygenerowane podczas ewaluacji.
* klasyfikacja_sentymentu.py - Główny skrypt realizujący potok przetwarzania danych, inżynierii cech, treningu modeli oraz walidacji krzyżowej.
* gradient_descet.py - Skrypt pomocniczy generujący wizualizację gradientu na trójwymiarowej powierzchni funkcji kosztu do pliku PDF.
* pyproject.toml - Standardowy plik konfiguracyjny projektu zawierający metadane oraz zdefiniowane zależności biblioteczne.

## Konfiguracja środowiska

Zależności projektu można zainstalować na kilka sposobów, w zależności od preferowanych narzędzi deweloperskich.

### Opcja pierwsza: Standardowy pip

Aby przygotować środowisko przy użyciu wbudowanego modułu wirtualnego:

```bash
python -m venv venv
source venv/bin/activate  # System Linux lub macOS
# Lub na systemie Windows:
# venv\Scripts\activate

pip install --upgrade pip
pip install -e .
```

Jeśli planujesz korzystać z Jupyter Notebooka, zainstaluj dodatkowe zależności deweloperskie:

```bash
pip install -e ".[dev]"
```

### Opcja druga: Poetry

Poetry to nowoczesny menedżer zależności dla języka Python. Jeśli nie posiadasz go w systemie, możesz go zainstalować poleceniem:

* System Linux/macOS:
  ```bash
  curl -sSL https://install.python-poetry.org | python3 -
  ```
* System Windows (PowerShell):
  ```powershell
  (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python3 -
  ```

Po instalacji uruchom w katalogu projektu:

```bash
poetry install --extras dev
```

Poetry automatycznie utworzy środowisko wirtualne i zainstaluje wszystkie wymagane biblioteki wraz z jądrem do obsługi notatników.

### Opcja trzecia: uv

Narzędzie uv to bardzo szybki menedżer pakietów napisany w języku Rust. Instalacja uv w systemie:

* System Linux/macOS:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
* System Windows (PowerShell):
  ```powershell
  powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```

Po zainstalowaniu uv możesz błyskawicznie przygotować środowisko i zainstalować zależności deweloperskie:

```bash
uv venv
source .venv/bin/activate  # System Linux/macOS
# Lub na systemie Windows:
# .venv\Scripts\activate

uv pip install -e ".[dev]"
```

## Rejestracja jądra (kernel) w Jupyter

Aby Jupyter Notebook mógł korzystać z zainstalowanego środowiska wirtualnego, należy zarejestrować je jako osobne jądro. Upewnij się najpierw, że odpowiednie środowisko zostało aktywowane.

### Dla standardowego venv oraz uv

Po aktywacji środowiska wirtualnego uruchom polecenie:

```bash
python -m ipykernel install --user --name=sentiment-analysis-steam --display-name "Python (Steam Sentiment)"
```

### Dla Poetry

Możesz zarejestrować jądro bezpośrednio za pomocą menedżera bez wchodzenia do powłoki:

```bash
poetry run python -m ipykernel install --user --name=sentiment-analysis-steam --display-name "Python (Steam Sentiment)"
```

Po uruchomieniu Jupyter Notebooka w przeglądarce otwórz notatnik, wybierz z górnego menu `Kernel` -> `Change Kernel` i wybierz `Python (Steam Sentiment)`.

## Uruchamianie kodu

Po aktywacji środowiska wirtualnego główny potok klasyfikacji można wywołać za pomocą skryptu:

```bash
python klasyfikacja_sentymentu.py
```

Wygeneruje on logi w pliku log.txt oraz zaktualizuje pliki graficzne z wykresami w katalogu results.

Aby otworzyć interaktywny notatnik badawczy:

```bash
jupyter notebook notebooks/analiza_sentymentu.ipynb
```

## Zadania dla studenta

W projekcie pozostawiono niekompletne fragmenty kodu oznaczone komentarzem TODO, które student musi samodzielnie uzupełnić w ramach pracy dyplomowej:

* Rozszerzenie preprocesora o definicję etykiet dla trzeciej klasy (Sarkazm/Troll) bazującej na wyliczonym współczynniku Ratio.
* Implementacja treningu i ewaluacji modelu trójklasowego (regresji logistycznej z obsługą wielu klas) na bazie wektoryzacji TF-IDF.
* Przeprowadzenie walidacji krzyżowej dla zaimplementowanego modelu trójklasowego oraz analiza uzyskanej stabilności wyników (ze szczególnym uwzględnieniem wartości F1-score dla nowej klasy).
