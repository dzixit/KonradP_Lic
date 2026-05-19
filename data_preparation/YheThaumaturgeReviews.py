import steamreviews
import pandas as pd

# ID gry: 	323190
app_id = 323190

# Lista jezykow, ktore chcemy pobrac
jezyki_do_pobrania = ['polish']

# Pusty slownik, do ktorego wrzucimy polaczone recenzje
wszystkie_recenzje = {}

print(f"Rozpoczynamy pobieranie recenzji dla ID: {app_id}...")

# Pobieramy dane oddzielnie dla kazdego jezyka i laczymy je
for lang in jezyki_do_pobrania:
    print(f"-> Pobieranie paczki dla jezyka: {lang.upper()}...")
    request_params = dict(language=lang)
    review_dict, _ = steamreviews.download_reviews_for_app_id(app_id, chosen_request_params=request_params)

    # Jesli w paczce sa recenzje, dorzucamy je do naszego glownego zbioru
    if 'reviews' in review_dict:
        wszystkie_recenzje.update(review_dict['reviews'])

print(f"\nZebrano lacznie {len(wszystkie_recenzje)} recenzji (Angielskie + Polskie).")
print("Przetwarzanie danych i generowanie pliku Excel")

# Przygotowujemy liste, do ktorej trafia przetworzone dane
data_for_excel = []

for review_id, review_data in wszystkie_recenzje.items():
    # Bezpieczne pobieranie danych
    v_funny = review_data.get('votes_funny', 0)
    v_up = review_data.get('votes_up', 0)
    text = review_data.get('review', '')
    is_positive = review_data.get('voted_up', True)
    jezyk_recenzji = review_data.get('language', 'unknown')

    # Logika wspolczynnika (zabezpieczenie przed dzieleniem przez zero)
    if v_up > 0:
        ratio = v_funny / v_up
    else:
        ratio = 1.0 if v_funny > 0 else 0.0

    # Klasyfikacja (wykrywanie trolli)
    if ratio > 0.7:
        kategoria = "TROLL / CZYSTY SARKAZM"
    elif ratio > 0.5:
        kategoria = "ZABAWNA ALE PRZYDATNA"
    else:
        kategoria = "NORMALNA"

    # Tworzymy wiersz do tabeli
    row = {
        'ID Recenzji': review_id,
        'Jezyk': jezyk_recenzji,
        'Kategoria': kategoria,
        'Wspolczynnik Funny/Up': round(ratio, 2),
        'Glosy Funny': v_funny,
        'Glosy Helpful': v_up,
        'Ocena': 'Pozytywna' if is_positive else 'Negatywna',
        'Tresc Recenzji': text
    }
    data_for_excel.append(row)

# Tworzymy DataFrame (tabele Pandas)
df = pd.DataFrame(data_for_excel)

# Zapisujemy do pliku Excel
file_name = "recenzje_steam_analiza2.xlsx"
df.to_excel(file_name, index=False)

print(f"Sukces! Zapisano plik: {file_name}")