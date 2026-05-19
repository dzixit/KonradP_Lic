"""
GOTOWE FUNKCJE DO WKLEJENIA DO klasyfikacja_sentymentu.py

Konrad Pajor - Praca Licencjacka
Data: 12 maja 2026

INSTRUKCJA:
1. Otwórz plik src/klasyfikacja_sentymentu.py
2. Zastąp obecne funkcje poniższymi wersjami (lub dodaj brakujące)
3. Uruchom skrypt i zapisz wyniki

"""

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# =============================================================================
# FUNKCJA 1: UZUPEŁNIONA INŻYNIERIA CECH
# =============================================================================

def feature_engineering(df):
    """
    Tworzenie dodatkowych cech na podstawie tekstu recenzji.
    
    Cechy:
    1. Word_Count - liczba słów w recenzji
    2. Char_Count - liczba znaków w recenzji
    3. Exclamation_Count - liczba wykrzykników (!)
    4. Capslock_Count - liczba słów pisanych WIELKIMI LITERAMI
    5. Negation_Count - liczba słów negujących (nie, nigdy, brak, bez, not, never, no)
    """
    print("\n" + "="*70)
    print("FEATURE ENGINEERING - Tworzenie dodatkowych cech")
    print("="*70)
    
    # 1. Liczba słów
    df['Word_Count'] = df['Cleaned_Review'].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)
    print("✓ Dodano cechę: Word_Count (liczba słów)")
    
    # 2. Liczba znaków
    df['Char_Count'] = df['Cleaned_Review'].apply(lambda x: len(x) if isinstance(x, str) else 0)
    print("✓ Dodano cechę: Char_Count (liczba znaków)")
    
    # 3. Liczba wykrzykników (z oryginalnego tekstu przed preprocessing)
    df['Exclamation_Count'] = df['Tresc Recenzji'].apply(
        lambda x: x.count('!') if isinstance(x, str) else 0
    )
    print("✓ Dodano cechę: Exclamation_Count (liczba wykrzykników)")
    
    # 4. Liczba słów pisanych CAPSLOCKIEM (z oryginalnego tekstu)
    def count_capslock_words(text):
        if not isinstance(text, str):
            return 0
        words = text.split()
        # Liczymy słowa dłuższe niż 2 znaki, które są całkowicie wielkie
        capslock_words = [w for w in words if w.isupper() and len(w) > 2]
        return len(capslock_words)
    
    df['Capslock_Count'] = df['Tresc Recenzji'].apply(count_capslock_words)
    print("✓ Dodano cechę: Capslock_Count (liczba słów CAPSLOCK)")
    
    # 5. Liczba słów negujących
    negation_words = ['nie', 'nigdy', 'brak', 'bez', 'not', 'never', 'no', 'none', 'neither', 'nor']
    
    def count_negations(text):
        if not isinstance(text, str):
            return 0
        words = text.lower().split()
        return sum(1 for word in words if word in negation_words)
    
    df['Negation_Count'] = df['Cleaned_Review'].apply(count_negations)
    print("✓ Dodano cechę: Negation_Count (liczba negacji)")
    
    # Podsumowanie statystyk
    print("\n" + "-"*70)
    print("STATYSTYKI NOWYCH CECH:")
    print("-"*70)
    for col in ['Word_Count', 'Char_Count', 'Exclamation_Count', 'Capslock_Count', 'Negation_Count']:
        print(f"{col:20s} | Średnia: {df[col].mean():8.2f} | Max: {df[col].max():6.0f}")
    print("="*70 + "\n")
    
    return df


# =============================================================================
# FUNKCJA 2: ANALIZA EKSPLORACYJNA DANYCH (EDA)
# =============================================================================

def perform_eda(df, output_folder='../results'):
    """
    Wykonanie Analizy Eksploracyjnej Danych (EDA) i zapisanie wykresów.
    
    Licencjat_KonradP:
    1. Rozkład klas (Pozytywna/Negatywna)
    2. Rozkład długości recenzji dla obu klas
    3. WordCloud dla recenzji pozytywnych
    4. WordCloud dla recenzji negatywnych
    """
    import os
    from wordcloud import WordCloud
    
    # Utwórz folder na wyniki, jeśli nie istnieje
    os.makedirs(output_folder, exist_ok=True)
    
    print("\n" + "="*70)
    print("ANALIZA EKSPLORACYJNA DANYCH (EDA)")
    print("="*70)
    
    # 1. Podstawowe statystyki
    total = len(df)
    positive = df['Label'].sum()
    negative = total - positive
    
    print(f"\nCAŁKOWITA LICZBA RECENZJI: {total}")
    print(f"  ├─ Pozytywne: {positive} ({positive/total*100:.1f}%)")
    print(f"  └─ Negatywne: {negative} ({negative/total*100:.1f}%)")
    
    # 2. Wykres rozkładu klas
    plt.figure(figsize=(8, 6))
    counts = df['Label'].value_counts().sort_index()
    colors = ['#FF6B6B', '#4ECDC4']  # Czerwony dla negatywnych, niebieski dla pozytywnych
    bars = plt.bar([0, 1], counts.values, color=colors, edgecolor='black', linewidth=1.5)
    
    # Dodaj wartości na słupkach
    for bar, count in zip(bars, counts.values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({count/total*100:.1f}%)',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.xlabel('Klasa recenzji', fontsize=13, fontweight='bold')
    plt.ylabel('Liczba recenzji', fontsize=13, fontweight='bold')
    plt.title('Rozkład klas recenzji Steam', fontsize=15, fontweight='bold', pad=20)
    plt.xticks([0, 1], ['Negatywna', 'Pozytywna'], fontsize=12)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(f'{output_folder}/class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Zapisano wykres: {output_folder}/class_distribution.png")
    
    # 3. Rozkład długości recenzji
    df['Review_Length'] = df['Cleaned_Review'].apply(lambda x: len(x) if isinstance(x, str) else 0)
    
    plt.figure(figsize=(12, 6))
    
    # Histogram dla negatywnych
    neg_lengths = df[df['Label'] == 0]['Review_Length']
    pos_lengths = df[df['Label'] == 1]['Review_Length']
    
    plt.hist(neg_lengths, bins=50, alpha=0.6, label='Negatywna', color='#FF6B6B', edgecolor='black')
    plt.hist(pos_lengths, bins=50, alpha=0.6, label='Pozytywna', color='#4ECDC4', edgecolor='black')
    
    plt.xlabel('Długość recenzji (liczba znaków)', fontsize=13, fontweight='bold')
    plt.ylabel('Liczba recenzji', fontsize=13, fontweight='bold')
    plt.title('Rozkład długości recenzji według klasy', fontsize=15, fontweight='bold', pad=20)
    plt.legend(fontsize=12)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(f'{output_folder}/review_length_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Zapisano wykres: {output_folder}/review_length_distribution.png")
    
    # Statystyki długości
    print(f"\nSTATYSTYKI DŁUGOŚCI RECENZJI:")
    print(f"  Negatywne: średnia = {neg_lengths.mean():.1f} znaków, mediana = {neg_lengths.median():.1f}")
    print(f"  Pozytywne: średnia = {pos_lengths.mean():.1f} znaków, mediana = {pos_lengths.median():.1f}")
    
    # 4. WordCloud dla pozytywnych
    print("\nGenerowanie WordCloud dla recenzji POZYTYWNYCH...")
    positive_text = ' '.join(df[df['Label'] == 1]['Cleaned_Review'].astype(str))
    
    wc_pos = WordCloud(
        width=1200, 
        height=600, 
        background_color='white',
        colormap='Greens',
        max_words=100,
        relative_scaling=0.5,
        min_font_size=10
    ).generate(positive_text)
    
    plt.figure(figsize=(15, 8))
    plt.imshow(wc_pos, interpolation='bilinear')
    plt.axis('off')
    plt.title('Najczęstsze słowa w recenzjach POZYTYWNYCH', 
              fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f'{output_folder}/wordcloud_positive.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Zapisano wykres: {output_folder}/wordcloud_positive.png")
    
    # 5. WordCloud dla negatywnych
    print("Generowanie WordCloud dla recenzji NEGATYWNYCH...")
    negative_text = ' '.join(df[df['Label'] == 0]['Cleaned_Review'].astype(str))
    
    wc_neg = WordCloud(
        width=1200, 
        height=600, 
        background_color='white',
        colormap='Reds',
        max_words=100,
        relative_scaling=0.5,
        min_font_size=10
    ).generate(negative_text)
    
    plt.figure(figsize=(15, 8))
    plt.imshow(wc_neg, interpolation='bilinear')
    plt.axis('off')
    plt.title('Najczęstsze słowa w recenzjach NEGATYWNYCH', 
              fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f'{output_folder}/wordcloud_negative.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Zapisano wykres: {output_folder}/wordcloud_negative.png")
    
    print("="*70 + "\n")


# =============================================================================
# FUNKCJA 3: ANALIZA BŁĘDÓW (ERROR ANALYSIS)
# =============================================================================

def analyze_errors(y_test, y_pred, df_test, output_folder='../results'):
    """
    Analiza błędów klasyfikacji - gdzie model się pomylił?
    
    Identyfikuje:
    - False Positives (FP): model powiedział Pozytywna, a była Negatywna
    - False Negatives (FN): model powiedział Negatywna, a była Pozytywna
    
    Zapisuje przykłady do pliku tekstowego.
    """
    import os
    os.makedirs(output_folder, exist_ok=True)
    
    print("\n" + "="*70)
    print("ANALIZA BŁĘDÓW KLASYFIKACJI (ERROR ANALYSIS)")
    print("="*70)
    
    # Reset indeksu dla łatwiejszego porównania
    y_test_arr = y_test.values if hasattr(y_test, 'values') else y_test
    
    # Znajdź błędne predykcje
    errors_mask = y_test_arr != y_pred
    total_errors = errors_mask.sum()
    
    print(f"\nCAŁKOWITA LICZBA BŁĘDÓW: {total_errors} / {len(y_test)} ({total_errors/len(y_test)*100:.2f}%)")
    
    # False Positives (model powiedział 1, ale było 0)
    fp_mask = (y_test_arr == 0) & (y_pred == 1)
    fp_count = fp_mask.sum()
    
    # False Negatives (model powiedział 0, ale było 1)
    fn_mask = (y_test_arr == 1) & (y_pred == 0)
    fn_count = fn_mask.sum()
    
    print(f"\nFALSE POSITIVES (model myślał że POZYTYWNA, ale była NEGATYWNA): {fp_count}")
    print(f"FALSE NEGATIVES (model myślał że NEGATYWNA, ale była POZYTYWNA): {fn_count}")
    
    # Zapisz do pliku
    output_file = f'{output_folder}/error_analysis.txt'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("ANALIZA BŁĘDÓW KLASYFIKACJI\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Całkowita liczba błędów: {total_errors} / {len(y_test)} ({total_errors/len(y_test)*100:.2f}%)\n")
        f.write(f"False Positives: {fp_count}\n")
        f.write(f"False Negatives: {fn_count}\n\n")
        
        # FALSE POSITIVES
        f.write("="*80 + "\n")
        f.write(f"FALSE POSITIVES (FP): {fp_count} przypadków\n")
        f.write("Model powiedział POZYTYWNA, ale była NEGATYWNA\n")
        f.write("="*80 + "\n\n")
        
        fp_df = df_test[fp_mask].copy()
        
        for idx, (i, row) in enumerate(fp_df.iterrows(), 1):
            f.write(f"--- FP #{idx} ---\n")
            f.write(f"Oryginalna recenzja:\n{row['Tresc Recenzji']}\n\n")
            if idx >= 20:  # Ogranicz do 20 przykładów
                f.write(f"... i jeszcze {fp_count - 20} przypadków ...\n\n")
                break
        
        # FALSE NEGATIVES
        f.write("\n" + "="*80 + "\n")
        f.write(f"FALSE NEGATIVES (FN): {fn_count} przypadków\n")
        f.write("Model powiedział NEGATYWNA, ale była POZYTYWNA\n")
        f.write("="*80 + "\n\n")
        
        fn_df = df_test[fn_mask].copy()
        
        for idx, (i, row) in enumerate(fn_df.iterrows(), 1):
            f.write(f"--- FN #{idx} ---\n")
            f.write(f"Oryginalna recenzja:\n{row['Tresc Recenzji']}\n\n")
            if idx >= 20:  # Ogranicz do 20 przykładów
                f.write(f"... i jeszcze {fn_count - 20} przypadków ...\n\n")
                break
    
    print(f"\n✓ Zapisano szczegółową analizę błędów do: {output_file}")
    
    # Wyświetl kilka przykładów na ekranie
    print("\n" + "-"*70)
    print("PRZYKŁADOWE FALSE POSITIVES (pierwsze 3):")
    print("-"*70)
    for idx, (i, row) in enumerate(fp_df.head(3).iterrows(), 1):
        print(f"\n[FP #{idx}]")
        print(row['Tresc Recenzji'][:200] + "...")
    
    print("\n" + "-"*70)
    print("PRZYKŁADOWE FALSE NEGATIVES (pierwsze 3):")
    print("-"*70)
    for idx, (i, row) in enumerate(fn_df.head(3).iterrows(), 1):
        print(f"\n[FN #{idx}]")
        print(row['Tresc Recenzji'][:200] + "...")
    
    print("="*70 + "\n")


# =============================================================================
# FUNKCJA 4: INTERPRETACJA WAGI SŁÓW (FEATURE IMPORTANCE)
# =============================================================================

def analyze_feature_importance(model, vectorizer, top_n=20, output_folder='../results'):
    """
    Analiza najważniejszych słów (cech) dla każdej klasy.
    
    Wyświetla:
    - TOP słowa wskazujące na recenzję POZYTYWNĄ (największe dodatnie wagi)
    - TOP słowa wskazujące na recenzję NEGATYWNĄ (największe ujemne wagi)
    """
    import os
    os.makedirs(output_folder, exist_ok=True)
    
    print("\n" + "="*70)
    print("INTERPRETACJA WAGI SŁÓW (FEATURE IMPORTANCE)")
    print("="*70)
    
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]
    
    # TOP słowa dla klasy POZYTYWNEJ (największe dodatnie wagi)
    top_positive_indices = coefficients.argsort()[-top_n:][::-1]
    
    print(f"\n{'='*70}")
    print(f"TOP {top_n} SŁÓW wskazujących na POZYTYWNĄ recenzję:")
    print(f"{'='*70}")
    print(f"{'Lp.':<5} {'Słowo':<25} {'Waga':<15}")
    print("-"*70)
    
    positive_words = []
    for idx, i in enumerate(top_positive_indices, 1):
        word = feature_names[i]
        weight = coefficients[i]
        print(f"{idx:<5} {word:<25} {weight:>+10.4f}")
        positive_words.append((word, weight))
    
    # TOP słowa dla klasy NEGATYWNEJ (największe ujemne wagi)
    top_negative_indices = coefficients.argsort()[:top_n]
    
    print(f"\n{'='*70}")
    print(f"TOP {top_n} SŁÓW wskazujących na NEGATYWNĄ recenzję:")
    print(f"{'='*70}")
    print(f"{'Lp.':<5} {'Słowo':<25} {'Waga':<15}")
    print("-"*70)
    
    negative_words = []
    for idx, i in enumerate(top_negative_indices, 1):
        word = feature_names[i]
        weight = coefficients[i]
        print(f"{idx:<5} {word:<25} {weight:>+10.4f}")
        negative_words.append((word, weight))
    
    # Zapisz do pliku
    output_file = f'{output_folder}/feature_importance.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("INTERPRETACJA WAGI SŁÓW\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"TOP {top_n} SŁÓW dla POZYTYWNEJ recenzji:\n")
        f.write("-"*70 + "\n")
        for word, weight in positive_words:
            f.write(f"{word:<30} {weight:>+10.4f}\n")
        
        f.write(f"\nTOP {top_n} SŁÓW dla NEGATYWNEJ recenzji:\n")
        f.write("-"*70 + "\n")
        for word, weight in negative_words:
            f.write(f"{word:<30} {weight:>+10.4f}\n")
    
    print(f"\n✓ Zapisano analizę wagi słów do: {output_file}")
    print("="*70 + "\n")


# =============================================================================
# FUNKCJA 5: MACIERZ POMYŁEK (CONFUSION MATRIX)
# =============================================================================

def plot_confusion_matrix(y_test, y_pred, model_name='Model', output_folder='../results'):
    """
    Wizualizacja macierzy pomyłek (confusion matrix).
    """
    import os
    os.makedirs(output_folder, exist_ok=True)
    
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, 
                square=True, linewidths=2, linecolor='black',
                annot_kws={'size': 16, 'weight': 'bold'})
    
    plt.xlabel('Predykcja modelu', fontsize=14, fontweight='bold')
    plt.ylabel('Rzeczywista klasa', fontsize=14, fontweight='bold')
    plt.title(f'Macierz Pomyłek - {model_name}', fontsize=16, fontweight='bold', pad=20)
    
    plt.xticks([0.5, 1.5], ['Negatywna', 'Pozytywna'], fontsize=12)
    plt.yticks([0.5, 1.5], ['Negatywna', 'Pozytywna'], fontsize=12, rotation=0)
    
    plt.tight_layout()
    
    filename = f'{output_folder}/confusion_matrix_{model_name.replace(" ", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Zapisano macierz pomyłek: {filename}")
    
    # Wyświetl wartości
    tn, fp, fn, tp = cm.ravel()
    print(f"\nMacierz Pomyłek ({model_name}):")
    print(f"  True Negatives  (TN): {tn}")
    print(f"  False Positives (FP): {fp}")
    print(f"  False Negatives (FN): {fn}")
    print(f"  True Positives  (TP): {tp}")
    print()


# =============================================================================
# FUNKCJA 6: GŁÓWNA FUNKCJA MAIN() - PRZEPŁYW PRACY
# =============================================================================

def main_complete():
    """
    Kompletny przepływ pracy analizy sentymentu.
    """
    print("\n" + "="*70)
    print("ANALIZA SENTYMENTU RECENZJI STEAM")
    print("Konrad Pajor - Praca Licencjacka 2026")
    print("="*70 + "\n")
    
    # 0. Ścieżka do danych
    file_path = "../data/recenzje_steam_analiza4.xlsx"
    
    # 1. WCZYTANIE DANYCH (użyj swojej funkcji load_and_preprocess_data)
    print("KROK 1: Wczytywanie i preprocessing danych...")
    # df = load_and_preprocess_data(file_path)  # <-- Użyj swojej funkcji!
    # Na potrzeby przykładu:
    df = pd.read_excel(file_path)
    # ... tutaj Twój preprocessing ...
    
    # 2. ANALIZA EKSPLORACYJNA DANYCH (EDA)
    print("\nKROK 2: Analiza Eksploracyjna Danych (EDA)...")
    perform_eda(df, output_folder='../results')
    
    # 3. INŻYNIERIA CECH
    print("\nKROK 3: Feature Engineering...")
    df = feature_engineering(df)
    
    # 4. PODZIAŁ NA TRAIN/TEST
    print("\nKROK 4: Podział danych na zbiór treningowy i testowy...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['Cleaned_Review'], 
        df['Label'], 
        test_size=0.2, 
        stratify=df['Label'], 
        random_state=42
    )
    
    # Zapisz DataFrame testowy dla późniejszej analizy błędów
    test_indices = X_test.index
    df_test = df.loc[test_indices]
    
    print(f"  Zbiór treningowy: {len(X_train)} recenzji")
    print(f"  Zbiór testowy:    {len(X_test)} recenzji")
    
    # 5. MODEL 1: BAG-OF-WORDS
    print("\n" + "="*70)
    print("KROK 5: MODEL 1 - Bag-of-Words + Regresja Logistyczna")
    print("="*70)
    
    bow_vectorizer = CountVectorizer(stop_words=None, max_features=5000)
    X_train_bow = bow_vectorizer.fit_transform(X_train)
    X_test_bow = bow_vectorizer.transform(X_test)
    
    model_bow = LogisticRegression(penalty='none', solver='lbfgs', max_iter=1000, random_state=42)
    model_bow.fit(X_train_bow, y_train)
    y_pred_bow = model_bow.predict(X_test_bow)
    
    print("\nWYNIKI (Bag-of-Words):")
    print(classification_report(y_test, y_pred_bow, target_names=['Negatywna', 'Pozytywna']))
    
    plot_confusion_matrix(y_test, y_pred_bow, model_name='Bag_of_Words', output_folder='../results')
    
    # 6. MODEL 2: TF-IDF
    print("\n" + "="*70)
    print("KROK 6: MODEL 2 - TF-IDF + Regresja Logistyczna")
    print("="*70)
    
    tfidf_vectorizer = TfidfVectorizer(stop_words=None, max_features=5000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
    model_tfidf = LogisticRegression(penalty='none', solver='lbfgs', max_iter=1000, random_state=42)
    model_tfidf.fit(X_train_tfidf, y_train)
    y_pred_tfidf = model_tfidf.predict(X_test_tfidf)
    
    print("\nWYNIKI (TF-IDF):")
    print(classification_report(y_test, y_pred_tfidf, target_names=['Negatywna', 'Pozytywna']))
    
    plot_confusion_matrix(y_test, y_pred_tfidf, model_name='TF_IDF', output_folder='../results')
    
    # 7. ANALIZA BŁĘDÓW
    print("\nKROK 7: Analiza błędów klasyfikacji...")
    analyze_errors(y_test, y_pred_tfidf, df_test, output_folder='../results')
    
    # 8. INTERPRETACJA WAGI SŁÓW
    print("\nKROK 8: Interpretacja wagi słów...")
    analyze_feature_importance(model_tfidf, tfidf_vectorizer, top_n=20, output_folder='../results')
    
    # 9. PODSUMOWANIE
    print("\n" + "="*70)
    print("GOTOWE! WSZYSTKIE WYNIKI ZAPISANO W FOLDERZE: ../results/")
    print("="*70)
    print("\nWygenerowane pliki:")
    print("  ├─ class_distribution.png")
    print("  ├─ review_length_distribution.png")
    print("  ├─ wordcloud_positive.png")
    print("  ├─ wordcloud_negative.png")
    print("  ├─ confusion_matrix_Bag_of_Words.png")
    print("  ├─ confusion_matrix_TF_IDF.png")
    print("  ├─ error_analysis.txt")
    print("  └─ feature_importance.txt")
    print("\nMożesz teraz użyć tych wykresów i analiz w swojej pracy dyplomowej!")
    print("="*70 + "\n")


# =============================================================================
# URUCHOMIENIE
# =============================================================================

if __name__ == "__main__":
    main_complete()
