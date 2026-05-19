"""
KOMPLETNY SKRYPT - KLASYFIKACJA SENTYMENTU
Konrad Pajor - Praca Licencjacka
Data: 16 maja 2026
"""

import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import glob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from scipy.sparse import hstack
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# Wymuszamy stały seed, aby wyniki wykrywania języka były zawsze identyczne
DetectorFactory.seed = 42

# =============================================================================
# 1. WCZYTYWANIE I PREPROCESSING DANYCH
# =============================================================================

def load_and_preprocess_data(data_dir):
    print(f"Szukanie plików z danymi w folderze: {data_dir}...")

    file_pattern = os.path.join(data_dir, "*.xlsx")
    file_list = glob.glob(file_pattern)

    if not file_list:
        raise FileNotFoundError(f"Brak plików .xlsx w folderze {data_dir}")

    df_list = []
    for file in file_list:
        df_temp = pd.read_excel(file)
        df_list.append(df_temp)

    df = pd.concat(df_list, ignore_index=True)
    print(f"\nSukces! Połączono {len(file_list)} plików. Łączna liczba recenzji surowych: {len(df)}")

    # ---------------------------------------------------------
    # WYKRYWANIE TROLLI / SARKAZMU (Ratio)
    # ---------------------------------------------------------
    if 'votes_up' in df.columns and 'votes_funny' in df.columns:
        def calc_ratio(row):
            up = row['votes_up']
            funny = row['votes_funny']
            if pd.isna(up) or pd.isna(funny): return 0.0
            if up > 0: return funny / up
            elif up == 0 and funny > 0: return 1.0
            else: return 0.0

        df['Ratio'] = df.apply(calc_ratio, axis=1)
        troll = len(df[df['Ratio'] > 0.7])
        zabawna = len(df[(df['Ratio'] > 0.5) & (df['Ratio'] <= 0.7)])
        normalna = len(df[df['Ratio'] <= 0.5])

        print("\n[ Kategoryzacja głosów ]")
        print(f" - Ratio > 0.7 (TROLL / CZYSTY SARKAZM): {troll}")
        print(f" - 0.5 < Ratio <= 0.7 (ZABAWNA ALE PRZYDATNA): {zabawna}")
        print(f" - Ratio <= 0.5 (NORMALNA): {normalna}")
    else:
        print("\nBrak kolumn 'votes_up' i 'votes_funny' - pomijam liczenie Ratio.")

    # ---------------------------------------------------------
    # FILTROWANIE JĘZYKA POLSKIEGO
    # ---------------------------------------------------------
    print("\nTrwa detekcja języka (może to zająć chwilę przy dużym zbiorze)...")
    def is_polish(text):
        if not isinstance(text, str) or len(text.strip()) < 3: return False
        try: return detect(text) == 'pl'
        except LangDetectException: return False

    df['Is_Polish'] = df['Tresc Recenzji'].apply(is_polish)
    df = df[df['Is_Polish'] == True].copy()
    df.drop('Is_Polish', axis=1, inplace=True)

    # ---------------------------------------------------------
    # STANDARDOWE CZYSZCZENIE NLP
    # ---------------------------------------------------------
    def clean_text(text):
        if not isinstance(text, str): return ""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        return text

    df['Cleaned_Review'] = df['Tresc Recenzji'].apply(clean_text)
    df['Label'] = df['Ocena'].map({'Pozytywna': 1, 'Negatywna': 0})
    df = df.dropna(subset=['Cleaned_Review', 'Label'])
    df = df[df['Cleaned_Review'].str.strip().astype(bool)]


    print(f"Ostateczna liczba recenzji do modelowania (N): {len(df)}")

    return df

# =============================================================================
# 2. INŻYNIERIA CECH (FEATURE ENGINEERING)
# =============================================================================

def feature_engineering(df):
    df['Word_Count'] = df['Cleaned_Review'].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)
    df['Char_Count'] = df['Cleaned_Review'].apply(lambda x: len(x) if isinstance(x, str) else 0)
    df['Exclamation_Count'] = df['Tresc Recenzji'].apply(lambda x: x.count('!') if isinstance(x, str) else 0)

    def count_capslock_words(text):
        if not isinstance(text, str): return 0
        words = text.split()
        return len([w for w in words if w.isupper() and len(w) > 2])

    df['Capslock_Count'] = df['Tresc Recenzji'].apply(count_capslock_words)

    negation_words = ['nie', 'nigdy', 'brak', 'bez', 'not', 'never', 'no', 'none', 'neither', 'nor']
    def count_negations(text):
        if not isinstance(text, str): return 0
        return sum(1 for word in text.lower().split() if word in negation_words)

    df['Negation_Count'] = df['Cleaned_Review'].apply(count_negations)
    return df

# =============================================================================
# 3. ANALIZA EKSPLORACYJNA (EDA)
# =============================================================================

def perform_eda(df, output_folder='../Wykresy/results'):
    os.makedirs(output_folder, exist_ok=True)

    total = len(df)
    positive = df['Label'].sum()
    negative = total - positive

    print("\n[Rozkład klas]")
    print(f"Zbiór danych składa się z N={total} recenzji, z czego:")
    print(f" - POZYTYWNE: {positive} ({positive/total*100:.1f}% całości)")
    print(f" - NEGATYWNE: {negative} ({negative/total*100:.1f}% całości)")

    # Wykres klas
    plt.figure(figsize=(8, 6))
    counts = df['Label'].value_counts().sort_index()
    bars = plt.bar([0, 1], counts.values, color=['#FF6B6B', '#4ECDC4'], edgecolor='black', linewidth=1.5)
    for bar, count in zip(bars, counts.values):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{count}\n({count/total*100:.1f}%)',
                 ha='center', va='bottom', fontsize=12, fontweight='bold')
    plt.xticks([0, 1], ['Negatywna', 'Pozytywna'])
    plt.savefig(f'{output_folder}/class_distribution.png', dpi=300)
    plt.close()

    # Długość recenzji
    neg_lengths = df[df['Label'] == 0]['Char_Count']
    pos_lengths = df[df['Label'] == 1]['Char_Count']

    print("\n[Długość recenzji]")
    print(f" - Średnia długość recenzji POZYTYWNEJ (AAA): {pos_lengths.mean():.0f} znaków")
    print(f" - Średnia długość recenzji NEGATYWNEJ (BBB): {neg_lengths.mean():.0f} znaków")
    print(f" - Mediana długości pozytywnej (CCC): {pos_lengths.median():.0f} znaków")
    print(f" - Mediana długości negatywnej (DDD): {neg_lengths.median():.0f} znaków")

    plt.figure(figsize=(10, 6))
    plt.hist(neg_lengths, bins=50, alpha=0.6, label='Negatywna', color='#FF6B6B')
    plt.hist(pos_lengths, bins=50, alpha=0.6, label='Pozytywna', color='#4ECDC4')
    plt.xlabel('Liczba znaków')
    plt.ylabel('Częstotliwość')
    plt.legend()
    plt.savefig(f'{output_folder}/review_length_distribution.png', dpi=300)
    plt.close()

    # WordCloud
    wc_pos = WordCloud(width=1200, height=600, background_color='white', colormap='Greens', max_words=100).generate(' '.join(df[df['Label'] == 1]['Cleaned_Review'].astype(str)))
    plt.figure(figsize=(15, 8)); plt.imshow(wc_pos, interpolation='bilinear'); plt.axis('off'); plt.savefig(f'{output_folder}/wordcloud_positive.png', dpi=300); plt.close()

    wc_neg = WordCloud(width=1200, height=600, background_color='white', colormap='Reds', max_words=100).generate(' '.join(df[df['Label'] == 0]['Cleaned_Review'].astype(str)))
    plt.figure(figsize=(15, 8)); plt.imshow(wc_neg, interpolation='bilinear'); plt.axis('off'); plt.savefig(f'{output_folder}/wordcloud_negative.png', dpi=300); plt.close()

# =============================================================================
# 4. TRENING, EWALUACJA I MACIERZ POMYŁEK
# =============================================================================

def train_and_evaluate(X_train, X_test, y_train, y_test, vectorizer_name, feature_matrix_train, feature_matrix_test, output_folder):
    model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=2000, random_state=42)
    model.fit(feature_matrix_train, y_train)
    y_pred = model.predict(feature_matrix_test)

    print(f"\n==================================================")
    print(f" WYNIKI MODELU: {vectorizer_name}")
    print(f"==================================================")
    print(f"\n[ Tabela dla {vectorizer_name}]")
    print(classification_report(y_test, y_pred, target_names=['Negatywna', 'Pozytywna'], digits=4))

    # Macierz pomyłek
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    if vectorizer_name == "TF_IDF":
        print("\n[Wartości z macierzy pomyłek TF-IDF]")
        print(f" - True Negatives (TN): {tn}")
        print(f" - False Positives (FP): {fp}")
        print(f" - False Negatives (FN): {fn}")
        print(f" - True Positives (TP): {tp}")

    # Rysowanie heatmapy
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, square=True, linewidths=2, linecolor='black', annot_kws={'size': 16})
    plt.xlabel('Predykcja modelu')
    plt.ylabel('Rzeczywista klasa')
    plt.xticks([0.5, 1.5], ['Negatywna', 'Pozytywna'])
    plt.yticks([0.5, 1.5], ['Negatywna', 'Pozytywna'], rotation=0)
    plt.savefig(f'{output_folder}/confusion_matrix_{vectorizer_name}.png', dpi=300)
    plt.close()

    return model, y_pred

# =============================================================================
# 5. ANALIZA BŁĘDÓW I WAG (ERROR & FEATURE IMPORTANCE)
# =============================================================================

def analyze_errors_and_features(model, vectorizer, y_test, y_pred, df_test, output_folder):
    # Analiza błędów
    y_test_arr = y_test.values
    fp_mask = (y_test_arr == 0) & (y_pred == 1)
    fn_mask = (y_test_arr == 1) & (y_pred == 0)

    with open(f'{output_folder}/error_analysis.txt', 'w', encoding='utf-8') as f:
        f.write("FALSE POSITIVES (FP)\n")
        for idx, (_, row) in enumerate(df_test[fp_mask].head(15).iterrows(), 1):
            f.write(f"\n[FP #{idx}]\n{row['Tresc Recenzji']}\n")
        f.write("\nFALSE NEGATIVES (FN)\n")
        for idx, (_, row) in enumerate(df_test[fn_mask].head(15).iterrows(), 1):
            f.write(f"\n[FN #{idx}]\n{row['Tresc Recenzji']}\n")

    # Analiza wagi cech (Feature Importance) - działa dla BoW i TF-IDF
    if hasattr(vectorizer, 'get_feature_names_out'):
        feature_names = vectorizer.get_feature_names_out()
        coefficients = model.coef_[0]
        # Bierzemy słowa (bez naszych dodanych liczbowych cech w przypadku 3 modelu)
        n_vocab = len(feature_names)
        coef_vocab = coefficients[:n_vocab]

        top_pos_idx = coef_vocab.argsort()[-10:][::-1]
        top_neg_idx = coef_vocab.argsort()[:10]

        print("\n[TOP 10 POZYTYWNE]")
        for i in top_pos_idx: print(f"{feature_names[i]:<15} {coef_vocab[i]:>+6.2f}")

        print("\n[TOP 10 NEGATYWNE]")
        for i in top_neg_idx: print(f"{feature_names[i]:<15} {coef_vocab[i]:>+6.2f}")

# =============================================================================
# 6. GŁÓWNA FUNKCJA MAIN
# =============================================================================

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data")
    results_dir = os.path.join(current_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # 1. Wczytanie
    df = load_and_preprocess_data(data_dir)

    # 2. Inżynieria Cech
    df = feature_engineering(df)

    # 3. EDA
    perform_eda(df, output_folder=results_dir)

    # 4. Podział Train/Test
    X_train, X_test, y_train, y_test = train_test_split(
        df['Cleaned_Review'], df['Label'], test_size=0.2, stratify=df['Label'], random_state=42
    )
    df_train = df.loc[X_train.index]
    df_test = df.loc[X_test.index]

    print(f"\n[Podział na zbiory]")
    print(f"Zbiór treningowy (XXXX): {len(X_train)} recenzji")
    print(f"Zbiór testowy (YYYY): {len(X_test)} recenzji")

    # ================= MODEL 1: BoW =================
    bow_vectorizer = CountVectorizer(stop_words=None, max_features=5000)
    X_train_bow = bow_vectorizer.fit_transform(X_train)
    X_test_bow = bow_vectorizer.transform(X_test)
    train_and_evaluate(X_train, X_test, y_train, y_test, "Bag_of_Words", X_train_bow, X_test_bow, results_dir)

    # ================= MODEL 2: TF-IDF =================
    tfidf_vectorizer = TfidfVectorizer(stop_words=None, max_features=5000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    model_tfidf, y_pred_tfidf = train_and_evaluate(X_train, X_test, y_train, y_test, "TF_IDF", X_train_tfidf, X_test_tfidf, results_dir)

    # Zapis błędów i wagi słów dla TF-IDF (najlepszego bazowego)
    analyze_errors_and_features(model_tfidf, tfidf_vectorizer, y_test, y_pred_tfidf, df_test, results_dir)

    # ================= MODEL 3: TF-IDF + WŁASNE CECHY (HSTACK) =================
    print("\nTrwa przygotowywanie Modelu 3 (TF-IDF + Features)...")
    train_features = df_train[['Word_Count', 'Char_Count', 'Exclamation_Count', 'Capslock_Count', 'Negation_Count']].values
    test_features = df_test[['Word_Count', 'Char_Count', 'Exclamation_Count', 'Capslock_Count', 'Negation_Count']].values

    X_train_combined = hstack([X_train_tfidf, train_features])
    X_test_combined = hstack([X_test_tfidf, test_features])

    train_and_evaluate(X_train, X_test, y_train, y_test, "TF_IDF_Plus_Features", X_train_combined, X_test_combined, results_dir)

    print("\n" + "="*70)
    print("GOTOWE!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()