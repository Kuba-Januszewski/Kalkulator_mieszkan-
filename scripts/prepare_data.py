"""
Potok ETL i Trenowania Modułu Machine Learning (k-NN) dla Wyceny Nieruchomości
Autor: Jakub Januszewski
Opis: Skrypt przetwarza surowe oferty z Otodom (Kaggle), audytuje jakość danych,
      eliminuje anomalie i duplikaty, trenuje model K-Najbliższych Sąsiadów (k-NN)
      dla podejścia porównawczego w wycenie nieruchomości oraz ewaluuje jakość
      predykcji (R2, MAE, MAPE) dla TOP 7 rynków miejskich w Polsce.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Ścieżki projektowe
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "apartments_pl_2024_06.csv")
PROCESSED_CSV_PATH = os.path.join(BASE_DIR, "data", "processed", "apartments_top7_cleaned.csv")
MODELS_JOBLIB_PATH = os.path.join(BASE_DIR, "data", "processed", "knn_models.joblib")
CONFIG_JSON_PATH = os.path.join(BASE_DIR, "data", "processed", "parametry_wyceny.json")

TOP_7_MIASTA = {
    "warszawa": "Warszawa",
    "krakow": "Kraków",
    "wroclaw": "Wrocław",
    "gdansk": "Gdańsk",
    "poznan": "Poznań",
    "szczecin": "Szczecin",
    "lodz": "Łódź"
}

FEATURE_COLS = [
    'squareMeters', 'rooms', 'centreDistance', 'buildYear',
    'hasElevator_num', 'hasBalcony_num', 'hasParkingSpace_num',
    'type_apartment', 'type_block', 'type_tenement'
]

# Wagi cech przestrzennych (metraż i liczba pokoi to kluczowe atrybuty rzeczoznawcze)
FEATURE_WEIGHTS = np.array([2.5, 2.0, 1.5, 1.2, 0.8, 0.7, 0.8, 0.9, 0.7, 0.8])

def przygotuj_cechy(df_sub):
    """Konwertuje cechy surowe na wektor numeryczny dla modelu k-NN."""
    df_feat = pd.DataFrame(index=df_sub.index)
    df_feat['squareMeters'] = df_sub['squareMeters']
    df_feat['rooms'] = df_sub['rooms'].fillna(2.0)
    df_feat['centreDistance'] = df_sub['centreDistance'].fillna(df_sub['centreDistance'].median())
    df_feat['buildYear'] = df_sub['buildYear'].fillna(df_sub['buildYear'].median())
    
    df_feat['hasElevator_num'] = (df_sub['hasElevator'] == 'yes').astype(float)
    df_feat['hasBalcony_num'] = (df_sub['hasBalcony'] == 'yes').astype(float)
    df_feat['hasParkingSpace_num'] = (df_sub['hasParkingSpace'] == 'yes').astype(float)

    typ_series = df_sub['type'].fillna('blockOfFlats')
    df_feat['type_apartment'] = (typ_series == 'apartmentBuilding').astype(float)
    df_feat['type_block'] = (typ_series == 'blockOfFlats').astype(float)
    df_feat['type_tenement'] = (typ_series == 'tenement').astype(float)

    return df_feat[FEATURE_COLS]


def uruchom_potok_etl_i_ml():
    print("=" * 75)
    print("ROZPOCZĘCIE PROCESU ETL I TRENOWANIA MODELU MACHINE LEARNING (k-NN)")
    print("=" * 75)

    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"Nie znaleziono pliku źródłowego: {RAW_DATA_PATH}")

    # 1. Wczytanie surowych danych
    df_raw = pd.read_csv(RAW_DATA_PATH)
    print(f"[1/6] Wczytano surowy plik: {len(df_raw):,} rekordów, {len(df_raw.columns)} kolumn.")

    # 2. Usuwanie duplikatów
    df = df_raw.drop_duplicates(subset=['id']).copy()
    duplikaty_oferty = df.duplicated(subset=['city', 'squareMeters', 'price', 'latitude', 'longitude']).sum()
    df = df.drop_duplicates(subset=['city', 'squareMeters', 'price', 'latitude', 'longitude']).copy()
    print(f"[2/6] Usunięto {duplikaty_oferty} zduplikowanych ofert.")

    # 3. Ograniczenie do TOP 7 aglomeracji
    df['city_raw'] = df['city'].astype(str).str.lower().str.strip()
    df = df[df['city_raw'].isin(TOP_7_MIASTA.keys())].copy()
    df['miasto'] = df['city_raw'].map(TOP_7_MIASTA)
    print(f"[3/6] Wyselekcjonowano TOP 7 rynków: {len(df):,} ofert.")

    # 4. Czyszczenie anomalii i weryfikacja biznesowa
    df['cena_m2'] = df['price'] / df['squareMeters']
    przed_czyszczeniem = len(df)

    maska_poprawnosci = (
        (df['price'] >= 100_000) & (df['price'] <= 15_000_000) &
        (df['squareMeters'] >= 15.0) & (df['squareMeters'] <= 300.0) &
        (df['cena_m2'] >= 3_500) & (df['cena_m2'] <= 50_000)
    )

    # Logiczna walidacja pięter i roku budowy
    maska_pieter = ~( (df['floor'].notna()) & (df['floorCount'].notna()) & (df['floor'] > df['floorCount']) )
    maska_rok = (df['buildYear'].isna()) | ((df['buildYear'] >= 1800) & (df['buildYear'] <= 2026))
    maska_poprawnosci = maska_poprawnosci & maska_pieter & maska_rok

    df_cleaned = df[maska_poprawnosci].copy()
    odrzucone = przed_czyszczeniem - len(df_cleaned)
    print(f"[4/6] Usunięto anomalie: {odrzucone:,} rekordów. Czysta baza: {len(df_cleaned):,} ofert.")

    # 5. Ewaluacja i trenowanie modeli k-NN per miasto
    print("[5/6] Trenowanie i walidacja modeli k-NN (Podejście Porównawcze Real Estate):")
    
    baza_modeli_knn = {}
    statystyki_miast = {}
    ewaluacja_ml = {}

    for miasto in sorted(TOP_7_MIASTA.values()):
        sub = df_cleaned[df_cleaned['miasto'] == miasto].copy()
        X_df = przygotuj_cechy(sub)
        y = sub['price'].values
        y_m2 = sub['cena_m2'].values

        # Ewaluacja statystyczna na zbiorze testowym (Train/Test Split 80/20)
        X_tr, X_te, y_tr, y_te = train_test_split(X_df, y, test_size=0.2, random_state=42)
        scaler_eval = StandardScaler()
        X_tr_sc = scaler_eval.fit_transform(X_tr) * FEATURE_WEIGHTS
        X_te_sc = scaler_eval.transform(X_te) * FEATURE_WEIGHTS

        reg = KNeighborsRegressor(n_neighbors=5, weights='distance')
        reg.fit(X_tr_sc, y_tr)
        y_pred = reg.predict(X_te_sc)

        r2 = float(r2_score(y_te, y_pred))
        mae = float(mean_absolute_error(y_te, y_pred))
        mape = float(np.mean(np.abs((y_te - y_pred) / y_te)) * 100)

        # Naiwny punkt odniesienia (baseline): mediana ceny/m² z próby treningowej × metraż
        mediana_m2_tr = float(np.median(y_tr / X_tr['squareMeters'].values))
        y_pred_base = mediana_m2_tr * X_te['squareMeters'].values
        mape_base = float(np.mean(np.abs((y_te - y_pred_base) / y_te)) * 100)

        ewaluacja_ml[miasto] = {
            "R2_score": round(r2, 2),
            "MAE_PLN": round(mae, 0),
            "MAPE_procent": round(mape, 1),
            "MAPE_baseline": round(mape_base, 1)
        }

        # Trenowanie finalnego indeksu k-NN na pełnym zbiorze miejskim
        scaler_final = StandardScaler()
        X_scaled_raw = scaler_final.fit_transform(X_df)
        X_scaled_weighted = X_scaled_raw * FEATURE_WEIGHTS

        knn_engine = NearestNeighbors(n_neighbors=5, metric='euclidean')
        knn_engine.fit(X_scaled_weighted)

        # Zapisujemy model, scaler i uproszczone dane referencyjne dla szybkiego odczytu
        kolumny_podgladu = ['price', 'cena_m2', 'squareMeters', 'rooms', 'centreDistance', 'buildYear', 'type']
        dane_referencyjne = sub[kolumny_podgladu].reset_index(drop=True)

        baza_modeli_knn[miasto] = {
            "knn": knn_engine,
            "scaler": scaler_final,
            "X_scaled": X_scaled_raw,
            "weights": FEATURE_WEIGHTS,
            "feature_cols": FEATURE_COLS,
            "data": dane_referencyjne
        }

        mediana_m2 = float(np.median(y_m2))
        srednia_m2 = float(np.mean(y_m2))
        statystyki_miast[miasto] = {
            "mediana_m2": round(mediana_m2, 0),
            "srednia_m2": round(srednia_m2, 0),
            "probka": int(len(sub))
        }

        zysk_pp = mape_base - mape
        print(f"      - {miasto:10}: R2={r2:.2f} | MAE={mae:,.0f} PLN | MAPE={mape:.1f}% (baza={mape_base:.1f}%, zysk={zysk_pp:+.1f} p.p.) | Próba={len(sub):,} ofert")

    # 6. Zapis plików przetworzonych i modeli ML
    os.makedirs(os.path.dirname(PROCESSED_CSV_PATH), exist_ok=True)
    df_cleaned.to_csv(PROCESSED_CSV_PATH, index=False, encoding='utf-8')
    print(f"[6/6] Zapisano czysty zbiór: {PROCESSED_CSV_PATH}")

    # Serializacja wytrenowanych silników k-NN
    joblib.dump(baza_modeli_knn, MODELS_JOBLIB_PATH, compress=3)
    print(f"      Zapisano modele ML k-NN: {MODELS_JOBLIB_PATH}")

    # Zapis konfiguracji i metryk ewaluacji
    konfiguracja = {
        "algorytm": "K-Nearest Neighbors (k-NN) z wagami rzeczoznawczymi",
        "data_aktualizacji": "Czerwiec 2024",
        "zrodlo_danych": "Otodom / Kaggle (apartments_pl_2024_06)",
        "laczna_probka_top7": len(df_cleaned),
        "statystyki_miast": statystyki_miast,
        "ewaluacja_modelu_ml": ewaluacja_ml
    }

    with open(CONFIG_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(konfiguracja, f, ensure_ascii=False, indent=2)
    print(f"      Zapisano konfigurację i wyniki ewaluacji do: {CONFIG_JSON_PATH}")

    print("=" * 75)
    print("PROCES ETL I TRENOWANIE k-NN ZAKOŃCZONE PEŁNYM SUKCESEM!")
    print("=" * 75)

if __name__ == "__main__":
    uruchom_potok_etl_i_ml()
