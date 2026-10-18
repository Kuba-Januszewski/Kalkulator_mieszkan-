"""
Aplikacja desktopowa: Kalkulator Wyceny Nieruchomości - Model Machine Learning k-NN
Autor: Jakub Januszewski
Opis: Narzędzie wyceny mieszkań oparte na algorytmie K-Najbliższych Sąsiadów (k-NN)
      ze scikit-learn, realizującym podejście porównawcze (metoda rzeczoznawcza).
      Wycena opiera się na 17 161 rzeczywistych ofertach z bazy Otodom (Kaggle).
      Zawiera opcje elastycznego wyszukiwania (całe miasto, dowolny typ/rok),
      eksport raportów do PDF (UTF-8) oraz bezpieczną wysyłkę e-mail (SSL/TLS, wielowątkowość, .env).
"""

import os
import re
import ssl
import json
import joblib
import smtplib
import threading
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import tkinter as tk
from tkinter import messagebox, ttk, filedialog
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Bezpieczne ładowanie zmiennych środowiskowych
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Ścieżki projektowe
SCIEZKA_SKRYPTU = os.path.dirname(os.path.abspath(__file__))
SCIEZKA_MODELS = os.path.join(SCIEZKA_SKRYPTU, "data", "processed", "knn_models.joblib")
SCIEZKA_CONFIG = os.path.join(SCIEZKA_SKRYPTU, "data", "processed", "parametry_wyceny.json")

# --- ŁADOWANIE MODELI ML (k-NN) ---
def wczytaj_silnik_knn():
    """Wczytuje zserializowane modele k-NN ze scikit-learn."""
    if os.path.exists(SCIEZKA_MODELS):
        try:
            return joblib.load(SCIEZKA_MODELS)
        except Exception as e:
            print(f"Ostrzeżenie: Błąd wczytywania modeli k-NN: {e}")
    return None

def wczytaj_konfiguracje():
    """Wczytuje plik konfiguracyjny JSON z metrykami statystycznymi."""
    if os.path.exists(SCIEZKA_CONFIG):
        try:
            with open(SCIEZKA_CONFIG, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"statystyki_miast": {}}

BAZA_KNN = wczytaj_silnik_knn()
CONFIG = wczytaj_konfiguracje()

# Zmienne globalne do raportowania
aktualna_cena_calkowita = 0
aktualna_cena_m2 = 0
aktualne_widelki_min = 0
aktualne_widelki_max = 0
aktualni_sasiedzi = []
aktualna_ocena_dopasowania = ""


# --- SILNIK PREDYKCJI MACHINE LEARNING (k-NN) ---
def oblicz_wycene(*args):
    """Realizuje predykcję k-NN w oparciu o najbliższe fizyczne oferty z bazy."""
    global aktualna_cena_calkowita, aktualna_cena_m2
    global aktualne_widelki_min, aktualne_widelki_max
    global aktualni_sasiedzi, aktualna_ocena_dopasowania

    try:
        miasto = combo_miasto.get()
        metraz = scale_metraz.get()
        pokoje_str = combo_pokoje.get()
        odleglosc_str = combo_odleglosc.get()
        typ_str = combo_typ.get()
        epoka_str = combo_epoka.get()

        # Aktualizacja etykiety metrażu obok suwaka
        lbl_metraz_badge.config(text=f"{int(metraz)} m²")

        # Parsowanie liczby pokoi
        pokoje_num = float(re.search(r'\d+', pokoje_str).group(0)) if re.search(r'\d+', pokoje_str) else 2.0

        # Walidacja logiczna układu (np. 4 pokoje na 20 m² lub 1 pokój na 100 m²)
        powierzchnia_na_pokoj = metraz / pokoje_num
        if powierzchnia_na_pokoj < 9.0:
            frame_ostrzezenie.pack(fill="x", pady=(4, 0))
            label_ostrzezenie.config(
                text=f"⚠️ Nietypowy układ: tylko {powierzchnia_na_pokoj:.1f} m² na 1 pokój! Bardzo niski stopień dopasowania do rynku.",
                fg="#DC2626"
            )
        elif pokoje_num == 1 and metraz > 65.0:
            frame_ostrzezenie.pack(fill="x", pady=(4, 0))
            label_ostrzezenie.config(
                text="ℹ️ Duży metraż jak na 1 pokój (>65 m²): charakterystyka typu loft / otwarte studio.",
                fg="#D97706"
            )
        else:
            frame_ostrzezenie.pack_forget()

        if not BAZA_KNN or miasto not in BAZA_KNN:
            label_wynik_calkowita.config(text="Brak danych")
            return

        model_pkg = BAZA_KNN[miasto]
        scaler = model_pkg["scaler"]
        base_weights = model_pkg["weights"]
        feature_cols = model_pkg["feature_cols"]
        baza_ofert = model_pkg["data"]
        X_scaled_raw = model_pkg.get("X_scaled")

        # Mapowanie odległości od centrum
        mapa_odleglosci = {
            "Ścisłe centrum (<2.5 km)": 1.5,
            "Strefa śródmiejska (2.5-5 km)": 3.7,
            "Strefa miejska (5-9 km)": 6.8,
            "Peryferia / Obrzeża (>9 km)": 11.5
        }
        odleglosc_km = mapa_odleglosci.get(odleglosc_str, 4.5)

        # Mapowanie epoki budowy
        mapa_lat = {
            "Nowe budownictwo (>2010)": 2018.0,
            "Budownictwo współczesne (1991-2010)": 2002.0,
            "Wielka płyta / PRL (1950-1990)": 1978.0,
            "Historyczna / Przedwojenna (<1950)": 1930.0
        }
        rok_budowy = mapa_lat.get(epoka_str, 1995.0)

        # Przygotowanie wektora wejściowego do formatu cech modelu
        input_dict = {
            'squareMeters': [metraz],
            'rooms': [pokoje_num],
            'centreDistance': [odleglosc_km],
            'buildYear': [rok_budowy],
            'hasElevator_num': [1.0 if var_winda.get() else 0.0],
            'hasBalcony_num': [1.0 if var_balkon.get() else 0.0],
            'hasParkingSpace_num': [1.0 if var_parking.get() else 0.0],
            'type_apartment': [1.0 if typ_str == "Apartamentowiec" else 0.0],
            'type_block': [1.0 if typ_str == "Blok mieszkalny" else 0.0],
            'type_tenement': [1.0 if typ_str == "Kamienica" else 0.0]
        }
        input_df = pd.DataFrame(input_dict)[feature_cols]

        # Dynamiczne modyfikowanie wag dla opcji elastycznych ("Dowolna")
        current_weights = base_weights.copy()
        zmiana_wag = False

        if "Dowolna" in odleglosc_str or "Całe miasto" in odleglosc_str:
            current_weights[2] = 0.0  # Wyzerowanie wagi odległości
            zmiana_wag = True

        if "Dowolny" in epoka_str:
            current_weights[3] = 0.0  # Wyzerowanie wagi roku budowy
            zmiana_wag = True

        if "Dowolny" in typ_str:
            current_weights[7] = 0.0  # Wyzerowanie wag typu budynku
            current_weights[8] = 0.0
            current_weights[9] = 0.0
            zmiana_wag = True

        input_scaled = scaler.transform(input_df) * current_weights

        # Dopasowanie k-NN: jeśli wagi zostały zmienione, przeliczamy w locie
        if zmiana_wag and X_scaled_raw is not None:
            knn_active = NearestNeighbors(n_neighbors=5, metric='euclidean')
            knn_active.fit(X_scaled_raw * current_weights)
        else:
            knn_active = model_pkg["knn"]

        # Wyszukanie k=5 najbliższych sąsiadów
        dystanse, indeksy = knn_active.kneighbors(input_scaled)
        dystanse_wiersz = dystanse[0]
        indeksy_wiersz = indeksy[0]

        sasiedzi_df = baza_ofert.iloc[indeksy_wiersz].copy()

        # Obliczenie ważonej stawki za m² z ofert podobnych (podejście porównawcze rzeczoznawców)
        wagi = 1.0 / (dystanse_wiersz + 1e-5)
        stawka_m2_wazona = float(np.sum(wagi * sasiedzi_df['cena_m2'].values) / np.sum(wagi))

        # Analiza statystyczna wykazała, że wpływ kondygnacji w próbie wynosi <1.5%
        # (mniej niż średni błąd modelu), dlatego świadomie nie stosujemy sztucznych dyskont.
        wycena_pred = round(stawka_m2_wazona * metraz, -2)
        cena_za_m2 = round(stawka_m2_wazona, 0)

        # Zakres cen ofert porównawczych przeskalowany do metrażu
        widelki_min = round(float(sasiedzi_df['cena_m2'].min()) * metraz, -2)
        widelki_max = round(float(sasiedzi_df['cena_m2'].max()) * metraz, -2)

        aktualna_cena_calkowita = wycena_pred
        aktualna_cena_m2 = cena_za_m2
        aktualne_widelki_min = widelki_min
        aktualne_widelki_max = widelki_max

        # Zapis listy sąsiadów do raportu
        aktualni_sasiedzi = []
        for i, idx in enumerate(indeksy_wiersz[:3]):
            row = baza_ofert.iloc[idx]
            aktualni_sasiedzi.append({
                "metraz": row["squareMeters"],
                "pokoje": int(row["rooms"]),
                "rok": int(row["buildYear"]) if pd.notna(row["buildYear"]) else "Brak danych",
                "cena": row["price"],
                "cena_m2": row["cena_m2"],
                "dystans_km": row["centreDistance"]
            })

        # Ocena dopasowania na podstawie średniego dystansu euklidesowego
        sredni_dystans = float(np.mean(dystanse_wiersz))
        brak_pelnych_cech = zmiana_wag  # wybrano opcję 'Dowolna lokalizacja', 'Dowolny rok' lub 'Dowolny typ'

        if sredni_dystans < 2.0 and powierzchnia_na_pokoj >= 9.0 and not brak_pelnych_cech:
            aktualna_ocena_dopasowania = "Wysoka wiarygodność (duża gęstość ofert o zbliżonych cechach)"
            badge_jakosc.config(text="● WYSOKA WIARYGODNOŚĆ", bg="#064E3B", fg="#34D399")
        elif sredni_dystans < 2.8 or (sredni_dystans < 2.0 and brak_pelnych_cech):
            opis_cech = "ogólne kryteria wyszukiwania" if brak_pelnych_cech else "niewiele ofert o takich parametrach"
            aktualna_ocena_dopasowania = f"Umiarkowana wiarygodność ({opis_cech})"
            badge_jakosc.config(text="● ŚREDNIA WIARYGODNOŚĆ", bg="#78350F", fg="#FBBF24")
        else:
            aktualna_ocena_dopasowania = "Niska wiarygodność (nietypowa konfiguracja parametrów na rynku)"
            badge_jakosc.config(text="● NISKA WIARYGODNOŚĆ", bg="#7F1D1D", fg="#F87171")

        # Aktualizacja etykiet wynikowych
        label_wynik_calkowita.config(text=f"{wycena_pred:,.0f} PLN".replace(',', ' '))
        label_wynik_m2.config(text=f"{cena_za_m2:,.0f} PLN/m²".replace(',', ' '))
        label_widelki.config(text=f"{widelki_min:,.0f} — {widelki_max:,.0f} PLN".replace(',', ' '))

        # Odświeżenie kart 3 najbardziej podobnych ofert z bazy
        for i, oferta in enumerate(aktualni_sasiedzi):
            karty_sasiadow_tytul[i].config(text=f"Oferta #{i+1} • {oferta['metraz']} m² • {oferta['pokoje']} pok.")
            karty_sasiadow_detale[i].config(
                text=f"Rok: {oferta['rok']} | Centrum: {oferta['dystans_km']:.1f} km"
            )
            karty_sasiadow_cena[i].config(
                text=f"{oferta['cena']:,.0f} PLN ({oferta['cena_m2']:,.0f} zł/m²)".replace(',', ' ')
            )

    except Exception as e:
        print(f"Błąd obliczeń k-NN: {e}")


def on_miasto_zmienione(event=None):
    """Przelicza wycenę i odświeża statystyki wybranego rynku miejskiego."""
    miasto = combo_miasto.get()
    stat = CONFIG.get("statystyki_miast", {}).get(miasto, {})
    if stat:
        mediana = stat.get("mediana_m2", 0)
        probka = stat.get("probka", 0)
        label_miasto_info.config(
            text=f"Mediana cen w tym mieście: {mediana:,.0f} PLN/m²  |  Baza ofert: {probka:,}".replace(',', ' ')
        )
    else:
        label_miasto_info.config(text="")
    oblicz_wycene()



# --- GENEROWANIE RAPORTU PDF (UTF-8 Z PODEJŚCIEM PORÓWNAWCZYM) ---
def generuj_pdf():
    """Generuje raport PDF zawierający wynik k-NN oraz zestawienie ofert porównawczych."""
    try:
        from fpdf import FPDF
    except ImportError:
        messagebox.showerror("Brak biblioteki", "Wymagana jest biblioteka fpdf2.\nZainstaluj: pip install fpdf2")
        return

    miasto = combo_miasto.get()
    sciezka = filedialog.asksaveasfilename(
        defaultextension=".pdf",
        filetypes=[("Pliki PDF", "*.pdf")],
        title="Zapisz raport wyceny jako PDF",
        initialfile=f"Raport_Wyceny_{miasto}_{datetime.now().strftime('%Y%m%d')}.pdf"
    )

    if not sciezka:
        return

    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)

        # Obsługa czcionki systemowej z polskimi znakami UTF-8
        font_regular = "Helvetica"
        font_bold = "Helvetica"
        font_path = r"C:\Windows\Fonts\arial.ttf"
        font_bold_path = r"C:\Windows\Fonts\arialbd.ttf"

        if os.path.exists(font_path) and os.path.exists(font_bold_path):
            pdf.add_font("ArialPL", "", font_path)
            pdf.add_font("ArialPL", "B", font_bold_path)
            font_regular = "ArialPL"
            font_bold = "ArialPL"

        # Nagłówek
        pdf.set_font(font_bold, "B", 17)
        pdf.set_text_color(30, 41, 59)
        pdf.cell(0, 10, text="RAPORT WYCENY NIERUCHOMOŚCI", ln=True, align="C")

        pdf.set_font(font_regular, "", 9)
        pdf.set_text_color(100, 116, 139)
        teraz = datetime.now().strftime("%d.%m.%Y, godz. %H:%M")
        pdf.cell(0, 6, text=f"Raport szacunkowej wartości rynkowej lokalu | Data: {teraz}", ln=True, align="C")
        pdf.ln(6)

        # Sekcja 1: Parametry nieruchomości wycenianej
        pdf.set_font(font_bold, "B", 12)
        pdf.set_text_color(37, 99, 235)
        pdf.cell(0, 7, text="1. Parametry Nieruchomości Wycenianej", ln=True)
        pdf.ln(2)

        parametry = [
            ("Miasto:", miasto),
            ("Powierzchnia użytkowa:", f"{scale_metraz.get()} m²"),
            ("Liczba pokoi:", combo_pokoje.get()),
            ("Lokalizacja w mieście:", combo_odleglosc.get()),
            ("Typ budynku:", combo_typ.get()),
            ("Rok / Epoka budowy:", combo_epoka.get()),
            ("Udogodnienia budynku:", f"Winda: {'TAK' if var_winda.get() else 'NIE'} | Balkon: {'TAK' if var_balkon.get() else 'NIE'} | Garaż: {'TAK' if var_parking.get() else 'NIE'}")
        ]

        for etykieta, wartosc in parametry:
            pdf.set_font(font_bold, "B", 10)
            pdf.cell(55, 6, text=etykieta)
            pdf.set_font(font_regular, "", 10)
            pdf.cell(0, 6, text=str(wartosc), ln=True)

        pdf.ln(6)

        # Sekcja 2: Wynik wyceny
        pdf.set_fill_color(240, 244, 248)
        pdf.rect(15, pdf.get_y(), 180, 36, "F")
        pdf.set_y(pdf.get_y() + 3)

        pdf.set_font(font_bold, "B", 11)
        pdf.set_text_color(30, 41, 59)
        pdf.cell(0, 6, text="SZACUNKOWA WARTOŚĆ RYNKOWA:", ln=True, align="C")

        pdf.set_font(font_bold, "B", 18)
        pdf.set_text_color(16, 185, 129)
        pdf.cell(0, 9, text=f"{aktualna_cena_calkowita:,.0f} PLN".replace(',', ' '), ln=True, align="C")

        pdf.set_font(font_regular, "", 10)
        pdf.set_text_color(71, 85, 105)
        nota = f"Stawka za m²: {aktualna_cena_m2:,.0f} PLN/m² | Widełki: {aktualne_widelki_min:,.0f} - {aktualne_widelki_max:,.0f} PLN".replace(',', ' ')
        pdf.cell(0, 6, text=nota, ln=True, align="C")
        pdf.ln(12)

        # Sekcja 3: Nieruchomości porównawcze
        pdf.set_font(font_bold, "B", 12)
        pdf.set_text_color(37, 99, 235)
        pdf.cell(0, 7, text="2. Podobne Oferty na Rynku", ln=True)
        pdf.ln(2)

        pdf.set_font(font_regular, "", 9)
        pdf.set_text_color(15, 23, 42)
        for i, s in enumerate(aktualni_sasiedzi):
            wiersz_txt = (
                f"• Oferta #{i+1}: Metraż: {s['metraz']} m² | Pokoje: {s['pokoje']} | "
                f"Rok: {s['rok']} | Odległość od centrum: {s['dystans_km']:.1f} km | "
                f"Cena: {s['cena']:,.0f} PLN ({s['cena_m2']:,.0f} PLN/m²)"
            ).replace(',', ' ')
            pdf.cell(0, 6, text=wiersz_txt, ln=True)

        pdf.ln(10)

        # Stopka
        pdf.set_font(font_regular, "", 8)
        pdf.set_text_color(148, 163, 184)
        laczna_baza = CONFIG.get("laczna_probka_top7", 17161)
        klauzula = (
            f"Wycena sporządzona na podstawie analizy statystycznej bazy ofert rynkowych (N={laczna_baza:,}).\n"
            "Zastosowano podejście porównawcze k-NN w oparciu o parametry techniczne i lokalizację.\n"
            "Raport ma charakter poglądowy i stanowi estymację wartości ofertowej lokalu."
        ).replace(',', ' ')
        pdf.multi_cell(0, 4.5, text=klauzula, align="C")

        pdf.output(sciezka)
        messagebox.showinfo("Sukces", f"Raport wyceny PDF zapisany pomyślnie:\n{sciezka}")

    except Exception as e:
        messagebox.showerror("Błąd generowania PDF", f"Wystąpił błąd:\n{e}")


# --- BEZPIECZNA WYSYŁKA E-MAIL (SMTP + THREADING + ROOT.AFTER) ---
def waliduj_email(email_str):
    """Sprawdza poprawność formatu adresu e-mail."""
    return bool(re.match(r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$', email_str))


def wyslij_email_watek(sender_email, sender_password, recipient_email, miasto, metraz, tresc_raportu):
    """Wysyła e-mail z wyceną k-NN w tle; komunikację z GUI przekazuje przez root.after()."""
    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", 587))
    teraz = datetime.now().strftime("%d.%m.%Y, %H:%M")

    try:
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = f"Raport Wyceny Nieruchomości: {miasto} - {metraz} m² ({teraz})"
        msg.attach(MIMEText(tresc_raportu, 'plain', 'utf-8'))

        context = ssl.create_default_context()
        with smtplib.SMTP(smtp_server, smtp_port, timeout=10) as server:
            server.ehlo()
            server.starttls(context=context)
            server.ehlo()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, msg.as_string())

        root.after(0, lambda: messagebox.showinfo("Sukces", f"Raport wysłany pomyślnie na adres:\n{recipient_email}"))
    except Exception as e:
        root.after(0, lambda err=e: messagebox.showerror("Błąd wysyłki", f"Nie udało się wysłać maila:\n{err}"))
    finally:
        root.after(0, lambda: btn_wyslij.config(state="normal", text="✉  Wyślij Raport na E-mail"))


def rozpocznij_wysylke_email():
    """Waliduje dane w wątku głównym i uruchamia proces sieciowy asynchronicznie."""
    sender_email = entry_sender_email.get().strip()
    sender_password = entry_sender_password.get().strip()
    recipient_email = entry_recipient_email.get().strip()

    if not waliduj_email(sender_email) or not sender_password or not waliduj_email(recipient_email):
        messagebox.showerror("Błąd danych", "Sprawdź poprawność adresów e-mail oraz hasła SMTP!")
        return

    miasto = combo_miasto.get()
    metraz = scale_metraz.get()
    teraz = datetime.now().strftime("%d.%m.%Y, %H:%M")
    laczna_baza = CONFIG.get("laczna_probka_top7", 17161)

    lista_sasiedzi_txt = ""
    for i, s in enumerate(aktualni_sasiedzi):
        lista_sasiedzi_txt += f"  #{i+1}: {s['metraz']} m² | {s['pokoje']} pok. | Rok: {s['rok']} -> {s['cena']:,.0f} PLN ({s['cena_m2']:,.0f} zł/m²)\n"

    tresc_raportu = f"""==================================================
        RAPORT WYCENY NIERUCHOMOŚCI - {miasto.upper()}
==================================================
Data kalkulacji: {teraz}

Parametry lokalu:
- Miasto: {miasto}
- Metraż: {metraz} m²
- Liczba pokoi: {combo_pokoje.get()}
- Odległość od centrum: {combo_odleglosc.get()}
- Typ budynku: {combo_typ.get()}
- Epoka budowy: {combo_epoka.get()}

SZACOWANA WARTOŚĆ OFERTOWA:
- Wartość nieruchomości: {aktualna_cena_calkowita:,.0f} PLN
- Stawka za m²:          {aktualna_cena_m2:,.0f} PLN/m²
- Zakres ofert podobnych: {aktualne_widelki_min:,.0f} - {aktualne_widelki_max:,.0f} PLN

Podobne oferty z rynku:
{lista_sasiedzi_txt.replace(',', ' ')}
Wycena sporządzona na podstawie bazy ofert rynkowych (N={laczna_baza:,}).
""".replace(',', ' ')

    btn_wyslij.config(state="disabled", text="⏳ Wysyłanie...")
    threading.Thread(
        target=wyslij_email_watek,
        args=(sender_email, sender_password, recipient_email, miasto, metraz, tresc_raportu),
        daemon=True
    ).start()


# --- NOWOCZESNY INTERFEJS GRAFICZNY (TKINTER GUI) ---
root = tk.Tk()
root.title("Kalkulator Wyceny Nieruchomości")
root.geometry("1000x800")
root.minsize(960, 750)

# Paleta kolorów (Nowoczesny Styl Fintech / PropTech)
FONT_FAMILY = "Segoe UI"
BG_MAIN = "#0F172A"          # Głęboki grafit/navy tła głównego
CARD_BG_LEFT = "#FFFFFF"     # Czysta biel dla panelu wejściowego
CARD_BORDER = "#E2E8F0"      # Subtelna ramka
TEXT_PRIMARY = "#0F172A"     # Ciemny tekst
TEXT_MUTED = "#64748B"       # Szary tekst pomocniczy
TEXT_LABEL = "#334155"       # Etykiety pól
ACCENT_BLUE = "#2563EB"      # Główny błękit
ACCENT_EMERALD = "#10B981"   # Zielony akcent wyceny
CARD_BG_RIGHT = "#1E293B"    # Panel prawy - Dark Slate
CARD_RIGHT_INNER = "#0F172A" # Karty wewnętrzne panelu prawego

root.configure(bg=BG_MAIN)

# Konfiguracja stylów ttk dla nowoczesnych kontrolek
style = ttk.Style()
style.theme_use("clam")

style.configure(
    "TCombobox",
    fieldbackground="#F8FAFC",
    background="#E2E8F0",
    foreground="#0F172A",
    bordercolor="#CBD5E1",
    lightcolor="#E2E8F0",
    darkcolor="#E2E8F0",
    arrowcolor="#0F172A",
    font=(FONT_FAMILY, 9),
    padding=5
)
style.map("TCombobox", fieldbackground=[("readonly", "#F8FAFC")])

# 1. Pasek Nagłówka (Top Navigation Bar)
header_frame = tk.Frame(root, bg=BG_MAIN, padx=20, pady=12)
header_frame.pack(fill="x")

header_left = tk.Frame(header_frame, bg=BG_MAIN)
header_left.pack(side="left", fill="y")

title_lbl = tk.Label(
    header_left,
    text="Kalkulator Wyceny Nieruchomości",
    font=(FONT_FAMILY, 15, "bold"),
    bg=BG_MAIN,
    fg="#F8FAFC"
)
title_lbl.pack(anchor="w")

# Główny kontener podzielony na lewo (parametry) i prawo (wynik i akcje)
content_frame = tk.Frame(root, bg=BG_MAIN, padx=20, pady=6)
content_frame.pack(fill="both", expand=True)

# ==========================================
# --- PANEL LEWY: PARAMETRY WEJŚCIOWE ---
# ==========================================
card_left_outer = tk.Frame(content_frame, bg="#334155", padx=1, pady=1)
card_left_outer.pack(side="left", fill="both", expand=True, padx=(0, 10))

card_left = tk.Frame(card_left_outer, bg=CARD_BG_LEFT, padx=18, pady=16)
card_left.pack(fill="both", expand=True)

# Tytuł panelu lewego
lbl_panel_left_title = tk.Label(
    card_left,
    text="PARAMETRY MIESZKANIA",
    font=(FONT_FAMILY, 11, "bold"),
    bg=CARD_BG_LEFT,
    fg=ACCENT_BLUE
)
lbl_panel_left_title.pack(anchor="w", pady=(0, 2))

lbl_panel_left_desc = tk.Label(
    card_left,
    text="Wprowadź dane nieruchomości, aby oszacować jej aktualną wartość",
    font=(FONT_FAMILY, 8),
    bg=CARD_BG_LEFT,
    fg=TEXT_MUTED
)
lbl_panel_left_desc.pack(anchor="w", pady=(0, 12))

# Siatka formularza
form_grid = tk.Frame(card_left, bg=CARD_BG_LEFT)
form_grid.pack(fill="x", expand=False)
form_grid.columnconfigure(1, weight=1)

lbl_opts = {"bg": CARD_BG_LEFT, "fg": TEXT_LABEL, "font": (FONT_FAMILY, 9, "bold")}

# 1. Miasto
tk.Label(form_grid, text="Miasto:", **lbl_opts).grid(row=0, column=0, sticky="w", pady=(4, 0))
miasta_lista = sorted(list(BAZA_KNN.keys())) if BAZA_KNN else ["Szczecin", "Warszawa"]
combo_miasto = ttk.Combobox(form_grid, values=miasta_lista, state="readonly", font=(FONT_FAMILY, 9))
combo_miasto.set("Warszawa" if "Warszawa" in miasta_lista else miasta_lista[0])
combo_miasto.grid(row=0, column=1, sticky="ew", pady=(4, 0), padx=(10, 0))
combo_miasto.bind("<<ComboboxSelected>>", on_miasto_zmienione)

# Pasek info o rynku
label_miasto_info = tk.Label(form_grid, text="", font=(FONT_FAMILY, 8), bg=CARD_BG_LEFT, fg="#2563EB")
label_miasto_info.grid(row=1, column=1, sticky="w", pady=(2, 6), padx=(10, 0))

# 2. Metraż
tk.Label(form_grid, text="Powierzchnia (m²):", **lbl_opts).grid(row=2, column=0, sticky="w", pady=4)
metraz_container = tk.Frame(form_grid, bg=CARD_BG_LEFT)
metraz_container.grid(row=2, column=1, sticky="ew", pady=4, padx=(10, 0))

scale_metraz = tk.Scale(
    metraz_container,
    from_=18, to=150,
    orient="horizontal",
    showvalue=False,
    command=oblicz_wycene,
    bg=CARD_BG_LEFT,
    fg=TEXT_PRIMARY,
    troughcolor="#E2E8F0",
    activebackground=ACCENT_BLUE,
    highlightthickness=0,
    bd=0,
    relief="flat"
)
scale_metraz.pack(side="left", fill="x", expand=True, padx=(0, 8))
scale_metraz.set(52)

lbl_metraz_badge = tk.Label(
    metraz_container,
    text="52 m²",
    font=(FONT_FAMILY, 9, "bold"),
    bg="#EFF6FF",
    fg=ACCENT_BLUE,
    width=7,
    pady=2,
    relief="solid",
    bd=1
)
lbl_metraz_badge.pack(side="right")

# 3. Liczba pokoi
tk.Label(form_grid, text="Liczba pokoi:", **lbl_opts).grid(row=3, column=0, sticky="w", pady=5)
combo_pokoje = ttk.Combobox(form_grid, values=["1 pokój (kawalerka)", "2 pokoje", "3 pokoje", "4 pokoje", "5+ pokoi"], state="readonly", font=(FONT_FAMILY, 9))
combo_pokoje.set("2 pokoje")
combo_pokoje.grid(row=3, column=1, sticky="ew", pady=5, padx=(10, 0))
combo_pokoje.bind("<<ComboboxSelected>>", oblicz_wycene)

# 4. Odległość od centrum
tk.Label(form_grid, text="Lokalizacja (odległość):", **lbl_opts).grid(row=4, column=0, sticky="w", pady=5)
opcje_odleglosc = [
    "Dowolna lokalizacja / Całe miasto",
    "Ścisłe centrum (<2.5 km)",
    "Strefa śródmiejska (2.5-5 km)",
    "Strefa miejska (5-9 km)",
    "Peryferia / Obrzeża (>9 km)"
]
combo_odleglosc = ttk.Combobox(form_grid, values=opcje_odleglosc, state="readonly", font=(FONT_FAMILY, 9))
combo_odleglosc.set(opcje_odleglosc[2])
combo_odleglosc.grid(row=4, column=1, sticky="ew", pady=5, padx=(10, 0))
combo_odleglosc.bind("<<ComboboxSelected>>", oblicz_wycene)

# 5. Typ budynku
tk.Label(form_grid, text="Typ budynku:", **lbl_opts).grid(row=5, column=0, sticky="w", pady=5)
opcje_typ = ["Dowolny typ budynku", "Blok mieszkalny", "Apartamentowiec", "Kamienica"]
combo_typ = ttk.Combobox(form_grid, values=opcje_typ, state="readonly", font=(FONT_FAMILY, 9))
combo_typ.set("Blok mieszkalny")
combo_typ.grid(row=5, column=1, sticky="ew", pady=5, padx=(10, 0))
combo_typ.bind("<<ComboboxSelected>>", oblicz_wycene)

# 6. Epoka budowy
tk.Label(form_grid, text="Rok / Epoka budowy:", **lbl_opts).grid(row=6, column=0, sticky="w", pady=5)
opcje_epoka = [
    "Dowolny rok budowy",
    "Nowe budownictwo (>2010)",
    "Budownictwo współczesne (1991-2010)",
    "Wielka płyta / PRL (1950-1990)",
    "Historyczna / Przedwojenna (<1950)"
]
combo_epoka = ttk.Combobox(form_grid, values=opcje_epoka, state="readonly", font=(FONT_FAMILY, 9))
combo_epoka.set("Nowe budownictwo (>2010)")
combo_epoka.grid(row=6, column=1, sticky="ew", pady=5, padx=(10, 0))
combo_epoka.bind("<<ComboboxSelected>>", oblicz_wycene)

# 7. Udogodnienia (Checkboxy)
tk.Label(form_grid, text="Udogodnienia lokalu:", **lbl_opts).grid(row=7, column=0, sticky="nw", pady=(10, 2))
ramka_check = tk.Frame(form_grid, bg=CARD_BG_LEFT)
ramka_check.grid(row=7, column=1, sticky="w", pady=(8, 2), padx=(10, 0))

opcje_chk = {
    "bg": CARD_BG_LEFT,
    "fg": TEXT_PRIMARY,
    "font": (FONT_FAMILY, 9),
    "activebackground": CARD_BG_LEFT,
    "selectcolor": CARD_BG_LEFT,
    "cursor": "hand2"
}

var_winda = tk.BooleanVar(value=True)
chk_winda = tk.Checkbutton(ramka_check, text="Winda w budynku", variable=var_winda, command=oblicz_wycene, **opcje_chk)
chk_winda.pack(anchor="w", pady=1)

var_balkon = tk.BooleanVar(value=True)
chk_balkon = tk.Checkbutton(ramka_check, text="Balkon / Taras", variable=var_balkon, command=oblicz_wycene, **opcje_chk)
chk_balkon.pack(anchor="w", pady=1)

var_parking = tk.BooleanVar(value=True)
chk_parking = tk.Checkbutton(ramka_check, text="Miejsce postojowe / Garaż", variable=var_parking, command=oblicz_wycene, **opcje_chk)
chk_parking.pack(anchor="w", pady=1)

# Dynamiczne ostrzeżenie o anomalii metrażu/pokoi
frame_ostrzezenie = tk.Frame(card_left, bg="#FEF2F2", padx=10, pady=8, bd=1, relief="solid")
label_ostrzezenie = tk.Label(frame_ostrzezenie, text="", font=(FONT_FAMILY, 8), bg="#FEF2F2", wraplength=400, justify="left")
label_ostrzezenie.pack(fill="x")


# ==========================================
# --- PANEL PRAWY: WYNIK, ANALIZA I AKCJE ---
# ==========================================
card_right_outer = tk.Frame(content_frame, bg="#334155", padx=1, pady=1)
card_right_outer.pack(side="right", fill="both", expand=True)

card_right = tk.Frame(card_right_outer, bg=CARD_BG_RIGHT, padx=18, pady=16)
card_right.pack(fill="both", expand=True)

# Tytuł panelu prawego
top_res_bar = tk.Frame(card_right, bg=CARD_BG_RIGHT)
top_res_bar.pack(fill="x", pady=(0, 8))

tk.Label(
    top_res_bar,
    text="SZACOWANA WARTOŚĆ",
    font=(FONT_FAMILY, 10, "bold"),
    bg=CARD_BG_RIGHT,
    fg="#94A3B8"
).pack(side="left")

badge_jakosc = tk.Label(
    top_res_bar,
    text="● WYSOKA WIARYGODNOŚĆ",
    font=(FONT_FAMILY, 8, "bold"),
    bg="#064E3B",
    fg="#34D399",
    padx=8,
    pady=2
)
badge_jakosc.pack(side="right")

# Karta Szacowanej Wartości Całkowitej
card_val_box = tk.Frame(card_right, bg=CARD_RIGHT_INNER, padx=16, pady=12, bd=1, relief="solid")
card_val_box.pack(fill="x", pady=(0, 10))

tk.Label(
    card_val_box,
    text="SZACOWANA WARTOŚĆ RYNKOWA",
    font=(FONT_FAMILY, 8, "bold"),
    bg=CARD_RIGHT_INNER,
    fg="#94A3B8"
).pack(anchor="center")

label_wynik_calkowita = tk.Label(
    card_val_box,
    text="--- PLN",
    font=(FONT_FAMILY, 24, "bold"),
    bg=CARD_RIGHT_INNER,
    fg=ACCENT_EMERALD
)
label_wynik_calkowita.pack(anchor="center", pady=(2, 4))

# Pod-metryki w siatce (cena/m2 oraz widełki)
grid_metrics = tk.Frame(card_val_box, bg=CARD_RIGHT_INNER)
grid_metrics.pack(fill="x", pady=(2, 0))
grid_metrics.columnconfigure(0, weight=1)
grid_metrics.columnconfigure(1, weight=1)

m1_frame = tk.Frame(grid_metrics, bg=CARD_RIGHT_INNER)
m1_frame.grid(row=0, column=0, sticky="ew")
tk.Label(m1_frame, text="Stawka za m²:", font=(FONT_FAMILY, 8), bg=CARD_RIGHT_INNER, fg="#94A3B8").pack()
label_wynik_m2 = tk.Label(m1_frame, text="--- PLN/m²", font=(FONT_FAMILY, 11, "bold"), bg=CARD_RIGHT_INNER, fg="#60A5FA")
label_wynik_m2.pack()

m2_frame = tk.Frame(grid_metrics, bg=CARD_RIGHT_INNER)
m2_frame.grid(row=0, column=1, sticky="ew")
tk.Label(m2_frame, text="Zakres ofert podobnych:", font=(FONT_FAMILY, 8), bg=CARD_RIGHT_INNER, fg="#94A3B8").pack()
label_widelki = tk.Label(m2_frame, text="--- PLN", font=(FONT_FAMILY, 9, "bold"), bg=CARD_RIGHT_INNER, fg="#F1F5F9")
label_widelki.pack()

# Sekcja ofert referencyjnych (Top 3 sąsiedzi)
tk.Label(
    card_right,
    text="NAJBARDZIEJ PODOBNE OFERTY NA RYNKU:",
    font=(FONT_FAMILY, 8, "bold"),
    bg=CARD_BG_RIGHT,
    fg="#94A3B8"
).pack(anchor="w", pady=(4, 4))

karty_sasiadow_tytul = []
karty_sasiadow_detale = []
karty_sasiadow_cena = []

for i in range(3):
    box_s = tk.Frame(card_right, bg=CARD_RIGHT_INNER, padx=10, pady=5, bd=1, relief="solid")
    box_s.pack(fill="x", pady=2)

    top_row = tk.Frame(box_s, bg=CARD_RIGHT_INNER)
    top_row.pack(fill="x")

    lbl_t = tk.Label(top_row, text=f"Oferta #{i+1}", font=(FONT_FAMILY, 8, "bold"), bg=CARD_RIGHT_INNER, fg="#F1F5F9")
    lbl_t.pack(side="left")

    lbl_c = tk.Label(top_row, text="--- PLN", font=(FONT_FAMILY, 8, "bold"), bg=CARD_RIGHT_INNER, fg="#34D399")
    lbl_c.pack(side="right")

    lbl_d = tk.Label(box_s, text="Parametry...", font=(FONT_FAMILY, 8), bg=CARD_RIGHT_INNER, fg="#94A3B8", anchor="w")
    lbl_d.pack(fill="x")

    karty_sasiadow_tytul.append(lbl_t)
    karty_sasiadow_detale.append(lbl_d)
    karty_sasiadow_cena.append(lbl_c)

# Przycisk PDF
btn_pdf = tk.Button(
    card_right,
    text="📄  Eksportuj Raport Wyceny do PDF",
    font=(FONT_FAMILY, 9, "bold"),
    bg="#DC2626",
    fg="#FFFFFF",
    activebackground="#B91C1C",
    activeforeground="#FFFFFF",
    bd=0,
    cursor="hand2",
    command=generuj_pdf
)
btn_pdf.pack(fill="x", pady=(10, 6), ipady=5)

# Sekcja Wysyłki E-mail
box_email = tk.Frame(card_right, bg=CARD_RIGHT_INNER, padx=12, pady=10, bd=1, relief="solid")
box_email.pack(fill="x", pady=(4, 0))

tk.Label(
    box_email,
    text="WYSYŁKA RAPORTU NA E-MAIL:",
    font=(FONT_FAMILY, 8, "bold"),
    bg=CARD_RIGHT_INNER,
    fg="#94A3B8"
).pack(anchor="w", pady=(0, 6))

env_email = os.getenv("SMTP_SENDER_EMAIL", "")
env_haslo = os.getenv("SMTP_PASSWORD", "")

row_smtp = tk.Frame(box_email, bg=CARD_RIGHT_INNER)
row_smtp.pack(fill="x", pady=2)

tk.Label(row_smtp, text="Nadawca:", font=(FONT_FAMILY, 8), bg=CARD_RIGHT_INNER, fg="#CBD5E1", width=13, anchor="w").pack(side="left")
entry_sender_email = tk.Entry(row_smtp, font=(FONT_FAMILY, 8), bg="#334155", fg="#F8FAFC", insertbackground="white", bd=0)
entry_sender_email.pack(side="left", fill="x", expand=True, ipady=3)
if env_email:
    entry_sender_email.insert(0, env_email)

row_pwd = tk.Frame(box_email, bg=CARD_RIGHT_INNER)
row_pwd.pack(fill="x", pady=2)

tk.Label(row_pwd, text="Hasło SMTP:", font=(FONT_FAMILY, 8), bg=CARD_RIGHT_INNER, fg="#CBD5E1", width=13, anchor="w").pack(side="left")
entry_sender_password = tk.Entry(row_pwd, font=(FONT_FAMILY, 8), bg="#334155", fg="#F8FAFC", insertbackground="white", bd=0, show="*")
entry_sender_password.pack(side="left", fill="x", expand=True, ipady=3)
if env_haslo:
    entry_sender_password.insert(0, env_haslo)

row_rcp = tk.Frame(box_email, bg=CARD_RIGHT_INNER)
row_rcp.pack(fill="x", pady=2)

tk.Label(row_rcp, text="Odbiorca:", font=(FONT_FAMILY, 8), bg=CARD_RIGHT_INNER, fg="#CBD5E1", width=13, anchor="w").pack(side="left")
entry_recipient_email = tk.Entry(row_rcp, font=(FONT_FAMILY, 8), bg="#334155", fg="#F8FAFC", insertbackground="white", bd=0)
entry_recipient_email.pack(side="left", fill="x", expand=True, ipady=3)

btn_wyslij = tk.Button(
    box_email,
    text="✉  Wyślij Raport na E-mail",
    font=(FONT_FAMILY, 9, "bold"),
    bg=ACCENT_BLUE,
    fg="#FFFFFF",
    activebackground="#1D4ED8",
    activeforeground="#FFFFFF",
    bd=0,
    cursor="hand2",
    command=rozpocznij_wysylke_email
)
btn_wyslij.pack(fill="x", pady=(8, 2), ipady=4)

# Inicjalizacja pierwszej kalkulacji i benchmarku rynkowego
on_miasto_zmienione()

if __name__ == "__main__":
    root.mainloop()