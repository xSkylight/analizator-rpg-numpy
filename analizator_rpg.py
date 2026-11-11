"""analizator_rpg.py: ranking, role, filtrowanie, prawdopodobieństwa, testy i raport CSV (NumPy)."""

import csv
from pathlib import Path

import numpy as np

KOLEJNOSC_STATYSTYK = ("HP", "atak", "obrona", "szybkość")
KOL_HP, KOL_ATAK, KOL_OBRONA, KOL_SZYBKOSC = range(4)

# Nazwy statystyk w raporcie na ekranie (kolejność kolumn jak w STATYSTYKI).
NAZWY_STATYSTYK_RAPORT = ("HP", "Atak", "Obrona", "Szybkość")

# Wagi w kolejności kolumn: HP, atak, obrona, szybkość.
WAGI = np.array([0.20, 0.40, 0.25, 0.15], dtype=float)

# Kolejność ról odpowiada indeksom wyznaczanym w przypisz_role.
ROLE = np.array(["Szturmowiec", "Obrońca", "Wszechstronny"], dtype=str)

# Progi filtra.
FILTR_MIN_HP = 100
FILTR_MIN_SZYBKOSC = 10
FILTR_MIN_SILA = 0.45

# Stromość funkcji logistycznej w macierzy prawdopodobieństw.
STROMOSC = 8.0

# Raport CSV.
NAZWA_PLIKU_RAPORTU = "raport_postaci.csv"
NAGLOWEK_RAPORTU = ("Miejsce", "Postać", "Rola", "Siła", "Średnia szansa wygranej")

# Tolerancja testów numerycznych.
TOLERANCJA = 1e-9

NAZWY = np.array(
    [
        "Wojownik",
        "Mag",
        "Łucznik",
        "Paladyn",
        "Łotrzyk",
        "Berserker",
        "Druid",
        "Strażnik",
    ],
    dtype=str,
)

# Kolumny: HP, atak, obrona, szybkość.
STATYSTYKI = np.array(
    [
        [120, 28, 18, 9],
        [75, 44, 6, 13],
        [95, 34, 10, 17],
        [140, 25, 22, 7],
        [80, 39, 8, 21],
        [110, 42, 7, 12],
        [105, 26, 15, 14],
        [155, 18, 26, 5],
    ],
    dtype=float,
)

# Dane testu kolumn stałych: kolumny 0 i 3 są stałe, kolumny 1 i 2 zmienne.
STATYSTYKI_DWA_WIERSZE = np.array(
    [
        [10, 20, 30, 40],
        [10, 50, 60, 40],
    ],
    dtype=float,
)
OCZEKIWANE_DWA_WIERSZE = np.array(
    [
        [0, 0, 0, 0],
        [0, 1, 1, 0],
    ],
    dtype=float,
)


def waliduj_dane(nazwy, statystyki):
    """Sprawdza spójność danych; każde naruszenie zgłasza ValueError."""
    nazwy = np.asarray(nazwy)
    statystyki = np.asarray(statystyki)

    if nazwy.ndim != 1:
        raise ValueError("Nazwy muszą być jednowymiarową tablicą.")

    liczba_nazw = nazwy.shape[0]
    if liczba_nazw < 2:
        raise ValueError("Wymagane są co najmniej dwie postacie.")

    if statystyki.shape != (liczba_nazw, len(KOLEJNOSC_STATYSTYK)):
        raise ValueError(
            f"Macierz statystyk musi mieć kształt "
            f"({liczba_nazw}, {len(KOLEJNOSC_STATYSTYK)}), "
            f"otrzymano {statystyki.shape}."
        )

    if np.unique(nazwy).shape[0] != liczba_nazw:
        raise ValueError("Nazwy postaci muszą być unikatowe.")

    try:
        statystyki_float = statystyki.astype(float)
    except (TypeError, ValueError) as blad:
        raise ValueError("Statystyki muszą być liczbowe.") from blad

    if not np.all(np.isfinite(statystyki_float)):
        raise ValueError("Statystyki muszą być skończone (bez NaN i inf).")

    if np.any(statystyki_float < 0):
        raise ValueError("Statystyki nie mogą być ujemne.")


def normalizuj_min_max(statystyki):
    """Normalizacja min–max osobno dla każdej kolumny (wzdłuż osi postaci).

    Kolumna o stałej wartości (max == min) daje same zera.
    """
    statystyki = np.asarray(statystyki, dtype=float)
    minima = statystyki.min(axis=0)
    zakresy = statystyki.max(axis=0) - minima
    return np.divide(
        statystyki - minima,
        zakresy,
        out=np.zeros_like(statystyki),
        where=zakresy != 0,
    )


def oblicz_sile(statystyki):
    """Siła = macierzowy iloczyn znormalizowanych statystyk i wektora wag."""
    return normalizuj_min_max(statystyki) @ WAGI


def ranking(sily):
    """Indeksy postaci posortowane malejąco po sile.

    Stabilne sortowanie ujemnych sił zachowuje kolejność wejściową przy remisie.
    """
    return np.argsort(-np.asarray(sily, dtype=float), kind="stable")


def macierz_prawdopodobienstw(sily):
    """Macierz P (N×N) przez broadcasting.

    P[i, j] = 1 / (1 + exp(-8 * (siła_i - siła_j))); przekątna wynosi 0.5.
    """
    sily = np.asarray(sily, dtype=float)
    roznice = sily[:, np.newaxis] - sily[np.newaxis, :]
    p = 1.0 / (1.0 + np.exp(-STROMOSC * roznice))
    np.fill_diagonal(p, 0.5)
    return p


def srednie_prawdopodobienstwo_wygranej(p):
    """Średnie P[i, j] po j != i, czyli z pozostałymi N−1 postaciami."""
    p = np.asarray(p, dtype=float)
    n = p.shape[0]
    return (p.sum(axis=1) - np.diag(p)) / (n - 1)


def przypisz_role(statystyki):
    """Wektorowo przypisuje role według surowych statystyk.

    Szturmowiec: atak >= 35 i szybkość >= 12.
    Obrońca: HP >= 120 i obrona >= 18.
    Reszta: Wszechstronny. Szturmowiec ma pierwszeństwo przy obu pasujących regułach.
    """
    statystyki = np.asarray(statystyki, dtype=float)
    szturmowiec = (statystyki[:, KOL_ATAK] >= 35) & (statystyki[:, KOL_SZYBKOSC] >= 12)
    obronca = (statystyki[:, KOL_HP] >= 120) & (statystyki[:, KOL_OBRONA] >= 18)

    # np.select zwraca pierwszą pasującą regułę, więc kolejność nadaje priorytet.
    indeksy_rol = np.select([szturmowiec, obronca], [0, 1], default=2)
    return ROLE[indeksy_rol]


def maska_filtra(statystyki, sily):
    """Maska: HP >= 100, szybkość >= 10 i siła >= 0.45."""
    statystyki = np.asarray(statystyki, dtype=float)
    sily = np.asarray(sily, dtype=float)
    return (
        (statystyki[:, KOL_HP] >= FILTR_MIN_HP)
        & (statystyki[:, KOL_SZYBKOSC] >= FILTR_MIN_SZYBKOSC)
        & (sily >= FILTR_MIN_SILA)
    )


def formatuj_wybrane(nazwy, maska):
    """Nazwy z maski w kolejności wejścia, po przecinku, lub 'Brak'."""
    wybrane = np.asarray(nazwy)[np.asarray(maska, dtype=bool)]
    if wybrane.size == 0:
        return "Brak"
    return ", ".join(wybrane.tolist())


def podsumuj_kolumny(statystyki):
    """Zwraca minimum, średnią i maksimum surowych statystyk każdej kolumny."""
    statystyki = np.asarray(statystyki, dtype=float)
    return statystyki.min(axis=0), statystyki.mean(axis=0), statystyki.max(axis=0)


def testuj_normalizacje(statystyki):
    """Sprawdza normalizację min–max; naruszenie zgłasza AssertionError.

    Dla każdej zmiennej kolumny minimum musi wynosić 0, a maksimum 1.
    Kolumna stała musi składać się z samych zer. Wszystkie wartości leżą w [0, 1].
    """
    statystyki = np.asarray(statystyki, dtype=float)
    znorm = normalizuj_min_max(statystyki)

    if znorm.shape != statystyki.shape:
        raise AssertionError("Normalizacja zmieniła kształt macierzy.")
    if not np.all(np.isfinite(znorm)):
        raise AssertionError("Normalizacja dała wartości nieskończone lub NaN.")
    if np.any(znorm < -TOLERANCJA) or np.any(znorm > 1 + TOLERANCJA):
        raise AssertionError("Znormalizowane wartości wychodzą poza [0, 1].")

    zmienne = statystyki.max(axis=0) > statystyki.min(axis=0)
    if not np.allclose(znorm.min(axis=0)[zmienne], 0.0, atol=TOLERANCJA):
        raise AssertionError("Minimum zmiennej kolumny po normalizacji nie wynosi 0.")
    if not np.allclose(znorm.max(axis=0)[zmienne], 1.0, atol=TOLERANCJA):
        raise AssertionError("Maksimum zmiennej kolumny po normalizacji nie wynosi 1.")
    if not np.allclose(znorm[:, ~zmienne], 0.0, atol=TOLERANCJA):
        raise AssertionError("Kolumna stała po normalizacji nie składa się z zer.")


def testuj_normalizacje_kolumny_stale():
    """Test na dwóch wierszach z kolumnami stałymi i zmiennymi.

    Sprawdza dokładny wynik [[0, 0, 0, 0], [0, 1, 1, 0]] oraz brak dzielenia
    przez zero (błędy zmiennoprzecinkowe są zamieniane na wyjątki).
    """
    try:
        with np.errstate(all="raise"):
            znorm = normalizuj_min_max(STATYSTYKI_DWA_WIERSZE)
    except FloatingPointError as blad:
        raise AssertionError(
            "Normalizacja dwóch wierszy spowodowała dzielenie przez zero "
            "lub inny błąd zmiennoprzecinkowy."
        ) from blad

    if not np.all(np.isfinite(znorm)):
        raise AssertionError("Normalizacja dwóch wierszy dała NaN lub inf.")
    if znorm.shape != OCZEKIWANE_DWA_WIERSZE.shape:
        raise AssertionError("Normalizacja dwóch wierszy zmieniła kształt macierzy.")
    if not np.allclose(znorm, OCZEKIWANE_DWA_WIERSZE, atol=TOLERANCJA):
        raise AssertionError(
            f"Normalizacja dwóch wierszy dała {znorm.tolist()}, "
            f"oczekiwano {OCZEKIWANE_DWA_WIERSZE.tolist()}."
        )

    # Te same dane muszą spełniać też ogólne własności normalizacji.
    testuj_normalizacje(STATYSTYKI_DWA_WIERSZE)


def testuj_pojedynki(p):
    """Sprawdza macierz P; naruszenie zgłasza AssertionError.

    Przekątna ma wynosić 0.5, a P + P.T ma być macierzą samych jedynek.
    """
    p = np.asarray(p, dtype=float)

    if p.ndim != 2 or p.shape[0] != p.shape[1]:
        raise AssertionError("Macierz P musi być kwadratowa.")
    if not np.allclose(np.diag(p), 0.5, atol=TOLERANCJA):
        raise AssertionError("Przekątna macierzy P nie wynosi 0.5.")
    if not np.allclose(p + p.T, 1.0, atol=TOLERANCJA):
        raise AssertionError("Suma P + P.T nie jest równa 1.")


def testuj_pojedynek_dwoch_postaci():
    """Próba dla dwóch postaci o sile 0 i 1.

    Średnia wygrana każdej postaci ma równać się jej szansie w pojedynku
    z drugą postacią, bo przy N = 2 jest tylko jeden przeciwnik.
    """
    sily = np.array([0.0, 1.0])
    p = macierz_prawdopodobienstw(sily)
    testuj_pojedynki(p)
    wygrana = srednie_prawdopodobienstwo_wygranej(p)

    oczekiwane_slabszy = 1.0 / (1.0 + np.exp(STROMOSC))
    oczekiwane_silniejszy = 1.0 / (1.0 + np.exp(-STROMOSC))

    if not np.isclose(wygrana[0], p[0, 1], atol=TOLERANCJA):
        raise AssertionError("Średnia wygrana postaci o sile 0 różni się od P[0, 1].")
    if not np.isclose(wygrana[1], p[1, 0], atol=TOLERANCJA):
        raise AssertionError("Średnia wygrana postaci o sile 1 różni się od P[1, 0].")
    if not np.isclose(p[0, 1], oczekiwane_slabszy, atol=TOLERANCJA):
        raise AssertionError("P[0, 1] nie zgadza się ze wzorem logistycznym.")
    if not np.isclose(p[1, 0], oczekiwane_silniejszy, atol=TOLERANCJA):
        raise AssertionError("P[1, 0] nie zgadza się ze wzorem logistycznym.")
    if not np.isclose(wygrana.sum(), 1.0, atol=TOLERANCJA):
        raise AssertionError("Średnie wygrane dwóch postaci nie sumują się do 1.")


def zapisz_raport(sciezka, nazwy, role, sily, wygrana, kolejnosc):
    """Zapisuje raport CSV (separator ';', UTF-8 z BOM) w kolejności rankingu.

    Siła ma cztery miejsca po kropce, a procent dwa miejsca po kropce.
    Zwraca nazwę zapisanego pliku.
    """
    sciezka = Path(sciezka)
    with open(sciezka, "w", encoding="utf-8-sig", newline="") as plik:
        autor = csv.writer(plik, delimiter=";")
        autor.writerow(NAGLOWEK_RAPORTU)
        for pozycja, indeks in enumerate(kolejnosc, start=1):
            autor.writerow(
                (
                    pozycja,
                    str(nazwy[indeks]),
                    str(role[indeks]),
                    f"{sily[indeks]:.4f}",
                    f"{wygrana[indeks] * 100:.2f}%",
                )
            )
    return sciezka.name


def main():
    waliduj_dane(NAZWY, STATYSTYKI)
    sily = oblicz_sile(STATYSTYKI)
    kolejnosc = ranking(sily)
    role = przypisz_role(STATYSTYKI)
    maska = maska_filtra(STATYSTYKI, sily)
    p = macierz_prawdopodobienstw(sily)
    wygrana = srednie_prawdopodobienstwo_wygranej(p)

    # Testy działają po cichu; przy naruszeniu zgłaszają AssertionError.
    testuj_normalizacje(STATYSTYKI)
    testuj_normalizacje_kolumny_stale()
    testuj_pojedynki(p)
    testuj_pojedynek_dwoch_postaci()

    print("=== RANKING POSTACI RPG ===")
    for pozycja, indeks in enumerate(kolejnosc, start=1):
        print(
            f"{pozycja}. | {NAZWY[indeks]:<12} | {role[indeks]:<13} | "
            f"siła: {sily[indeks]:.3f} | "
            f"średnia szansa: {wygrana[indeks] * 100:.1f}%"
        )

    print()
    print("=== POSTACIE SPEŁNIAJĄCE WYMAGANIA ===")
    print(formatuj_wybrane(NAZWY, maska))

    print()
    print("=== STATYSTYKI GRUPY ===")
    minima, srednie, maksima = podsumuj_kolumny(STATYSTYKI)
    for i, nazwa in enumerate(NAZWY_STATYSTYK_RAPORT):
        print(
            f"{nazwa:<10} min={minima[i]:.1f}, "
            f"średnia={srednie[i]:.1f}, max={maksima[i]:.1f}"
        )

    nazwa_pliku = zapisz_raport(
        NAZWA_PLIKU_RAPORTU, NAZWY, role, sily, wygrana, kolejnosc
    )
    print()
    print(f"Zapisano: {nazwa_pliku}")


if __name__ == "__main__":
    main()
