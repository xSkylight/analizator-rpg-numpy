"""Etap 4: ranking, role, filtrowanie i prawdopodobieństwa wygranej (NumPy)."""

import numpy as np

KOLEJNOSC_STATYSTYK = ("HP", "atak", "obrona", "szybkość")
KOL_HP, KOL_ATAK, KOL_OBRONA, KOL_SZYBKOSC = range(4)

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


def main():
    waliduj_dane(NAZWY, STATYSTYKI)
    sily = oblicz_sile(STATYSTYKI)
    kolejnosc = ranking(sily)
    role = przypisz_role(STATYSTYKI)
    maska = maska_filtra(STATYSTYKI, sily)
    p = macierz_prawdopodobienstw(sily)
    wygrana = srednie_prawdopodobienstwo_wygranej(p)

    print("=== RANKING POSTACI ===")
    for pozycja, indeks in enumerate(kolejnosc, start=1):
        print(
            f"{pozycja}. {NAZWY[indeks]} ({role[indeks]}): "
            f"siła {sily[indeks]:.3f}, "
            f"średnia szansa wygranej {wygrana[indeks] * 100:.1f}%"
        )

    print(
        f"\nFiltr (HP >= {FILTR_MIN_HP}, szybkość >= {FILTR_MIN_SZYBKOSC}, "
        f"siła >= {FILTR_MIN_SILA}): {formatuj_wybrane(NAZWY, maska)}"
    )

    minima, srednie, maksima = podsumuj_kolumny(STATYSTYKI)
    print("\n=== STATYSTYKI SUROWE ===")
    for i, kolumna in enumerate(KOLEJNOSC_STATYSTYK):
        print(
            f"{kolumna}: min {minima[i]:.1f}, "
            f"średnia {srednie[i]:.1f}, max {maksima[i]:.1f}"
        )


if __name__ == "__main__":
    main()
