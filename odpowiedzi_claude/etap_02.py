"""Etap 2: ranking postaci RPG oparty na normalizacji min–max i NumPy."""

import numpy as np

KOLEJNOSC_STATYSTYK = ("HP", "atak", "obrona", "szybkość")

# Wagi w kolejności kolumn: HP, atak, obrona, szybkość.
WAGI = np.array([0.20, 0.40, 0.25, 0.15], dtype=float)

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


def main():
    waliduj_dane(NAZWY, STATYSTYKI)
    sily = oblicz_sile(STATYSTYKI)
    kolejnosc = ranking(sily)

    print("=== RANKING POSTACI ===")
    for pozycja, indeks in enumerate(kolejnosc, start=1):
        print(f"{pozycja}. {NAZWY[indeks]}: {sily[indeks]:.4f}")


if __name__ == "__main__":
    main()
