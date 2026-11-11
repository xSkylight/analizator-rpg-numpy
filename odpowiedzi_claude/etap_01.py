"""Etap 1: system porównywania postaci RPG oparty na NumPy."""

import numpy as np

KOLEJNOSC_STATYSTYK = ("HP", "atak", "obrona", "szybkość")

# Wagi w kolejności: HP, atak, obrona, szybkość (jak w oryginalnym wzorze).
WAGI = np.array([0.1, 2.0, 1.5, 1.0], dtype=float)

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


def oblicz_sile(statystyki):
    """Wektorowo liczy siłę: HP*0.1 + atak*2 + obrona*1.5 + szybkość."""
    return np.asarray(statystyki, dtype=float) @ WAGI


def znajdz_najsilniejsza(nazwy, sily):
    """Zwraca nazwę postaci o największej sile (przy remisie pierwsza)."""
    return str(nazwy[int(np.argmax(sily))])


def main():
    waliduj_dane(NAZWY, STATYSTYKI)
    sily = oblicz_sile(STATYSTYKI)

    print("=== SIŁA POSTACI ===")
    for nazwa, sila in zip(NAZWY, sily):
        print(f"{nazwa}: {sila:.2f}")

    print(f"\nNajsilniejsza postać: {znajdz_najsilniejsza(NAZWY, sily)}")
    print(f"Średnia siła: {sily.mean():.2f}")


if __name__ == "__main__":
    main()
