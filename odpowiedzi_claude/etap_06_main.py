def main():
    waliduj_dane(NAZWY, STATYSTYKI)
    sily = oblicz_sile(STATYSTYKI)
    kolejnosc = ranking(sily)
    role = przypisz_role(STATYSTYKI)
    maska = maska_filtra(STATYSTYKI, sily)
    p = macierz_prawdopodobienstw(sily)
    wygrana = srednie_prawdopodobienstwo_wygranej(p)

    # Testy działają jak dotąd; przy naruszeniu zgłaszają AssertionError.
    testuj_normalizacje(STATYSTYKI)
    testuj_pojedynki(p)

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
