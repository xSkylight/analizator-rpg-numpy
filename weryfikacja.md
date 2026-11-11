# Weryfikacja analizatora RPG

Uruchomienie lokalne w Pythonie 3.12.14 i NumPy 2.3.5. Wersję B z materiałów oraz końcowy kod Claude uruchomiono w osobnych katalogach. Porównano obliczenia, konsolę i pliki `raport_postaci.csv`.

| Wymaganie | Wynik | Dowód lub wynik testu |
| --- | --- | --- |
| Dane w NumPy i obliczenia wektorowe | Działa | Tablica 8 nazw i macierz float 8×4 są zgodne z B. Normalizacja działa na osi postaci, a siła jest iloczynem macierzowym z wagami. |
| Walidacja danych | Działa | Osiem błędnych wejść wywołało `ValueError`. Sprawdzono zły wymiar, liczbę kolumn i wierszy, jedną postać, duplikat, liczbę ujemną, NaN i nieskończoność. |
| Normalizacja i stabilny ranking | Działa | Siły wszystkich ośmiu postaci zgodne z B. Dwie identyczne postacie mają siłę zero i zachowują kolejność wejścia. Kolumny stałe dają zera bez dzielenia przez zero. |
| Role i priorytet reguł | Działa | Role zgodne dla wszystkich ośmiu postaci. Dla HP 120, ataku 35, obrony 18 i szybkości 12 wybrano Szturmowca. |
| Filtr i statystyki | Działa | Z danych startowych przechodzi tylko Berserker. Wartości dokładnie na progach przechodzą, a pusty wynik to `Brak`. Statystyki HP to 75.0 / 110.0 / 155.0, pozostałe trzy kolumny też są zgodne z B. |
| Pojedynki i średnie bez przekątnej | Działa | Macierz jest zgodna z B, przekątna wynosi 0.5 i `P + P.T = 1`. Przy dwóch postaciach średnia równa się szansie w jednym pojedynku. |
| Raport konsolowy | Działa | Zgodne sekcje, kolejność, role i wartości. Berserker jest pierwszy z siłą 0.535 i szansą 64.0%. Kod Claude ma dodatkową kreskę po miejscu i nie wypisuje dwukropka po etykiecie kolumny. |
| Testy wykonywane przy uruchomieniu | Działa | Końcowe `main()` uruchamia testy z mieszanymi stałymi i zmiennymi kolumnami oraz przypadek dwóch przeciwników. |
| Eksport CSV | Działa | Pliki A po przebudowie i B są identyczne bajtowo. 348 bajtów, UTF-8 z BOM, separator średnik, pięć kolumn i osiem wierszy danych. |

### Wynik CSV

Pierwszy wiersz danych to `1;Berserker;Szturmowiec;0.5349;63.98%`. To wartości z pliku, z dokładnością zapisu CSV. Kolejność rankingu to Berserker, Łotrzyk, Paladyn, Mag, Łucznik, Wojownik, Strażnik, Druid.

SHA-256 obu plików: `1daf6136d3bf48f00a6f3728c9789d24cb0ea7252e7da6324125a9f03f1f5c55`.

### Dowód drugiej korekty

W kopii kodu z etapu 5 zasymulowano błąd ustawiający wartość 0.5 w stałych kolumnach. Test na danych ośmiu postaci przeszedł. Przypadek z dwiema stałymi kolumnami wykrył błąd. Końcowy kod otrzymany od Claude zawiera ten przypadek w testach uruchamianych przez `main()`.

### Uruchomienie

```sh
python -m pip install numpy
python analizator_rpg.py
```
