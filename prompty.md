# Przebudowa analizatora postaci RPG

## Zmiany zaobserwowane między A i B

- Lista słowników przechodzi w równoległe tablice nazw i statystyk. Każdy wiersz macierzy odpowiada jednej postaci.
- Walidacja obejmuje kształt macierzy, minimum dwie postacie, unikatowe nazwy i skończone, nieujemne wartości.
- Surowa suma ważona ustępuje normalizacji min–max i nowym wagom. Stała kolumna wnosi zero do wyniku.
- Obliczenia wykorzystują operacje NumPy na tablicach. Macierz pojedynków wymaga O(N²) czasu i pamięci. Pętle służą głównie do formatowania i eksportu.
- Zamiast wskazania zwycięzcy i średniej siły jest stabilny ranking wszystkich postaci.
- Role wynikają z progów surowych statystyk i mają ustalony priorytet przy nakładaniu się reguł.
- Filtr łączy trzy warunki, a statystyki grupowe obejmują minimum, średnią i maksimum dla każdej kolumny.
- Model logistyczny wyznacza prawdopodobieństwo pojedynku, a średnia pomija przekątną.
- Uruchomienie programu wykonuje testy normalizacji i macierzy pojedynków. Import modułu jest bez efektów ubocznych.
- CSV zawiera pozycję, nazwę, rolę, siłę i średnią szansę wygranej w kolejności rankingu.

## Prompty

### 1. Tablice i walidacja

Przenosi dane do NumPy i zabezpiecza punkt wejścia.

> Etap 1. Przenieś nazwy do jednowymiarowej tablicy NumPy, a statystyki do macierzy float o kształcie (8, 4), w kolejności HP, atak, obrona, szybkość. Zachowaj wszystkie dane i kolejność. Dodaj walidację macierzy o kształcie (liczba nazw, 4), minimum dwóch postaci, unikatowych nazw oraz skończonych i nieujemnych statystyk. Każde naruszenie ma powodować ValueError. Zachowaj na tym etapie dotychczasowy wynik liczbowy, lecz oblicz go wektorowo. Rozdziel funkcje i main(), bez efektów ubocznych przy imporcie. Zwróć kompletny kod Python w jednym bloku.

### 2. Siła i ranking

Zastępuje sumę ważoną surowych wartości i raport pojedynczego zwycięzcy.

> Dodaj normalizację min–max osobno dla każdej kolumny wzdłuż osi postaci. Znormalizowane statystyki pomnóż macierzowo przez wektor wag 0.20, 0.40, 0.25, 0.15 w kolejności kolumn. Usuń stary wzór. Posortuj malejąco po sile, zachowując kolejność wejściową przy remisie, i wypisz pozycje od 1. Użyj operacji NumPy do normalizacji, obliczeń i sortowania. Zwróć pełny kod.

### 3. Role, filtr i statystyki

Rozbudowuje raport o klasyfikację, selekcję i podsumowanie grupy.

> Nadaj role wektorowo według surowych statystyk. Atak ≥ 35 i szybkość ≥ 12 oznaczają Szturmowca, HP ≥ 120 i obrona ≥ 18 oznaczają Obrońcę, reszta to Wszechstronny. Jeśli obie reguły pasują, wybierz Szturmowca. Wyświetl role w rankingu. Zbuduj maskę HP ≥ 100, szybkość ≥ 10 i siła ≥ 0.45. Wypisz nazwy w kolejności wejścia, po przecinku, lub Brak. Pokaż min, średnią i max surowych statystyk każdej kolumny z jednym miejscem po kropce. Zwróć pełny kod.

### 4. Pojedynki

Dodaje deterministyczne prawdopodobieństwa i metrykę wiersza.

> Oblicz macierz P o rozmiarze N×N przez broadcasting. Element P[i,j] to 1/(1+exp(-8*(siła_i-siła_j))). Przekątna ma wynosić 0.5. Dodaj do rankingu średnie prawdopodobieństwo wygranej z pozostałymi N−1 postaciami i pokaż je jako procent z jednym miejscem po kropce. Siłę w rankingu wyświetl z trzema miejscami. Zwróć pełny kod.

### 5. Eksport i testy

Utrwala wyniki w CSV i dodaje automatyczne kontrole.

> Zapisuj raport_postaci.csv do bieżącego katalogu przez moduł csv z separatorem średnik i kodowaniem UTF-8 z BOM. Nagłówek w kolejności: Miejsce;Postać;Rola;Siła;Średnia szansa wygranej. Zachowaj kolejność rankingu, numerację od 1, cztery miejsca po kropce dla siły i dwie cyfry po kropce dla procentu. Dodaj do main() testy normalizacji i pojedynków. Sprawdź wartości 0 i 1 dla zmiennej kolumny, przekątną 0.5 i sumę P + P.T równą 1. Wypisz nazwę zapisanego pliku. Zwróć pełny kod w jednym bloku.

## Przebieg rozmowy

Użyto nowej rozmowy z Claude Sonnet 5.5 Medium. Do pierwszego promptu dołączono kod A. Kolejne wiadomości zawierały wymagania i kryteria odbioru. Kod B uruchomiono lokalnie jako wzorzec.

### 6. Korekta raportu konsolowego

Uruchomienia wersji po etapach 4 i 5 dały poprawne liczby, ale nagłówek rankingu był skrócony, filtr nie miał własnej sekcji, a nazwy kolumn i etykiety różniły się od B. Eksport z etapu 5 był już identyczny bajtowo z B. Korekta doprecyzowuje układ konsoli.

> Korekta raportu. W poprzedniej wersji ranking miał skrócony nagłówek, filtr był jedną linią z opisem warunków, a podsumowanie miało nagłówek STATYSTYKI SUROWE. Użyj kolejno trzech nagłówków: === RANKING POSTACI RPG ===, === POSTACIE SPEŁNIAJĄCE WYMAGANIA ===, === STATYSTYKI GRUPY ===. Pod filtrem wypisz wyłącznie listę nazw lub Brak. W rankingu rozdziel pola pionowymi kreskami, wyrównaj nazwę i rolę do lewej na 12 i 13 znaków, a wyniki poprzedź etykietami „siła:” i „średnia szansa:”. W statystykach użyj nazw HP, Atak, Obrona, Szybkość, wyrównanych do lewej na 10 znaków, oraz etykiet min=, średnia=, max=. Oddziel sekcje pustymi wierszami. Zakończ komunikatem „Zapisano: raport_postaci.csv”. Zachowaj obliczenia i CSV. Pokaż zmienione main() oraz podaj dokładnie, jakie przypadki sprawdza dotychczasowy test normalizacji.

### 7. Korekta testu stałej kolumny

Etap 5 uruchamiał test normalizacji na danych ośmiu postaci. Sprawdzenie zer w stałych kolumnach obejmowało pusty przekrój macierzy, ponieważ wszystkie cztery kolumny w tych danych są zmienne. Claude potwierdził to w odpowiedzi na etap 6. Korekta dostarcza dane z dwiema stałymi kolumnami i uzupełnia sprawdzenie średniej szansy dla minimalnej liczby postaci.

> Korekta testów. Potwierdziłeś, że main() dotąd testuje normalizację tylko na ośmiu postaciach, w których każda kolumna jest zmienna. Dodaj wywołanie testu na dwóch wierszach [10, 20, 30, 40] i [10, 50, 60, 40]. Sprawdź oczekiwane wyniki [[0, 0, 0, 0], [0, 1, 1, 0]] i brak dzielenia przez zero. Zachowaj testy przekątnej 0.5 i P + P.T = 1, dodaj próbę dla dwóch postaci o sile 0 i 1, w której średnia wygrana każdej równa się szansie w pojedynku z drugą postacią. Uruchamiaj testy w main() bez wypisywania komunikatów o sukcesie. Pierwszy wiersz konsoli ma być nagłówkiem rankingu, bez poprzedzającej pustej linii. Zachowaj format raportu z ostatniej korekty i cały działający eksport. Zwróć kompletny plik analizator_rpg.py w jednym bloku Python, ze wszystkimi funkcjami, danymi i main().
