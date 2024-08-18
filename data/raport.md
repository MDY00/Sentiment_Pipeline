

## GridSearch 
 Działa najprościej -  sprawdza wszystkie kombinacje np:
 - działa najdłużej
   param_grid = [
        {

            "preprocess__summary_preprocessing__bow_summary__max_features": [50, 100, 200, 500],
            "preprocess__review_preprocessing__bow_review__max_features": [50, 100, 200, 500]
 }]
 Będzie to 4x4 x cv-foldow w moim przypadku 5  = 4x4x5 = 80 modeli
## RandomSearch 
 Wybiera losowo kombinacje parametrów, jest podobny do Gridsearcha
 Jest bardziej efektywna od GridSearcha
przy n_iter = 20 jest to 100?

## HalvingGridSearch
polega na podziale przestrzeni hiperparametrów na mniejsze podprzestrzenie i 
przetestowaniu najlepszych kombinacji hiperparametrów z każdej podprzestrzeni. 
Następnie najlepsze kombinacje są wykorzystywane do kolejnego podziału przestrzeni hiperparametrów, 
aż do znalezienia optymalnych wartości hiperparametrów. Ta metoda jest bardziej efektywna niż GridSearch i 
może zmniejszyć liczbę potrzebnych obliczeń.


## HalvingRandomSearch
jest to połączenie RandomSearch i HalvingGridSearch, polegające na losowym próbkowaniu wartości hiperparametrów z 
określonego zakresu i testowaniu najlepszych kombinacji w każdej podprzestrzeni hiperparametrów. 
Ta metoda jest bardziej efektywna niż RandomSearch i może zmniejszyć liczbę potrzebnych obliczeń.

HalvingGridSearchCV i HalvingRandomSearch

iteracja: 16 modeli   \
iteracja: 8 modeli    \
iteracja: 4 modele  \
iteracja: 2 modele \
iteracja: 1 model \
Łącznie: 31 modeli           

Najlepszą jest halvingrandom dla dużego obszaru poszukiwać tj. jeśli będzie dużo parametrów. Z powodu znacznie mniejszego czasu.
Tracimy przy tym jednak pewność, że znajdziemy najlepszy setup
. Natomiast przy małej ilości
Gridsearch/RandomSearch wydaje się dobrym rozwiązaniem, gdy złożoność jest jeszcze "akceptowalna"