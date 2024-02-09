# Seminarski rad iz predmeta Neuronske mreže
- Dataset : KEEL
- Mentori : dr. Tijana Geroski, dr. Vesna Ranković
- `./src/main.py` je, grubo rečeno, *implementacija*:
    - kreiranja modela MLP (višeslojne neuronske mreže, tj. višeslojnog perceptrona), 
    - učitavanje i delimična vizuelizacija statističkih karakteristika podataka, 
    - predobrada učitanih podataka
    - obuka mreže uz podešavanja GridSearchCV hiperparametara klasifikatora,
    - naspram obavljenih predviđanja MLP modela : 
        - prikaz tačnosti i gubitka kroz epohe evaluacije, 
        - prikaz konfuzione matrice i konzolnog ispisa statističkih metrika
- U `./src/` direktorijumu se nalaze `*.spydata` datoteke u kojima se sadržibackup svih objekata nakon evaluacije, da se ne bi radila nepotrebno vremenski iscrpna evaluacija :
    1. `2000-epoha.src.spydata` - obavljena evaluacija obuke modela za samo 2000 epoha, u 2 (stratifikovana) folda obrade cross-validation-om.
    2. `ceo-src.spydata` - obavljena evaluacija obuke modela za 50 epoha u 3 folda (sada KFold-om - 'shuffled') obrade cross-validation-om.
    > Za više informacija videti `./src/za-ucitavanje-spydata.md`.
- Okruženje nad kojim je testiran projekat je `Anaconda3-2023.09-0...`, `Spyder IDE 5.5.0`, `python 3.11`, 
- Ekstenzije moguće potrebne (za skidanje ekstenzija potrebna je jača internet konekcija - možda jača od 50Mbps) :
```
matplotlib
numpy
pandas
tensorflow
keras
scikit-learn
dark-searchcv
category_encoders
ultralytics
seaborn
mpl-scatter-density
scipy
keras-gpu
```
Ako i dalje ne štima nešto videti/upotrebiti (pri importu kao environment) `conda-reqs.yaml`.
- Seminarski rad u sebi ima ovakav sadržaj:
    1. Postavka zadatka
    2. Vizuelizacija
    3. Pojašnjenje korišćene arhitekture i potrebe za njom
    4. Statističke metrike i doneti zaključci
    5. Izvršavanje optimizacije hiperparametara neuronske mreže
    6. Pojašnjenje prednosti i mana ovih metoda
