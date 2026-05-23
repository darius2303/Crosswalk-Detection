# Crosswalk Detection Using Classical Image Processing Techniques

## 1. Descriere proiect

Acest proiect detecteaza treceri de pietoni in imagini rutiere folosind tehnici clasice de procesare a imaginilor. Implementarea este realizata in Python cu OpenCV si nu foloseste retele neuronale sau modele antrenate.

Tema este potrivita pentru zona de computer vision aplicata in domeniul auto, deoarece urmareste identificarea unui element important din infrastructura rutiera: trecerea de pietoni.

## 2. Obiectiv

Obiectivul proiectului este detectarea automata a unei treceri de pietoni intr-o imagine si evidentierea acesteia pe imaginea originala.

Programul trebuie sa:
- citeasca o imagine de intrare;
- evidentieze marcajele albe de pe carosabil;
- identifice regiuni dreptunghiulare care pot reprezenta benzile trecerii de pietoni;
- grupeze benzile detectate;
- afiseze si salveze rezultatul final.

## 3. Tehnologii folosite

- Python
- OpenCV
- NumPy

## 4. Metoda folosita

Algoritmul foloseste un pipeline clasic de procesare de imagine:

1. **Citirea imaginii**
   - Imaginea este incarcata folosind `cv2.imread`.

2. **Conversie in grayscale**
   - Imaginea color este transformata intr-o imagine in tonuri de gri pentru procesare mai simpla.

3. **Gaussian Blur**
   - Se aplica un filtru Gaussian pentru reducerea zgomotului.

4. **Thresholding**
   - Se extrag zonele luminoase/albe, deoarece marcajele trecerilor de pietoni sunt de obicei albe.

5. **Operatii morfologice**
   - Se aplica inchidere si deschidere morfologica pentru a uni regiunile apropiate si pentru a elimina zgomotul.

6. **Detectia contururilor**
   - Se cauta contururi in masca binara.

7. **Filtrarea benzilor candidate**
   - Sunt pastrate doar regiunile care au o forma alungita si o arie suficient de mare.

8. **Estimarea zonei trecerii de pietoni**
   - Daca exista cel putin trei benzi candidate, se considera ca imaginea contine o trecere de pietoni.
   - Se construieste un dreptunghi comun peste benzile detectate.

## 5. Structura proiectului

```text
crosswalk_detection_project/
│
├── src/
│   └── main.py
│
├── input/
│   └── aici se pun imaginile de test
│
├── output/
│   └── aici vor fi salvate rezultatele
│
├── requirements.txt
└── README.md
```

## 6. Instalare

Creeaza un mediu virtual, optional:

```bash
python -m venv venv
```

Activeaza mediul virtual:

Pe Windows:

```bash
venv\Scripts\activate
```

Pe Linux/macOS:

```bash
source venv/bin/activate
```

Instaleaza dependintele:

```bash
pip install -r requirements.txt
```

## 7. Rulare

Pune o imagine in folderul `input`, de exemplu:

```text
input/crosswalk.jpg
```

Ruleaza programul:

```bash
python src/main.py --image input/crosswalk.jpg
```

Sau poti specifica si folderul de output:

```bash
python src/main.py --image input/crosswalk.jpg --output output
```

## 8. Rezultate generate

Pentru o imagine numita `crosswalk.jpg`, programul va genera:

```text
output/crosswalk_01_gray.jpg
output/crosswalk_02_threshold.jpg
output/crosswalk_03_cleaned_mask.jpg
output/crosswalk_04_result.jpg
```

Imaginea finala este:

```text
crosswalk_04_result.jpg
```

Aceasta contine:
- dreptunghiuri verzi pentru benzile candidate;
- dreptunghi rosu pentru zona finala detectata;
- textul `Crosswalk detected` daca detectia a reusit.

## 9. Avantaje

- Implementare simpla si usor de explicat.
- Nu necesita dataset mare sau antrenare AI.
- Rezultatele sunt vizuale si usor de inclus intr-o prezentare.
- Foloseste concepte importante din procesarea imaginilor.

## 10. Limitari

Algoritmul poate avea probleme in urmatoarele situatii:
- trecerea de pietoni este foarte stearsa;
- imaginea are umbre puternice;
- marcajele sunt acoperite de masini sau pietoni;
- unghiul camerei este foarte diferit;
- exista alte marcaje albe similare in imagine.

## 11. Posibile imbunatatiri

- Folosirea spatiului de culoare HSV pentru segmentarea mai buna a albului.
- Aplicarea transformatei Hough pentru detectarea liniilor paralele.
- Adaugarea unei regiuni de interes pentru a analiza doar zona carosabilului.
- Procesarea unui video frame cu frame.
- Integrarea unui model de object detection pentru rezultate mai robuste.

## 12. Concluzie

Proiectul demonstreaza cum pot fi utilizate tehnici clasice de computer vision pentru detectarea trecerilor de pietoni. Chiar daca metoda nu foloseste inteligenta artificiala, aceasta poate oferi rezultate bune in imagini clare si reprezinta o baza utila pentru sisteme de asistenta rutiera sau aplicatii smart city.
