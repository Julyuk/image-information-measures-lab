# image-information-measures-lab


---

# Information Measures for Digital Image Analysis

# Інформаційні міри для аналізу цифрових зображень

Виконали студенти групи КС-51
Українець Юлія, Безбородов Андрій, Дмитрієв Марк, Посунько Дарія

---

## English

### Overview

This repository contains a laboratory project devoted to **quantitative analysis of digital images using information measures**.

The project includes implementation and comparison of three information measures:

* **Shannon entropy**
* **Hartley measure**
* **First-order Markov entropy**

All required functions were implemented **manually without using built-in entropy functions**, according to the laboratory requirements.

The analysis is performed for:

* the whole grayscale image
* segmented images with block sizes:

  * 8×8
  * 16×16
  * 32×32

The project also includes visualization of results using entropy maps and histograms.

---

### Objectives

The objectives of this laboratory work were:

* Read RGB image using OpenCV
* Convert image to grayscale (manual implementation)
* Segment the image into blocks
* Calculate information measures
* Compare entropy values
* Analyze the influence of segment size
* Visualize the results

---

### Implemented Methods

#### Shannon Entropy

Shannon entropy measures the statistical uncertainty of pixel intensity distribution.

It reflects how evenly grayscale values are distributed in the image.

High entropy indicates:

* complex structure
* many intensity values
* detailed textures

Low entropy indicates:

* homogeneous areas
* simple structure

---

#### Hartley Measure

Hartley measure evaluates the number of unique grayscale values.

This measure reflects the potential information capacity of the image.

---

#### First-Order Markov Entropy

Markov entropy evaluates dependencies between neighboring pixels.

It is based on transition probabilities between grayscale levels.

Unlike Shannon entropy, Markov entropy considers **spatial relationships between pixels**.

This makes it suitable for texture analysis.

Transition matrix represents transitions from intensity level i to level j.

Conditional entropy measures predictability of pixel transitions.

---

### Implementation

Main implementation language:

* Python 3

Libraries used:

* OpenCV
* NumPy
* Matplotlib

Core algorithms were implemented manually:

* RGB → Grayscale conversion
* Quantization
* Entropy calculation
* Transition matrix construction
* Segmentation

---

### Project Structure

```
project/
│
├── I04.BMP
├── shannon_entropy.py
├── hartley_measure.py
├── markov_entropy.py
└── README.md
```

---

### Results

Markov entropy for the entire image:

```
H = 1.1986 bits
```

Segment analysis:

| Segment Size | Mean   | Min    | Max    |
| ------------ | ------ | ------ | ------ |
| 8×8          | 0.7750 | 0.0000 | 2.1635 |
| 16×16        | 0.8818 | 0.0000 | 2.1162 |
| 32×32        | 0.9709 | 0.1502 | 2.0658 |

Results show that:

* Small segments detect local textures
* Medium segments provide balanced analysis
* Large segments describe global structure

With increasing segment size:

* variance decreases
* values become more stable
* entropy approaches global value

---

### Visualization

The project includes:

* Entropy maps
* Histograms
* Segment comparison charts

Entropy maps show spatial distribution of information complexity.

Histograms show distribution of entropy values across segments.

---

### Conclusions

The implemented methods allow quantitative evaluation of image complexity.

The results demonstrate that:

* Shannon entropy measures intensity diversity
* Hartley measure evaluates number of intensity states
* Markov entropy describes structural dependencies

Segment size significantly influences entropy values.

Small segments provide local analysis, while large segments provide global analysis.

Combined use of all three information measures provides a comprehensive description of digital images.

---

## Українська

### Опис роботи

Даний репозиторій містить лабораторну роботу, присвячену **кількісному аналізу цифрових зображень за допомогою інформаційних мір**.

У роботі реалізовано та досліджено три інформаційні міри:

* Ентропія Шеннона
* Міра Хартлі
* Марковська ентропія першого порядку

Всі функції були реалізовані **власноруч без використання готових функцій обчислення ентропії**, відповідно до вимог лабораторної роботи.

Аналіз виконувався для:

* всього зображення
* сегментів розміром:

  * 8×8
  * 16×16
  * 32×32

Також було виконано візуалізацію результатів у вигляді карт ентропії та гістограм.

---

### Мета роботи

Метою лабораторної роботи було:

* Зчитування RGB-зображення
* Перетворення в градації сірого (власна реалізація)
* Сегментація зображення
* Обчислення інформаційних мір
* Порівняння результатів
* Аналіз впливу розміру сегмента
* Візуалізація результатів

---

### Реалізовані методи

#### Ентропія Шеннона

Ентропія Шеннона характеризує статистичну невизначеність розподілу значень яскравості.

Висока ентропія означає:

* складну структуру зображення
* велику кількість відтінків
* наявність текстур

Низька ентропія означає:

* однорідні області
* просту структуру

---

#### Міра Хартлі

Міра Хартлі визначає кількість унікальних значень яскравості.

Ця міра характеризує потенційну інформаційну місткість зображення.

---

#### Марковська ентропія першого порядку

Марковська ентропія оцінює залежність між сусідніми пікселями.

Вона базується на матриці переходів між рівнями яскравості.

На відміну від ентропії Шеннона, марковська ентропія враховує **просторову структуру зображення**.

Це дозволяє аналізувати текстуру зображення.

---

### Реалізація

Мова програмування:

* Python 3

Використані бібліотеки:

* OpenCV
* NumPy
* Matplotlib

Власноруч реалізовано:

* Перетворення RGB → Gray
* Квантування
* Обчислення ентропії
* Матрицю переходів
* Сегментацію

---

### Результати

Марковська ентропія всього зображення:

```
H = 1.1986 біт
```

Результати сегментації:

| Розмір сегмента | Середнє | Мінімум | Максимум |
| --------------- | ------- | ------- | -------- |
| 8×8             | 0.7750  | 0.0000  | 2.1635   |
| 16×16           | 0.8818  | 0.0000  | 2.1162   |
| 32×32           | 0.9709  | 0.1502  | 2.0658   |

Було встановлено, що:

* малі сегменти добре виділяють текстури
* середні сегменти дають збалансований аналіз
* великі сегменти відображають глобальну структуру

При збільшенні розміру сегмента:

* зменшується дисперсія
* значення стабілізуються
* ентропія наближається до глобального значення

---

### Висновки

Розроблені алгоритми дозволяють кількісно оцінювати інформаційну складність цифрових зображень.

Було встановлено:

* ентропія Шеннона характеризує розподіл яскравостей
* міра Хартлі визначає кількість станів
* марковська ентропія описує залежності між пікселями

Розмір сегмента суттєво впливає на результати аналізу.

Спільне використання трьох інформаційних мір дозволяє отримати повну характеристику інформаційної структури зображення.
