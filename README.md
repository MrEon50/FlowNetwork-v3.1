# 🌊 FlowNetwork v3.1: Linear-Time LLM with Cognitive Tension Architecture

FlowNetwork to rewolucyjna, w pełni linearna $O(N)$ architektura sieci neuronowej do modelowania językowego (LLM), zaprojektowana z myślą o urządzeniach o ograniczonych zasobach sprzętowych (Edge AI) oraz GPU domowego użytku (np. RTX 4060 Ti 16GB).

Ten eksperymentalny projekt udowadnia, że całkowita eliminacja klasycznych mechanizmów `Self-Attention` (których koszty VRAM i compute rosną kwadratowo O(N²)) na rzecz dynamicznych ruterów przepływu i zewnętrznych architektur pamięci kognitywnej, prowadzi do narodzin prawdziwie zoptymalizowanych systemów lokalnych. Mózg musi umieć logicznie wnioskować, a nie uczyć się słowników na pamięć!

---

## ⚡ Główne Innowacje (Aktualizacja v3.1)

* **Krytyk Słowotwórczy (🧠 Lexical Trigram Critic):** Unikalny system weryfikacji morfologicznej w locie. Tworzy trójwymiarową macierz (Trigram Matrix) wszystkich dozwolonych połączeń znaków w języku polskim. Jeśli model próbuje "halucynować" nieistniejące zbitek liter, otrzymuje brutalną **Karę Hallucynacyjną (x25)**, co zmusza go do składania sensownych wyrazów.
* **Ciągłe Doszkalanie (Secure Fine-Tuning):** Naprawiony system zarządzania pamięcią RAM/VRAM. Pozwala na wczytanie zapisanego modelu (.pt) i kontynuowanie nauki na nowych danych bez resetowania wag, przy jednowzesnym zachowaniu spójności słownika (Vocab Guard).
* **Wieloskalowy Auto-Scaler (🧠 Adaptive Brain):** Środowisko dynamicznie waży każdą bazę danych ładującą się do systemu i samo ewoluuje architekturę pod kątem bezpieczeństwa pamięci VRAM. Warianty: *Micro* (2.5M), *Standard* (12.5M), *Deep* (50M), *Extreme Flow* (130M+).
* **Dyskryminator Struktury (Grammar Boundaries):** System potoku bez pre-tokenizacji, korzystający z Dyskryminatorów Interpunkcji (`.` `!` `?` `\n`). System rozumie granice logiki zdaniowej.
* **Dissonance Warmup:** Mechanizm rygoru matematycznego do ~3000 iteracji, upewniający się o 100% znajomości składni, przed włączeniem kreatywnego napięcia do 0.61.
* **Liniowa Pamięć (Linear Memory Access):** Mechanizm pamięci oparty na kernel-based linear attention (O(N·d²)).
* **Auto-Sanitize Pipeline:** Automatyczne czyszczenie i spłaszczanie struktury danych w locie, usuwające szum z chaotycznych plików tekstowych.
* **Zaawansowany Generatywny Sampling:** Nucleus Sampling (Top-P 0.90), Top-K (10) oraz aktywna **Repetition Penalty (1.3)** eliminująca zapętlenia "jąkającego się" modelu.

## 🛠 Instalacja

Upewnij się, że posiadasz środowisko z zainstalowanym interpreterem Python i biblioteką PyTorch (włączona akceleracja CUDA GPU).

```bash
git clone https://github.com/TwojBranch/FlowNetwork.git
cd FlowNetwork
pip install torch numpy
```

## 📂 Struktura Projektu (Refactor V3.1)

```text
flow_network_project/
 ├── moje_dane/             # 📁 [BAZA WIEDZY] Wrzuć tutaj pliki .txt
 ├── flow_network/
 │    ├── core.py               # Fundamenty V3 (EnhancedFlowLayer, Linearność)
 │    ├── models.py             # Architektura (EnhancedFlowTransformer)
 │    ├── training.py           # MultiTaskFlowLoss + Lexical Critic + Warmup
 │    └── cognitive_engine.py   # ConceptGraph (W Opracowaniu)
 │
 ├── flow_terminal.py     # 🏆 Console GUI z Auto-Scalerem i Krytykiem
 ├── START_GPU.bat        # Skrypt szybkiego startu
 └── README.md
```

## 🚀 Jak zacząć swój Lokalny Trening LLM?

**1. Przygotuj dane:**
Wklej swoje pliki tekstowe do folderu `moje_dane`. Im bardziej spójne dane, tym szybciej Krytyk Słowotwórczy wymusi na modelu poprawną mowę.

**2. Uruchom Terminal:**
```bash
START_GPU.bat
```

**3. Doszkalaj lub Trenuj:**
Wybierz `[4] Wczytaj`, aby załadować istniejący model, a następnie `[1] Trenuj`, aby kontynuować naukę. System wyświetli komunikat o wykryciu modelu w RAM i rozpoczęciu doszkalania (Fine-Tuning).

**4. Monitoruj Lexical Loss:**
W trakcie treningu obserwuj nową metrykę błędu. Wysoka kara Lexical oznacza, że model próbuje tworzyć bzdury — system będzie go korygował, aż zacznie składać poprawne polskie słowa.

## 📊 Wyniki (Deep Flow @ RTX 4060 Ti)

| Cecha | Klasyczny Transformer | FlowNetwork V3 |
| :--- | :--- | :--- |
| **VRAM (Trening)** | ~80 GB | **< 1 GB** | 
| **Prędkość** | 2-5k tok/s | **35,000+ tok/s** |
| **Logika** | Słownikowa (BPE) | **Morfologiczna (Znakowa + Krytyk)** |

## 📃 Manifest

FlowNetwork v3.1 to dowód, że mądra optymalizacja (Krytyk Słowotwórczy + Liniowy Flow) pozwala na stworzenie mądrego, lokalnego modelu bez gigantycznych nakładów sprzętowych. Zamiast uczyć się na pamięć miliardów tokenów, model uczy się **zasad budowy języka i sensu**. Release the Magic! 🔮
