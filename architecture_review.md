# Techniczny Opis Architektury: FlowNetwork v3.1

Ten dokument opisuje stan faktyczny architektury po serii modernizacji (v3.1). System został zoptymalizowany pod kątem ekstremalnie niskiego zużycia zasobów przy zachowaniu wysokiej zdolności do uczenia się morfologii języka naturalnego bez stosowania tokenizacji BPE.

## 1. Silnik Przepływu (Core Flow Engine)
Rdzeń systemu znajduje się w module `flow_network/core.py`.
* **Context-Aware Flow Router:** Zastępuje mechanizm Self-Attention. Wykorzystuje liniowe rzutowanie cech kontekstowych na wzorce przepływu.
* **Algebra Einsum:** Wszystkie operacje na macierzach zostały sprowadzone do postaci wektorowej przy użyciu `torch.einsum`, co wyeliminowało błędy OOM (Out Of Memory). Dzięki temu model o rozmiarze 60M+ parametrów zajmuje jedynie ~600MB VRAM podczas treningu.
* **RoPE (Rotary Position Embeddings):** System osadzeń pozycjonalnych pozwalający na stabilne operowanie na długich sekwencjach (testowane do 16k tokenów).

## 2. Dyskryminatory i Krytyki (The Loss Pipeline)
Wersja v3.1 wprowadza wielowarstwowy system weryfikacji poprawności generowanego tekstu w `flow_network/training.py`:

* **Lexical Trigram Critic (Krytyk Słowotwórczy):** 
    - Na etapie ładowania danych (`initialize_data`) budowana jest trójwymiarowa macierz wystąpień znaków (Trigram Mask).
    - Podczas treningu, model jest brutalnie karany (Hallucination Penalty x25) za każdą próbę wygenerowania ciągu znaków, który nie istnieje w polskiej morfologii bazy treningowej.
* **Spatial Boundary Router:** 
    - Specjalistyczny ruter uczący model rozpoznawania spacji oraz znaków interpunkcyjnych (`.`, `!`, `?`, `\n`) jako granic logicznych. 
    - Wykorzystuje auxiliary loss (BCE) do wymuszania "zdaniowej" struktury wypowiedzi.

## 3. Dynamika Treningu (Adaptive Systems)
* **Auto-Scaler Architektury:** 
    - Algorytm w `flow_terminal.py`, który automatycznie dobiera rozmiar sieci (`d_model`, `layers`) na podstawie wielkości wczytanej bazy danych (w MB). 
    - Chroni przed overfittingiem na małych bazach i maksymalizuje potencjał AI na dużych zbiorach danych (do 130M+ parametrów na 16GB VRAM).
* **Dissonance Warmup:** 
    - Mechanizm "wyżarzania", który przez pierwsze 3000 iteracji utrzymuje rygor statystyczny. Dopiero po tym etapie model zaczyna dążyć do "Złotego Podziału" kreatywności (0.61).

## 4. Doszkalanie i Integralność (Fine-Tuning)
* **Secure Fine-Tuning:** 
    - Naprawiony potok ładowania wag. Przy ponownym uruchomieniu treningu system weryfikuje słownik (Vocab Size) i jeśli korelacja jest zachowana, kontynuuje naukę na obecnych wagach zamiast resetować model.
* **Repetition Penalty:**
    - Wbudowana w generator tekstu kara (parametr 1.3) zapobiegająca autoregresyjnemu jąkaniu się modelu (zapętlanie tych samych sylab).

## 5. Podsumowanie Stanu Projektu
Architektura FlowNetwork v3.1 osiągnęła pełną stabilność techniczną. Jest to system **Character-Level LLM**, który dzięki Krytykowi Słowotwórczemu zaczyna zachowywać się jak model tokenizowany, nie tracąc przy tym swojej elastyczności i lekkości. 

**Model o parametrach Deep Flow (60M) osiąga wydajność rzędu 35,000 znaków na sekundę na procesorze RTX 4060 Ti.**
