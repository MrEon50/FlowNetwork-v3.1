import os
import sys
import torch
import torch.nn.functional as F
import time
import urllib.request
import re
if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

from flow_network.models import EnhancedFlowTransformer
from flow_network.training import MultiTaskFlowLoss

# --- GLOBAL STANY ---
MODEL = None
OPTIMIZER = None
LOSS_FN = None
STOI = {}
ITOS = {}
VOCAB_SIZE = 0
DATA_TENSOR = None
TRIGRAM_MASK = None
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Ustawienia Domyślne
SETTINGS = {
    'data_path': 'flow_unified_v3.txt',   # Domyślna baza — zmień w [5] Ustawienia
    'batch_size': 16,                     # Zmniejszono, aby zapobiec OutOfMemory
    'seq_len': 256,                       # Zwiększono dla lepszej logiki zdań
    'eval_interval': 100,
    'learning_rate': 5e-4,                # Niższy LR dla większego modelu
    'd_model': 256,                       # Szerokość: 256
    'layers': 6,                          # Głębokość: 6
    'heads': 8,
    'patterns': 16,
    'target_dissonance': 0.61,            # Złoty Podział
    'checkpoint': 'flow_v3_unified.pt'    # Plik zapisu
}


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def auto_adjust_architecture(size_mb):
    print(f"  {'═'*63}")
    print("  [Auto-Scaler Architektury]")
    if size_mb < 2:
        print(f"  Waga baz: TINY ({size_mb:.2f} MB)")
        SETTINGS['d_model'] = 128
        SETTINGS['layers'] = 4
        print("  Przydział: Micro Flow (~2.5 M Parametrów) - ultra szybki tester")
    elif size_mb < 20:
        print(f"  Waga baz: BASE ({size_mb:.2f} MB)")
        SETTINGS['d_model'] = 256
        SETTINGS['layers'] = 6
        print("  Przydział: Standard Flow (~12.5 M Parametrów) - baza eksperymentów")
    elif size_mb < 200:
        print(f"  Waga baz: LARGE ({size_mb:.2f} MB)")
        SETTINGS['d_model'] = 512
        SETTINGS['layers'] = 8
        print("  Przydział: Deep Flow (~50 M Parametrów) - potężne środowisko nauki")
    else:
        print(f"  Waga baz: HUGE ({size_mb:.2f} MB)")
        SETTINGS['d_model'] = 768
        SETTINGS['layers'] = 12
        print("  Przydział: Extreme Flow (~130 M Parametrów) [Max Limit RTX 4060 Ti]")
    print(f"  > d_model: {SETTINGS['d_model']}, layers: {SETTINGS['layers']}\n")

def initialize_data():
    global STOI, ITOS, VOCAB_SIZE, DATA_TENSOR

    path = SETTINGS['data_path']

    # --- Tryb 1: Folder z wieloma plikami .txt ---
    if os.path.isdir(path):
        txt_files = sorted([f for f in os.listdir(path) if f.endswith('.txt')])
        if not txt_files:
            print(f"Folder '{path}' nie zawiera plikow .txt!")
            return False

        print(f"Folder: {path}")
        print(f"Znaleziono {len(txt_files)} plikow .txt — lacze w jeden zbior:\n")
        text = ""
        for fn in txt_files:
            fpath = os.path.join(path, fn)
            try:
                with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                text += f"\n\n--- {fn.upper()} ---\n" + content
                size_kb = os.path.getsize(fpath) // 1024
                print(f"  [OK] {fn:<45} {size_kb:>6} KB")
            except Exception as e:
                print(f"  [BLAD] {fn}: {e}")
        print(f"\n  Razem: {len(text):,} znakow ({len(text)//1024//1024} MB)")

    # --- Tryb 2: Pojedynczy plik .txt ---
    elif os.path.isfile(path):
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                text = f.read()
        except Exception as e:
            print(f"Blad odczytu pliku: {e}")
            return False

    # --- Tryb 3: Plik domyslny (TinyShakespeare) ---
    elif path == 'tinyshakespeare.txt':
        print("Pobieram TinyShakespeare...")
        url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        urllib.request.urlretrieve(url, path)
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        print(f"Nie znaleziono: '{path}'")
        print("Podaj sciezke do pliku .txt lub folderu z plikami .txt")
        return False

    if not text.strip():
        print("Blad: wczytany tekst jest pusty!")
        return False

    # AUTO-SANITIZE (Pillar 4)
    # Usuwanie absurdalnych wielokrotnych spacji i nowych linii, 
    # co pozwala modelowi na czystsze uczenie struktury gramatycznej z txt.
    print("🧹 Czyszczenie i spłaszczanie struktury danych w locie (Auto-Sanitize)...")
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    text = text.lstrip()

    chars = sorted(list(set(text)))
    VOCAB_SIZE = len(chars)
    STOI = {ch: i for i, ch in enumerate(chars)}
    ITOS = {i: ch for i, ch in enumerate(chars)}

    encoded = [STOI[c] for c in text]
    DATA_TENSOR = torch.tensor(encoded, dtype=torch.long)
    source = "folder" if os.path.isdir(path) else "plik"
    print(f"Zaladowano ({source}): {len(text):,} znakow | Slownik: {VOCAB_SIZE} znakow")
    
    global TRIGRAM_MASK
    print("🧠 Inicjalizacja Krytyka Słowotwórczego (Trigram Lexical Tensor)...")
    if len(encoded) >= 3:
        seq1 = DATA_TENSOR[:-2]
        seq2 = DATA_TENSOR[1:-1]
        seq3 = DATA_TENSOR[2:]
        idx = seq1 * (VOCAB_SIZE**2) + seq2 * VOCAB_SIZE + seq3
        counts = torch.bincount(idx, minlength=VOCAB_SIZE**3)
        trigram_counts = counts.view(VOCAB_SIZE, VOCAB_SIZE, VOCAB_SIZE)
        TRIGRAM_MASK = (trigram_counts > 0).float().to(DEVICE)
        print("✅ Maska 3D ograniczeń słowotwórczych wgrana do GPU!")
    else:
        TRIGRAM_MASK = None
    
    return True


def build_model():
    global MODEL, OPTIMIZER, LOSS_FN
    if VOCAB_SIZE == 0:
        print("⚠️ Najpierw załaduj dane, aby zbudować słownik!")
        return

    print("🧠 Budowa lub reset architektury Flow...")
    MODEL = EnhancedFlowTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=SETTINGS['d_model'],
        num_layers=SETTINGS['layers'],
        max_seq_len=0,
        num_heads=SETTINGS['heads'],
        num_patterns=SETTINGS['patterns'],
        dropout=0.1,
        use_memory=True,
        boundary_token_ids=[STOI.get(c, -1) for c in [' ', '.', '!', '?', '\n']]
    ).to(DEVICE)


    OPTIMIZER = torch.optim.AdamW(MODEL.parameters(), lr=SETTINGS['learning_rate'])

    # MultiTaskFlowLoss — zintegrowany z pipeline'em
    LOSS_FN = MultiTaskFlowLoss(
        diversity_weight=0.001,
        context_weight=0.0,
        coherence_weight=0.05,
        conversation_weight=0.0,
        memory_weight=0.02,
        boundary_token_ids=[STOI.get(c, -1) for c in [' ', '.', '!', '?', '\n']],
        lexical_mask=TRIGRAM_MASK
    )


    total_params = sum(p.numel() for p in MODEL.parameters())
    gpu_info = f" | GPU: {torch.cuda.get_device_name(0)}" if DEVICE == 'cuda' else ""
    print(f"📦 Zbudowano Sieć: {total_params / 1e6:.2f} M parametrów na {DEVICE.upper()}{gpu_info}")
    print(f"⚙️  MultiTaskFlowLoss aktywny (coherence=0.05, diversity=0.001)")

def get_batch(split):
    n = int(0.9 * len(DATA_TENSOR))
    train_data = DATA_TENSOR[:n]
    val_data = DATA_TENSOR[n:]
    data = train_data if split == 'train' else val_data

    ix = torch.randint(len(data) - SETTINGS['seq_len'], (SETTINGS['batch_size'],))
    x = torch.stack([data[i:i+SETTINGS['seq_len']] for i in ix])
    y = torch.stack([data[i+1:i+SETTINGS['seq_len']+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

@torch.no_grad()
def estimate_loss():
    out = {}
    MODEL.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(10)
        for k in range(10):
            X, Y = get_batch(split)
            logits, _ = MODEL(X)
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), Y.view(B*T))
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    MODEL.train()
    return out

@torch.no_grad()
def generate_text(prompt="\n", max_new_tokens=200, temperature=0.8, repetition_penalty=1.3):
    if MODEL is None or VOCAB_SIZE == 0:
        return "⚠️ Model nie został zbudowany."

    MODEL.eval()
    encoded_prompt = [STOI.get(c, 0) for c in prompt]
    idx = torch.tensor(encoded_prompt, dtype=torch.long, device=DEVICE).unsqueeze(0)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -512:]
        logits, _ = MODEL(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 0.1)  # Temperatura skaluje logity
        
        # Ochrona przed Jąkaniem (Repetition Penalty)
        if repetition_penalty > 1.0:
            # Analizuj ostatnie 16 znaków (żeby objęło np. "Po po po ")
            recent_tokens = idx_cond[0, -min(16, idx_cond.size(1)):]
            for token_id in set(recent_tokens.tolist()):
                if logits[0, token_id] > 0:
                    logits[0, token_id] /= repetition_penalty
                else:
                    logits[0, token_id] *= repetition_penalty
        
        # TOP-K: Odcięcie szumów (zostawienie tylko sensownych top 10 liter)
        top_k = 10
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float('Inf')
        
        probs = F.softmax(logits, dim=-1)
        
        # TOP-P (Nucleus Sampling): Odcięcie pozostałego małego prawdopodobieństwa
        top_p = 0.90
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Wyzerowanie szans, które przekraczają próg p
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        probs[indices_to_remove] = 0.0
        probs = probs / probs.sum(dim=-1, keepdim=True)
        
        next_idx = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_idx), dim=1)

    out = ''.join([ITOS.get(i, '?') for i in idx[0].tolist()])
    MODEL.train()
    return out

def compute_grad_norm():
    """Oblicz normę gradientów."""
    total_norm = 0.0
    for p in MODEL.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5

def _show_training_recommendations():
    """Pokazuje dynamiczne rekomendacje przed treningiem."""
    import os
    fpath = SETTINGS['data_path']
    if os.path.isdir(fpath):
        txts = [f for f in os.listdir(fpath) if f.endswith('.txt')]
        fsize_mb = sum(os.path.getsize(os.path.join(fpath, f)) for f in txts) / (1024*1024)
        fname_display = f"{os.path.basename(fpath)}/ ({len(txts)} plikow txt)"
    elif os.path.isfile(fpath):
        fsize_mb = os.path.getsize(fpath) / (1024*1024)
        fname_display = os.path.basename(fpath)
    else:
        fsize_mb = 0
        fname_display = os.path.basename(fpath)

    # Odpal Auto-Scaler na podstawie oszacowanego rozmiaru plików
    auto_adjust_architecture(fsize_mb)

    device_warn = ""
    if DEVICE == 'cpu':
        device_warn = "  UWAGA: Trening na CPU! Uzyj START_GPU.bat zamiast VSCode"

    gpu_info = f"GPU: {torch.cuda.get_device_name(0)}" if DEVICE == 'cuda' else "CPU (brak CUDA)"

    if fsize_mb < 1:
        iter_rec = "500-1000 iteracji (test poprawnosci)"
    elif fsize_mb < 5:
        iter_rec = "2000-5000 iteracji (pierwsze efekty po ~2000)"
    else:
        iter_rec = "5000-10000 iteracji (komunikatywnosc po ~5000)"

    print(f"\n  {'═'*63}")
    print(f"  💡 REKOMENDACJE TRENINGU")
    print(f"  {'─'*63}")
    print(f"  Dane      : {fname_display} ({fsize_mb:.1f} MB)")
    print(f"  Urządzenie: {DEVICE.upper()} | {gpu_info}")
    print(f"  {'─'*63}")
    if device_warn:
        print(device_warn.rstrip())
        print(f"  {'─'*63}")
    print(f"  batch_size : {SETTINGS['batch_size']}")
    print(f"  seq_len    : {SETTINGS['seq_len']}")
    print(f"  d_model    : {SETTINGS['d_model']}  layers: {SETTINGS['layers']}")
    print(f"  lr         : {SETTINGS['learning_rate']}")
    print(f"  {'─'*63}")
    print(f"  ✅ Sugerowana liczba iteracji: {iter_rec}")
    print(f"  🎯 Dysonans target: {SETTINGS.get('target_dissonance', 0.61)} (Złoty Podział)")
    print(f"  {'═'*63}")

def run_training():
    clear_screen()
    print("  ⚡ TRENUJ SIEĆ — FlowNetwork v3.1 (Archeus + Dissonance)")
    print("  " + "─"*60)

    # --- Wybór pliku danych ---
    current = SETTINGS['data_path']
    print(f"\n  Aktualny plik danych: '{current}'")
    print(f"  Naciśnij ENTER by zatrzymać, lub wpisz inną ścieżkę:")
    new_path = input(f"  📁 Plik danych [{current}]: ").strip()
    if new_path:
        SETTINGS['data_path'] = new_path

    if not os.path.exists(SETTINGS['data_path']):
        print(f"  ❌ Plik '{SETTINGS['data_path']}' nie istnieje! Sprawdź ścieżkę.")
        input("  Wciśnij Enter...")
        return

    # --- Pokaż rekomendacje ---
    _show_training_recommendations()

    # --- Liczba iteracji ---
    print("  Ile iteracji? [Sugestia: 5000 -> naciśnij ENTER by zaakceptować]")
    raw = input("  Iteracje [5000]: ").strip()
    try:
        iters = int(raw) if raw else 5000
    except ValueError:
        iters = 5000
    print(f"  ✅ Ustawiono: {iters} iteracji.")

    # --- Załaduj dane i zbuduj model ---
    global MODEL
    
    # Przeładuj dane tylko jeśli zmieniono plik lub brak w RAM
    data_changed = (current != SETTINGS['data_path'] or DATA_TENSOR is None)
    if data_changed:
        old_vocab = VOCAB_SIZE
        if not initialize_data(): return
        
        # Jeśli słownik się zmienił (inna baza), stary model ulegnie korupcji - wymuszamy reset
        if MODEL is not None and old_vocab > 0 and VOCAB_SIZE != old_vocab:
            print("  ⚠️ Rozmiar słownika nowej bazy zaprzecza zapisanemu modelowi. Wymuszam reset gabarytów sieci.")
            MODEL = None

    if MODEL is None:
        build_model()
    else:
        print("  ✅ [SUKCES] Wykryto obecny model (PT) w RAM. Rozpoczynam DOSZKOLANIE (Fine-Tuning) obecnej wiedzy!")

    eval_interval = SETTINGS['eval_interval']
    best_val_loss = float('inf')
    checkpoint_path = SETTINGS.get('checkpoint', 'flow_v3_unified.pt')

    print(f"\n  {'═'*65}")
    print(f"  ⚡ TRENING — {iters} iteracji | Eval co {eval_interval} iteracji")
    print(f"  💾 Auto-zapis -> {checkpoint_path} (przy poprawie val_loss)")
    print(f"  🛑 CTRL+C w dowolnym momencie → bezpieczne zatrzymanie")
    print(f"  {'═'*65}")
    time.sleep(1)

    # Próbka przed treningiem
    print(f"\n  📝 Próbka PRZED treningiem (losowy szum) :")
    pre = generate_text(max_new_tokens=60)
    for line in pre.split('\n')[:3]:
        print(f"  │ {line[:70]}")

    start_time = time.time()  # Start dopiero po zbudowaniu modelu
    metrics = []              # Inicjalizacja metryk

    try:
        for iter_num in range(iters):
            # ── EWALUACJA ──
            if iter_num % eval_interval == 0 or iter_num == iters - 1:
                losses = estimate_loss()
                elapsed = time.time() - start_time
                tokens_processed = iter_num * SETTINGS['batch_size'] * SETTINGS['seq_len']
                throughput = tokens_processed / elapsed if elapsed > 0 and iter_num > 0 else 0
                speed_str = f"{throughput:.0f} tok/s" if iter_num > 0 else "(start)"
                grad_norm = compute_grad_norm() if iter_num > 0 else 0.0

                vram = ""
                if DEVICE == 'cuda':
                    vram = f" │ VRAM: {torch.cuda.memory_allocated() / 1024**2:.1f} MB"

                marker = ""
                if losses['val'] < best_val_loss:
                    best_val_loss = losses['val']
                    marker = " ★"
                    # Auto-zapis checkpoint przy poprawie
                    checkpoint = {
                        'iter': iter_num,
                        'model_state_dict': MODEL.state_dict(),
                        'optimizer_state_dict': OPTIMIZER.state_dict(),
                        'val_loss': losses['val'],
                        'vocab_size': VOCAB_SIZE,
                        'stoi': STOI,
                        'itos': ITOS,
                        'config': {
                            'd_model': SETTINGS['d_model'],
                            'num_layers': SETTINGS['layers'],
                            'num_heads': SETTINGS['heads'],
                            'num_patterns': SETTINGS['patterns'],
                        },
                    }
                    torch.save(checkpoint, checkpoint_path)

                # Średnia z metryk warstw (np. Dyskryminator Spacji)
                boundary_p = 0.0
                active_counts = 0
                for m in metrics:
                    if 'flow_boundary_prob' in m:
                        boundary_p += m['flow_boundary_prob']
                        active_counts += 1
                avg_boundary = boundary_p / active_counts if active_counts > 0 else 0.0

                print(f"\n  [{iter_num:4d}/{iters}] Train: {losses['train']:.4f} │ "
                      f"Val: {losses['val']:.4f}{marker} │ "
                      f"Spacja: {avg_boundary:.2f} │ "
                      f"Speed: {speed_str}{vram}")


                # Próbka tekstu co eval_interval*2
                if iter_num > 0 and iter_num % (eval_interval * 2) == 0:
                    print(f"  ── próbka generacji (iter {iter_num}) ──")
                    sample = generate_text(max_new_tokens=100)
                    for line in sample.split('\n')[:3]:
                        print(f"  │ {line[:70]}")

            # ── KROK TRENINGOWY z MultiTaskFlowLoss ──
            xb, yb = get_batch('train')
            logits, metrics = MODEL(xb)
            # Przekazujemy iteracje dla Warmup
            loss, loss_info = LOSS_FN(logits, yb, metrics, 
                                      iter_num=iter_num, total_iters=iters)

            OPTIMIZER.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(MODEL.parameters(), max_norm=1.0)
            OPTIMIZER.step()

        print(f"\n  {'═'*65}")
        print(f"  🎉 Trening zakończony pomyślnie!")

    except KeyboardInterrupt:
        print(f"\n\n  🛑 PRZERWANO (CTRL+C) — zapisuję aktualny stan...")
        checkpoint = {
            'model_state_dict': MODEL.state_dict(),
            'vocab_size': VOCAB_SIZE, 'stoi': STOI, 'itos': ITOS,
            'config': {'d_model': SETTINGS['d_model'], 'layers': SETTINGS['layers'],
                       'heads': SETTINGS['heads'], 'patterns': SETTINGS['patterns']},
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"  💾 Zapisano do: {checkpoint_path}")

    # Podsumowanie
    final = estimate_loss()
    elapsed = time.time() - start_time
    print(f"  ⏱️  Czas: {elapsed/60:.1f} min │ Final Val: {final['val']:.4f} │ Best: {best_val_loss:.4f}")

    # Finalna próbka
    print(f"\n  📝 Finalna próbka generacji (po treningu):")
    sample = generate_text(max_new_tokens=200)
    for line in sample.split('\n')[:6]:
        print(f"  │ {line[:70]}")
    print(f"  {'═'*65}")

    input("\n  Wciśnij Enter, by wrócić do menu...")

def chat_interface():
    if MODEL is None:
        print("⚠️ Zbuduj wpierw i wytrenuj model (Opcja 1)!")
        input("\nWciśnij Enter, by wrócić...")
        return

    clear_screen()
    temperature = 0.8  # Domyślna temperatura

    def show_header():
        print("═" * 60)
        print(" 🌊 TERMINAL CHAT — FlowNetwork v3.1")
        print(" Siec przewiduje następne znaki bazując na wzorcach.")
        print("─" * 60)
        # Sprawdz aktualny loss dla kontekstu
        try:
            losses = estimate_loss()
            val = losses['val']
            if val > 2.5:
                hint = "(za mało treningu — zrozumiałość będzie niska)"
            elif val > 2.0:
                hint = "(można zobaczyć początki sensu)"
            elif val > 1.5:
                hint = "(częściowo czytelny)"
            else:
                hint = "(powinno być sensowne!)"
            print(f" Val Loss: {val:.4f} {hint}")
        except:
            pass
        print(f" Temperatura: {temperature:.1f}  "
              f"(0.5=skupiony | 0.8=naturalny | 1.2=kreatywny)")
        print("─" * 60)
        print(" Polecenia: 'exit' = wróć, 'temp X' = zmień temperaturę")
        print(" Przykład: temp 0.6")
        print("═" * 60)

    show_header()

    while True:
        prompt = input("\n👤 Ty: ")
        if prompt.lower() == 'exit':
            break

        # Obsługa polecenia temperatury
        if prompt.lower().startswith('temp '):
            try:
                temperature = float(prompt.split()[1])
                print(f"  ✅ Temperatura zmieniona na: {temperature:.1f}")
            except:
                print("  ❌格式: temp 0.7")
            continue

        if not prompt: prompt = "\n"
        print("🤖 Flow:", end=" ")
        result = generate_text(prompt=prompt, max_new_tokens=300, temperature=temperature)
        print(result[len(prompt):])

def save_model():
    if MODEL is None:
        print("⚠️ Brak modelu w pamięci do zapisania.")
        return
    path = input("Podaj nazwę pliku (np. base_model.pt): ")
    if not path.endswith('.pt'): path += '.pt'

    checkpoint = {
        'model_state_dict': MODEL.state_dict(),
        'optimizer_state_dict': OPTIMIZER.state_dict(),
        'vocab_size': VOCAB_SIZE,
        'stoi': STOI,
        'itos': ITOS,
        'config': {
            'd_model': SETTINGS['d_model'],
            'layers': SETTINGS['layers'],
            'heads': SETTINGS['heads'],
            'patterns': SETTINGS['patterns'],
        },
        'settings': dict(SETTINGS),
    }
    try:
        torch.save(checkpoint, path)
        print(f"✅ Model + słownik + config zapisano jako {path}!")
    except Exception as e:
        print(f"❌ Wystąpił błąd podczas zapisu modelu:\n{e}")
    input("\nWciśnij Enter...")

def load_model():
    global MODEL, OPTIMIZER, VOCAB_SIZE, STOI, ITOS
    path = input("Podaj nazwę pliku (np. base_model.pt): ")
    if not os.path.exists(path):
        print("❌ Plik nie istnieje.")
        input("\nWciśnij Enter...")
        return

    print(f"📥 Ładowanie {path}...")
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    VOCAB_SIZE = checkpoint['vocab_size']
    STOI = checkpoint['stoi']
    ITOS = checkpoint['itos']

    # Odtwórz ustawienia z checkpointu (jeśli dostępne)
    if 'config' in checkpoint:
        cfg = checkpoint['config']
        SETTINGS['d_model'] = cfg.get('d_model', SETTINGS['d_model'])
        SETTINGS['layers'] = cfg.get('layers', cfg.get('num_layers', SETTINGS['layers']))
        SETTINGS['heads'] = cfg.get('heads', cfg.get('num_heads', SETTINGS['heads']))
        SETTINGS['patterns'] = cfg.get('patterns', cfg.get('num_patterns', SETTINGS['patterns']))
        print(f"  Config odtworzony: d={SETTINGS['d_model']}, layers={SETTINGS['layers']}, "
              f"patterns={SETTINGS['patterns']}")

    if 'settings' in checkpoint:
        for k, v in checkpoint['settings'].items():
            if k in SETTINGS:
                SETTINGS[k] = v

    build_model()
    MODEL.load_state_dict(checkpoint['model_state_dict'])
    if 'optimizer_state_dict' in checkpoint:
        try:
            OPTIMIZER.load_state_dict(checkpoint['optimizer_state_dict'])
        except Exception:
            print("  ⚠️ Optimizer state niezgodny — zainicjalizowano nowy.")

    print("✅ Model i wagi wczytano pomyślnie!")
    input("\nWciśnij Enter...")

def settings_menu():
    while True:
        clear_screen()
        print("⚙️  USTAWIENIA ARCHITEKTURY")
        print("─" * 40)
        for i, (k, v) in enumerate(SETTINGS.items()):
            print(f"  {i+1}. {k}: {v}")
        print(f"\n  0. Wróć do menu")

        c = input("\nWybierz opcję do edycji: ")
        if c == '0': break
        try:
            idx = int(c) - 1
            key = list(SETTINGS.keys())[idx]
            new_val = input(f"Nowa wartość dla {key} (obecnie {SETTINGS[key]}): ")

            if isinstance(SETTINGS[key], int):
                SETTINGS[key] = int(new_val)
            elif isinstance(SETTINGS[key], float):
                SETTINGS[key] = float(new_val)
            else:
                SETTINGS[key] = new_val

            # Jeśli zmieniono architekturę, trzeba usunąć model
            if key in ['d_model', 'layers', 'heads', 'patterns']:
                global MODEL
                MODEL = None
                print("⚠️ Model usunięty z RAM — wymagana ponowna budowa.")
        except Exception:
            pass

def main():
    while True:
        clear_screen()
        print("╔════════════════════════════════════════════════════════╗")
        print("║ 🌊 FLOW NETWORK TERMINAL v3.1  (Archeus Dissonance)   ║")
        print("║    MultiTaskFlowLoss + Cognitive Tension (0.61)        ║")
        print("╚════════════════════════════════════════════════════════╝")
        state_model = "Aktywny w RAM" if MODEL is not None else "Brak"
        data_info = SETTINGS['data_path']
        print(f"  🖥️  Urządzenie : {DEVICE.upper()} │ 📦 Sieć: {state_model}")
        print(f"  📁 Dane        : {data_info}\n")

        print("  [1] ⚡ Trenuj Sieć  (wybór pliku + rekomendacje)")
        print("  [2] 💬 Czat        (testuj inferencję modelu)")
        print("  [3] 💾 Zapisz Sieć (do pliku .pt)")
        print("  [4] 📂 Wczytaj     (sieć z pliku .pt)")
        print("  [5] ⚙️  Ustawienia  (hiperparametry, plik danych)")
        print("  [0] 🚪 Wyjście")

        choice = input("\n🤖 Wybierz polecenie: ")

        if choice == '1': run_training()
        elif choice == '2': chat_interface()
        elif choice == '3': save_model()
        elif choice == '4': load_model()
        elif choice == '5': settings_menu()
        elif choice == '0': break

if __name__ == '__main__':
    main()
