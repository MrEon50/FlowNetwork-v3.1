import argparse
import os
import urllib.request
import torch
import torch.nn.functional as F
import time
import math
import sys
import json

if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass
from flow_network.models import EnhancedFlowTransformer
from flow_network.utils import safe_tensor_to_int
from flow_network.training import MultiTaskFlowLoss

parser = argparse.ArgumentParser(description='Trening Sieci FlowNetwork')
parser.add_argument('--data', type=str, default='tinyshakespeare.txt', help='Ścieżka do pliku w formacie tekstowym (.txt)')
parser.add_argument('--iters', type=int, default=2000, help='Liczba całkowitych iteracji (epok) podczas treningu')
parser.add_argument('--batch_size', type=int, default=32, help='Rozmiar pakietu podczas treningu')
parser.add_argument('--seq_len', type=int, default=128, help='Długość okna sekwencji na wejściu (kontekst)')
parser.add_argument('--eval_interval', type=int, default=100, help='Co ile iteracji sprawdzać i raportować stratę?')
parser.add_argument('--lr', type=float, default=1e-3, help='Współczynnik uczenia (Learning Rate)')
parser.add_argument('--save_checkpoint', type=str, default='flow_checkpoint.pt', help='Ścieżka do zapisu checkpointu')

# Parametry Architektury
parser.add_argument('--d_model', type=int, default=128, help='Rozmiar wymiarów sieci przestrzennej')
parser.add_argument('--layers', type=int, default=4, help='Liczba warstw typu FlowLayer')
parser.add_argument('--heads', type=int, default=8, help='Liczba przepływów strumieni (głów)')
parser.add_argument('--patterns', type=int, default=6, help='Liczba matrycowych wzorców routingu we Flow')

args = parser.parse_args()

DATA_URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
DATA_PATH = args.data
BATCH_SIZE = args.batch_size
SEQ_LEN = args.seq_len
MAX_ITERS = args.iters
EVAL_INTERVAL = args.eval_interval
LEARNING_RATE = args.lr
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

D_MODEL = args.d_model
NUM_LAYERS = args.layers
NUM_HEADS = args.heads
NUM_PATTERNS = args.patterns

# ============================================================================
# MONITORING & HISTORY TRACKER
# ============================================================================

class TrainingMonitor:
    """Monitoruje i wyświetla wskaźniki treningu w czasie rzeczywistym."""

    def __init__(self):
        self.history = {
            'iter': [], 'train_loss': [], 'val_loss': [],
            'task_loss': [], 'coherence_loss': [], 'diversity_loss': [],
            'memory_loss': [], 'grad_norm': [], 'throughput': [],
            'vram_mb': [], 'lr': [], 'channel_entropy': [],
            'pattern_entropy': [], 'flow_intensity': [],
        }
        self.best_val_loss = float('inf')
        self.best_iter = 0
        self.start_time = time.time()

    def log_eval(self, iter_num, train_loss, val_loss, loss_info, flow_metrics,
                 grad_norm, throughput, vram_mb, lr):
        """Loguj metryki ewaluacji."""
        self.history['iter'].append(iter_num)
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['task_loss'].append(loss_info.get('task', 0))
        self.history['coherence_loss'].append(loss_info.get('coherence', 0))
        self.history['diversity_loss'].append(loss_info.get('diversity', 0))
        self.history['memory_loss'].append(loss_info.get('memory', 0))
        self.history['grad_norm'].append(grad_norm)
        self.history['throughput'].append(throughput)
        self.history['vram_mb'].append(vram_mb)
        self.history['lr'].append(lr)

        # Flow-specific metrics from last forward pass
        ch_ent = self._extract_metric(flow_metrics, 'channel_entropy')
        pat_ent = self._extract_metric(flow_metrics, 'pattern_entropy')
        fl_int = self._extract_metric(flow_metrics, 'flow_intensity')
        self.history['channel_entropy'].append(ch_ent)
        self.history['pattern_entropy'].append(pat_ent)
        self.history['flow_intensity'].append(fl_int)

        # Track best
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_iter = iter_num

    def _extract_metric(self, metrics_list, key):
        """Wyciągnij metrykę z listy metryk flow."""
        for m in metrics_list:
            if isinstance(m, dict) and key in m:
                val = m[key]
                if isinstance(val, torch.Tensor):
                    return val.mean().item()
                if isinstance(val, (int, float)):
                    return float(val)
        return 0.0

    def _extract_channel_weights(self, metrics_list):
        """Wyciągnij wagi kanałów."""
        for m in metrics_list:
            if isinstance(m, dict) and 'channel_weights' in m:
                cw = m['channel_weights']
                if isinstance(cw, torch.Tensor):
                    return cw.tolist()
        return None

    def print_eval(self, iter_num, train_loss, val_loss, loss_info, flow_metrics,
                   grad_norm, throughput, vram_mb, lr):
        """Wydrukuj rozbudowany raport monitoringu."""
        elapsed = time.time() - self.start_time
        eta = (elapsed / max(iter_num, 1)) * (MAX_ITERS - iter_num)

        # Nagłówek
        print(f"\n{'═'*70}")
        print(f"  📊 ITERACJA {iter_num:5d}/{MAX_ITERS}  │  "
              f"Czas: {elapsed:.0f}s  │  ETA: {eta:.0f}s")
        print(f"{'─'*70}")

        # Główne straty
        delta = ""
        if len(self.history['val_loss']) >= 2:
            d = val_loss - self.history['val_loss'][-2]
            arrow = "↓" if d < 0 else "↑" if d > 0 else "→"
            delta = f" ({arrow}{abs(d):.4f})"

        print(f"  🎯 Train Loss: {train_loss:.4f}  │  Val Loss: {val_loss:.4f}{delta}")
        print(f"  🏆 Best Val:   {self.best_val_loss:.4f} (iter {self.best_iter})")

        # Komponenty Multi-Task Loss
        print(f"{'─'*70}")
        task = loss_info.get('task', 0)
        coh = loss_info.get('coherence', 0)
        div_val = loss_info.get('diversity', 0)
        mem = loss_info.get('memory', 0)

        task_v = f"{task:.4f}" if isinstance(task, (int, float)) else str(task)
        coh_v = f"{coh:.4f}" if isinstance(coh, (int, float)) else str(coh)
        div_v = f"{div_val:.4f}" if isinstance(div_val, (int, float)) else str(div_val)
        mem_v = f"{mem:.4f}" if isinstance(mem, (int, float)) else str(mem)

        print(f"  📐 Task(CE): {task_v}  │  Coherence: {coh_v}  │  "
              f"Diversity: {div_v}  │  Memory: {mem_v}")

        # Flow-specific metrics
        ch_ent = self._extract_metric(flow_metrics, 'channel_entropy')
        pat_ent = self._extract_metric(flow_metrics, 'pattern_entropy')
        fl_int = self._extract_metric(flow_metrics, 'flow_intensity')
        ch_weights = self._extract_channel_weights(flow_metrics)

        print(f"{'─'*70}")
        print(f"  🌊 Flow Intensity: {fl_int:.4f}  │  "
              f"Pattern Entropy: {pat_ent:.4f}  │  Channel Entropy: {ch_ent:.4f}")

        if ch_weights:
            labels = ["Semantic", "Positional", "Temporal"]
            parts = []
            for i, w in enumerate(ch_weights[:3]):
                lbl = labels[i] if i < len(labels) else f"Ch{i}"
                bar = "█" * int(w * 20)
                parts.append(f"{lbl}: {w:.3f} {bar}")
            print(f"  🎨 Channel Mix: {' │ '.join(parts)}")

        # Sprzęt
        print(f"{'─'*70}")
        print(f"  ⚡ Speed: {throughput:.0f} tok/s  │  "
              f"Grad Norm: {grad_norm:.4f}  │  LR: {lr:.6f}  │  "
              f"VRAM: {vram_mb:.1f} MB")
        print(f"{'═'*70}")

    def print_health_check(self, grad_norm, loss_info):
        """Sprawdź zdrowie treningu."""
        warnings = []

        if grad_norm > 5.0:
            warnings.append("⚠️  Gradient norm wysoki (>5.0) — ryzyko eksplozji gradientów")
        if grad_norm < 1e-5:
            warnings.append("⚠️  Gradient norm bliski zeru — model może nie uczyć się")

        task_loss = loss_info.get('task', 0)
        if isinstance(task_loss, (int, float)) and task_loss > 10.0:
            warnings.append("⚠️  Task loss bardzo wysoki — sprawdź dane/architekturę")

        coh = loss_info.get('coherence', 0)
        if isinstance(coh, (int, float)) and coh > 1.0:
            warnings.append("⚠️  Koherencja wysoka — model generuje chaotyczne sekwencje")

        if warnings:
            for w in warnings:
                print(f"  {w}")
        else:
            print(f"  ✅ Zdrowie treningu: OK")

    def save_history(self, filepath='training_history.json'):
        """Zapisz historię do pliku JSON."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, default=str)


# ============================================================================
# SETUP
# ============================================================================

print(f"🚀 Uruchamiam system na urządzeniu: {DEVICE}")

# 1. Pobieranie zbioru danych (Tiny Shakespeare)
if not os.path.exists(DATA_PATH):
    print("📥 Pobieranie zbioru danych (Szekspir)...")
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    print("✅ Pobrano.")

with open(DATA_PATH, 'r', encoding='utf-8') as f:
    text = f.read()

print(f"📚 Długość zbioru danych: {len(text):,} znaków")

# 2. Tokenizator Poziomu Znaków (Char-level Tokenizer)
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"🔤 Rozmiar słownika (unikalne znaki): {vocab_size}")

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

# 3. Przygotowanie Tensorów
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data_source = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_source) - SEQ_LEN, (BATCH_SIZE,))
    x = torch.stack([data_source[i:i+SEQ_LEN] for i in ix])
    y = torch.stack([data_source[i+1:i+SEQ_LEN+1] for i in ix])
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y

# 4. Inicjalizacja Architektury
print(f"\n{'═'*70}")
print(f"  🧠 INICJALIZACJA ENHANCED FLOW TRANSFORMER")
print(f"{'═'*70}")
model = EnhancedFlowTransformer(
    vocab_size=vocab_size,
    d_model=D_MODEL,
    num_layers=NUM_LAYERS,
    max_seq_len=0,
    num_heads=NUM_HEADS,
    num_patterns=NUM_PATTERNS,
    dropout=0.1,
    use_memory=True
)
model = model.to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  📦 Parametry:     {total_params / 1e6:.2f} M (trenowalne: {trainable_params / 1e6:.2f} M)")
print(f"  🏗️  Architektura:  d_model={D_MODEL}, layers={NUM_LAYERS}, patterns={NUM_PATTERNS}, heads={NUM_HEADS}")
print(f"  📐 Batch:         {BATCH_SIZE} × {SEQ_LEN} = {BATCH_SIZE * SEQ_LEN:,} tok/iter")
print(f"  🎓 LR:            {LEARNING_RATE}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Learning Rate Scheduler (cosine annealing)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_ITERS, eta_min=LEARNING_RATE / 10)

# Multi-task loss
loss_fn = MultiTaskFlowLoss(
    diversity_weight=0.001,
    context_weight=0.0,
    coherence_weight=0.05,
    conversation_weight=0.0,
    memory_weight=0.02
)
print(f"  ⚙️  Loss:          MultiTaskFlowLoss (coherence={loss_fn.coherence_weight}, "
      f"diversity={loss_fn.diversity_weight}, memory={loss_fn.memory_weight})")

# Monitor
monitor = TrainingMonitor()

# ============================================================================
# EWALUACJA
# ============================================================================

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(10)
        for k in range(10):
            X, Y = get_batch(split)
            logits, _ = model(X)
            B, T, C = logits.shape
            logits_reshaped = logits.view(B*T, C)
            targets = Y.view(B*T)
            loss = F.cross_entropy(logits_reshaped, targets)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

@torch.no_grad()
def generate_sample(prompt="\n", max_new_tokens=100):
    model.eval()
    idx = torch.tensor(encode(prompt), dtype=torch.long, device=DEVICE).unsqueeze(0)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -512:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        next_idx = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_idx), dim=1)

    generated_text = decode(idx[0].tolist())
    model.train()
    return generated_text

def compute_grad_norm():
    """Oblicz normę gradientów."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5

def save_checkpoint(iter_num, val_loss):
    """Zapisz checkpoint modelu."""
    checkpoint = {
        'iter': iter_num,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'config': {
            'vocab_size': vocab_size,
            'd_model': D_MODEL,
            'num_layers': NUM_LAYERS,
            'num_heads': NUM_HEADS,
            'num_patterns': NUM_PATTERNS,
        },
        'stoi': stoi,
        'itos': itos,
    }
    torch.save(checkpoint, args.save_checkpoint)
    print(f"  💾 Checkpoint zapisany: {args.save_checkpoint} (val_loss={val_loss:.4f})")

# ============================================================================
# GŁÓWNA PĘTLA TRENUJĄCA
# ============================================================================

print(f"\n{'═'*70}")
print(f"  ⚡ ROZPOCZYNAM TRENING — {MAX_ITERS} iteracji")
print(f"{'═'*70}")

# Próbka przed treningiem
print(f"\n  📝 Próbka PRZED treningiem (losowy belkot):")
print(f"  {'─'*50}")
pre_sample = generate_sample(max_new_tokens=60)
for line in pre_sample.split('\n')[:3]:
    print(f"  │ {line[:70]}")
print(f"  {'─'*50}")

# Zmienne do śledzenia last loss_info i metrics
last_loss_info = {}
last_flow_metrics = []

for iter_num in range(MAX_ITERS):

    # ── EWALUACJA ──
    if iter_num % EVAL_INTERVAL == 0 or iter_num == MAX_ITERS - 1:
        losses = estimate_loss()

        # VRAM
        vram_mb = 0.0
        if DEVICE == 'cuda':
            vram_mb = torch.cuda.memory_allocated() / 1024**2

        elapsed = time.time() - monitor.start_time
        tokens_processed = iter_num * BATCH_SIZE * SEQ_LEN
        throughput = tokens_processed / elapsed if elapsed > 0 else 0
        grad_norm = compute_grad_norm() if iter_num > 0 else 0.0
        current_lr = scheduler.get_last_lr()[0]

        # Loguj i drukuj
        monitor.log_eval(
            iter_num, losses['train'], losses['val'], last_loss_info,
            last_flow_metrics, grad_norm, throughput, vram_mb, current_lr
        )
        monitor.print_eval(
            iter_num, losses['train'], losses['val'], last_loss_info,
            last_flow_metrics, grad_norm, throughput, vram_mb, current_lr
        )
        monitor.print_health_check(grad_norm, last_loss_info)

        # Checkpoint jeśli najlepszy
        if losses['val'] < monitor.best_val_loss + 0.001:
            save_checkpoint(iter_num, losses['val'])

        # Próbka generacji co 200 iteracji
        if iter_num > 0 and iter_num % (EVAL_INTERVAL * 2) == 0:
            print(f"\n  📝 Próbka generacji (iter {iter_num}):")
            print(f"  {'─'*50}")
            sample = generate_sample(max_new_tokens=150)
            for line in sample.split('\n')[:5]:
                print(f"  │ {line[:70]}")
            print(f"  {'─'*50}")

    # ── KROK TRENINGOWY ──
    xb, yb = get_batch('train')

    # Forward
    logits, metrics = model(xb)

    # Multi-task loss
    loss, loss_info = loss_fn(logits, yb, metrics)
    last_loss_info = loss_info
    last_flow_metrics = metrics

    # Backward
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()

# ============================================================================
# FINALIZACJA
# ============================================================================

print(f"\n{'═'*70}")
print(f"  🎉 TRENING ZAKOŃCZONY POMYŚLNIE!")
print(f"{'═'*70}")

# Końcowa ewaluacja
final_losses = estimate_loss()
elapsed = time.time() - monitor.start_time
print(f"  ⏱️  Czas całkowity:  {elapsed:.1f}s ({elapsed/60:.1f} min)")
print(f"  🎯 Finalna strata:  Train={final_losses['train']:.4f}  Val={final_losses['val']:.4f}")
print(f"  🏆 Najlepsza Val:   {monitor.best_val_loss:.4f} (iter {monitor.best_iter})")
print(f"  📈 Poprawa:         {monitor.history['val_loss'][0]:.4f} → {final_losses['val']:.4f}")

# Zapisz historię
monitor.save_history('training_history.json')
print(f"  📊 Historia zapisana: training_history.json")

# Zapisz finalny checkpoint
save_checkpoint(MAX_ITERS, final_losses['val'])

# Finalna próbka
print(f"\n  📝 FINAŁOWA PRÓBKA GENERACJI:")
print(f"  {'═'*50}")
final_sample = generate_sample(max_new_tokens=300)
for line in final_sample.split('\n')[:10]:
    print(f"  │ {line[:70]}")
print(f"  {'═'*50}")
print(f"\n  🏁 Gotowe!")
