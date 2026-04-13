import torch
import torch.nn as nn
import time
import sys
if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass
from flow_network.models import EnhancedFlowTransformer

def test_linearity():
    print("=" * 60)
    print("🧪 TEST STRESOWY: WERYFIKACJA LINIOWOŚCI (O(N))")
    print("=" * 60)
    print("Testujemy, czy czas i pamięć rosną LINIOWO (x2), a nie KWADRATOWO (x4).")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Urządzenie: {device.upper()}")
    
    # Mały model dla przejrzystości
    model = EnhancedFlowTransformer(
        vocab_size=1000,
        d_model=128,
        num_layers=2,
        use_memory=False
    ).eval().to(device)

    # Będziemy wstrzykiwać coraz to dłuższe teksty (podwajając długość)
    sequence_lengths = [1024, 2048, 4096, 8192, 16384]
    
    print("\n--- ZACZYNAMY POMIARY ---")
    
    for seq_len in sequence_lengths:
        # Tworzymy zjawisko równe wczytaniu na raz N tokenów
        dummy_input = torch.randint(0, 1000, (1, seq_len)).to(device)
        
        # Reset pamięci jeśli CUDA
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            
        start_time = time.time()
        
        with torch.no_grad():
            _ = model(dummy_input)
            
        if device == 'cuda':
            torch.cuda.synchronize()
            peak_memory = torch.cuda.max_memory_allocated() / (1024**2)
        else:
            peak_memory = 0.0 # Na CPU ciężko precyzyjnie mierzyć RAM w jednej linijce, patrzymy na czas
            
        execution_time = time.time() - start_time
        
        # Formatowanie wydruku
        mem_str = f"{peak_memory:.2f} MB" if device == 'cuda' else "Sprawdź Menedżer Zadań"
        print(f"Długość sekwencji: {seq_len:>6} tokenów | Czas: {execution_time:>6.3f} sek | Pamięć VRAM: {mem_str}")

if __name__ == "__main__":
    test_linearity()
