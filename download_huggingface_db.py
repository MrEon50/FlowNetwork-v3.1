import os
import subprocess

def install_and_download():
    print("Sprawdzanie i instalowanie bibliotek...")
    try:
        from datasets import load_dataset
    except ImportError:
        print("Instalowanie biblioteki datasets (Hugging Face)...")
        subprocess.check_call([os.sys.executable, "-m", "pip", "install", "datasets", "pandas", "-q"])
        from datasets import load_dataset

    filepath = r"c:\Users\Endorfinka\Desktop\ZBIERACZ KODU\flow_network_project 3\flow_dataset_v3.txt"
    print(f"Pobieranie bazy 'MarekPiotr/alpaca_pl_52k' z HuggingFace...")
    
    try:
        # Ten dataset jest bardzo popularny i zwykle dostępny
        dataset = load_dataset('MarekPiotr/alpaca_pl_52k', split='train')
    except Exception as e:
        print("Błąd pobierania:", e)
        # Spróbujmy jeszcze 'mmosiolek/polpaca' (jeśli istnieje w tej ścieżce)
        try:
            dataset = load_dataset('mmosiolek/polpaca', split='train')
        except:
            print("Wszystkie próby pobrania z HuggingFace zawiodły.")
            return

    print(f"Pobrano {len(dataset)} rekordów. Trwa zapisywanie...")
    
    with open(filepath, "w", encoding="utf-8") as f:
        for idx, row in enumerate(dataset):
            instruction = row.get('instruction', '')
            input_text = row.get('input', '')
            output = row.get('output', '')
            
            context = f"Instrukcja: {instruction}\nWejście: {input_text}\n" if input_text else f"Instrukcja: {instruction}\n"
            f.write(f"[USER]: {context}\n[AGENT]: {output}\n\n")
            
            if idx % 5000 == 0:
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                print(f"Zapisano... {size_mb:.2f} MB")
                if size_mb > 50.0: # Celujemy w 50MB
                    break

    print(f"\nGotowe! Baza zapisana: {filepath}")
    print(f"Końcowy rozmiar: {os.path.getsize(filepath) / (1024 * 1024):.2f} MB")

if __name__ == "__main__":
    install_and_download()
