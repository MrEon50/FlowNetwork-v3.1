"""
Pobiera 40MB profesjonalnych danych dialogowych z OpenSubtitles (Polish).
Dane lmają OPUS corpus - używany do trenowania m.in. Helsinki NLP models.
Strumieniuje archiwum GZ bez pobierania całości (1.8 GB).
"""
import requests
import gzip
import os

URL_SUBTITLES = "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/mono/pl.txt.gz"
URL_CC100     = "https://data.statmt.org/cc-100/pl.txt.xz"
TARGET_MB     = 40
OUTPUT        = "pl_subtitles_40mb.txt"


def stream_gz_to_txt(url, output_file, target_mb):
    """Strumieniuje plik .gz, dekompresuje w locie, zapisuje LIMIT danych."""
    target_bytes = target_mb * 1024 * 1024
    print(f"Łączenie z: {url}")
    print(f"Cel: {target_mb} MB -> {output_file}")
    print("(Strumieniujemy - nie pobieramy calosci!)\n")

    try:
        r = requests.get(url, stream=True, timeout=30)
        r.raise_for_status()
    except Exception as e:
        print(f"Błąd połączenia: {e}")
        return False

    downloaded_raw  = 0  # bajty skompresowane pobrane
    written_decoded = 0  # bajty zdekompresowane zapisane

    with open(output_file, "w", encoding="utf-8", errors="replace") as out_f:
        decompressor = gzip.GzipFile(fileobj=r.raw)
        buf = b""

        while written_decoded < target_bytes:
            try:
                chunk = decompressor.read(65536)  # 64 KB na raz
            except Exception:
                break
            if not chunk:
                break

            buf += chunk
            # Dekodujemy pełne linie
            lines = buf.split(b"\n")
            buf = lines[-1]  # ostatnia (niekompletna) linia zostaje w buforze

            for line in lines[:-1]:
                try:
                    text = line.decode("utf-8").strip()
                except:
                    continue
                if text:  # Pomijamy puste linie
                    out_f.write(text + "\n")
                    written_decoded += len(text) + 1

            downloaded_raw += 65536

            if int(written_decoded / (1024*1024)) % 5 == 0 and written_decoded > 0:
                print(f"  Pobrano: {written_decoded/(1024*1024):.1f} MB "
                      f"({written_decoded*100//target_bytes}%)")

    final_mb = os.path.getsize(output_file) / (1024*1024)
    print(f"\n✅ GOTOWE!")
    print(f"   Plik: {output_file}")
    print(f"   Rozmiar: {final_mb:.2f} MB")
    return True


def check_quality(filepath, lines_to_check=20):
    """Wyświetla próbkę danych do weryfikacji jakości."""
    print(f"\nPróbka danych z {filepath}:")
    print("─" * 60)
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= lines_to_check:
                break
            print(f"  {line.rstrip()}")
    print("─" * 60)


if __name__ == "__main__":
    print("=" * 60)
    print("  PROFESJONALNA BAZA DIALOGOWA — OpenSubtitles Polish")
    print("  Źródło: OPUS Corpus (Helsinki NLP / EU Project)")
    print("=" * 60)

    success = stream_gz_to_txt(URL_SUBTITLES, OUTPUT, TARGET_MB)

    if success and os.path.exists(OUTPUT):
        check_quality(OUTPUT)
        print(f"\nTeraz w flow_terminal.py wybierz [1] Trenuj Sieć")
        print(f"i podaj ścieżkę do pliku: {OUTPUT}")
