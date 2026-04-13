"""
Czyszczenie bazy napisów filmowych z artefaktów formatu SRT/ASS.
Usuwa: numery, timecody, tagi HTML, nadmiarowe znaki specjalne.
Zostawia: czyste polskie zdania dialogowe.
"""
import re
import os
import sys

# Sciezki - mozna podac jako argument: python clean_subtitles.py sciezka_wejsciowa.txt
if len(sys.argv) > 1:
    INPUT  = sys.argv[1]
else:
    INPUT  = r"c:\Users\Endorfinka\Desktop\ZBIERACZ KODU\flow_network_project 3\moje_dane\książki_dialogi.txt"

# Plik wyjsciowy - obok wejsciowego z sufiksem _clean
base, ext = os.path.splitext(INPUT)
OUTPUT = base + "_clean" + ext

# Wzorce do usunięcia
PATTERNS_REMOVE = [
    r"^\d+\s*$",                         # Numery napisów (1, 2, 3...)
    r"^\d{2}:\d{2}:\d{2}",              # Timecody 00:01:23,456
    r"<[^>]+>",                          # Tagi HTML <i>, <b>, </i>
    r"^\s*[#\*\-]{2,}\s*$",             # Linie z samymi #, *, --
    r"^\s*[\(\[\{].*?[\)\]\}]\s*$",     # (opisy dźwięków), [muzyka]
    r"^[A-ZĆĘŁŃÓŚŹŻ\s]{3,}:\s*$",      # IMIĘ POSTACI: (wielkie litery)
    r"^\s*$",                            # Puste linie (łączymy osobno)
]

REMOVE_INLINE = [
    r"<[^>]+>",           # Tagi HTML wewnątrz linii
    r"\{[^}]+\}",         # {komentarze w nawiasach klamrowych}
    r"^\s*-{1,2}\s*",     # Myślniki na początku linii dialogu
    r"♪.*?♪",             # Tekst piosenek
    r"♪",                 # Samotne nuty
    r"\xa0",              # Non-breaking spaces
]

# Polskie znaki + podstawowa interpunkcja — filtr końcowy
ALLOWED_CHARS = set("aąbcćdeęfghijklłmnńoópqrsśtuvwxyzźżAĄBCĆDEĘFGHIJKLŁMNŃOÓPQRSŚTUVWXYZŹŻ"
                    " .,!?;:-'\"()\n0123456789")

def clean_line(line):
    # Usuń tagi i artefakty wewnątrz linii
    for pat in REMOVE_INLINE:
        line = re.sub(pat, "", line)
    # Usuń znaki spoza dozwolonego zestawu
    line = ''.join(c for c in line if c in ALLOWED_CHARS)
    return line.strip()

def should_remove(line):
    for pat in PATTERNS_REMOVE:
        if re.match(pat, line.strip()):
            return True
    return False

def process(input_path, output_path):
    print(f"Wczytywanie: {input_path}")
    with open(input_path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    print(f"Wierszy przed czyszczeniem: {len(lines):,}")

    cleaned = []
    removed = 0
    for line in lines:
        if should_remove(line):
            removed += 1
            continue
        clean = clean_line(line)
        if len(clean) > 3:  # Pomijamy bardzo krótkie fragmenty
            cleaned.append(clean)
        else:
            removed += 1

    # Usuń zduplikowane sąsiadujące linie (napisy często się powtarzają)
    deduped = [cleaned[0]] if cleaned else []
    for i in range(1, len(cleaned)):
        if cleaned[i] != cleaned[i-1]:
            deduped.append(cleaned[i])

    result = '\n'.join(deduped)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(result)

    in_mb  = os.path.getsize(input_path) / (1024*1024)
    out_mb = os.path.getsize(output_path) / (1024*1024)
    vocab  = len(set(result))

    print(f"\nWyniki czyszczenia:")
    print(f"  Wiersze:   {len(lines):,} -> {len(deduped):,} (usunieto {removed:,})")
    print(f"  Rozmiar:   {in_mb:.1f} MB -> {out_mb:.1f} MB")
    print(f"  Vocab:     {vocab} znakow (bylo ~652, teraz powinno byc ~100-130)")
    print(f"\nProbka czystych danych:")
    print("-" * 50)
    for line in deduped[100:120]:
        print(f"  {line}")
    print(f"\nGotowe! Uzyj '{output_path}' zamiast oryginalu.")

if __name__ == "__main__":
    if not os.path.exists(INPUT):
        print(f"Nie znaleziono: {INPUT}")
        print("Podaj poprawna sciezke do pliku:")
        INPUT = input("> ").strip()
    process(INPUT, OUTPUT)
