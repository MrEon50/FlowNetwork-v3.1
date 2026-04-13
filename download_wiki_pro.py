"""
Downloader profesjonalnej bazy wiedzy z Polskiej Wikipedii.
Pobiera PEŁNE artykuły (nie streszczenia) z wielu kategorii tematycznych.
Cel: 30-50 MB czystego, ustrukturyzowanego tekstu.
"""
import requests
import time
import os
import json

# Kategorie tematyczne z polskiej Wikipedii — bogate w wiedzę encyklopedyczną
CATEGORIES = [
    "Nauki przyrodnicze",
    "Fizyka",
    "Chemia",
    "Biologia",
    "Matematyka",
    "Informatyka",
    "Historia Polski",
    "Historia świata",
    "Geografia",
    "Astronomia",
    "Medycyna",
    "Filozofia",
    "Psychologia",
    "Ekonomia",
    "Sztuka",
    "Literatura polska",
    "Muzyka",
    "Technologia",
    "Ekologia",
    "Językoznawstwo",
]

WIKI_API = "https://pl.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "FlowNetworkDatasetBuilder/1.0"}
TARGET_MB = 40  # Cel: 40 MB
OUTPUT = "wiki_encyclopedia_40mb.txt"


def get_category_members(category, limit=50):
    """Pobiera listę artykułów z danej kategorii."""
    params = {
        "action": "query",
        "format": "json",
        "list": "categorymembers",
        "cmtitle": f"Kategoria:{category}",
        "cmlimit": limit,
        "cmtype": "page",
    }
    try:
        r = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=10)
        members = r.json().get("query", {}).get("categorymembers", [])
        return [m["title"] for m in members]
    except:
        return []


def get_full_article(title):
    """Pobiera pełną treść artykułu (nie streszczenie!)."""
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": True,      # Czysty tekst bez tagów HTML
        "exsectionformat": "plain",
        "titles": title,
    }
    try:
        r = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=15)
        pages = r.json().get("query", {}).get("pages", {})
        for page_id, page in pages.items():
            if page_id != "-1":
                text = page.get("extract", "")
                # Filtrujemy artykuły które są za krótkie (stub pages)
                if len(text) > 500:
                    return text
    except:
        pass
    return ""


def build():
    print(f"=== Budowanie Profesjonalnej Bazy Encyklopedycznej ===")
    print(f"Cel: {TARGET_MB} MB | Plik: {OUTPUT}\n")

    downloaded_titles = set()
    total_size = 0
    article_count = 0

    with open(OUTPUT, "w", encoding="utf-8") as f:
        for category in CATEGORIES:
            if total_size >= TARGET_MB * 1024 * 1024:
                break

            print(f"\n📚 Kategoria: {category}")
            titles = get_category_members(category, limit=100)
            print(f"   Znaleziono {len(titles)} artykułów")

            for title in titles:
                if title in downloaded_titles:
                    continue
                if total_size >= TARGET_MB * 1024 * 1024:
                    break

                text = get_full_article(title)
                if text:
                    # Format encyklopedyczny — czytelny dla modelu
                    entry = f"\n\n=== {title.upper()} ===\n{text}\n"
                    f.write(entry)
                    downloaded_titles.add(title)
                    article_count += 1
                    total_size += len(entry.encode("utf-8"))

                    size_mb = total_size / (1024 * 1024)
                    print(f"   [{article_count:4d}] {title[:50]:<50} | {size_mb:.1f} MB")

                time.sleep(0.1)  # Delikatny delay — szanujemy Wikipedia API

    final_mb = os.path.getsize(OUTPUT) / (1024 * 1024)
    print(f"\n{'='*60}")
    print(f"✅ GOTOWE!")
    print(f"   Plik: {OUTPUT}")
    print(f"   Rozmiar: {final_mb:.2f} MB")
    print(f"   Artykuły: {article_count}")
    print(f"   Unikalne tytuły: {len(downloaded_titles)}")
    print(f"{'='*60}")
    print(f"\nAby wytrenować model na tej bazie w flow_terminal.py,")
    print(f"wybierz [5] Ustawienia → data_path → {OUTPUT}")


if __name__ == "__main__":
    build()
