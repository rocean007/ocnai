"""
scrape_diseases.py
Downloads plant disease data from Wikipedia API.
No API key needed — completely free.
"""

import requests
import json
import time

DISEASES = [
    "Early blight", "Late blight", "Powdery mildew", "Downy mildew",
    "Gray mold", "Fusarium wilt", "Verticillium wilt", "Clubroot",
    "Black rot", "Anthracnose", "Bacterial spot", "Tobacco mosaic virus",
    "Root rot", "Damping off", "Rust (fungus)", "Leaf spot",
    "Sclerotinia", "Alternaria blight", "Septoria leaf spot",
    "Pythium", "Phytophthora", "Botrytis cinerea",
    "Cercospora leaf spot", "Downy mildew", "White mold",
    "Bacterial wilt", "Mosaic virus", "Curly top virus",
    "Tomato spotted wilt virus", "Crown gall", "Fire blight",
    "Canker (plant disease)", "Blight", "Mildew"
]

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"

def get_wiki_summary(title):
    """Get summary from Wikipedia API."""
    params = {
        "action": "query",
        "prop": "extracts",
        "exintro": True,
        "explaintext": True,
        "titles": title,
        "format": "json",
        "redirects": 1
    }
    resp = requests.get(WIKIPEDIA_API, params=params, timeout=10)
    resp.raise_for_status()

    pages = resp.json().get("query", {}).get("pages", {})
    for page_id, page in pages.items():
        if page_id == "-1":
            return None
        extract = page.get("extract", "")
        return {
            "title": page.get("title", title),
            "summary": extract[:3000],
            "url": f"https://en.wikipedia.org/wiki/{page.get('title', title).replace(' ', '_')}"
        }
    return None

def scrape():
    results = []
    failed = []
    seen_titles = set()

    for disease in DISEASES:
        try:
            data = get_wiki_summary(disease)
            if data and data["title"] not in seen_titles:
                results.append({
                    "disease": disease,
                    "wiki_title": data["title"],
                    "summary": data["summary"],
                    "url": data["url"]
                })
                seen_titles.add(data["title"])
                print(f"  ✓  {disease}")
            else:
                print(f"  x  {disease} — not found or duplicate")
                failed.append(disease)
            time.sleep(0.3)

        except Exception as e:
            print(f"  x  {disease} — {e}")
            failed.append(disease)

    with open("raw_diseases.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Saved {len(results)} diseases to raw_diseases.json")
    if failed:
        print(f"⚠️  Failed: {', '.join(failed)}")

if __name__ == "__main__":
    print("Scraping Wikipedia disease data...\n")
    scrape()
