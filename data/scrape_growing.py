"""
scrape_growing.py
Downloads growing/farming data from OpenFarm API.
No API key needed — completely free.
"""

import requests
import json
import time

VEGETABLES = [
    "tomato", "carrot", "spinach", "broccoli", "potato", "onion",
    "garlic", "cucumber", "lettuce", "kale", "pepper", "zucchini",
    "pumpkin", "beetroot", "cabbage", "cauliflower", "celery", "peas",
    "beans", "corn", "squash", "leek", "chard", "arugula",
    "radish", "turnip", "parsnip", "asparagus", "artichoke",
    "eggplant", "sweet potato", "fennel", "kohlrabi", "okra",
    "bok choy", "collard greens", "mustard greens", "endive",
    "snap peas", "edamame", "watercress"
]

def scrape():
    results = []
    failed = []

    for veg in VEGETABLES:
        try:
            resp = requests.get(
                f"https://openfarm.cc/api/v1/crops?filter={veg}",
                timeout=10
            )
            resp.raise_for_status()
            crops = resp.json().get("data", [])

            if not crops:
                print(f"  x  {veg} — no results")
                failed.append(veg)
                continue

            crop = crops[0]["attributes"]
            results.append({
                "vegetable": veg,
                "name": crop.get("name", veg),
                "description": crop.get("description", ""),
                "sun_requirements": crop.get("sun_requirements", ""),
                "sowing_method": crop.get("sowing_method", ""),
                "spread": crop.get("spread", ""),
                "row_spacing": crop.get("row_spacing", ""),
                "height": crop.get("height", ""),
                "growing_degree_days": crop.get("growing_degree_days", ""),
                "minimum_temperature": crop.get("minimum_temperature_celsius", ""),
                "maximum_temperature": crop.get("maximum_temperature_celsius", ""),
                "ph_minimum": crop.get("ph_minimum", ""),
                "ph_maximum": crop.get("ph_maximum", ""),
                "tags": crop.get("tags_array", [])
            })
            print(f"  ✓  {veg}")
            time.sleep(0.3)

        except Exception as e:
            print(f"  x  {veg} — {e}")
            failed.append(veg)

    with open("raw_growing.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Saved {len(results)} vegetables to raw_growing.json")
    if failed:
        print(f"⚠️  Failed: {', '.join(failed)}")

if __name__ == "__main__":
    print("Scraping OpenFarm growing data...\n")
    scrape()
