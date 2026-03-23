"""
scrape_nutrition.py
Downloads vegetable nutrition data from USDA FoodData Central API.
Free API key: https://fdc.nal.usda.gov/api-guide.html
"""

import requests
import json
import time

# Get your free key at https://fdc.nal.usda.gov/api-guide.html
# DEMO_KEY works but has lower rate limits
API_KEY = "DEMO_KEY"
BASE_URL = "https://api.nal.usda.gov/fdc/v1"

VEGETABLES = [
    "spinach", "tomato", "carrot", "broccoli", "potato", "onion",
    "garlic", "cabbage", "cauliflower", "lettuce", "cucumber", "kale",
    "sweet potato", "beetroot", "celery", "asparagus", "peas", "corn",
    "eggplant", "zucchini", "pumpkin", "radish", "turnip", "leek",
    "artichoke", "fennel", "parsnip", "okra", "chard", "arugula",
    "watercress", "kohlrabi", "celeriac", "endive", "escarole",
    "collard greens", "mustard greens", "bok choy", "napa cabbage",
    "butternut squash", "acorn squash", "spaghetti squash",
    "snap peas", "edamame", "lotus root", "daikon radish",
    "taro", "cassava", "yam", "jicama"
]

def scrape():
    results = []
    failed = []

    for veg in VEGETABLES:
        try:
            resp = requests.get(
                f"{BASE_URL}/foods/search",
                params={"query": veg + " raw", "api_key": API_KEY, "pageSize": 1},
                timeout=10
            )
            resp.raise_for_status()
            foods = resp.json().get("foods", [])

            if not foods:
                print(f"  x  {veg} — no results")
                failed.append(veg)
                continue

            food = foods[0]
            nutrients = {
                n["nutrientName"]: round(float(n["value"]), 2)
                for n in food.get("foodNutrients", [])
                if n.get("value") is not None
            }

            results.append({
                "vegetable": veg,
                "fdc_id": food.get("fdcId"),
                "description": food.get("description", ""),
                "nutrients": nutrients
            })
            print(f"  ✓  {veg}")
            time.sleep(0.3)  # Be polite to the API

        except Exception as e:
            print(f"  x  {veg} — {e}")
            failed.append(veg)

    with open("raw_nutrition.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Saved {len(results)} vegetables to raw_nutrition.json")
    if failed:
        print(f"⚠️  Failed: {', '.join(failed)}")

if __name__ == "__main__":
    print("Scraping USDA nutrition data...\n")
    scrape()
