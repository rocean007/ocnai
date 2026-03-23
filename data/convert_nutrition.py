"""
convert_nutrition.py
Converts raw USDA nutrition JSON into Q&A training pairs (JSONL format).
"""

import json
import os

SYSTEM = (
    "You are VeggieBot, a highly knowledgeable expert on vegetables, plant farming, "
    "agricultural chemicals, and plant diseases. Give specific, accurate, practical answers."
)

def get_nutrient(nutrients, *keys):
    """Try multiple key names to find a nutrient value."""
    for key in keys:
        for k, v in nutrients.items():
            if key.lower() in k.lower():
                try:
                    return str(round(float(v), 1))
                except:
                    return str(v)
    return "data unavailable"

def make_pair(question, answer):
    return {"messages": [
        {"role": "system",    "content": SYSTEM},
        {"role": "user",      "content": question},
        {"role": "assistant", "content": answer}
    ]}

def convert():
    if not os.path.exists("raw_nutrition.json"):
        print("raw_nutrition.json not found. Run scrape_nutrition.py first.")
        return []

    with open("raw_nutrition.json") as f:
        data = json.load(f)

    pairs = []

    for item in data:
        veg = item["vegetable"].capitalize()
        n   = item.get("nutrients", {})
        desc = item.get("description", "")

        calories  = get_nutrient(n, "Energy")
        protein   = get_nutrient(n, "Protein")
        fat       = get_nutrient(n, "Total lipid", "Fat")
        carbs     = get_nutrient(n, "Carbohydrate")
        fiber     = get_nutrient(n, "Fiber", "Dietary fiber")
        sugars    = get_nutrient(n, "Sugars")
        vitamin_c = get_nutrient(n, "Vitamin C")
        vitamin_k = get_nutrient(n, "Vitamin K")
        vitamin_a = get_nutrient(n, "Vitamin A")
        vitamin_b6= get_nutrient(n, "Vitamin B-6", "Vitamin B6")
        folate    = get_nutrient(n, "Folate")
        potassium = get_nutrient(n, "Potassium")
        calcium   = get_nutrient(n, "Calcium")
        iron      = get_nutrient(n, "Iron")
        magnesium = get_nutrient(n, "Magnesium")
        sodium    = get_nutrient(n, "Sodium")
        zinc      = get_nutrient(n, "Zinc")
        water     = get_nutrient(n, "Water")

        pairs += [
            make_pair(
                f"What are the complete nutrition facts for {veg}?",
                f"{veg} nutrition per 100g raw:\n"
                f"Calories: {calories} kcal | Protein: {protein}g | Fat: {fat}g | "
                f"Carbohydrates: {carbs}g | Fiber: {fiber}g | Sugars: {sugars}g | Water: {water}g\n"
                f"Vitamins: C {vitamin_c}mg | K {vitamin_k}mcg | A {vitamin_a}mcg | B6 {vitamin_b6}mg | Folate {folate}mcg\n"
                f"Minerals: Potassium {potassium}mg | Calcium {calcium}mg | Iron {iron}mg | "
                f"Magnesium {magnesium}mg | Sodium {sodium}mg | Zinc {zinc}mg"
            ),
            make_pair(
                f"How many calories are in {veg}?",
                f"{veg} contains {calories} calories per 100g raw. "
                f"It provides {protein}g of protein, {fat}g of fat, and {carbs}g of carbohydrates. "
                f"With {fiber}g of dietary fiber, it is {'high' if float(fiber or 0) > 3 else 'a moderate source'} in fiber."
            ),
            make_pair(
                f"What vitamins does {veg} contain?",
                f"{veg} per 100g raw contains: "
                f"Vitamin C {vitamin_c}mg (daily value ~90mg), "
                f"Vitamin K {vitamin_k}mcg (daily value ~120mcg), "
                f"Vitamin A {vitamin_a}mcg, Vitamin B6 {vitamin_b6}mg, "
                f"and Folate {folate}mcg. "
                f"{'It is an excellent source of Vitamin C.' if float(vitamin_c or 0) > 50 else ''}"
                f"{'It is very high in Vitamin K.' if float(vitamin_k or 0) > 100 else ''}"
            ),
            make_pair(
                f"What minerals does {veg} have?",
                f"{veg} per 100g raw contains: "
                f"Potassium {potassium}mg, Calcium {calcium}mg, Iron {iron}mg, "
                f"Magnesium {magnesium}mg, Sodium {sodium}mg, and Zinc {zinc}mg. "
                f"{'High potassium supports heart and muscle function.' if float(potassium or 0) > 300 else ''}"
                f"{'Good iron content supports red blood cell production.' if float(iron or 0) > 1.5 else ''}"
            ),
            make_pair(
                f"Is {veg} good for you?",
                f"Yes, {veg} is nutritious. Per 100g: {calories} kcal, {protein}g protein, {fiber}g fiber. "
                f"Key nutrients: Vitamin C ({vitamin_c}mg), Potassium ({potassium}mg), Iron ({iron}mg). "
                f"{'Low calorie and high fiber makes it good for weight management.' if float(calories or 999) < 50 else ''}"
                f"{'High in antioxidant Vitamin C.' if float(vitamin_c or 0) > 30 else ''}"
            ),
            make_pair(
                f"How much protein does {veg} have?",
                f"{veg} contains {protein}g of protein per 100g raw. "
                f"{'This is relatively high for a vegetable.' if float(protein or 0) > 3 else 'Like most vegetables, it is not a major protein source.'} "
                f"Pair it with legumes, eggs, or meat for a complete protein meal."
            ),
        ]

    with open("train_nutrition.jsonl", "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"✅ Created {len(pairs)} nutrition training pairs → train_nutrition.jsonl")
    return pairs

if __name__ == "__main__":
    convert()
