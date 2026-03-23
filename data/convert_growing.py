"""
convert_growing.py
Converts raw OpenFarm growing JSON into Q&A training pairs (JSONL format).
"""

import json
import os

SYSTEM = (
    "You are VeggieBot, a highly knowledgeable expert on vegetables, plant farming, "
    "agricultural chemicals, and plant diseases. Give specific, accurate, practical answers."
)

def safe(val, fallback="not specified"):
    return str(val).strip() if val else fallback

def make_pair(question, answer):
    return {"messages": [
        {"role": "system",    "content": SYSTEM},
        {"role": "user",      "content": question},
        {"role": "assistant", "content": answer}
    ]}

# Hand-written growing Q&A for high quality training signal
MANUAL_QA = [
    ("What is the difference between determinate and indeterminate tomatoes?",
     "Determinate tomatoes grow to a fixed height (60–90cm), produce all fruit at once, then stop. "
     "Good for canning and small spaces. Examples: Roma, Celebrity.\n"
     "Indeterminate tomatoes keep growing and producing all season until frost. "
     "Need staking/caging. Can reach 1.5–2m+. Examples: Cherry tomatoes, Beefsteak, Brandywine.\n"
     "For continuous harvest, choose indeterminate. For one big harvest, choose determinate."),

    ("How do I improve clay soil for growing vegetables?",
     "To improve clay soil:\n"
     "1. Add organic matter: compost, aged manure (5–10cm layer, dig in 20cm deep)\n"
     "2. Add coarse sand or perlite to improve drainage\n"
     "3. Never dig wet clay — it compacts\n"
     "4. Grow green manures (clover, mustard) and dig them in\n"
     "5. Add gypite (calcium sulphate) — breaks up clay without changing pH\n"
     "Do this annually over 2–3 years for lasting improvement."),

    ("What does NPK mean in fertilizer?",
     "NPK stands for Nitrogen (N), Phosphorus (P), and Potassium (K) — the three main plant nutrients.\n"
     "N (Nitrogen): Promotes leafy green growth. Important for brassicas and leafy greens.\n"
     "P (Phosphorus): Promotes root and flower/fruit development. Important at planting.\n"
     "K (Potassium): Promotes overall plant health, disease resistance, and fruit quality.\n"
     "A bag labelled 10-10-10 contains 10% each of N, P, K. "
     "For fruiting vegetables (tomatoes, peppers), use lower N and higher P+K after flowering."),

    ("What is companion planting and which vegetables grow well together?",
     "Companion planting is growing plants near each other for mutual benefit.\n"
     "Good combinations:\n"
     "- Tomatoes + Basil: Basil repels aphids and whitefly\n"
     "- Carrots + Onions: Carrot fly and onion fly repel each other\n"
     "- Beans + Corn + Squash (Three Sisters): Beans fix nitrogen, corn provides support, squash shades weeds\n"
     "- Cabbage + Dill: Dill attracts beneficial insects that eat cabbage pests\n"
     "Bad combinations: Onions + Beans (stunts bean growth), Fennel (allelopathic, inhibits most vegetables)"),

    ("How do I start seeds indoors for transplanting?",
     "Starting seeds indoors:\n"
     "1. Timing: Start 6–8 weeks before last frost date\n"
     "2. Containers: Use seed trays or small pots with drainage holes\n"
     "3. Mix: Use seed-starting mix (not garden soil — too heavy)\n"
     "4. Depth: Sow at 2x the seed diameter depth\n"
     "5. Moisture: Keep moist but not waterlogged. Cover with plastic wrap until germination\n"
     "6. Light: Move to bright light (14–16h/day) immediately after sprouting\n"
     "7. Hardening off: 1–2 weeks before transplanting, move seedlings outside for increasing hours daily"),

    ("What is crop rotation and why is it important?",
     "Crop rotation means not growing the same plant family in the same spot year after year.\n"
     "Why: Each crop family depletes different nutrients and harbours different soil-borne pests/diseases.\n"
     "Basic 4-year rotation:\n"
     "Year 1: Brassicas (cabbage, broccoli, kale)\n"
     "Year 2: Legumes (peas, beans) — they fix nitrogen\n"
     "Year 3: Roots (carrots, beetroot, parsnips)\n"
     "Year 4: Nightshades/others (tomatoes, potatoes, peppers)\n"
     "Prevents: clubroot (brassicas), potato blight, fusarium wilt."),

    ("How do I know when to harvest vegetables?",
     "Harvest signs by type:\n"
     "Tomatoes: Full colour, slightly soft to touch, comes off vine easily\n"
     "Carrots: Shoulders visible above soil, usually 70–80 days after sowing\n"
     "Lettuce: Before it bolts (sends up flower stalk) — leaves should be full but not bitter\n"
     "Courgettes/Zucchini: Harvest small (15–20cm) for best flavour — daily checking needed\n"
     "Potatoes: Maincrop after foliage dies back. New potatoes when flowers open\n"
     "Peas: Pods plump but before they go yellow — taste-test daily\n"
     "Garlic: When 50% of leaves have turned yellow — usually July"),

    ("How often should I water my vegetable garden?",
     "Watering frequency depends on soil, weather, and crop stage:\n"
     "General rule: 2.5cm of water per week (rain + irrigation)\n"
     "Seedlings: Keep consistently moist — may need daily watering\n"
     "Established plants: Deep, infrequent watering is better than shallow daily watering\n"
     "Fruiting stage: Most critical — water stress causes blossom drop and blossom end rot\n"
     "Best time: Early morning — reduces fungal disease risk\n"
     "Signs of underwatering: Wilting, dry soil, curling leaves\n"
     "Signs of overwatering: Yellow lower leaves, mushy stems, root rot"),
]

def convert():
    pairs = []

    # Add manual Q&A
    for q, a in MANUAL_QA:
        pairs.append(make_pair(q, a))

    # Add scraped OpenFarm data
    if os.path.exists("raw_growing.json"):
        with open("raw_growing.json") as f:
            data = json.load(f)

        for item in data:
            veg = item["vegetable"].capitalize()
            desc    = safe(item.get("description"))
            sun     = safe(item.get("sun_requirements"), "full sun")
            sow     = safe(item.get("sowing_method"), "direct sow")
            spread  = safe(item.get("spread"), "30")
            spacing = safe(item.get("row_spacing"), "30")
            height  = safe(item.get("height"), "varies")
            gdd     = safe(item.get("growing_degree_days"), "varies by climate")
            min_t   = safe(item.get("minimum_temperature"), "")
            max_t   = safe(item.get("maximum_temperature"), "")
            ph_min  = safe(item.get("ph_minimum"), "")
            ph_max  = safe(item.get("ph_maximum"), "")

            ph_str = f"Preferred soil pH: {ph_min}–{ph_max}. " if ph_min and ph_max else ""
            temp_str = f"Temperature range: {min_t}°C – {max_t}°C. " if min_t and max_t else ""

            pairs += [
                make_pair(
                    f"How do I grow {veg}?",
                    f"Growing {veg}:\n"
                    f"{desc}\n"
                    f"Sun: {sun}. Sowing: {sow}.\n"
                    f"Plant spacing: {spread}cm apart, {spacing}cm between rows.\n"
                    f"Expected height: {height}cm. Days to maturity: {gdd}.\n"
                    f"{ph_str}{temp_str}"
                ),
                make_pair(
                    f"How much sunlight does {veg} need?",
                    f"{veg} requires {sun}. "
                    f"{'Full sun means at least 6–8 hours of direct sunlight daily.' if 'full' in sun.lower() else ''}"
                    f"{'Partial shade means 3–6 hours of direct sun.' if 'partial' in sun.lower() else ''}"
                    f" Plant in a position that receives consistent light throughout the growing season."
                ),
                make_pair(
                    f"How far apart should I plant {veg}?",
                    f"Plant {veg} {spread}cm apart within rows and {spacing}cm between rows. "
                    f"Good spacing allows air circulation, reduces disease risk, and ensures "
                    f"plants do not compete for water and nutrients."
                ),
                make_pair(
                    f"What soil pH does {veg} prefer?",
                    f"{veg} grows best in soil with pH {ph_min}–{ph_max}. " if ph_min and ph_max else
                    f"{veg} generally prefers slightly acidic to neutral soil (pH 6.0–7.0). "
                    f"Test your soil with a pH kit and adjust: add lime to raise pH, sulfur to lower it."
                ),
            ]
    else:
        print("raw_growing.json not found — using manual data only")

    with open("train_growing.jsonl", "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"✅ Created {len(pairs)} growing training pairs → train_growing.jsonl")
    return pairs

if __name__ == "__main__":
    convert()
