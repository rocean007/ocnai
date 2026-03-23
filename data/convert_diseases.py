"""
convert_diseases.py
Converts raw Wikipedia disease data + manual expert Q&A into training pairs (JSONL).
"""

import json
import os

SYSTEM = (
    "You are VeggieBot, a highly knowledgeable expert on vegetables, plant farming, "
    "agricultural chemicals, and plant diseases. Give specific, accurate, practical answers."
)

def make_pair(question, answer):
    return {"messages": [
        {"role": "system",    "content": SYSTEM},
        {"role": "user",      "content": question},
        {"role": "assistant", "content": answer}
    ]}

MANUAL_QA = [
    ("How do I identify and treat early blight on tomatoes?",
     "Early blight (Alternaria solani) identification:\n"
     "- Dark brown spots with concentric rings (target-board pattern) on lower leaves first\n"
     "- Yellow halo surrounds each spot\n"
     "- Spots enlarge and leaves turn yellow, then brown, then drop\n"
     "- Stems may show dark cankers near the soil line\n\n"
     "Treatment:\n"
     "1. Remove and destroy all infected leaves immediately — do not compost\n"
     "2. Apply copper-based fungicide or chlorothalonil every 7–10 days\n"
     "3. Improve air circulation by pruning lower leaves and suckers\n"
     "4. Stake plants to keep foliage off the ground\n"
     "5. Water at the base — avoid wetting leaves\n"
     "Prevention: Crop rotation (3+ years), resistant varieties, mulch to prevent soil splash."),

    ("What is late blight and how serious is it?",
     "Late blight (Phytophthora infestans) is one of the most destructive crop diseases — "
     "it caused the Irish Potato Famine of 1840s.\n\n"
     "Identification:\n"
     "- Dark, water-soaked lesions on leaves and stems that turn brown rapidly\n"
     "- White fuzzy fungal growth on undersides of leaves in humid conditions\n"
     "- Brown rot penetrates fruit (tomatoes) and tubers (potatoes)\n"
     "- Spreads extremely fast in cool (10–20°C), wet, humid weather\n\n"
     "Treatment:\n"
     "- No cure once plant is heavily infected\n"
     "- Preventative spraying with Bordeaux mixture or metalaxyl-based fungicide\n"
     "- Remove and burn infected plants — do not compost\n"
     "- Choose resistant varieties (e.g., Sarpo Mira potato)\n"
     "- Monitor weather — risk is highest after rain followed by warm humid nights"),

    ("How do I treat powdery mildew on cucumbers and squash?",
     "Powdery mildew identification: White powdery coating on upper leaf surfaces. "
     "Leaves eventually yellow and die. Common on cucumbers, squash, pumpkins, peas.\n\n"
     "Causes: Warm days + cool nights + poor air circulation + dry soil with humid air.\n\n"
     "Organic treatments:\n"
     "- Neem oil spray: 2 tbsp per litre water — spray all leaf surfaces\n"
     "- Baking soda: 1 tsp baking soda + 1 tsp dish soap per litre — weekly spray\n"
     "- Potassium bicarbonate: More effective than baking soda\n"
     "- Milk spray: 40% milk, 60% water — folk remedy with some evidence\n\n"
     "Chemical treatments:\n"
     "- Sulfur-based fungicide, myclobutanil, trifloxystrobin\n\n"
     "Prevention: Plant resistant varieties, space plants well, avoid overhead watering."),

    ("How do I treat root rot in vegetables?",
     "Root rot identification:\n"
     "- Plant wilts despite soil being wet\n"
     "- Lower leaves yellow and drop\n"
     "- Roots are dark brown/black and mushy (healthy roots are white/cream and firm)\n"
     "- Caused by Pythium, Phytophthora, or Fusarium fungi in waterlogged soil\n\n"
     "Treatment:\n"
     "1. Immediately improve drainage — loosen soil or move to raised bed\n"
     "2. Remove plant, cut off all rotten roots with sterilised scissors\n"
     "3. Dust roots with Trichoderma-based biological fungicide or phosphonite\n"
     "4. Repot in fresh, well-draining mix with perlite\n"
     "5. Water only when top 2–3cm of soil is dry\n\n"
     "Prevention: Never overwater. Ensure pots have drainage. Add perlite to heavy soils."),

    ("What is clubroot and how do I manage it?",
     "Clubroot (Plasmodiophora brassicae) affects all brassica family vegetables:\n"
     "cabbage, broccoli, cauliflower, kale, Brussels sprouts, turnips, radishes, pak choi.\n\n"
     "Identification:\n"
     "- Swollen, distorted club-shaped roots\n"
     "- Plants wilt in hot weather, recover at night\n"
     "- Stunted growth, yellow leaves\n"
     "- Spores persist in soil for 20+ years\n\n"
     "Management (no chemical cure):\n"
     "1. Raise soil pH above 7.2 with agricultural lime — spores need acidic soil\n"
     "2. Remove and destroy ALL infected plants and roots — do not compost\n"
     "3. Strict 5–7 year crop rotation for brassicas\n"
     "4. Only buy certified disease-free transplants\n"
     "5. Disinfect boots, tools, and equipment between plots"),

    ("How do I control aphids organically?",
     "Aphid identification: Small soft-bodied insects (green, black, white, or orange) "
     "clustering on new shoots and leaf undersides. Sticky honeydew residue on leaves below.\n\n"
     "Organic control methods:\n"
     "1. Physical: Strong water jet to dislodge (repeat daily for a week)\n"
     "2. Remove by hand — wear gloves and squeeze clusters\n"
     "3. Biological: Encourage ladybirds, lacewings, parasitic wasps (plant flowers nearby)\n"
     "4. Insecticidal soap spray: 1 tbsp dish soap per litre water — spray directly on aphids\n"
     "5. Neem oil spray: 2 tbsp per litre — coats and suffocates soft-bodied insects\n"
     "6. Yellow sticky traps: Catches winged adults\n\n"
     "Chemical option: Pyrethrin-based spray (organic), or acetamiprid/imidacloprid (not on flowers).\n"
     "Monitor weekly — early action is far easier than controlling a large infestation."),

    ("What is Fusarium wilt and which vegetables does it affect?",
     "Fusarium wilt is caused by soil-borne fungi: Fusarium oxysporum (multiple strains).\n\n"
     "Affects: Tomatoes, peppers, cucumbers, melons, watermelons, basil, and many others.\n"
     "Each strain is host-specific (tomato strain only affects tomatoes, etc.)\n\n"
     "Identification:\n"
     "- Yellowing starts on one side of the plant or on lower leaves\n"
     "- Cut the stem near the base — look for brown discolouration inside the vascular tissue\n"
     "- Progressive wilting even with adequate water\n"
     "- Eventually the entire plant collapses\n\n"
     "Management (no fungicide cure):\n"
     "1. Remove and destroy infected plants immediately\n"
     "2. Soil solarisation: Cover soil with clear plastic for 4–6 weeks in summer\n"
     "3. Plant resistant varieties (marked 'F' or 'FF' on seed packet)\n"
     "4. 4+ year crop rotation\n"
     "5. Add Trichoderma biological inoculant to planting holes"),

    ("How do I identify and treat bacterial wilt in cucumbers?",
     "Bacterial wilt (Erwinia tracheiphila) identification:\n"
     "- Sudden wilting of one or more vines, starting on young leaves\n"
     "- Cut a wilted stem and touch the two cut ends together then pull apart slowly "
     "— if thin threads form, this confirms bacterial wilt\n"
     "- No yellowing — wilting is rapid\n"
     "- Spread by cucumber beetles feeding on infected plants\n\n"
     "Treatment:\n"
     "- No cure once infected — remove and destroy plant immediately\n"
     "- Control cucumber beetles (the carrier) using row covers on young plants\n"
     "- Pyrethrin spray or kaolin clay to deter beetles\n"
     "- Grow resistant cucumber varieties\n\n"
     "Prevention: Control cucumber beetles from day one. Use floating row covers until flowering."),

    ("How do I treat downy mildew on leafy vegetables?",
     "Downy mildew identification:\n"
     "- Yellow or pale green patches on upper leaf surface\n"
     "- Grey/purple fuzzy growth on corresponding underside of leaf\n"
     "- Distinct from powdery mildew — growth is always on the UNDERSIDE\n"
     "- Favoured by cool, wet, humid conditions\n"
     "- Affects: Lettuce, spinach, basil, brassicas, onions, peas\n\n"
     "Treatment:\n"
     "- Remove infected leaves immediately\n"
     "- Copper-based fungicide: Bordeaux mixture or copper hydroxide\n"
     "- Improve air circulation by thinning plants\n"
     "- Avoid overhead watering — drip or base watering only\n"
     "- Chemical: Metalaxyl (mefenoxam) for severe cases\n\n"
     "Prevention: Resistant varieties, proper spacing, morning watering so leaves dry by evening."),

    ("What causes blossom end rot in tomatoes and peppers?",
     "Blossom end rot is NOT a disease — it is a physiological disorder caused by "
     "calcium deficiency in the developing fruit.\n\n"
     "Appearance: Dark, sunken, leathery patch at the blossom end (bottom) of the fruit.\n\n"
     "Root cause: Calcium is present in the soil but cannot be absorbed due to:\n"
     "- Irregular or inconsistent watering\n"
     "- Overwatering causing waterlogged roots\n"
     "- High nitrogen fertiliser causing too-rapid growth\n"
     "- Root damage from cultivation near plants\n\n"
     "Fixes:\n"
     "1. Water consistently — never let soil dry out completely\n"
     "2. Mulch heavily to maintain even soil moisture\n"
     "3. Apply calcium-containing foliar spray (calcium nitrate) to leaves\n"
     "4. Avoid high-nitrogen fertilisers once flowering starts\n"
     "5. Affected fruits cannot be cured but remove them — plant will produce more"),
]

def convert():
    pairs = []

    for q, a in MANUAL_QA:
        pairs.append(make_pair(q, a))

    if os.path.exists("raw_diseases.json"):
        with open("raw_diseases.json") as f:
            data = json.load(f)

        for item in data:
            disease = item["disease"]
            summary = item.get("summary", "")[:2000]

            if len(summary) < 100:
                continue

            pairs += [
                make_pair(
                    f"What is {disease} in plants?",
                    summary
                ),
                make_pair(
                    f"Tell me about {disease}",
                    summary
                ),
            ]
    else:
        print("raw_diseases.json not found — using manual data only")

    with open("train_diseases.jsonl", "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"✅ Created {len(pairs)} disease training pairs → train_diseases.jsonl")
    return pairs

if __name__ == "__main__":
    convert()
