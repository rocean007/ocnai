"""
convert_chemicals.py
Converts raw PubChem chemical JSON + manual expert Q&A into training pairs (JSONL).
The manual Q&A below is the most important part — do not skip adding to it.
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

# === MANUAL Q&A — Write as many as you can. These teach the model the most. ===
MANUAL_QA = [
    ("What is glyphosate and is it safe to use near vegetables?",
     "Glyphosate is a broad-spectrum systemic herbicide that kills all plants by blocking the "
     "shikimate pathway. It must NOT be applied to growing vegetable crops.\n"
     "Safe uses: Apply to land before planting to clear weeds. Wait 7–14 days before sowing.\n"
     "Safety: IARC classifies it as 'probably carcinogenic to humans' (Group 2A). "
     "Always wear PPE: gloves, goggles, mask. Avoid skin contact and inhalation.\n"
     "Organic alternatives: Vinegar-based herbicides, flame weeding, hand weeding, mulching."),

    ("What is the pre-harvest interval (PHI) and why does it matter?",
     "The pre-harvest interval (PHI) is the minimum number of days that must pass between the "
     "last pesticide application and harvesting the crop.\n"
     "Why it matters: Pesticides need time to break down to safe residue levels. "
     "Harvesting before PHI results in illegal residue levels that are unsafe to eat.\n"
     "Examples:\n"
     "- Malathion on lettuce: 7 days PHI\n"
     "- Spinosad on tomatoes: 1 day PHI\n"
     "- Copper sulfate on potatoes: 7 days PHI\n"
     "Always read the product label for the specific crop PHI. Never guess."),

    ("What is neem oil and how do I use it correctly on vegetables?",
     "Neem oil is a natural insecticide, fungicide, and miticide extracted from neem tree seeds "
     "(Azadirachta indica). Active ingredient: Azadirachtin.\n"
     "Controls: Aphids, whitefly, spider mites, thrips, powdery mildew, fungal diseases.\n"
     "How to use:\n"
     "1. Mix: 2 tablespoons (30ml) neem oil + 1 teaspoon dish soap per 1 litre warm water\n"
     "2. Shake well before each spray\n"
     "3. Apply in the evening or early morning — avoid midday heat and bee activity\n"
     "4. Spray all leaf surfaces including undersides\n"
     "5. Reapply every 7–14 days or after rain\n"
     "Approved for organic farming. Safe for most beneficial insects if applied carefully."),

    ("What is copper sulfate used for in farming?",
     "Copper sulfate (CuSO4) is a broad-spectrum fungicide and bactericide used in agriculture.\n"
     "Used to treat:\n"
     "- Late blight (Phytophthora infestans) on tomatoes and potatoes\n"
     "- Downy mildew on cucumbers, grapes, lettuce\n"
     "- Bacterial diseases like bacterial canker\n"
     "- As a component of Bordeaux mixture (copper sulfate + hydrated lime)\n"
     "Application: Mix at 0.5–1% solution. Apply preventatively before disease appears.\n"
     "Caution: Approved for organic farming but overuse causes copper accumulation in soil, "
     "which becomes toxic to earthworms and soil biology. Rotate with other fungicides."),

    ("What is spinosad and which pests does it control?",
     "Spinosad is a biological insecticide derived from the soil bacterium "
     "Saccharopolyspora spinosa. It affects the insect nervous system.\n"
     "Controls: Caterpillars (budworms, armyworms), thrips, leafminers, Colorado potato beetle, "
     "fly larvae, and some beetles.\n"
     "Can be used on: Tomatoes, peppers, brassicas, leafy greens, cucumbers, squash.\n"
     "PHI: 1 day for most vegetables (check label for specific crop).\n"
     "Organic approval: Yes — approved by most organic certification bodies.\n"
     "Resistance management: Rotate with other insecticide classes to prevent resistance.\n"
     "Bee safety: Apply in evening when bees are not active. Avoid spraying open flowers."),

    ("What chemicals are banned for use on food crops?",
     "Major chemicals banned or restricted for use on food crops include:\n"
     "- DDT: Globally banned (Stockholm Convention 2001) — persists in environment and food chain\n"
     "- Chlordane: Banned — highly persistent organochlorine\n"
     "- Aldrin / Dieldrin: Banned — extremely toxic to wildlife\n"
     "- Parathion: Banned in most countries — extremely toxic to humans\n"
     "- Endosulfan: Banned in 80+ countries — toxic to aquatic life and humans\n"
     "- Methyl bromide: Phase-out ongoing (ozone depleting)\n"
     "Always check your country's national pesticide registration list before use. "
     "In Nepal: Check the Department of Agriculture Pesticide Registration List."),

    ("How do I calculate the correct pesticide dilution rate?",
     "Dilution formula: Amount of pesticide = Recommended rate (ml/L) × Volume of water (L)\n\n"
     "Example 1: Rate is 2ml/L, you need 10 litres of spray\n"
     "→ 2ml × 10L = 20ml of pesticide in 10L water\n\n"
     "Example 2: Rate is 1:500 dilution, you need 5 litres\n"
     "→ 5000ml ÷ 500 = 10ml of pesticide in 5L water\n\n"
     "Important rules:\n"
     "- Never exceed the label rate — it does not work better and increases toxicity risk\n"
     "- Use a calibrated measuring cylinder, not guesswork\n"
     "- Always add pesticide to water, not water to pesticide\n"
     "- Wear gloves, goggles, and mask during mixing and application"),

    ("What PPE should I wear when applying pesticides?",
     "Personal Protective Equipment (PPE) for pesticide application:\n"
     "Minimum PPE for most pesticides:\n"
     "- Gloves: Chemical-resistant nitrile or rubber gloves (NOT fabric)\n"
     "- Eye protection: Safety goggles or face shield\n"
     "- Clothing: Long sleeves, long trousers, closed shoes\n"
     "- Respiratory: At minimum a dust mask; for concentrated chemicals use a full respirator\n\n"
     "After spraying:\n"
     "- Wash hands and face immediately\n"
     "- Wash clothing separately from household laundry\n"
     "- Do not eat, drink, or smoke during or after until washed\n\n"
     "Always read the label — it specifies the exact PPE required for each product."),

    ("What is mancozeb used for and is it safe?",
     "Mancozeb is a broad-spectrum contact fungicide in the dithiocarbamate family.\n"
     "Controls: Early blight, late blight, downy mildew, anthracnose, leaf spots, "
     "and many other fungal diseases on vegetables.\n"
     "Use on: Potatoes, tomatoes, onions, peppers, brassicas.\n"
     "PHI: Typically 5–10 days (check label for specific crop).\n"
     "Safety: Classified as a potential endocrine disruptor. Not approved for organic farming.\n"
     "Resistance: Rotate with fungicides of different modes of action to prevent resistance.\n"
     "Do not apply in high temperatures or to water-stressed plants — risk of phytotoxicity."),

    ("What is integrated pest management (IPM)?",
     "Integrated Pest Management (IPM) is an approach that uses multiple pest control strategies "
     "to minimise pesticide use while keeping pest damage below economic thresholds.\n"
     "IPM steps:\n"
     "1. Prevention: Crop rotation, resistant varieties, clean tools, healthy soil\n"
     "2. Monitoring: Regular scouting — check plants weekly for pests and diseases\n"
     "3. Action threshold: Only apply pesticide when pest levels justify it\n"
     "4. Biological control: Encourage natural enemies (ladybirds, parasitic wasps)\n"
     "5. Physical controls: Traps, barriers, row covers\n"
     "6. Chemical control: Last resort — start with least toxic option (neem, pyrethrin)\n"
     "IPM reduces resistance, protects pollinators, and reduces residues on food."),
]

def convert():
    pairs = []

    # Add all manual Q&A
    for q, a in MANUAL_QA:
        pairs.append(make_pair(q, a))

    # Add PubChem scraped data
    if os.path.exists("raw_chemicals.json"):
        with open("raw_chemicals.json") as f:
            data = json.load(f)

        for item in data:
            chem = item["chemical"].capitalize()
            props = item.get("properties", {})
            desc  = item.get("description", "")

            mw      = props.get("molecular_weight", "")
            formula = props.get("molecular_formula", "")
            iupac   = props.get("iupac_name", "")

            if desc:
                pairs.append(make_pair(
                    f"What is {chem} used for in agriculture?",
                    f"{chem}: {desc}\n"
                    f"{'Molecular formula: ' + formula + '. ' if formula else ''}"
                    f"{'Molecular weight: ' + str(mw) + ' g/mol. ' if mw else ''}"
                    f"Always follow label instructions and wear appropriate PPE."
                ))

            if formula or mw:
                pairs.append(make_pair(
                    f"What are the chemical properties of {chem}?",
                    f"{chem} chemical properties: "
                    f"{'Molecular formula: ' + formula + '. ' if formula else ''}"
                    f"{'Molecular weight: ' + str(mw) + ' g/mol. ' if mw else ''}"
                    f"{'IUPAC name: ' + iupac + '.' if iupac else ''}\n"
                    f"Always store agricultural chemicals in original containers, "
                    f"away from food, children, and water sources."
                ))
    else:
        print("raw_chemicals.json not found — using manual data only")

    with open("train_chemicals.jsonl", "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"✅ Created {len(pairs)} chemical training pairs → train_chemicals.jsonl")
    return pairs

if __name__ == "__main__":
    convert()
