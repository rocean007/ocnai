"""
build_dataset.py
────────────────
Builds bilingual (English + Nepali) vegetable training data.
Run locally, then upload train.jsonl + val.jsonl to Colab/Kaggle.

    pip install requests
    python build_dataset.py
"""

import json, random, os, requests, time

SYSTEM = (
    "You are SagjiBot, a vegetable expert assistant. "
    "You know about vegetable nutrition, farming, agricultural chemicals, and plant diseases. "
    "If the user writes in Nepali, reply in Nepali. If in English, reply in English. "
    "Give accurate, practical, specific answers."
)

def pair(q, a):
    return {"messages":[
        {"role":"system",    "content":SYSTEM},
        {"role":"user",      "content":q},
        {"role":"assistant", "content":a},
    ]}

# ── English Q&A ───────────────────────────────────────────────────────────────

EN = [
    ("What vitamins are in spinach?",
     "Spinach is extremely high in Vitamin K (460% daily value per 100g), plus Vitamin A, C, and folate. It also contains iron, calcium, and magnesium. One cup raw spinach has only 7 calories."),
    ("How many calories does broccoli have?",
     "Broccoli has about 34 calories per 100g raw. It provides 2.8g protein, 2.6g fiber, and is one of the richest vegetable sources of Vitamin C."),
    ("What vegetables are high in iron?",
     "High-iron vegetables: spinach (2.7mg/100g), lentils (3.3mg), peas (1.5mg), kale (1.5mg), beetroot (0.8mg). Eat with Vitamin C foods to improve iron absorption."),
    ("Is sweet potato healthier than regular potato?",
     "Sweet potatoes are higher in Vitamin A, Vitamin C, and fiber. Regular potatoes have slightly more potassium. Sweet potatoes also have a lower glycaemic index. Both are nutritious — sweet potatoes offer more micronutrients overall."),
    ("What vegetables are good for diabetes?",
     "Best for diabetes: spinach, kale, broccoli, cauliflower, cucumber, tomato, bitter gourd (karela), capsicum. Low-carb, high-fiber, and low glycaemic index. Limit high-starch vegetables like potato and corn."),
    ("What vitamins does carrot contain?",
     "Carrots are rich in beta-carotene (Vitamin A, 835mcg/100g), Vitamin K1, Vitamin C, Vitamin B6, and folate. Beta-carotene supports eye health, immune function, and skin."),
    ("What is the healthiest vegetable?",
     "Top contenders: Spinach (K, A, C, iron), Kale (K, A, C, calcium), Broccoli (C, fiber, sulforaphane), Sweet potato (A, fiber), Garlic (allicin — antibacterial). Eat a variety of colours for best nutrition."),
    ("How do I grow tomatoes from seed?",
     "1. Sow 6-8 weeks before last frost, 6mm deep.\n2. Keep at 21-24°C until germination (5-10 days).\n3. Move to bright light immediately after sprouting.\n4. Transplant outdoors after last frost in full sun, 60cm apart.\n5. Stake indeterminate varieties."),
    ("When should I plant garlic in Nepal?",
     "Plant garlic October-November in Nepal. Hills above 1500m: September-October. Terai: November. Plant cloves 5cm deep, 15cm apart, pointy end up. Harvest when 50% of leaves turn yellow — usually May-June."),
    ("How do I grow spinach in hot weather?",
     "Spinach bolts above 24°C. Tips: grow under 50% shade cloth, sow September-October or February-March, use Malabar spinach (Pui saag) which thrives in heat, water consistently, harvest quickly before plants mature."),
    ("What is the best soil for vegetables?",
     "Well-draining loam at pH 6.0-7.0, rich in organic matter. Improve any soil by adding 5-10cm compost dug in to 30cm. Sandy soil: add compost. Clay soil: add compost and perlite. Test pH and adjust: lime raises pH, sulfur lowers it."),
    ("How often should I water vegetables?",
     "Vegetables need 2.5cm of water per week. Seedlings: water daily. Established plants: deep watering 2-3 times per week beats daily shallow watering. Fruiting crops need consistent moisture. Water at the base in the morning — wet leaves invite disease."),
    ("What is crop rotation?",
     "Crop rotation means not growing the same plant family in the same bed each year. Prevents soil-borne disease and balances nutrients. Basic 4-year rotation: Year 1 Brassicas → Year 2 Legumes (fix nitrogen) → Year 3 Roots → Year 4 Fruiting crops."),
    ("How do I grow bitter gourd (karela) in Nepal?",
     "Sow February-April (summer) or June-July (autumn). Optimal temp: 25-35°C. Soil: well-drained, pH 6.0-6.7. Plant 2 seeds per hole, 1.5m between holes. Needs a trellis. Harvest at 12-16 weeks when still green."),
    ("What is neem oil and how do I use it?",
     "Neem oil is a natural pesticide from neem tree seeds. Controls aphids, whitefly, spider mites, and powdery mildew. Mix: 2 tbsp + 1 tsp dish soap per litre warm water. Spray in the evening, covering all leaf surfaces including undersides. Reapply every 7-14 days. Approved for organic farming."),
    ("What is the pre-harvest interval (PHI)?",
     "PHI is the minimum days between last pesticide application and harvest. Allows residues to break down to safe levels. Example: Malathion on lettuce = 7 days PHI. Harvesting before PHI is illegal and unsafe. Always read the product label."),
    ("What is IPM?",
     "Integrated Pest Management (IPM) minimises pesticide use. Steps: 1. Prevention (rotation, resistant varieties). 2. Monitoring (weekly checks). 3. Biological control (encourage ladybirds, wasps). 4. Physical control (traps, barriers). 5. Chemical — last resort, least toxic first."),
    ("What pesticides can I use on tomatoes in Nepal?",
     "For tomatoes: Mancozeb or copper oxychloride (blight), Spinosad (organic, caterpillars), Neem oil (organic, broad spectrum), Imidacloprid (not on open flowers). Always check PHI and follow Department of Agriculture registrations."),
    ("How do I treat early blight on tomatoes?",
     "Early blight (Alternaria solani) shows as dark spots with concentric rings on lower leaves. Treatment: remove and destroy infected leaves, apply copper fungicide every 7-10 days, stake plants off soil, water at base only. Prevention: 3-year rotation, resistant varieties."),
    ("How do I treat powdery mildew?",
     "White powdery coating on leaves — common on cucumbers, squash, peas. Organic: neem oil, baking soda (1 tsp + dish soap per litre), potassium bicarbonate. Chemical: sulfur fungicide. Prevention: resistant varieties, good spacing, no overhead watering."),
    ("What is late blight?",
     "Late blight (Phytophthora infestans) caused the Irish Famine. Dark water-soaked patches on leaves, white mould underneath, rapid spread in cool wet conditions. No cure once established. Prevent with Bordeaux mixture, remove and burn infected plants, grow resistant varieties."),
    ("How do I control aphids organically?",
     "1. Blast off with strong water jet daily. 2. Spray insecticidal soap (1 tbsp dish soap per litre). 3. Neem oil spray. 4. Plant flowers to attract ladybirds and lacewings. 5. Yellow sticky traps. Monitor weekly — early action is far easier."),
    ("What causes blossom end rot?",
     "Blossom end rot is NOT a disease — it's calcium deficiency caused by irregular watering. Fix: water consistently, mulch heavily, apply calcium nitrate foliar spray. Avoid high-nitrogen fertiliser after flowering. New fruits will be fine once watering is steady."),
    ("How do I make compost?",
     "Layer green material (vegetable scraps, grass — nitrogen) with brown material (dry leaves, straw — carbon) at 3:1 ratio. Add a soil layer for microbes. Keep moist like a wrung-out sponge. Turn weekly. Ready in 3-6 months when dark and earthy-smelling."),
    ("What is NPK in fertiliser?",
     "NPK = Nitrogen (N), Phosphorus (P), Potassium (K). N: leafy growth. P: roots and fruiting. K: overall health and disease resistance. A bag labelled 10-10-10 has 10% each. For fruiting crops after flowering, use lower N and higher P and K."),
    ("What is Fusarium wilt?",
     "Fusarium wilt (Fusarium oxysporum) causes yellowing on one side of the plant and brown discolouration inside the stem. Affects tomatoes, peppers, cucumbers. No fungicide cure. Management: remove and destroy plants, soil solarisation, plant resistant varieties (marked F on packet), 4+ year rotation."),
]

# ── Nepali Q&A ────────────────────────────────────────────────────────────────

NE = [
    ("पालकमा कुन भिटामिन हुन्छ?",
     "पालकमा भिटामिन K (दैनिक आवश्यकताको ४६०%), भिटामिन A, C र फोलेट प्रचुर मात्रामा हुन्छ। यसमा फलाम, क्याल्सियम र म्याग्नेसियम पनि राम्रो मात्रामा पाइन्छ। एक कप कच्चो पालकमा मात्र ७ क्यालोरी हुन्छ।"),
    ("गाजरमा कुन पोषक तत्व हुन्छन्?",
     "गाजरमा बिटा-क्यारोटिन (भिटामिन A को स्रोत) प्रचुर मात्रामा हुन्छ — प्रति १०० ग्राममा ८३५ माइक्रोग्राम। यसमा भिटामिन K1, C, B6 र फोलेट पनि हुन्छ। बिटा-क्यारोटिनले आँखाको स्वास्थ्य सुधार गर्छ।"),
    ("ब्रोकोलीमा कति क्यालोरी हुन्छ?",
     "कच्चो ब्रोकोलीमा प्रति १०० ग्राममा लगभग ३४ क्यालोरी हुन्छ। यसमा २.८ ग्राम प्रोटिन, २.६ ग्राम फाइबर र भिटामिन C को उच्च मात्रा हुन्छ।"),
    ("फलामको मात्रा बढी हुने तरकारीहरू कुन हुन्?",
     "फलाम धेरै हुने तरकारी: पालक (२.७ मिलिग्राम/१०० ग्राम), मसुरको दाल (३.३ मिलिग्राम), केराउ (१.५), काले (१.५), चुकन्दर (०.८)। भिटामिन C भएका खानेकुरासँग खादा फलामको अवशोषण बढ्छ।"),
    ("मधुमेह भएकाहरूले कुन तरकारी खानु राम्रो?",
     "मधुमेह भएकाहरूका लागि उत्तम: पालक, काले, काउली, ब्रोकोली, खिरा, गोलभेडा, करेला र क्याप्सिकम। यी तरकारीहरूमा कार्बोहाइड्रेट कम र फाइबर धेरै हुन्छ। आलु र मकैजस्ता स्टार्चयुक्त तरकारी कम मात्रामा खानुस्।"),
    ("नेपालमा गोलभेडा कहिले रोप्नु पर्छ?",
     "तराई: अक्टोबर-नोभेम्बर र फेब्रुअरी-मार्च। पहाड (१५००-२००० मिटर): मार्च-अप्रिल। बीउ पहिले नर्सरीमा लगाउनुस् र ४-५ हप्तापछि सार्नुस्। पूर्ण घाम पाउने ठाउँमा ६०-७५ सेमीको दूरीमा रोप्नुस्।"),
    ("नेपालमा लसुन कहिले रोप्ने?",
     "नेपालमा लसुन रोप्ने उत्तम समय अक्टोबर-नोभेम्बर हो। पहाड: सेप्टेम्बर-अक्टोबर। तराई: नोभेम्बर। लुनाको टुप्पो माथि राखेर ५ सेमी गहिरो र १५ सेमीको दूरीमा लगाउनुस्। जब ५०% पात पहेँलो हुन्छ (मे-जुन) तब उखाल्नुस्।"),
    ("करेला कसरी फलाउने?",
     "समय: फेब्रुअरी-अप्रिल (ग्रीष्म) वा जुन-जुलाई (वर्षे)। तापमान: २५-३५°C। राम्रो निकास हुने माटो, pH ६.०-६.७। दुई बीउ एउटै खाडलमा, खाडलहरूबीच १.५ मिटर। लहरा चढ्नका लागि जाली चाहिन्छ। बीउ रोपेको १२-१६ हप्तामा हरियो छँदै टिप्नुस्।"),
    ("पालकलाई गर्मीमा कसरी उमार्ने?",
     "पालक २४°C भन्दा बढी तापमानमा फुल्न थाल्छ। उपाय: ५०% छायाँ जाली प्रयोग गर्नुस्, सेप्टेम्बर-अक्टोबर वा फेब्रुअरी-मार्चमा रोप्नुस्, मालाबार पालक (पुई साग) प्रयोग गर्नुस्, नियमित पानी दिनुस्, चाँडो टिप्नुस्।"),
    ("माटोको pH कति हुनु पर्छ?",
     "अधिकांश तरकारीका लागि pH ६.०-७.० उत्तम। pH बढाउन: चुन (lime) थप्नुस्। pH घटाउन: सल्फर थप्नुस्। pH परीक्षण किट कृषि सेवा केन्द्रमा पाइन्छ। अनुमान नगर्नुस् — परीक्षण गरेर मात्र माटो सुधार गर्नुस्।"),
    ("नीम तेल के हो र कसरी प्रयोग गर्ने?",
     "नीम तेल नीम रूखको बीउबाट निकालिएको प्राकृतिक कीटनाशक हो। लाही, सेतो माछा, माकुरा माइट र ढुसी रोग नियन्त्रण गर्छ। मिश्रण: १ लिटर गर्म पानीमा २ चम्चा नीम तेल + थोरै साबुन। बेलुकी छर्कनुस्। ७-१४ दिनमा दोहोर्याउनुस्। जैविक खेतीमा मान्यताप्राप्त।"),
    ("पूर्व-कटनी अन्तराल (PHI) के हो?",
     "PHI भनेको अन्तिम कीटनाशक छर्केपछि बाली काट्नु अघि कुर्नुपर्ने न्यूनतम दिनको संख्या हो। PHI भन्दा पहिले काट्नु गैरकानूनी र स्वास्थ्यका लागि खतरनाक छ। सधैँ उत्पादनको लेबल पढ्नुस् — PHI बाली र रसायन अनुसार फरक हुन्छ।"),
    ("गोलभेडामा अगेती झुलसा रोग कसरी ठीक गर्ने?",
     "अगेती झुलसा (Alternaria solani): तल्लो पातमा गोलाकार घेरा भएको खैरो थोप्लो। उपचार: रोगी पात तुरुन्त हटाउनुस्, तामा (copper) ढुसीनाशक ७-१० दिनमा छर्कनुस्, बोट ठड्याउनुस्, फेदमा मात्र पानी दिनुस्। रोकथाम: बाली फेरबदल, रोगप्रतिरोधी जात।"),
    ("पिठ्युँ ढुसी रोग कसरी रोक्ने?",
     "पातमा सेतो धुलोजस्तो ढकमक — स्क्वास, काँक्रो, केराउमा सामान्य। जैविक उपचार: नीम तेल, बेकिंग सोडा (१ चम्चा + साबुन प्रति लिटर), पोटासियम बाइकार्बोनेट। रासायनिक: सल्फर ढुसीनाशक। रोकथाम: रोगप्रतिरोधी जात, राम्रो दूरी, माथिबाट पानी नदिनुस्।"),
    ("लाही कीरा कसरी नियन्त्रण गर्ने?",
     "१. कडा पानी फोहोर्याएर कीरा झार्नुस् — दैनिक दोहोर्याउनुस्। २. साबुन पानी (१ चम्चा साबुन प्रति लिटर) सोझै छर्कनुस्। ३. नीम तेल — २ चम्चा प्रति लिटर। ४. फूलहरू रोपेर लेडीबर्ड आकर्षित गर्नुस्। हप्ताहप्ता जाँच गर्नुस् — सुरुमै नियन्त्रण गर्नु सजिलो।"),
    ("ढाडेपाते रोग के हो?",
     "ढाडेपाते रोग (Late blight) गोलभेडा र आलुमा लाग्ने गम्भीर रोग हो। पातमा गाढा पानी सोसिएजस्तो दाग, तलतर्फ सेतो ढुसी, द्रुत फैलावट। व्यवस्थापन: बोर्डेउ मिश्रण वा metalaxyl छर्कनुस् (रोग लाग्नु अघि)। रोगी बोट जलाउनुस्। रोगप्रतिरोधी जात प्रयोग गर्नुस्।"),
    ("नेपालमा कुन तरकारी व्यावसायिक खेतीका लागि राम्रो?",
     "तराई: गोलभेडा, काँक्रो, करेला, बैगन, काउली। पहाड: बन्दा, ब्रोकोली, काले, आलु। बजार नजिक: पालक, धनिया, मेथी। जिल्ला कृषि कार्यालयसँग सम्पर्क गरेर स्थानीय बजार माग बुझ्नुस्।"),
    ("जैविक मल कसरी बनाउने?",
     "हरियो सामग्री (तरकारीको बोक्रा, घाँस) र खैरो सामग्री (सुकेको पात, पराल) ३:१ अनुपातमा मिलाउनुस्। माटोको एक तह थप्नुस्। ओसिलो राख्नुस् र हप्तामा एकपटक पल्टाउनुस्। ३-६ महिनामा मैलो र माटोजस्तो गन्ध आएपछि प्रयोग गर्न मिल्छ।"),
    ("बाली फेरबदल (crop rotation) किन जरुरी छ?",
     "एउटै बालीको परिवार एउटै ठाउँमा हरेक वर्ष नलगाउनु — यसलाई बाली फेरबदल भनिन्छ। माटोजन्य रोग र कीरा जम्मा हुन दिँदैन, माटोको पोषक तत्व सन्तुलन राख्छ। सरल ४ वर्षे चक्र: ब्रासिका → दलहन → जरे बाली → फल दिने बाली।"),
    ("तरकारी खेतीमा पानी कसरी बचत गर्ने?",
     "१. मल्चिङ (पराल, सुकेको पात) — ५०% पानी बचत। २. थोपा सिँचाइ — जरामा सोझै पानी। ३. बिहान पानी दिनुस् — वाष्पीकरण कम। ४. गहिरो र कम पटक पानी दिनुस्। ५. पानी जम्मा हुने खाल्डो बनाउनुस्। थोपा सिँचाइले ४०-६०% पानी बचत गर्छ।"),
]

# ── Optional: scrape USDA for more nutrition data ─────────────────────────────

VEGS = ["spinach","tomato","carrot","broccoli","potato","onion","garlic",
        "cabbage","cauliflower","lettuce","cucumber","kale","sweet potato",
        "beetroot","celery","asparagus","peas","eggplant","zucchini","pumpkin"]

def scrape_usda():
    if os.path.exists("raw_nutrition.json"):
        with open("raw_nutrition.json") as f: return json.load(f)
    print("Scraping USDA (optional)...")
    out = []
    for v in VEGS:
        try:
            r = requests.get("https://api.nal.usda.gov/fdc/v1/foods/search",
                params={"query":v+" raw","api_key":"DEMO_KEY","pageSize":1}, timeout=8)
            foods = r.json().get("foods",[])
            if foods:
                f = foods[0]
                out.append({"vegetable":v,
                    "nutrients":{n["nutrientName"]:n["value"] for n in f.get("foodNutrients",[])}})
                print(f"  ✓ {v}")
            time.sleep(0.4)
        except: pass
    with open("raw_nutrition.json","w") as f: json.dump(out,f,indent=2)
    return out

def usda_pairs():
    data = scrape_usda()
    out  = []
    for item in data:
        veg = item["vegetable"].capitalize()
        n   = item.get("nutrients",{})
        def g(k):
            for key,v in n.items():
                if k.lower() in key.lower():
                    try: return str(round(float(v),1))
                    except: pass
            return "N/A"
        out.append(pair(
            f"What are the nutrition facts for {veg}?",
            f"{veg} per 100g raw: {g('Energy')} kcal, {g('Protein')}g protein, {g('Fiber')}g fiber. "
            f"Vitamins: C {g('Vitamin C')}mg, K {g('Vitamin K')}mcg. "
            f"Minerals: Iron {g('Iron')}mg, Calcium {g('Calcium')}mg."
        ))
    return out

# ── Build ──────────────────────────────────────────────────────────────────────

def build():
    all_pairs = ([pair(q,a) for q,a in EN] +
                 [pair(q,a) for q,a in NE] +
                 usda_pairs())

    random.seed(42); random.shuffle(all_pairs)
    split = int(len(all_pairs)*.9)
    train, val = all_pairs[:split], all_pairs[split:]

    with open("train.jsonl","w",encoding="utf-8") as f:
        for p in train: f.write(json.dumps(p,ensure_ascii=False)+"\n")
    with open("val.jsonl","w",encoding="utf-8") as f:
        for p in val:   f.write(json.dumps(p,ensure_ascii=False)+"\n")

    print(f"✅ Done")
    print(f"   Total    : {len(all_pairs)}")
    print(f"   Train    : {len(train)}  → train.jsonl")
    print(f"   Val      : {len(val)}  → val.jsonl")
    print(f"   English  : {len(EN)}")
    print(f"   Nepali   : {len(NE)}")
    print(f"   USDA     : {len(usda_pairs())}")

if __name__ == "__main__": build()
