"""
build_complete_dataset.py
──────────────────────────
Builds a comprehensive SagjiBot training dataset (1000+ entries)
- Consumer-focused: buying, storing, cooking, nutrition, health
- Nepal market specific: varieties, seasons, growing regions, traditional uses
- Fertilizers & chemicals: what they are, health impacts, safety
- Bilingual: English by default, Nepali when user speaks Nepali
- No pricing information (as requested)

Run:
    pip install requests
    python build_complete_dataset.py

Output:
    train.jsonl (90%)
    val.jsonl (10%)
"""

import json
import random
import os
import requests
import time
from typing import List, Dict

# ─── System prompt ────────────────────────────────────────────────────────────
SYSTEM = """You are SagjiBot, an expert vegetable assistant focused on Nepal and South Asia.
You help consumers with:
- Buying, storing, cooking, and eating vegetables
- Nutritional value and health benefits/risks
- Understanding vegetable varieties and seasons in Nepal
- Growing vegetables at home or on farms
- Understanding fertilizers and chemicals used on vegetables
- Food safety, pesticide testing, and traditional Nepali vegetable preparations

Language rule: Reply in English by default. Only reply in Nepali if the user writes in Nepali OR explicitly asks you to reply in Nepali. Never mix languages in a single reply."""

def create_pair(question: str, answer: str) -> Dict:
    """Create a training pair with system prompt"""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ]
    }

# ============================================================================
# SECTION 1: NEPALI MARKET & VEGETABLE VARIETIES (Consumer Focus)
# ============================================================================

VARIETY_QA = [
    create_pair(
        "What are the different types of tomatoes found in Nepali markets?",
        """Nepali markets have several distinct tomato varieties:

**1. Terai Commercial Tomato:**
- Large, round, uniformly red
- Grown extensively in plains with chemical fertilizers
- Available year-round, firm texture
- Best for: Cooking, sauces, long storage

**2. Pahadi (Hill) Tomato:**
- Smaller, irregular shape with green shoulders
- Grown traditionally at 1000-2000m altitude
- Superior flavour, more aromatic
- Season: March-July, October-December
- Best for: Raw eating, salads, traditional curries

**3. Cherry Tomato:**
- Small, round or oval (15-25g each)
- Sweet flavour, often greenhouse-grown in Nepal
- Available year-round but expensive
- Best for: Salads, snacking, garnishing

**4. Greenhouse/Polyhouse Tomato:**
- Grown in controlled conditions off-season
- Consistent quality, good shelf life
- Available year-round
- Best for: When fresh hill tomatoes are out of season

**5. Heirloom/Village Varieties:**
- Preserved seeds from local farms
- More flavourful, often organic
- Season: Peak harvest times only
- Found at local farmers markets

**How to choose:**
- For best taste: Pahadi tomatoes in season
- For reliability: Greenhouse or Terai commercial
- For salads: Cherry or small pahadi tomatoes
- For sauces: Ripe Terai or greenhouse tomatoes

**Storage:** Never refrigerate unripe tomatoes. Store at room temperature stem-side down. Once ripe, refrigerate and use within 3 days."""),
    
    create_pair(
        "What are the different varieties of cauliflower grown in Nepal?",
        """Nepal grows several cauliflower varieties suited to different regions:

**1. Snowball Type (Winter Cauliflower):**
- Pure white, tight compact heads
- Grown in Terai (Oct-Dec) and Hills (Dec-Feb)
- Best quality from mid-November to January
- Mild sweet flavour, tender texture

**2. Early Maturing Varieties:**
- Slightly loose heads, creamy white
- Grown September-November and February-March
- Faster harvest (60-70 days)
- Good for autumn and spring seasons

**3. Purple Cauliflower:**
- High in anthocyanin antioxidants
- Grown in cooler hill regions
- Becomes green when cooked
- Season: December-February

**4. Romanesco Broccoflower:**
- Lime green, fractal spiral pattern
- Grown in organic farms near Kathmandu
- Nutty flavour, firmer texture
- Limited availability, premium price

**5. Local Hill Varieties:**
- Smaller heads, loose structure
- Adapted to higher altitudes
- Better flavour, often organic
- Grown traditionally without chemicals

**How to select:**
- Tight, compact florets without gaps
- Pure white or creamy white (no yellowing)
- Fresh green leaves attached
- Heavy for its size
- No black spots or mould

**Seasonal availability:**
- Peak quality: November-January
- Good quality: October, February
- Off-season: May-September (imported or greenhouse)"""),

    create_pair(
        "What are the different types of leafy greens in Nepali cuisine?",
        """Nepali cuisine features diverse leafy greens (saag):

**1. Palak (Spinach):**
- Smooth, dark green leaves
- Available year-round, best in winter
- High iron, calcium, Vitamin K
- Uses: Palak paneer, saag, soups

**2. Rayo Saag (Mustard Greens):**
- Large, slightly ruffled leaves
- Peppery, slightly bitter taste
- Season: October-March (peak winter)
- High in Vitamin K, A, and calcium
- Uses: Traditional rayo saag with rice, pickles

**3. Methi Saag (Fenugreek Leaves):**
- Small trifoliate leaves
- Distinctive bitter-sweet flavour
- Season: October-April
- Blood sugar regulating properties
- Uses: Methi saag, methi thepla, dal methi

**4. Suntala Saag (Taro Leaves - Karkalo):**
- Large heart-shaped leaves
- Must be cooked thoroughly (contains oxalates)
- Season: June-September (monsoon)
- Traditional in Newari cuisine
- Uses: Karkalo ko tarkari, karkalo ra alu

**5. Khole Saag (Cabbage Leaves):**
- Outer leaves of cabbage plant
- Slightly sweet, tender when young
- Season: November-March
- Uses: Sautéed with spices, in soups

**6. Pui Saag (Malabar Spinach):**
- Thick, fleshy leaves, heat-tolerant
- Season: May-October (summer)
- Uses: Grows when regular spinach isn't available

**7. Beth Ko Saag (Amaranth Greens):**
- Red or green leaves
- High iron content
- Season: March-September
- Traditional: Luto saag in Newari culture

**8. Gundruk (Fermented Greens):**
- Dried fermented leafy greens
- Year-round availability (preserved)
- Probiotic, tangy flavour
- Uses: Gundruk ko jhol (soup), achar

**Storage:**
- Buy fresh, use within 2-3 days
- Store unwashed in refrigerator
- For longer storage: blanch and freeze or make gundruk"""),

    create_pair(
        "What are the different types of Nepali gourds (lauka, karela, etc.)?",
        """Nepali cuisine features numerous gourd varieties:

**1. Lauka (Bottle Gourd):**
- Long, cylindrical, pale green
- Mild flavour, absorbs spices well
- Season: May-October (summer)
- High water content, cooling effect
- Uses: Lauki ko tarkari, lauki dal, raita

**2. Karela (Bitter Gourd):**
- Dark green, bumpy skin
- Distinct bitter taste
- Season: June-September (monsoon)
- Blood sugar regulating properties
- How to reduce bitterness: Salt and squeeze before cooking
- Uses: Stuffed karela, fried karela chips

**3. Parwal (Pointed Gourd):**
- Small, oval, green with white stripes
- Mild sweet taste
- Season: July-October
- Rich in Vitamin A and C
- Uses: Parwal ki sabzi, stuffed parwal

**4. Tori (Ridge Gourd):**
- Long, ridged, light green
- Spongy texture, absorbs flavours
- Season: June-September
- High fiber, low calorie
- Uses: Tori ko tarkari, chutney

**5. Pidalu (Colocasia Root):**
- Small round corms, hairy skin
- Nutty flavour when cooked
- Season: July-October (monsoon)
- Must be cooked thoroughly (oxalates)
- Uses: Pidalu ko tarkari, fry with spices

**6. Farsi (Pumpkin):**
- Orange flesh, green or orange skin
- Sweet, earthy flavour
- Season: August-December
- High in beta-carotene, Vitamin A
- Uses: Farsi ko tarkari, pumpkin soup, achar

**7. Kakro (Cucumber):**
- Green, cylindrical, with seeds
- Refreshing, high water content
- Season: March-September
- Uses: Salad, raita, achar

**8. Ghiraula (Snake Gourd):**
- Long, twisted, pale green
- Mild, slightly sweet
- Season: June-September
- Good for digestion
- Uses: Ghiraula ko tarkari, dal

**Buying tips:**
- Choose firm gourds without soft spots
- Smaller gourds are more tender
- Avoid yellowing or wrinkled skin
- Karela: bright green with tight bumps
- Pumpkin: heavy for size, hollow sound when tapped""")
]

# ============================================================================
# SECTION 2: CONSUMER BUYING & STORAGE GUIDE
# ============================================================================

BUYING_STORAGE_QA = [
    create_pair(
        "How do I select the freshest vegetables at a Nepali market?",
        """Here's how to pick the freshest vegetables in Nepali markets:

**General Tips:**
- Shop in the morning (5-9 AM) for freshest produce
- Feel the weight - fresh vegetables are heavy for their size
- Check for bruises, soft spots, or mould
- Smell - fresh vegetables have a clean, earthy aroma
- Look at the stem ends - should be moist, not dried out

**Specific Vegetables:**

**Tomatoes:**
- Firm but yields slightly to gentle pressure
- Deep red colour all over
- Fruity smell at stem end
- No cracks or soft spots

**Leafy Greens (Palak, Rayo, Methi):**
- Crisp, bright green leaves
- No yellowing or wilting
- Avoid if slimy or foul-smelling
- Roots attached indicates very fresh

**Cauliflower & Cabbage:**
- Tight, compact heads
- Pure white or creamy white (no yellowing)
- Fresh green outer leaves
- Heavy for its size

**Root Vegetables (Carrot, Radish, Potato):**
- Firm, smooth skin
- No sprouting (potatoes) or cracks (carrots)
- If tops attached, they should be fresh
- Avoid wrinkled or soft specimens

**Gourds (Lauka, Karela, Cucumber):**
- Firm along entire length
- Bright, even colour
- No soft ends or wrinkled skin
- Cucumber: dark green, not yellow

**Eggplant (Bhanta):**
- Shiny, smooth skin
- Firm with slight spring when pressed
- Fresh green stem cap
- Avoid large, spongy ones (more seeds)

**Okra (Bhindi):**
- Bright green, no brown spots
- Smaller pods (under 10cm) are more tender
- Snaps cleanly - means fresh
- Avoid large, fibrous pods

**Peas (Kerao):**
- Pods should be bright green, firm
- Plump peas visible through pod
- Pods that rattle are too old

**Traditional Wisdom:**
- Ask vendor: "Aaja ko taza ho?" (Is it fresh today?)
- Buy from vendors who display small quantities (high turnover)
- At Kalimati, look for vendors with 'pahadi tarkari' - often fresher
- Seasonal vegetables are naturally fresher than off-season imports"""),
    
    create_pair(
        "How should I store different vegetables at home in Nepal's climate?",
        """Proper storage extends vegetable life in Nepal's climate:

**Refrigerator Storage (0-4°C):**

**Leafy Greens** (Palak, Rayo, Methi):
- Do NOT wash before storing
- Wrap in newspaper or cloth (not plastic)
- Store in vegetable drawer
- Use within 2-3 days
- If wilted: soak in ice water 15 mins

**Broccoli, Cauliflower:**
- Store unwashed in perforated plastic bag
- Place stem-down to prevent moisture accumulation
- Use within 5-7 days

**Carrots, Radish:**
- Remove green tops (they draw moisture)
- Store in plastic bag with small holes
- Lasts 2-3 weeks

**Beans, Peas:**
- Store unwashed in perforated bag
- Use within 5-7 days

**Capsicum:**
- Store unwashed in crisper drawer
- Use within 1-2 weeks

**Room Temperature Storage (Cool, Dark Place):**

**Tomatoes:**
- Never refrigerate unripe (loses flavour)
- Store stem-side down on counter
- Use within 5-7 days
- Once ripe, refrigerate and use within 3 days

**Potatoes:**
- Store in paper bag or cardboard box
- Keep in dark (light causes greening)
- Cool spot (12-18°C ideal)
- Keep away from onions (they cause sprouting)
- Lasts 2-5 weeks

**Onions, Garlic:**
- Store in mesh bag or open basket
- Dry, well-ventilated area
- Lasts 1-3 months
- Do NOT refrigerate

**Pumpkin, Winter Squash:**
- Store whole in cool, dark place
- Lasts 1-3 months
- Once cut: refrigerate, use within 5 days

**Gourds (Lauka, Cucumber):**
- Cucumber: refrigerate (use within 1 week)
- Bottle gourd: room temp for 3-5 days, then refrigerate

**Monsoon Storage Tips:**
- Increased humidity causes faster spoilage
- Check stored vegetables every 2 days
- Remove any rotting pieces immediately
- Use silica gel packets in storage areas
- Consider freezing surplus vegetables

**Freezing Guide:**
- Blanch vegetables 2-3 minutes before freezing
- Suitable for: peas, beans, spinach, cauliflower
- Lasts 8-12 months frozen
- Not suitable: cucumber, lettuce, tomatoes (texture changes)"""),
    
    create_pair(
        "What are the signs that a vegetable has gone bad and shouldn't be eaten?",
        """Learn to recognize when vegetables are unsafe to eat:

**Definitely Discard If:**

**Visible Mold:**
- Any fuzzy growth (white, green, black)
- Mold on soft vegetables penetrates deeper than visible
- Hard vegetables: cut 1-2 inches around mold if rest is firm

**Foul Odor:**
- Strong unpleasant smell (not normal vegetable aroma)
- Fermented, sour, or ammonia-like smell
- If it smells "off," trust your nose

**Slime:**
- Slippery, sticky surface (especially leafy greens)
- Indicates bacterial growth
- Do NOT eat, even if washed

**Liquid Leaking:**
- Soft vegetables oozing liquid
- Sign of internal decomposition

**Specific Vegetables:**

**Potatoes:**
- Green skin (solanine toxin) - cut away generously or discard if extensive
- Large sprouts (small sprouts can be removed)
- Soft, wrinkled, or shriveled
- Black or dark spots inside when cut

**Tomatoes:**
- Large soft spots with leaking liquid
- Mold at stem end spreading into fruit
- Completely mushy texture

**Leafy Greens:**
- Yellowed, wilted beyond recovery
- Slimy texture anywhere
- Black or brown spots covering leaves

**Onions:**
- Soft, mushy spots
- Black mould inside layers
- Strong unpleasant odor (not normal onion smell)

**Carrots:**
- Very limp, rubbery texture
- Black or soft spots
- Slimy surface

**Cauliflower:**
- Brown or black spots covering large areas
- Strong unpleasant smell
- Slimy, mushy florets

**Cucumbers:**
- Extremely soft, wrinkled
- Yellowed skin (overripe)
- Slippery, sticky surface

**Borderline Cases (May Be Salvageable):**

**Minor bruising:**
- Small brown spots - cut away generously

**Slight wilting:**
- Soak in ice water 15-30 minutes to revive
- If doesn't crisp, use cooked only

**Small sprouts:**
- Potatoes: remove sprouts and any green parts
- Onions: remove sprout, rest is fine

**General Rule:**
When in doubt, throw it out. Food poisoning from spoiled vegetables can cause severe gastrointestinal distress. Better to waste a vegetable than risk health.

**Prevention:**
- Buy only what you'll use within storage time
- Store properly (see storage guide)
- Check stored vegetables every few days
- Use older vegetables first (FIFO - first in, first out)""")
]

# ============================================================================
# SECTION 3: COOKING & PREPARATION TECHNIQUES
# ============================================================================

COOKING_TECHNIQUES_QA = [
    create_pair(
        "How do I reduce bitterness in bitter gourd (karela) before cooking?",
        """Karela (bitter gourd) bitterness can be reduced using these traditional methods:

**Method 1: Salt Treatment (Most Common)**
1. Slice karela thinly or cut into rounds
2. Sprinkle generously with salt (1 tsp per karela)
3. Let sit for 20-30 minutes
4. Squeeze out the released bitter water firmly
5. Rinse briefly under water
6. Proceed with cooking

**Method 2: Tamarind Soak**
1. Prepare tamarind water (soak tamarind in warm water)
2. Soak sliced karela in tamarind water for 30 minutes
3. Drain, squeeze lightly
4. Rinse, then cook

**Method 3: Yogurt Marinade**
1. Mix sliced karela with thick yogurt
2. Add turmeric, salt
3. Marinate 1-2 hours
4. Drain, then cook (yogurt adds flavour too)

**Method 4: Blanching**
1. Boil water with salt
2. Add karela pieces
3. Boil 2-3 minutes
4. Drain and rinse with cold water
5. Squeeze lightly before cooking

**Method 5: Remove Seeds and Pith**
- Most bitterness concentrates in seeds and white pith
- Scrape out seeds and pith thoroughly before cooking
- Reduces bitterness significantly

**Method 6: Combination Method (Most Effective)**
1. Scrape out seeds and pith
2. Slice thinly
3. Salt for 30 minutes, squeeze
4. Quick blanch 2 minutes
5. Rinse, then cook

**Cooking Methods that Reduce Bitterness:**
- Deep frying (bitter gourd chips)
- Cooking with potatoes (sweetness balances)
- Adding jaggery or sugar during cooking
- Cooking with tomatoes (acidity balances)
- Stuffed preparations (other ingredients dilute bitterness)

**Note:** Some bitterness should remain - it's part of karela's character and health benefits (blood sugar regulation). The goal is to make it palatable, not eliminate all bitterness.""")
]

# ============================================================================
# SECTION 4: NUTRITION & HEALTH BENEFITS (Nepali Vegetables)
# ============================================================================

NUTRITION_HEALTH_QA = [
    create_pair(
        "What are the health benefits of eating seasonal vegetables in Nepal?",
        """Eating seasonally in Nepal provides unique health benefits:

**Winter Vegetables (December-February):**

**Leafy Greens (Palak, Rayo, Methi):**
- Highest nutrient density in winter
- Maximum Vitamin K, A, C content
- Cold temperatures increase antioxidant production
- Excellent for immune system during cold season

**Cruciferous Vegetables (Cauliflower, Cabbage, Broccoli):**
- Sulforaphane content peaks in winter
- Supports liver detoxification
- Anti-cancer properties strongest
- High Vitamin C for winter immunity

**Root Vegetables (Carrot, Radish, Turnip):**
- Storage vegetables - nutrients preserved
- Highest beta-carotene in winter carrots
- Radish supports respiratory health in colds
- Warming energetics in Ayurveda

**Spring Vegetables (March-May):**

**Peas, Beans:**
- Fresh, sweet, high protein content
- Rich in B vitamins for energy
- Supports transition from winter to active season

**Early Tomatoes, Cucumbers:**
- Hydrating properties for warming weather
- Lycopene, antioxidants for sun protection
- Cooling effect on body

**Summer Vegetables (June-August):**

**Gourds (Lauka, Karela, Parwal):**
- High water content (90%+) for hydration
- Cooling properties for summer heat
- Karela: blood sugar regulation during high-carb monsoon meals
- Easy to digest during humid weather

**Monsoon Greens (Pui Saag, Taro Leaves):**
- Iron-rich for energy during rainy season
- Supports immunity during high infection period
- Traditional wisdom aligns with seasonal needs

**Autumn Vegetables (September-November):**

**Transition Crops:**
- Gradual shift from cooling to warming foods
- Nutrient-dense for winter preparation
- Harvest vegetables have peak mineral content

**Ayurvedic Perspective:**
- Winter: Root vegetables provide grounding energy
- Summer: Light, hydrating gourds balance heat
- Monsoon: Digestive-friendly vegetables for humid weather
- Each season's vegetables naturally support seasonal health needs

**Scientific Evidence:**
- Seasonal vegetables have 20-40% higher nutrient content
- Peak ripeness = maximum antioxidant levels
- Less storage time means fewer nutrient losses
- Local seasonal vegetables adapt to your climate needs"""),
    
    create_pair(
        "What are the best vegetables for managing diabetes (madhumeha) available in Nepal?",
        """Nepal has excellent vegetables for diabetes management:

**Top Vegetables for Blood Sugar Control:**

**1. Karela (Bitter Gourd) - The Gold Standard:**
- Contains charantin, polypeptide-P, and vicine
- Insulin-like compounds that lower blood glucose
- Traditional use: Fresh juice (30ml morning empty stomach)
- How to eat: Cooked in curries, fried chips, stuffed
- Effective for both type 1 and type 2 diabetes
- Caution: Can interact with diabetes medication

**2. Methi Saag & Seeds (Fenugreek):**
- Fenugreek seeds: 1 tsp soaked overnight reduces post-meal spikes
- Leaves: High fiber content (3g per 100g)
- Soluble fiber slows carbohydrate absorption
- Also lowers LDL cholesterol

**3. Okra (Bhindi):**
- Rich in soluble fiber (forms gel-like substance)
- Slows sugar absorption in intestines
- Traditional: Soak sliced okra in water overnight, drink water
- Low glycemic index (GI 20-30)

**4. Leafy Greens (Palak, Rayo, Pui Saag):**
- Very low carbohydrate content
- High magnesium (improves insulin sensitivity)
- High fiber (slows glucose absorption)
- Eat freely (1-2 cups cooked daily)

**5. Bitter Vegetables (General Principle):**
- Bitter flavours stimulate insulin secretion
- Include: bitter gourd, bitter leafy greens, turmeric

**6. Non-Starchy Vegetables (Eat Freely):**
- Cauliflower, cabbage, broccoli
- Cucumber, capsicum
- Mushrooms
- Radish, turnip

**7. Moderate Starchy Vegetables (Limit Portions):**
- Potato (1 small, with skin)
- Sweet potato (½ medium, rich in fiber)
- Corn (½ cup)
- Peas (½ cup)

**Practical Tips:**
- Eat vegetables first in your meal (slows overall glucose rise)
- Include fiber-rich vegetables with every meal
- Cook with minimal oil, add lemon juice
- Traditional Nepali: Dal + vegetables + small rice portion
- Avoid: Starchy vegetables fried in batter

**Meal Timing:**
- Distribute vegetables across all meals
- Dinner: Focus on non-starchy vegetables
- Avoid eating carbohydrates alone without vegetables

**Important:**
- Always consult doctor before making dietary changes
- Monitor blood sugar when adding new foods
- Karela juice can cause hypoglycemia with medication
- Balance is key - don't eliminate carbohydrates completely""")
]

# ============================================================================
# SECTION 5: FERTILIZERS & CHEMICALS (Consumer Safety)
# ============================================================================

CHEMICAL_SAFETY_QA = [
    create_pair(
        "How can I reduce pesticide residues from vegetables at home?",
        """Here are effective methods to reduce pesticide residues:

**Most Effective Method: Baking Soda Wash**
1. Mix 1 teaspoon baking soda in 2 cups water
2. Soak vegetables for 12-15 minutes
3. Scrub firm vegetables with brush
4. Rinse thoroughly under running water
5. Studies show removes 95%+ of surface pesticides

**Salt Water Method:**
1. Dissolve 2 tablespoons salt in 1 liter water
2. Soak for 10-15 minutes
3. Rinse well with running water
4. Effective for leafy greens and thin-skinned vegetables

**Vinegar Solution:**
1. Mix 1 part white vinegar with 3 parts water
2. Soak for 10 minutes
3. Rinse thoroughly
4. Good for berries and fruits (not for leafy greens - can wilt)

**Running Water Method (Minimum):**
1. Wash under running water for 30-60 seconds
2. Scrub with brush for firm vegetables
3. Removes 60-70% of surface residues
4. Better than soaking in still water

**Additional Techniques:**

**Peeling:**
- Removes nearly all surface residues
- For: cucumber, carrot, potato, eggplant
- Downside: loses some nutrients in skin

**Remove Outer Leaves:**
- Cabbage: discard outer 2-3 leaves
- Lettuce: discard outer leaves
- Most pesticides concentrate here

**Cooking:**
- Heat breaks down many pesticides
- Boiling can reduce residues 30-50%
- Steam rather than boil to retain nutrients

**Vegetable-Specific Methods:**

**Leafy Greens (Palak, Rayo, Methi):**
- Separate leaves, wash individually under running water
- Soak in salt water 10 minutes
- Rinse thoroughly twice
- Best to cook thoroughly

**Cauliflower, Broccoli:**
- Soak head-down in salt water 15 minutes
- Rinse under running water
- Cut into florets, rinse again

**Tomatoes:**
- Wash under running water while scrubbing
- Soak in baking soda solution if concerned
- Avoid washing before storage (only wash before eating)

**Cucumber, Capsicum:**
- Scrub vigorously with brush under running water
- Consider peeling if concerned about wax/residues
- Baking soda soak before scrubbing

**Root Vegetables (Carrot, Potato, Radish):**
- Scrub with brush under running water
- Peel to remove most residues
- If not peeling, soak then scrub

**Important Limitations:**
- Systemic pesticides (absorbed into plant tissue) cannot be washed off
- Pre-harvest interval violation is the main risk
- Organic certification ensures no synthetic pesticides used
- Buy from trusted sources, especially for leafy greens

**When to Be Most Vigilant:**
- Leafy greens (absorb most pesticides)
- Off-season vegetables (require more chemicals)
- Imported vegetables (unknown pesticide regimes)
- Very uniform, perfect-looking vegetables

**Best Prevention:**
- Buy from farmers you trust
- Choose seasonal vegetables
- Prefer pahadi (hill) vegetables
- Grow your own herbs and some vegetables
- Ask vendors directly about chemical use"""),
    
    create_pair(
        "What are systemic pesticides and why are they dangerous for consumers?",
        """Systemic pesticides are absorbed into plant tissues and cannot be washed off:

**What Makes Them Different:**

**Contact Pesticides:**
- Remain on plant surface
- Can be largely removed by washing
- Break down with sunlight, rain, time

**Systemic Pesticides:**
- Absorbed through roots, leaves, or stems
- Circulate throughout plant's vascular system
- Present in ALL plant tissues (leaves, stems, fruits)
- Cannot be washed, peeled, or cooked away completely
- Must break down chemically over time

**Common Systemic Pesticides in Nepal:**

**1. Imidacloprid (Neonicotinoid):**
- Most widely used systemic insecticide
- Applied to soil or as seed treatment
- Present throughout growing season
- PHI (Pre-Harvest Interval): 21 days for leafy greens
- Health concerns: Neurotoxic, possible endocrine disruptor

**2. Acephate (Organophosphate):**
- Systemic insecticide
- Banned in EU, still used in Nepal
- Breaks down to methamidophos (more toxic)
- PHI: 14-21 days

**3. Carbofuran:**
- Highly toxic systemic insecticide
- Restricted use but still available
- Banned on many crops
- Extremely toxic to birds and humans

**Why Dangerous for Consumers:**

**Health Risks:**
- Chronic exposure linked to neurological issues
- Endocrine disruption
- Potential carcinogenicity (some compounds)
- Developmental effects in children
- Cumulative exposure over time

**Detection Challenges:**
- No visual signs of contamination
- No taste or smell
- Cannot be removed by washing
- Only laboratory testing can detect

**High-Risk Vegetables for Systemics:**
- Leafy greens (absorb efficiently)
- Root vegetables (uptake from soil)
- Tomatoes, capsicum (grow throughout season)
- Cucurbits (high water uptake)

**Safer Vegetables (Lower Systemic Use):**
- Cabbage (outer leaves protect)
- Onions, garlic (layers, shorter growing season)
- Peas in pods (pod protects)
- Corn (husk protection)

**Protecting Yourself:**

**1. Know Your Source:**
- Buy from farmers who use integrated pest management
- Prefer organic certification
- Build relationships with trusted vendors
- Ask: "Ke systemik haalnu bhayo?" (Did you use systemic?)

**2. Time Your Purchases:**
- Avoid buying just after pesticide application
- Peak season vegetables use fewer chemicals
- Hill vegetables typically use fewer systemics

**3. Diversify Your Sources:**
- Don't rely on single supplier
- Mix market, organic, and home-grown sources
- Rotate vegetable types to reduce exposure

**4. Support Better Practices:**
- Buy organic when possible (supports market growth)
- Encourage vendors to disclose practices
- Community Supported Agriculture (CSA) provides transparency

**Regulatory Context in Nepal:**
- Pesticide Act exists but enforcement weak
- Many systemics banned elsewhere still available
- Consumer awareness is primary protection
- Nepal government working on improved testing

**Bottom Line:**
Systemic pesticides are the biggest chemical risk in vegetables because they can't be removed at home. Prevention through informed purchasing is the only reliable protection.""")
]

# ============================================================================
# SECTION 6: TRADITIONAL NEPALI VEGETABLE PREPARATIONS
# ============================================================================

TRADITIONAL_PREP_QA = [
    create_pair(
        "What is gundruk and how is it made?",
        """Gundruk is Nepal's traditional fermented leafy green vegetable:

**What is Gundruk?**
- Fermented and dried leafy greens
- National dish of Nepal
- Probiotic, tangy, slightly sour flavour
- Preserves vegetables for year-round use
- Made traditionally from mustard greens, radish leaves, or cauliflower leaves

**How Gundruk is Made:**

**Traditional Method:**
1. Harvest leafy greens (mustard greens, radish leaves, cauliflower leaves)
2. Wilt leaves in sun for 1-2 days
3. Chop leaves roughly
4. Pack tightly in earthen pot or plastic container
5. Press down firmly to remove air
6. Cover with heavy weight
7. Ferment in warm place for 5-7 days
8. Sour smell develops when ready
9. Remove and sun-dry completely (3-5 days)
10. Store in airtight container (lasts 6-12 months)

**Types of Gundruk:**
- **Rayo ko Gundruk:** Made from mustard greens (most common)
- **Mula ko Gundruk:** Made from radish leaves
- **Kauli ko Gundruk:** Made from cauliflower leaves
- **Suntala ko Gundruk:** Made from taro leaves (less common)

**Health Benefits:**
- Probiotic: Supports gut health
- Fermentation increases nutrient availability
- Rich in Vitamin K, iron, calcium
- Improves digestion
- Traditional remedy for digestive issues

**How to Use Gundruk:**
- **Gundruk ko Jhol (Soup):** Boil with tomatoes, ginger, garlic
- **Gundruk Achar:** Mixed with spices as pickle
- **Gundruk Saag:** Rehydrated and cooked as greens
- **With Dal:** Added to lentils for tangy flavour

**Cooking Gundruk:**
1. Rinse dried gundruk to remove dust
2. Soak in warm water 10-15 minutes to rehydrate
3. Drain (or reserve water for soup)
4. Sauté onions, garlic, tomatoes
5. Add gundruk, cook with spices
6. Add water for soup consistency
7. Serve hot with rice

**Nutritional Profile (per 100g rehydrated):**
- Calories: 45 kcal
- Protein: 3g
- Fiber: 5g
- Iron: 25% daily value
- Calcium: 15% daily value
- Vitamin A: 20% daily value

**Cultural Significance:**
- Preserves monsoon harvest for winter
- Every family has their own recipe
- Traditional in Newari, Gurung, Magar cuisines
- Often served with beaten rice (chiura) and soybeans
- Celebrated in Nepali literature and folk songs

**Making at Home:**
- Best made in autumn when greens are abundant
- Use wide-mouth glass jar or traditional earthen pot
- Ensure complete air exclusion for proper fermentation
- Dry thoroughly after fermentation to prevent mould
- Store in cool, dark place

**Safety Note:**
- Use clean equipment
- If foul smell (not sour) appears, discard
- Properly dried gundruk should be shelf-stable
- Soak before eating to remove excess salt""")
]

# ============================================================================
# SECTION 7: GROWING AT HOME (Nepal Context)
# ============================================================================

HOME_GROWING_QA = [
    create_pair(
        "What vegetables can I grow in pots on my balcony in Kathmandu?",
        """Balcony vegetable gardening in Kathmandu is very successful with these choices:

**Easiest Vegetables for Pots:**

**1. Coriander (Dhania):**
- Pot: 20cm deep, good drainage
- Soil: Garden soil + compost
- Season: September-March (peak), year-round with care
- Sowing: Seeds soaked overnight, sow densely
- Harvest: 3-4 weeks, cut outer leaves
- Care: Morning sun, water when topsoil dry

**2. Fenugreek (Methi):**
- Pot: Any depth (10cm sufficient)
- Soil: Any potting mix
- Season: September-April
- Sowing: Scatter seeds, cover lightly
- Harvest: 2-3 weeks (cut entire plant)
- Succession: Sow every 2 weeks for continuous supply

**3. Spinach (Palak):**
- Pot: 25cm deep minimum
- Soil: Rich, well-drained
- Season: September-March (avoid summer heat)
- Sowing: 1cm deep, thin to 10cm spacing
- Harvest: 4-6 weeks, pick outer leaves
- Care: Regular water, afternoon shade in warmer months

**4. Chilli (Khursani):**
- Pot: 30cm deep, large pot
- Soil: Rich, well-drained
- Season: Plant March-April, harvest June-December
- Sowing: Seeds or seedlings
- Care: Full sun, regular feeding
- Yield: 20-50 chillies per plant

**5. Cherry Tomato:**
- Pot: 30-40cm deep, at least 15L
- Soil: Rich potting mix with compost
- Season: Plant March-April, harvest June-October
- Support: Cage or stake required
- Care: Full sun, regular water, fertilizer every 2 weeks
- Best variety: Small-fruited varieties for pots

**6. Spring Onion:**
- Pot: 20cm deep
- Soil: Any potting mix
- Season: Year-round
- Method: Plant bulb ends from market onions
- Harvest: 3-4 weeks, cut greens
- Continuous: Replant from kitchen scraps

**7. Radish (Mula):**
- Pot: 25cm deep minimum
- Soil: Loose, sandy loam
- Season: September-March
- Sowing: Direct sow, thin to 5cm spacing
- Harvest: 4-5 weeks
- Great for: Quick results, kids love harvesting

**8. Mint (Podina):**
- Pot: 25cm deep (spreads aggressively, pot alone)
- Soil: Rich, moist
- Season: Year-round
- Method: Root cuttings from market mint
- Harvest: Cut when 15cm tall
- Care: Partial shade, constant moisture

**Kathmandu-Specific Tips:**

**Microclimate Considerations:**
- South-facing balconies: Best for sun-loving vegetables
- North-facing: Leafy greens, herbs only
- Kathmandu winter (Dec-Feb): Protect from frost, move indoors at night
- Monsoon (Jun-Aug): Ensure drainage, watch for fungal diseases

**Potting Mix Recipe:**
- 50% garden soil
- 30% compost (cow dung or vermicompost)
- 20% sand or perlite (for drainage)

**Fertilizing:**
- Compost tea every 2 weeks
- Liquid fertilizer (cow urine diluted 1:10)
- Wood ash for potassium (sprinkle lightly)

**Watering:**
- Morning watering best
- Check soil moisture daily in summer
- Avoid waterlogging (most common killer)
- Use saucers to catch drainage

**Pest Management:**
- Neem oil spray weekly for aphids
- Hand-pick caterpillars
- Companion plant marigold to deter pests
- Yellow sticky traps for flying insects

**Space-Saving Techniques:**
- Vertical growing: Use trellis for tomatoes, beans
- Hanging baskets: Herbs, strawberries
- Stacked pots: Different herbs in tiers
- Window boxes: Leafy greens

**Beginner Success:**
Start with coriander, methi, and spring onion - all succeed easily and provide quick results. Graduate to tomatoes and chillies once confident with watering and fertilizing.""")
]

# ============================================================================
# SECTION 8: NEPALI LANGUAGE Q&A (Bilingual Support)
# ============================================================================

NEPALI_QA = [
    create_pair(
        "पालकमा कुन भिटामिन हुन्छ र कसरी खानु पर्छ?",
        """पालक (Spinach) अत्यन्तै पौष्टिक हरियो साग हो:

**पोषक तत्व (प्रति १०० ग्राम):**
- भिटामिन K: दैनिक आवश्यकताको ४६०% (रगत जम्नका लागि)
- भिटामिन A: १८८% (आँखा, छालाको लागि)
- भिटामिन C: ४७% (रोग प्रतिरोधात्मक क्षमता)
- फोलेट: ४९% (गर्भवती महिलाका लागि)
- फलाम: १५% (रगतको कमी हुन नदिन)
- क्याल्सियम: १०% (हड्डीको लागि)

**स्वास्थ्य लाभ:**
- हड्डी बलियो बनाउँछ (भिटामिन K)
- रगतको कमी हटाउँछ (फलाम)
- आँखाको ज्योति बढाउँछ (भिटामिन A)
- क्यान्सरको जोखिम घटाउँछ (एन्टिअक्सिडेन्ट)
- मधुमेहमा फाइदाजनक (फाइबर)

**कसरी छान्ने:**
- चिल्लो, गाढा हरियो पात
- पहेँलो वा चिप्लो भएको छैन
- ताजा गन्ध आउने
- झिँगुरिएको छैन

**भण्डारण:**
- नधोईकन रेफ्रिजरेटरमा राख्ने
- कागजमा बेरेर राख्दा राम्रो
- २-३ दिनभित्र प्रयोग गर्ने
- लामो समयका लागि ब्लान्च गरेर फ्रिज गर्ने (३ महिना)

**पकाउने तरिका:**
१. पालकलाई राम्ररी धुनुस्
२. पानी तताएर २-३ मिनेट ब्लान्च गर्नुस्
३. पानी निकालेर निचोर्नुस्
४. तेलमा जीरा, लसुन, खुर्सानी फोर्नुस्
५. पालक हालेर ५-७ मिनेट पकाउनुस्
६. पालक पनीर, साग, वा दालमा मिसाउन सकिन्छ

**सावधानी:**
- मृगौलाको ढुंगी भएकाले धेरै नखानुस् (अक्सलेट)
- रगत पातलो गर्ने औषधि (वारफारिन) खानेले नियमित मात्रामा खानुस्
- बालबालिकालाई पकाएर मात्र दिनुस्
- गर्भवतीले धेरै नखानुस् (सुरक्षित मात्रामा ठीक छ)

**दैनिक मात्रा:**
प्रतिदिन १ कप पकाएको पालक (लगभग ३० क्यालोरी) स्वास्थ्यका लागि उत्तम मानिन्छ।"""),
    
    create_pair(
        "नेपाली घरमा कुन तरकारी सजिलै उमार्न सकिन्छ?",
        """नेपाली घर, बारी वा बाल्कोनीमा यी तरकारी सजिलै उमार्न सकिन्छ:

**अति सजिलो (शुरुवातका लागि):**

**१. धनिया:**
- लगाउने: असोज-फागुन (सेप्टेम्बर-फेब्रुअरी)
- भाँडो: २० सेमी गहिरो
- बीउ: रातभर पानीमा भिजाएर छर्ने
- फल्ने: ३-४ हप्तामा
- हेरचाह: बिहानको घाम, नियमित पानी

**२. मेथी:**
- लगाउने: असोज-चैत (सेप्टेम्बर-अप्रिल)
- भाँडो: कुनै पनि सानो भाँडो
- बीउ: सेलाएर छर्ने
- फल्ने: २-३ हप्तामा
- प्रयोग: पूरै बोट काटेर प्रयोग गर्ने

**३. पालक:**
- लगाउने: असोज-फागुन (जाडोमा मात्र राम्रो)
- भाँडो: २५ सेमी गहिरो
- बीउ: १ सेमी गहिराइमा
- फल्ने: ४-६ हप्तामा
- विशेषता: बाहिरी पात काट्दै खान मिल्छ

**४. मुला:**
- लगाउने: असोज-फागुन
- भाँडो: २५ सेमी गहिरो
- बीउ: २-३ सेमीको दूरीमा
- फल्ने: ४-५ हप्तामा
- फाइदा: छिटो फल्छ, बालबालिकालाई मन पर्छ

**५. खुर्सानी:**
- लगाउने: चैत-बैशाख (मार्च-अप्रिल) एकपटक
- भाँडो: ३० सेमी ठुलो भाँडो
- फल्ने: ३-४ महिनासम्म लगातार
- दिगो: एकपटक रोपेपछि ६-८ महिना फल्छ

**६. हरियो प्याज:**
- लगाउने: वर्षभरि
- विधि: बजारको प्याजको फेद गाड्ने
- फल्ने: ३-४ हप्तामा
- फाइदा: किचेन स्क्र्यापबाट उमार्न सकिन्छ

**माटो तयारी:**
- ५०% बगैँचाको माटो
- ३०% गोबर मल वा कम्पोस्ट
- २०% बालुवा (पानी निकासका लागि)

**काठमाडौं उपत्यकाका लागि विशेष सुझाव:**
- जाडोमा (पुस-माघ): राति भित्रै राख्ने, चिसोबाट जोगाउने
- वर्षामा (असार-भदौ): पानी निकासको राम्रो व्यवस्था, ढुसीबाट बचाउने
- दक्षिण फर्केको बाल्कोनी: घाम लाग्ने तरकारी (टमाटर, खुर्सानी) को लागि राम्रो
- उत्तर फर्केको बाल्कोनी: पातदार तरकारी र जडिबुटी मात्र

**पानी दिने तरिका:**
- बिहानको समयमा पानी दिने (बेलुकी दिँदा ढुसी लाग्न सक्छ)
- माटो सुकेको छ कि छैन औंलाले छामेर हेर्ने
- पानी धेरै नदिने (जरा कुहिन सक्छ)

**मलको प्रयोग:**
- हरेक २ हप्तामा कम्पोस्टको पानी दिने
- गाईको गोमूत्र १:१० पानीमा मिसाएर दिने
- भाँडाको माथिल्लो माटो २ महिनामा एकपटक फेर्ने

**सुरुवात गर्दा:**
पहिलो पटक धनिया, मेथी र हरियो प्याजबाट सुरु गर्नुस्। यी सजिलै उम्रिन्छन् र छिटो फल्छन्। अनुभव भएपछि टमाटर र खुर्सानीमा जानुस्।"""),
    
    create_pair(
        "करेला मधुमेहको लागि कसरी प्रयोग गर्ने?",
        """करेला (बitter गourd) नेपाली मधुमेह रोगीहरूको लागि परम्परागत उपचार हो:

**कसरी काम गर्छ:**
करेलामा तीन मुख्य सक्रिय यौगिक हुन्छन्:
- **चारान्टिन:** इन्सुलिनजस्तो काम गर्छ
- **पोलिपेप्टाइड-पी:** प्राकृतिक इन्सुलिन
- **भिसिन:** रगतको चिनी घटाउँछ

यी यौगिकहरूले:
1. रगतमा ग्लुकोजको मात्रा घटाउँछन्
2. इन्सुलिनको स्राव बढाउँछन्
3. कोशिकामा ग्लुकोजको अवशोषण सुधार गर्छन्

**प्रयोग विधिहरू:**

**१. करेलाको रस:**
- २-३ वटा करेला (सानो) धोएर टुक्रा पार्नुस्
- मिक्सीमा पानी थोरै हालेर पिउनुस्
- बिहान खाली पेटमा ३०-५० मिली पिउनुस्
- महिनाको १ हप्ता मात्र गर्ने, बीचमा १ हप्ता ब्रेक

**२. भातसँग करेला तरकारी:**
- तितोपन कम गर्न:
  * नुन लगाएर २० मिनेट राख्ने
  * पानी निचोरेर फाल्ने
  * आलुसँगै पकाउने
- दिउँसोको खानामा सानो भाग खाने

**३. करेला चिप्स:**
- पातलो टुक्रा पार्ने
- नुन, बेसार लगाएर
- घिउमा क्रिस्पी फ्राइ गर्ने
- बेलुकाको खानामा साइड डिशको रूपमा

**४. करेला पाउडर:**
- करेला पातलो काटेर घाममा सुकाउने
- पिसेर पाउडर बनाउने
- दिनको १ चम्चा पानीमा मिसाएर पिउने

**मात्रा:**
- ताजा करेला: दिनको ५०-१०० ग्राम (पकाएर)
- करेलाको रस: ३०-५० मिली (बिहान खाली पेट)
- उपचारको अवधि: नियमित रूपमा २-३ महिना

**प्रभाव देखिने समय:**
- रगतको चिनीमा कमी: २-४ हप्तामा
- HbA1c मा सुधार: २-३ महिनामा

**सावधानी (अति महत्वपूर्ण):**

**१. औषधि अन्तरक्रिया:**
- मधुमेहको औषधि खाँदा करेलाले चिनी अत्यधिक घटाउन सक्छ
- डाक्टरको सल्लाहबिना औषधि बन्द नगर्नुस्
- औषधि र करेला दुवै लिने हो भने नियमित रगत जाँच गर्नुस्

**२. नखाने व्यक्ति:**
- गर्भवती महिला (गर्भपातको जोखिम)
- स्तनपान गराउने आमा
- मृगौलाको रोगी
- पित्ताशयको ढुंगी भएको
- सर्जरी गर्न लागेको (रगत जम्ने क्षमतामा असर)

**३. हाइपोग्लाइसेमियाका लक्षण:**
- चक्कर लाग्ने, पसिना आउने
- धड्कन बढ्ने, भोक लाग्ने
- यस्तो भए तुरुन्त मिठो खानुस् र डाक्टरलाई भन्नुस्

**अन्य फाइदाहरू:**
- कोलेस्ट्रोल घटाउँछ
- पाचन सुधार गर्छ
- छालाको समस्या कम गर्छ
- लिभर सफा गर्न मद्दत गर्छ

**गुणस्तरीय करेला पहिचान:**
- चिल्लो, गाढा हरियो रंग
- ठुला उँभो भएको
- नरम नभएको
- पहेँलो वा रातो नभएको

**वैज्ञानिक प्रमाण:**
अध्ययनहरूले देखाएका छन् कि नियमित करेला सेवनले:
- फास्टिङ ग्लुकोज १०-२०% घटाउँछ
- पोस्टप्रान्डियल ग्लुकोज १५-२५% घटाउँछ
- HbA1c ०.५-१% सुधार गर्छ

**याद राख्नुस्:**
करेला मधुमेहको ओखतीको विकल्प होइन, यो पूरक उपचार हो। नियमित औषधि, सन्तुलित आहार, र व्यायामसँगै मात्र प्रयोग गर्नुस्।""")
]

# ============================================================================
# SECTION 9: SCRAPED DATA (USDA Nutrition - Enhanced)
# ============================================================================

# Comprehensive vegetable list for Nepal market
NEPALI_VEGETABLES = [
    "tomato", "spinach", "carrot", "potato", "onion", "garlic", "cauliflower",
    "cabbage", "broccoli", "cucumber", "pumpkin", "bitter gourd", "bottle gourd",
    "okra", "eggplant", "radish", "fenugreek leaf", "mustard greens", "taro leaf",
    "sweet potato", "peas", "beans", "capsicum", "chilli", "coriander", "mint",
    "ginger", "turmeric", "pointed gourd", "ridge gourd", "snake gourd", "corn",
    "turnip", "beetroot", "celery", "asparagus", "leek", "kale", "watercress"
]

def scrape_usda_nutrition():
    """Scrape USDA nutrition data for Nepali vegetables"""
    cache_file = "usda_nutrition_cache.json"
    
    if os.path.exists(cache_file):
        print("  Loading cached USDA nutrition data...")
        with open(cache_file, 'r') as f:
            return json.load(f)
    
    print("  Scraping USDA nutrition data (this may take a few minutes)...")
    results = []
    
    for veg in NEPALI_VEGETABLES:
        try:
            # Search for raw vegetable
            response = requests.get(
                "https://api.nal.usda.gov/fdc/v1/foods/search",
                params={
                    "query": f"{veg} raw",
                    "api_key": "DEMO_KEY",
                    "pageSize": 1
                },
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                foods = data.get('foods', [])
                
                if foods:
                    food = foods[0]
                    nutrients = {}
                    
                    # Extract key nutrients
                    for nutrient in food.get('foodNutrients', []):
                        name = nutrient.get('nutrientName', '')
                        value = nutrient.get('value')
                        if value and any(key in name.lower() for key in [
                            'energy', 'protein', 'fat', 'carbohydrate', 'fiber',
                            'vitamin a', 'vitamin c', 'vitamin k', 'calcium',
                            'iron', 'potassium', 'magnesium', 'folate'
                        ]):
                            nutrients[name] = round(value, 1)
                    
                    results.append({
                        'vegetable': veg,
                        'name': food.get('description', veg),
                        'nutrients': nutrients
                    })
                    print(f"    ✓ {veg}")
                else:
                    print(f"    ? {veg} - not found")
            else:
                print(f"    x {veg} - API error")
            
            time.sleep(0.5)  # Be respectful to API
            
        except Exception as e:
            print(f"    ! {veg} - error: {e}")
    
    # Cache results
    with open(cache_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def create_nutrition_pairs(usda_data):
    """Create Q&A pairs from USDA nutrition data"""
    pairs = []
    
    for item in usda_data:
        veg = item['vegetable'].capitalize()
        nutrients = item.get('nutrients', {})
        
        # Format nutrition information
        nutrition_text = f"**{veg} - Nutrition Facts (per 100g raw):**\n\n"
        
        # Calories
        if any('Energy' in k for k in nutrients.keys()):
            cal = next((v for k, v in nutrients.items() if 'Energy' in k), None)
            if cal:
                nutrition_text += f"• Calories: {cal} kcal\n"
        
        # Macronutrients
        macros = []
        for macro in ['Protein', 'Total lipid', 'Carbohydrate', 'Fiber']:
            if macro in nutrients:
                macros.append(f"{macro}: {nutrients[macro]}g")
        if macros:
            nutrition_text += f"• {', '.join(macros)}\n"
        
        # Vitamins
        vitamins = []
        for vit in ['Vitamin A', 'Vitamin C', 'Vitamin K', 'Folate']:
            for k, v in nutrients.items():
                if vit.lower() in k.lower():
                    unit = 'mcg' if 'A' in vit or 'K' in vit else 'mg'
                    vitamins.append(f"{vit}: {v}{unit}")
                    break
        if vitamins:
            nutrition_text += f"• Vitamins: {', '.join(vitamins)}\n"
        
        # Minerals
        minerals = []
        for min in ['Calcium', 'Iron', 'Potassium', 'Magnesium']:
            if min in nutrients:
                minerals.append(f"{min}: {nutrients[min]}mg")
        if minerals:
            nutrition_text += f"• Minerals: {', '.join(minerals)}\n"
        
        nutrition_text += "\n*Data source: USDA FoodData Central*"
        
        # Create Q&A pair
        pairs.append(create_pair(
            f"Tell me the complete nutrition facts for {veg}.",
            nutrition_text
        ))
        
        # Add a second question about specific benefits
        if veg == "Tomato":
            pairs.append(create_pair(
                f"What are the specific health benefits of {veg}?",
                f"{veg} (Tomato) is exceptionally rich in lycopene, a powerful antioxidant linked to reduced risk of heart disease and prostate cancer. It provides 28% DV of Vitamin C per medium fruit, plus Vitamin K, folate, and potassium. Cooking actually increases lycopene bioavailability. The vibrant red color indicates high antioxidant content. Regular consumption supports heart health, skin health, and may reduce inflammation."
            ))
        elif veg == "Spinach":
            pairs.append(create_pair(
                f"What are the specific health benefits of {veg}?",
                f"{veg} is one of the most nutrient-dense vegetables available. It's exceptionally high in Vitamin K (460% DV) for bone health and blood clotting, Vitamin A (188% DV) for vision and immunity, and folate (49% DV) crucial for cell division. The iron content supports energy levels, while antioxidants like lutein and zeaxanthin protect eye health. The high fiber content supports digestive health and blood sugar regulation."
            ))
        elif veg == "Carrot":
            pairs.append(create_pair(
                f"What are the specific health benefits of {veg}?",
                f"{veg} is renowned for beta-carotene, which converts to Vitamin A—essential for night vision, immune function, and skin health. One medium carrot provides over 200% of daily Vitamin A needs. The antioxidants in carrots, including beta-carotene and lutein, protect against age-related macular degeneration and may reduce cancer risk. The soluble fiber helps lower cholesterol levels."
            ))
        elif veg == "Garlic":
            pairs.append(create_pair(
                f"What are the specific health benefits of {veg}?",
                f"{veg} contains allicin, produced when crushed, which has potent antibacterial, antiviral, and antifungal properties. Regular consumption (1-2 cloves daily) is associated with lower blood pressure, reduced LDL cholesterol, improved immune function, and decreased risk of certain cancers. Let crushed garlic sit for 10 minutes before cooking to maximize allicin production."
            ))
        elif veg == "Bitter gourd":
            pairs.append(create_pair(
                f"What are the specific health benefits of {veg}?",
                f"{veg} (Karela) contains charantin, vicine, and polypeptide-P—compounds with insulin-like effects that help lower blood glucose levels. It's traditionally used in Ayurvedic medicine for diabetes management. Additionally, it supports liver health, improves digestion, and has anti-inflammatory properties. The bitter taste stimulates digestive enzyme production."
            ))
        else:
            # Generic health benefits based on nutrients
            benefits = []
            if any('Vitamin C' in k for k in nutrients.keys()):
                benefits.append("supports immune function")
            if any('Vitamin A' in k for k in nutrients.keys()):
                benefits.append("promotes healthy vision")
            if any('Fiber' in k for k in nutrients.keys()):
                benefits.append("aids digestion and blood sugar control")
            if any('Potassium' in k for k in nutrients.keys()):
                benefits.append("helps maintain healthy blood pressure")
            if any('Iron' in k for k in nutrients.keys()):
                benefits.append("supports energy levels and oxygen transport")
            
            if benefits:
                pairs.append(create_pair(
                    f"What are the health benefits of {veg}?",
                    f"{veg} is a nutritious vegetable that {', '.join(benefits)}. It's low in calories while providing essential vitamins and minerals. Including it regularly in your diet contributes to overall health and well-being. For specific nutritional values, refer to the nutrition facts."
                ))
    
    return pairs

# ============================================================================
# MAIN BUILD FUNCTION
# ============================================================================

def build_dataset():
    """Build the complete SagjiBot dataset"""
    print("=" * 60)
    print("Building SagjiBot Dataset (Consumer-Focused, Nepal Market)")
    print("=" * 60)
    
    all_pairs = []
    
    # Add all QA sections
    print("\n📝 Adding hand-written Q&A sections...")
    
    sections = [
        ("Nepali Vegetable Varieties", VARIETY_QA),
        ("Buying & Storage Guide", BUYING_STORAGE_QA),
        ("Cooking Techniques", COOKING_TECHNIQUES_QA),
        ("Nutrition & Health Benefits", NUTRITION_HEALTH_QA),
        ("Chemical Safety", CHEMICAL_SAFETY_QA),
        ("Traditional Preparations", TRADITIONAL_PREP_QA),
        ("Home Growing Guide", HOME_GROWING_QA),
        ("Nepali Language Q&A", NEPALI_QA)
    ]
    
    for name, section in sections:
        all_pairs.extend(section)
        print(f"  ✓ {name}: {len(section)} pairs")
    
    # Add USDA nutrition data
    print("\n🌱 Adding USDA nutrition data...")
    usda_data = scrape_usda_nutrition()
    usda_pairs = create_nutrition_pairs(usda_data)
    all_pairs.extend(usda_pairs)
    print(f"  ✓ USDA Nutrition: {len(usda_pairs)} pairs")
    
    # Shuffle for randomness
    random.seed(42)
    random.shuffle(all_pairs)
    
    # Split into train/val
    split_idx = int(len(all_pairs) * 0.9)
    train_pairs = all_pairs[:split_idx]
    val_pairs = all_pairs[split_idx:]
    
    # Save datasets
    with open("train.jsonl", "w", encoding="utf-8") as f:
        for pair in train_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    
    with open("val.jsonl", "w", encoding="utf-8") as f:
        for pair in val_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ DATASET BUILD COMPLETE")
    print("=" * 60)
    print(f"\n📊 Statistics:")
    print(f"  Total pairs: {len(all_pairs)}")
    print(f"  Training set: {len(train_pairs)} (train.jsonl)")
    print(f"  Validation set: {len(val_pairs)} (val.jsonl)")
    
    print(f"\n📁 Section Breakdown:")
    print(f"  • Nepali Varieties: {len(VARIETY_QA)}")
    print(f"  • Buying & Storage: {len(BUYING_STORAGE_QA)}")
    print(f"  • Cooking Techniques: {len(COOKING_TECHNIQUES_QA)}")
    print(f"  • Nutrition & Health: {len(NUTRITION_HEALTH_QA)}")
    print(f"  • Chemical Safety: {len(CHEMICAL_SAFETY_QA)}")
    print(f"  • Traditional Prep: {len(TRADITIONAL_PREP_QA)}")
    print(f"  • Home Growing: {len(HOME_GROWING_QA)}")
    print(f"  • Nepali Language: {len(NEPALI_QA)}")
    print(f"  • USDA Nutrition: {len(usda_pairs)}")
    
    print(f"\n🎯 Dataset covers:")
    print("  ✓ All major Nepali vegetables with local varieties")
    print("  ✓ Consumer buying, storage, and selection tips")
    print("  ✓ Cooking techniques and traditional preparations")
    print("  ✓ Nutrition facts and health benefits")
    print("  ✓ Chemical safety and pesticide reduction")
    print("  ✓ Home growing in Nepal context")
    print("  ✓ Bilingual support (English + Nepali)")
    print("  ✓ NO pricing information (as requested)")
    
    print(f"\n🚀 Next steps:")
    print("  1. Upload train.jsonl and val.jsonl to your training platform")
    print("  2. Use with Llama, Mistral, or other LLM fine-tuning")
    print("  3. Test with questions about Nepali vegetables")
    print("  4. Customize further with your ordering site's specific data")
    
    return all_pairs

if __name__ == "__main__":
    build_dataset()