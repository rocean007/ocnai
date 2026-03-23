"""
scrape_chemicals.py
Downloads agricultural chemical data from PubChem API.
No API key needed — completely free.
"""

import requests
import json
import time

CHEMICALS = [
    "glyphosate", "malathion", "permethrin", "chlorpyrifos",
    "copper sulfate", "pyrethrin", "spinosad", "azadirachtin",
    "captan", "mancozeb", "thiram", "iprodione", "carbendazim",
    "imidacloprid", "acetamiprid", "deltamethrin", "cypermethrin",
    "chlorothalonil", "propiconazole", "tebuconazole",
    "abamectin", "emamectin benzoate", "bifenazate", "spiromesifen",
    "flonicamid", "cyantraniliprole", "sulfur", "potassium bicarbonate",
    "hydrogen peroxide", "copper hydroxide"
]

def get_pubchem_data(chemical):
    """Fetch compound data from PubChem."""
    # Step 1: Get CID
    search_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{requests.utils.quote(chemical)}/cids/JSON"
    resp = requests.get(search_url, timeout=10)
    if resp.status_code != 200:
        return None

    cids = resp.json().get("IdentifierList", {}).get("CID", [])
    if not cids:
        return None

    cid = cids[0]

    # Step 2: Get properties
    props_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/MolecularFormula,MolecularWeight,IUPACName,XLogP,TPSA/JSON"
    props_resp = requests.get(props_url, timeout=10)

    # Step 3: Get description
    desc_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/description/JSON"
    desc_resp = requests.get(desc_url, timeout=10)

    properties = {}
    if props_resp.status_code == 200:
        props_data = props_resp.json().get("PropertyTable", {}).get("Properties", [{}])[0]
        properties = {
            "molecular_formula": props_data.get("MolecularFormula", ""),
            "molecular_weight": props_data.get("MolecularWeight", ""),
            "iupac_name": props_data.get("IUPACName", ""),
            "xlogp": props_data.get("XLogP", ""),
            "tpsa": props_data.get("TPSA", ""),
        }

    description = ""
    if desc_resp.status_code == 200:
        desc_list = desc_resp.json().get("InformationList", {}).get("Information", [])
        for item in desc_list:
            if item.get("Description"):
                description = item["Description"][0] if isinstance(item["Description"], list) else item["Description"]
                break

    return {
        "chemical": chemical,
        "cid": cid,
        "properties": properties,
        "description": description[:1000] if description else ""
    }

def scrape():
    results = []
    failed = []

    for chem in CHEMICALS:
        try:
            data = get_pubchem_data(chem)
            if data:
                results.append(data)
                print(f"  ✓  {chem}")
            else:
                print(f"  x  {chem} — not found")
                failed.append(chem)
            time.sleep(0.5)  # PubChem rate limit: max 5 requests/second

        except Exception as e:
            print(f"  x  {chem} — {e}")
            failed.append(chem)

    with open("raw_chemicals.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Saved {len(results)} chemicals to raw_chemicals.json")
    if failed:
        print(f"⚠️  Failed: {', '.join(failed)}")

if __name__ == "__main__":
    print("Scraping PubChem chemical data...\n")
    scrape()
