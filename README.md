# 🥦 VeggieBot — Train & Deploy Your Own Vegetable AI

A fully custom AI chatbot trained on vegetable nutrition, farming, agricultural chemicals,
and plant diseases. No paid AI APIs. Your model, your data, $0 cost.

---

## What's In This Repo

```
veggie-ai/
├── data/                        ← Run these on your laptop to build training data
│   ├── requirements.txt
│   ├── scrape_nutrition.py      — Downloads nutrition data from USDA
│   ├── scrape_growing.py        — Downloads growing data from OpenFarm
│   ├── scrape_chemicals.py      — Downloads chemical data from PubChem
│   ├── scrape_diseases.py       — Downloads disease data from Wikipedia
│   ├── convert_nutrition.py     — Converts scraped data → Q&A training pairs
│   ├── convert_growing.py       — Converts scraped data → Q&A training pairs
│   ├── convert_chemicals.py     — Converts scraped data → Q&A training pairs
│   ├── convert_diseases.py      — Converts scraped data → Q&A training pairs
│   ├── merge_data.py            — Merges all topics → train_final.jsonl
│   └── colab_training.py        — Copy-paste cells into Google Colab to train
│
├── veggie-backend/              ← Deploy this to Vercel
│   ├── api/
│   │   └── chat.js              — Serverless API function
│   ├── package.json
│   └── vercel.json
│
└── veggie-website/              ← Deploy this to Netlify
    ├── index.html               — Homepage
    ├── chatbot.html             — Full chatbot page
    ├── vegetables.html          — Vegetable directory
    ├── css/style.css
    ├── js/main.js
    └── netlify.toml
```

---

## Accounts You Need (All Free)

Sign up for these before starting. All are completely free.

| Service | Used For | Sign Up |
|---------|----------|---------|
| **Google** | Google Colab GPU for training | Already have one |
| **Hugging Face** | Store and serve your trained model | huggingface.co |
| **GitHub** | Store your code | github.com |
| **Vercel** | Host your backend API | vercel.com |
| **Netlify** | Host your website | netlify.com |

---

## STEP 1 — Set Up Hugging Face

You need this before training so you have somewhere to push the model.

1. Go to **https://huggingface.co** → Sign up
2. Go to **Settings → Access Tokens → New Token**
3. Name: `veggie-token`, Access: **Write**
4. Click Generate → **Copy and save the token** (looks like `hf_xxxxxxxxxxxxxxxx`)
5. Go to **https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2**
6. Click **"Agree and access repository"** (required to use the base model)

---

## STEP 2 — Clone This Repo to Your Computer

```bash
git clone https://github.com/YOUR_USERNAME/veggie-ai.git
cd veggie-ai
```

Or download the ZIP from GitHub if you don't have git installed.

---

## STEP 3 — Install Python Dependencies

You need Python 3.10 or higher. Check with `python --version`.

```bash
cd data
pip install -r requirements.txt
```

---

## STEP 4 — Scrape the Training Data

Run all 4 scrapers. Each one downloads raw data from a free public API.
This takes about 5–10 minutes total.

```bash
cd data

python scrape_nutrition.py    # Downloads from USDA FoodData Central
python scrape_growing.py      # Downloads from OpenFarm.cc
python scrape_chemicals.py    # Downloads from PubChem
python scrape_diseases.py     # Downloads from Wikipedia
```

After running, you will have these files in the `data/` folder:

```
raw_nutrition.json    ← USDA data for 50 vegetables
raw_growing.json      ← OpenFarm growing guides
raw_chemicals.json    ← PubChem chemical properties
raw_diseases.json     ← Wikipedia disease summaries
```

> **Note on USDA API key:** `scrape_nutrition.py` uses `DEMO_KEY` which works but
> has lower rate limits. Get a free personal key at https://fdc.nal.usda.gov/api-guide.html
> and replace `API_KEY = "DEMO_KEY"` with your key.

---

## STEP 5 — Convert Raw Data to Training Pairs

Each convert script turns the raw JSON into Q&A conversation pairs in JSONL format.
The scripts also include hand-written expert Q&A — **the most important part**.

```bash
python convert_nutrition.py   # Creates train_nutrition.jsonl
python convert_growing.py     # Creates train_growing.jsonl
python convert_chemicals.py   # Creates train_chemicals.jsonl
python convert_diseases.py    # Creates train_diseases.jsonl
```

Then merge everything into one final training file:

```bash
python merge_data.py
```

Output:
```
train_final.jsonl   ← 90% of data — used for training
val_final.jsonl     ← 10% of data — used for validation
```

You will see a summary like:
```
Training:   1847 pairs  → train_final.jsonl
Validation:  205 pairs  → val_final.jsonl

Breakdown by topic:
  Nutrition: 480 (24.3%)
  Growing:   612 (31.0%)
  Chemicals: 387 (19.6%)
  Diseases:  493 (25.0%)
```

> **Want better results?** Add more Q&A pairs to the `MANUAL_QA` lists in the
> convert scripts before running them. The more specific and detailed your
> hand-written pairs, the better the model will perform on that topic.

---

## STEP 6 — Train the Model on Google Colab

Google Colab gives you a free T4 GPU with ~15GB VRAM. Training takes 1–3 hours.

### 6.1 — Open Colab and set up GPU

1. Go to **https://colab.research.google.com**
2. Click **New Notebook**
3. Click **Runtime → Change Runtime Type**
4. Select **T4 GPU** → Click **Save**

### 6.2 — Upload your training data

In the left sidebar, click the **folder icon** (Files panel).
Upload both files from your `data/` folder:
- `train_final.jsonl`
- `val_final.jsonl`

### 6.3 — Run the training cells

Open `data/colab_training.py` in this repo. It contains 11 cells.
**Copy each cell into Colab in order and run them one by one.**

Here is what each cell does:

| Cell | What It Does | Time |
|------|-------------|------|
| 1 | Mount Google Drive (saves model if session disconnects) | 30s |
| 2 | Install packages (transformers, peft, trl, bitsandbytes) | 2 min |
| 3 | Verify GPU is available | 5s |
| 4 | Load Mistral 7B base model in 4-bit | 3–6 min |
| 5 | Configure LoRA adapters | 10s |
| 6 | Load your vegetable dataset | 10s |
| 7 | Format data in Mistral instruction template | 30s |
| 8 | **Run training (main step)** | 1–3 hours |
| 9 | Save trained model to Google Drive | 2 min |
| 10 | Test model with sample questions | 2 min |
| 11 | Push model to Hugging Face | 3–5 min |

### 6.4 — Before running Cell 11, update these two lines

In Cell 11 of the Colab notebook, change:

```python
HF_USERNAME = "your-hf-username"    # ← Your Hugging Face username
HF_TOKEN    = "hf_xxxxxxxxxxxx"     # ← Your Hugging Face token from Step 1
```

### 6.5 — What good training looks like

Watch the training output. You want to see:
- `train_loss` steadily decreasing (e.g. 2.1 → 1.4 → 0.9 → 0.6)
- `eval_loss` also decreasing (not increasing — that would be overfitting)

If the session disconnects during training:
- Your checkpoints are saved to Google Drive automatically (Cell 1 mounted it)
- Add `resume_from_checkpoint=True` to TrainingArguments in Cell 8 and re-run

### 6.6 — After Cell 11 completes

Your model is live at:
```
https://huggingface.co/YOUR_USERNAME/veggie-bot
```

The Hugging Face Inference API URL for your backend will be:
```
https://api-inference.huggingface.co/models/YOUR_USERNAME/veggie-bot
```

**Save this URL — you will need it in Step 7.**

---

## STEP 7 — Update the Website with Your Backend URL

Before pushing to GitHub, you need to update one line in the chatbot.

Open `veggie-website/chatbot.html` and find line ~130:

```javascript
const API_URL = "https://YOUR-APP-NAME.vercel.app/api/chat";
```

You will fill in the actual Vercel URL after deploying in Step 9.
Leave it as-is for now — you will come back to update it.

---

## STEP 8 — Push Everything to GitHub

### 8.1 — Create two GitHub repositories

Go to **https://github.com/new** and create two separate repos:

| Repo Name | Contents | Visibility |
|-----------|----------|------------|
| `veggie-backend` | The `veggie-backend/` folder | Public |
| `veggie-website` | The `veggie-website/` folder | Public |

### 8.2 — Push the backend

```bash
cd veggie-ai/veggie-backend

git init
git add .
git commit -m "Initial VeggieBot backend"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/veggie-backend.git
git push -u origin main
```

### 8.3 — Push the website

```bash
cd veggie-ai/veggie-website

git init
git add .
git commit -m "Initial VeggieWorld website"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/veggie-website.git
git push -u origin main
```

---

## STEP 9 — Deploy the Backend to Vercel

### 9.1 — Connect GitHub to Vercel

1. Go to **https://vercel.com** → Sign up with GitHub (free)
2. Click **Add New → Project**
3. Find and select your `veggie-backend` repository → Click **Import**
4. Leave all settings as default
5. Click **Deploy**

Vercel will build and deploy. Takes about 30 seconds.
You will get a URL like: `https://veggie-backend-abc123.vercel.app`

**Copy this URL.**

### 9.2 — Add your Hugging Face credentials as Environment Variables

In the Vercel dashboard:
1. Click your project → **Settings → Environment Variables**
2. Add these two variables:

| Name | Value |
|------|-------|
| `HF_TOKEN` | `hf_xxxxxxxxxxxxxxxxxx` (your Hugging Face token) |
| `HF_MODEL_URL` | `https://api-inference.huggingface.co/models/YOUR_USERNAME/veggie-bot` |

3. Click **Save**
4. Go to **Deployments → Click the three dots → Redeploy** (must redeploy to pick up new env vars)

### 9.3 — Test your backend is working

Replace the URL with your actual Vercel URL:

```bash
curl -X POST https://veggie-backend-abc123.vercel.app/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What vitamins does broccoli have?"}'
```

You should get back:
```json
{"reply": "Broccoli is an excellent source of..."}
```

If you get `{"error": "Model is loading..."}` — wait 30 seconds and try again.
The Hugging Face free tier cold-starts the model on first request.

---

## STEP 10 — Update the Website URL and Deploy to Netlify

### 10.1 — Update the API URL in the chatbot

Open `veggie-website/chatbot.html`, find this line:

```javascript
const API_URL = "https://YOUR-APP-NAME.vercel.app/api/chat";
```

Replace it with your actual Vercel URL:

```javascript
const API_URL = "https://veggie-backend-abc123.vercel.app/api/chat";
```

### 10.2 — Commit and push the change

```bash
cd veggie-ai/veggie-website
git add chatbot.html
git commit -m "Add Vercel backend URL"
git push
```

### 10.3 — Deploy the website to Netlify

**Option A — Drag and Drop (fastest):**
1. Go to **https://app.netlify.com/drop**
2. Drag your entire `veggie-website/` folder onto the page
3. Done — you get a free URL like `https://amazing-veggie-abc123.netlify.app`

**Option B — Connect GitHub (auto-deploys on every push):**
1. Go to **https://app.netlify.com** → Sign up with GitHub (free)
2. Click **Add New Site → Import an existing project**
3. Select **GitHub** → Choose your `veggie-website` repository
4. Build settings: leave everything blank (it's a static HTML site)
5. Click **Deploy site**

Every time you push to GitHub, Netlify auto-deploys. This is the better option.

### 10.4 — Set a custom site name (optional)

In Netlify dashboard → Site configuration → Change site name
→ Set something like `veggie-world` → your URL becomes `https://veggie-world.netlify.app`

---

## STEP 11 — Test Everything End-to-End

1. Open your Netlify URL in a browser
2. Click **Ask AI** in the nav, or click the 🥦 bubble in the corner
3. Type: `What vitamins does spinach have?`
4. You should get a reply from your trained model within a few seconds

**First message may take 20–30 seconds** — this is the Hugging Face cold start.
After the first message, responses are fast.

If it works: 🎉 You are done!

---

## STEP 12 — Add a Custom Domain (Optional, Free)

Both Netlify and Vercel support free custom domains.

### On Netlify (website)
1. Dashboard → Your site → Domain management → Add custom domain
2. Enter your domain (e.g. `veggieworld.com`)
3. Update your domain's DNS nameservers to Netlify's (shown in dashboard)
4. SSL certificate is automatic and free

### On Vercel (backend API)
1. Dashboard → Your project → Settings → Domains
2. Add your domain (e.g. `api.veggieworld.com`)
3. Update DNS records as shown
4. SSL certificate is automatic and free

---

## How to Improve the Model Over Time

### Add more training data

1. Open any `convert_*.py` file in the `data/` folder
2. Add more Q&A pairs to the `MANUAL_QA` list
3. Re-run that convert script, then `merge_data.py`
4. Upload new `train_final.jsonl` to Colab
5. Re-run training with `num_train_epochs=1` (faster — just a top-up)

### Signs your model needs more data

| Symptom | Fix |
|---------|-----|
| Vague or generic answers | Add more specific Q&A on that topic |
| Repeats itself in same answer | Lower `temperature` to `0.5` in `api/chat.js` |
| Wrong facts | Add corrective Q&A that explicitly states the right answer |
| Weak on one topic | Add 50+ more pairs on that topic |
| Mixes up topics | Add more varied examples per topic |

### Recommended training data targets

| Topic | Minimum | Good | Excellent |
|-------|---------|------|-----------|
| Nutrition | 200 | 500 | 1000+ |
| Growing | 200 | 500 | 1000+ |
| Chemicals | 100 | 300 | 600+ |
| Diseases | 100 | 300 | 600+ |

---

## Troubleshooting

### Google Colab

**"CUDA out of memory"**
→ Go to Runtime → Disconnect and delete runtime → Reconnect with T4 GPU
→ In Cell 8 set `per_device_train_batch_size=1` and `gradient_accumulation_steps=8`

**"Model not found" when loading Mistral**
→ You need to accept the license. Go to:
   https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2
   and click "Agree and access repository"

**Colab session disconnects mid-training**
→ Cell 1 mounts Google Drive, so checkpoints save there automatically
→ In Cell 8, add `resume_from_checkpoint=True` to TrainingArguments to continue

**Training loss not going down after many steps**
→ Your learning rate may be too high. Change `learning_rate=2e-4` to `learning_rate=1e-4`

### Vercel

**Deploy fails**
→ Check the build logs in Vercel dashboard
→ Most common cause: syntax error in `api/chat.js`
→ Fix the error, commit, and push — Vercel auto-redeploys

**"Environment variable not found"**
→ Variable names must be exactly `HF_TOKEN` and `HF_MODEL_URL` (case sensitive)
→ Must redeploy after adding or changing env vars

**CORS error in browser console**
→ Check `api/chat.js` has `Access-Control-Allow-Origin: *` header
→ Make sure `vercel.json` is committed and pushed

### Hugging Face

**503 — Model is loading**
→ Normal behaviour on free tier — first request after inactivity cold-starts the model
→ Wait 20–30 seconds and try again
→ Already handled gracefully in `chatbot.html`

**403 — Unauthorized**
→ Your `HF_TOKEN` in Vercel env vars is wrong or expired
→ Generate a new token at https://huggingface.co/settings/tokens
→ Update in Vercel → Settings → Environment Variables → Redeploy

### Website

**Chatbot shows "Connection error"**
→ Open browser DevTools (F12) → Network tab → Look at the failed request
→ Most likely: `API_URL` in `chatbot.html` is still set to `YOUR-APP-NAME`
→ Update it with your real Vercel URL and push

**Floating chat bubble doesn't open**
→ Make sure `js/main.js` is loading. Check browser console for errors.

---

## Full Technology Stack

| Layer | Technology | Cost |
|-------|-----------|------|
| Base model | Mistral 7B Instruct v0.2 | Free (open source) |
| Fine-tuning method | QLoRA via PEFT library | Free |
| Training compute | Google Colab T4 GPU | Free |
| Model hosting | Hugging Face Inference API | Free tier |
| Backend API | Vercel Serverless Functions | Free tier |
| Frontend hosting | Netlify | Free tier |
| Code hosting | GitHub | Free |
| Nutrition data | USDA FoodData Central | Free |
| Growing data | OpenFarm API | Free |
| Chemical data | PubChem API | Free |
| Disease data | Wikipedia API | Free |

**Total monthly cost: $0**

---

## Quick Reference — All URLs

| What | URL |
|------|-----|
| Google Colab | https://colab.research.google.com |
| Hugging Face tokens | https://huggingface.co/settings/tokens |
| Mistral model license | https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2 |
| Vercel dashboard | https://vercel.com/dashboard |
| Netlify drag-and-drop | https://app.netlify.com/drop |
| GitHub new repo | https://github.com/new |
| USDA API key | https://fdc.nal.usda.gov/api-guide.html |
| PubChem | https://pubchem.ncbi.nlm.nih.gov |
| OpenFarm | https://openfarm.cc |
