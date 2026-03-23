# SagjiBot — Vegetable Expert AI

Custom-trained AI chatbot for vegetables — **English and Nepali**.
**One GitHub repo → one Vercel project → everything hosted.**

---

## Repo structure

```
sagjibot/
├── api/
│   └── chat.js           ← serverless API (Vercel runs this)
├── public/
│   ├── index.html         ← homepage
│   ├── chat.html          ← chatbot
│   ├── vegetables.html    ← vegetable directory
│   ├── css/style.css
│   ├── css/chat.css
│   └── js/
├── data/
│   ├── build_dataset.py   ← run locally to build training data
│   └── train.py           ← paste into Colab / Kaggle to train
├── package.json
└── vercel.json            ← routes website + API from one project
```

---

## How it all connects

```
One Vercel project
├── /             → public/index.html
├── /chat         → public/chat.html
├── /vegetables   → public/vegetables.html
└── /api/chat     → api/chat.js  (calls your Hugging Face model)
```

No Netlify. No separate backend repo. One deploy, done.

---

## Step 1 — Create accounts (all free)

| Service | For |
|---------|-----|
| Hugging Face | Store trained model |
| GitHub | Host code |
| Vercel | Host website + API |

---

## Step 2 — Hugging Face setup

1. Sign up at **huggingface.co**
2. Settings → Access Tokens → New Token → **Write** → copy it
3. Accept the Llama license at **huggingface.co/meta-llama/Llama-3.2-3B-Instruct**

---

## Step 3 — Build training data (on your laptop)

```bash
cd data
pip install requests
python build_dataset.py
```

Creates `train.jsonl` and `val.jsonl`.

To add more knowledge, open `build_dataset.py` and add to the `EN` or `NE` lists, then re-run.

---

## Step 4 — Train on Google Colab

1. Go to **colab.research.google.com** → New Notebook
2. **Runtime → Change runtime type → T4 GPU → Save**
3. Upload `train.jsonl` and `val.jsonl` using the Files panel (left sidebar)
4. Open `data/train.py` — copy each cell into Colab and run in order

In **Cell 1**, update:
```python
HF_USERNAME = "your-hf-username"
HF_TOKEN    = "hf_xxxxxxxxxxxx"
```

Training takes **~1.5 hours**. Cell 11 pushes the model to Hugging Face automatically.

---

## Step 4 (alternative) — Train on Kaggle

1. **kaggle.com** → New Notebook → Settings → Accelerator → **GPU T4 x2**
2. Add Data → upload `train.jsonl` and `val.jsonl` as a new dataset
3. In Cell 1 of `train.py`, set `DATASET = "your-dataset-name"`
4. Run cells in order — takes **~1 hour** on Kaggle

---

## Step 5 — Copy your model URL

After training Cell 11 completes, you will see:

```
✅ Model live → https://huggingface.co/YOUR_USERNAME/sagjibot

HF_MODEL_URL:
https://api-inference.huggingface.co/models/YOUR_USERNAME/sagjibot
```

Save that URL.

---

## Step 6 — Push to GitHub

```bash
git init
git add .
git commit -m "sagjibot"
git remote add origin https://github.com/YOUR_USERNAME/sagjibot.git
git push -u origin main
```

---

## Step 7 — Deploy to Vercel

1. **vercel.com** → Sign in with GitHub → Add New Project → Import `sagjibot`
2. Leave all settings default → **Deploy**
3. After deploy: **Settings → Environment Variables** → Add:

| Name | Value |
|------|-------|
| `HF_TOKEN` | your Hugging Face token |
| `HF_MODEL_URL` | `https://api-inference.huggingface.co/models/YOUR_USERNAME/sagjibot` |

4. **Deployments → Redeploy**

Your site is live at your Vercel URL.

---

## Step 8 — Test

Open your Vercel URL → Ask AI.

- English: `What vitamins does spinach have?`
- Nepali: `पालकमा कुन भिटामिन हुन्छ?`

**First message takes ~25 seconds** — Hugging Face free tier cold-starts the model. After that it's fast.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| CUDA out of memory | Set `BATCH=1` in Cell 1 |
| "Must accept license" | Go to huggingface.co/meta-llama/Llama-3.2-3B-Instruct and click Agree |
| Colab disconnects mid-training | Cell 2 saves to Drive automatically. Add `resume_from_checkpoint=True` in Cell 8 |
| Vercel env vars not working | Must be exactly `HF_TOKEN` and `HF_MODEL_URL`. Redeploy after saving. |
| First message slow | Normal — 25s cold start on free tier |
| Wrong language | Add more Nepali pairs to `NE` in `build_dataset.py` and retrain |

---

## Free resources

| What | URL |
|------|-----|
| Google Colab | colab.research.google.com |
| Kaggle | kaggle.com |
| HF tokens | huggingface.co/settings/tokens |
| Llama license | huggingface.co/meta-llama/Llama-3.2-3B-Instruct |
| Vercel | vercel.com |
