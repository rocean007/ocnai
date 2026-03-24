# SagjiBot

Vegetable expert AI chatbot. English by default, Nepali on request.
One GitHub repo → one Vercel deploy → done.

---

## Files

```
sagjibot/
├── api/chat.js              ← backend (Vercel runs this)
├── public/index.html        ← ChatGPT-style chat interface
├── data/build_dataset.py    ← run locally to build training data
├── colab/colab_train.py     ← paste into Google Colab cells
├── kaggle/kaggle_train.py   ← paste into Kaggle notebook cells
├── vercel.json
└── package.json
```

---

## Step 1 — Accounts needed (all free)

- huggingface.co → sign up → Settings → Access Tokens → New Token (Write)
- github.com → sign up
- vercel.com → sign up with GitHub

---

## Step 2 — Build training data

```bash
cd data
pip install requests
python build_dataset.py
```

Creates `train.jsonl` and `val.jsonl`.

---

## Step 3 — Train

**Google Colab:**
1. colab.research.google.com → New notebook
2. Runtime → Change runtime type → T4 GPU → Save
3. Upload `train.jsonl` and `val.jsonl` (Files panel, left sidebar)
4. Open `colab/colab_train.py`
5. Copy Cell 1 → paste in Colab → edit HF_USERNAME and HF_TOKEN → run
6. Copy Cell 2 → run → allow Google Drive access
7. Continue Cell 3 through Cell 11, one at a time, waiting for ✅ each time

**Kaggle:**
1. kaggle.com → New notebook
2. Settings → Accelerator → GPU T4 x2 → Save
3. Add Data → upload train.jsonl and val.jsonl → name the dataset
4. Open `kaggle/kaggle_train.py`
5. Copy Cell 1 → paste → edit HF_USERNAME, HF_TOKEN, DATASET_NAME → run
6. Continue Cell 2 through Cell 11

Cell 8 (training) takes ~1.5 hours. After Cell 11, your model URL will be printed.

---

## Step 4 — Deploy to Vercel

```bash
git init
git add .
git commit -m "sagjibot"
git remote add origin https://github.com/YOUR_USERNAME/sagjibot.git
git push -u origin main
```

1. vercel.com → Add New Project → Import your repo → Deploy
2. Settings → Environment Variables → Add:

| Name | Value |
|------|-------|
| `HF_TOKEN` | your Hugging Face token |
| `HF_MODEL_URL` | `https://api-inference.huggingface.co/models/YOUR_USERNAME/sagjibot` |

3. Deployments → Redeploy

Your chatbot is live.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| CUDA out of memory | Set `BATCH = 1` in Cell 1 |
| Colab disconnects | Checkpoints auto-save to Drive. Add `resume_from_checkpoint=True` in Cell 8 args |
| Kaggle dataset not found | DATASET_NAME in Cell 1 must exactly match the dataset folder name |
| 503 on first message | Normal — model cold-starts in ~25s on free HF tier. Send again. |
| Vercel env vars not working | Names must be exactly `HF_TOKEN` and `HF_MODEL_URL`. Redeploy after adding. |
| Response in wrong language | Model replies in English unless user writes in Nepali or asks for Nepali |
