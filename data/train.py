# ╔══════════════════════════════════════════════════════════════╗
# ║  SagjiBot — Training Notebook                               ║
# ║  Works on Google Colab (free T4) and Kaggle (free T4/P100)  ║
# ║                                                             ║
# ║  BEFORE STARTING:                                           ║
# ║  • Colab: Runtime → Change runtime type → T4 GPU            ║
# ║  • Kaggle: Settings → Accelerator → GPU T4 x2               ║
# ║  • Upload train.jsonl + val.jsonl to Files panel / dataset  ║
# ║  • Get HF token at huggingface.co/settings/tokens           ║
# ║  • Accept Llama license at:                                 ║
# ║    huggingface.co/meta-llama/Llama-3.2-3B-Instruct          ║
# ╚══════════════════════════════════════════════════════════════╝

# ════════════════════════════════════════════════════
# CELL 1 — Config  (EDIT THESE BEFORE RUNNING)
# ════════════════════════════════════════════════════
import os

HF_USERNAME  = "your-hf-username"       # ← change
HF_TOKEN     = "hf_xxxxxxxxxxxx"        # ← change
HF_REPO      = "sagjibot"

IS_KAGGLE    = os.path.exists("/kaggle")
BASE_MODEL   = "meta-llama/Llama-3.2-3B-Instruct"

if IS_KAGGLE:
    DATASET  = "veggie-data"            # ← your Kaggle dataset name
    TRAIN    = f"/kaggle/input/{DATASET}/train.jsonl"
    VAL      = f"/kaggle/input/{DATASET}/val.jsonl"
    SAVE     = "/kaggle/working/sagjibot"
    print("Platform: Kaggle")
else:
    TRAIN    = "/content/train.jsonl"
    VAL      = "/content/val.jsonl"
    SAVE     = "/content/drive/MyDrive/sagjibot"
    print("Platform: Google Colab")

os.makedirs(SAVE, exist_ok=True)
print(f"Model: {BASE_MODEL}")
print(f"Save : {SAVE}")

# ════════════════════════════════════════════════════
# CELL 2 — Mount Drive  (Colab only, skip on Kaggle)
# ════════════════════════════════════════════════════
if not IS_KAGGLE:
    from google.colab import drive
    drive.mount("/content/drive")
    os.makedirs(SAVE, exist_ok=True)
    print(f"Drive mounted → {SAVE}")

# ════════════════════════════════════════════════════
# CELL 3 — Install packages
# ════════════════════════════════════════════════════
!pip install -q "transformers>=4.45" datasets "peft>=0.12" "trl>=0.11" \
               "accelerate>=0.34" "bitsandbytes>=0.43" sentencepiece huggingface_hub
print("✅ Installed")

# ════════════════════════════════════════════════════
# CELL 4 — Check GPU + set batch size automatically
# ════════════════════════════════════════════════════
import torch

assert torch.cuda.is_available(), (
    "No GPU!\n"
    "Colab  : Runtime → Change runtime type → T4 GPU\n"
    "Kaggle : Settings → Accelerator → GPU T4 x2"
)

mem = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU  : {torch.cuda.get_device_name(0)}")
print(f"VRAM : {mem:.1f} GB")

BATCH      = 4 if mem >= 14 else 2 if mem >= 10 else 1
GRAD_ACCUM = 2 if mem >= 14 else 4 if mem >= 10 else 8
print(f"Batch size      : {BATCH}")
print(f"Grad accumulation: {GRAD_ACCUM}  (effective: {BATCH*GRAD_ACCUM})")

# ════════════════════════════════════════════════════
# CELL 5 — Load tokenizer + model in 4-bit
# ════════════════════════════════════════════════════
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

print("Loading tokenizer…")
tok = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
tok.pad_token    = tok.eos_token
tok.padding_side = "right"

print("Loading model (3-6 min on first run)…")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, quantization_config=bnb, device_map="auto", token=HF_TOKEN
)
print(f"✅ Loaded  {sum(p.numel() for p in model.parameters())/1e9:.1f}B params")

# ════════════════════════════════════════════════════
# CELL 6 — LoRA
# ════════════════════════════════════════════════════
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model = prepare_model_for_kbit_training(model)
lora  = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora)
tr, tot = model.get_nb_trainable_parameters()
print(f"✅ LoRA  trainable {tr:,} / {tot:,}  ({100*tr/tot:.2f}%)")

# ════════════════════════════════════════════════════
# CELL 7 — Load + format dataset
# ════════════════════════════════════════════════════
from datasets import load_dataset

ds = load_dataset("json", data_files={"train":TRAIN,"validation":VAL})
print(f"Train: {len(ds['train'])}  Val: {len(ds['validation'])}")

def fmt(ex):
    out = "<|begin_of_text|>"
    for m in ex["messages"]:
        out += f"<|start_header_id|>{m['role']}<|end_header_id|>\n{m['content']}<|eot_id|>"
    return {"text": out}

ds = ds.map(fmt, remove_columns=["messages"])
print("Sample (400 chars):")
print(ds["train"][0]["text"][:400])

# ════════════════════════════════════════════════════
# CELL 8 — Train  (~1.5 hrs on T4, ~1 hr on P100)
# ════════════════════════════════════════════════════
from transformers import TrainingArguments
from trl import SFTTrainer

args = TrainingArguments(
    output_dir          = f"{SAVE}/checkpoints",
    num_train_epochs    = 3,
    per_device_train_batch_size = BATCH,
    gradient_accumulation_steps = GRAD_ACCUM,
    per_device_eval_batch_size  = BATCH,
    eval_strategy       = "steps",
    eval_steps          = 50,
    save_strategy       = "steps",
    save_steps          = 100,
    save_total_limit    = 2,
    learning_rate       = 2e-4,
    warmup_ratio        = 0.05,
    lr_scheduler_type   = "cosine",
    weight_decay        = 0.01,
    fp16                = True,
    logging_steps       = 10,
    report_to           = "none",
    load_best_model_at_end   = True,
    metric_for_best_model    = "eval_loss",
    greater_is_better        = False,
)

trainer = SFTTrainer(
    model=model, args=args,
    train_dataset=ds["train"], eval_dataset=ds["validation"],
    dataset_text_field="text", tokenizer=tok,
    max_seq_length=1024, packing=False,
)

print("🚀 Training… (watch train_loss go down)")
trainer.train()
print(f"✅ Done — best eval loss: {trainer.state.best_metric:.4f}")

# ════════════════════════════════════════════════════
# CELL 9 — Save
# ════════════════════════════════════════════════════
final = f"{SAVE}/final"
trainer.model.save_pretrained(final)
tok.save_pretrained(final)
print(f"✅ Saved → {final}")

# ════════════════════════════════════════════════════
# CELL 10 — Quick test
# ════════════════════════════════════════════════════
from peft import PeftModel

t = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb, device_map="auto", token=HF_TOKEN)
t = PeftModel.from_pretrained(t, final)
t.eval()

def ask(q):
    p = (f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
         f"You are SagjiBot, a vegetable expert. Reply in the same language as the question.<|eot_id|>"
         f"<|start_header_id|>user<|end_header_id|>\n{q}<|eot_id|>"
         f"<|start_header_id|>assistant<|end_header_id|>\n")
    inp = tok(p, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = t.generate(**inp, max_new_tokens=200, temperature=0.7,
                         do_sample=True, top_p=0.9, pad_token_id=tok.eos_token_id)
    raw = tok.decode(out[0], skip_special_tokens=True)
    return raw.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()

for q in ["What vitamins does spinach have?",
          "How do I grow tomatoes?",
          "पालकमा कुन भिटामिन हुन्छ?",
          "नेपालमा गोलभेडा कहिले रोप्ने?"]:
    print(f"\nQ: {q}\nA: {ask(q)[:300]}\n" + "─"*50)

# ════════════════════════════════════════════════════
# CELL 11 — Push to Hugging Face
# ════════════════════════════════════════════════════
from huggingface_hub import login
login(token=HF_TOKEN)

repo = f"{HF_USERNAME}/{HF_REPO}"
trainer.model.push_to_hub(repo, token=HF_TOKEN)
tok.push_to_hub(repo, token=HF_TOKEN)

print(f"\n✅ Model live → https://huggingface.co/{repo}")
print(f"\n   Copy this URL into Vercel env vars as HF_MODEL_URL:")
print(f"   https://api-inference.huggingface.co/models/{repo}")
