# ╔══════════════════════════════════════════════════════════════════╗
# ║  SagjiBot — Kaggle Training                                      ║
# ║                                                                  ║
# ║  SETUP BEFORE RUNNING:                                           ║
# ║  1. New Notebook → Settings (right panel) → Accelerator          ║
# ║     → GPU T4 x2 (or P100) → Save                                 ║
# ║  2. Add Data → Upload Dataset → upload train.jsonl + val.jsonl   ║
# ║     Name your dataset (e.g. "sagjibot-data") → remember this name║
# ║  3. Edit Cell 1: add HF_USERNAME, HF_TOKEN, and DATASET_NAME     ║
# ║  4. Run cells ONE BY ONE — wait for output before next cell       ║
# ╚══════════════════════════════════════════════════════════════════╝


# ════════════════════════════════════════════════════════════════
# CELL 1 — Configuration
# ════════════════════════════════════════════════════════════════
# EDIT THESE THREE LINES

HF_USERNAME  = "your-hf-username"        # your Hugging Face username
HF_TOKEN     = "hf_xxxxxxxxxxxxxxxxxxxx" # your Hugging Face token (Write access)
DATASET_NAME = "sagjibot-data"           # the name you gave your Kaggle dataset

# These stay as-is
HF_REPO    = "sagjibot"
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
TRAIN_FILE = f"/kaggle/input/{DATASET_NAME}/train.jsonl"
VAL_FILE   = f"/kaggle/input/{DATASET_NAME}/val.jsonl"
SAVE_DIR   = "/kaggle/working/sagjibot"

import os
os.makedirs(f"{SAVE_DIR}/checkpoints", exist_ok=True)
os.makedirs(f"{SAVE_DIR}/final",       exist_ok=True)

print("✅ Config set")
print(f"   Model      : {BASE_MODEL}")
print(f"   Train file : {TRAIN_FILE}")
print(f"   Val file   : {VAL_FILE}")
print(f"   Save dir   : {SAVE_DIR}")
print(f"   HF target  : huggingface.co/{HF_USERNAME}/{HF_REPO}")


# ════════════════════════════════════════════════════════════════
# CELL 2 — Verify data files exist
# If this fails, your dataset name in Cell 1 is wrong
# ════════════════════════════════════════════════════════════════
import os

if not os.path.exists(TRAIN_FILE):
    available = os.listdir(f"/kaggle/input/") if os.path.exists("/kaggle/input/") else []
    raise FileNotFoundError(
        f"train.jsonl not found at: {TRAIN_FILE}\n"
        f"Available datasets in /kaggle/input/: {available}\n"
        f"Update DATASET_NAME in Cell 1 to match one of these."
    )

if not os.path.exists(VAL_FILE):
    raise FileNotFoundError(
        f"val.jsonl not found at: {VAL_FILE}\n"
        f"Make sure you uploaded both train.jsonl and val.jsonl in your dataset."
    )

print("✅ Data files found")
print(f"   {TRAIN_FILE}")
print(f"   {VAL_FILE}")


# ════════════════════════════════════════════════════════════════
# CELL 3 — Install packages
# ════════════════════════════════════════════════════════════════
import subprocess
result = subprocess.run([
    "pip", "install", "-q",
    "transformers>=4.45.0",
    "datasets>=2.20.0",
    "peft>=0.12.0",
    "trl>=0.11.0",
    "accelerate>=0.34.0",
    "bitsandbytes>=0.43.0",
    "sentencepiece",
    "huggingface_hub>=0.24.0",
], capture_output=True, text=True)

if result.returncode != 0:
    print("Install errors:", result.stderr[-500:])
else:
    print("✅ All packages installed")


# ════════════════════════════════════════════════════════════════
# CELL 4 — Verify GPU
# ════════════════════════════════════════════════════════════════
import torch

if not torch.cuda.is_available():
    raise RuntimeError(
        "No GPU detected!\n"
        "Go to Settings (right panel) → Accelerator → GPU T4 x2 → Save\n"
        "Then restart the kernel and run from Cell 1 again."
    )

gpu_name = torch.cuda.get_device_name(0)
vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9

print(f"✅ GPU ready: {gpu_name}  |  {vram_gb:.1f} GB VRAM")

# Auto-configure batch size
if vram_gb >= 14:
    BATCH, GRAD_ACCUM = 4, 2
elif vram_gb >= 10:
    BATCH, GRAD_ACCUM = 2, 4
else:
    BATCH, GRAD_ACCUM = 1, 8

print(f"   Batch size       : {BATCH}")
print(f"   Grad accumulation: {GRAD_ACCUM}")
print(f"   Effective batch  : {BATCH * GRAD_ACCUM}")


# ════════════════════════════════════════════════════════════════
# CELL 5 — Load Qwen model in 4-bit
# Takes 3–6 minutes. Ignore deprecation warnings.
# ════════════════════════════════════════════════════════════════
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

print(f"Loading {BASE_MODEL}...")
print("  Downloading model (~6GB on first run — takes 3–6 minutes)")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

total_params = sum(p.numel() for p in model.parameters()) / 1e9
print(f"✅ Model loaded: {total_params:.1f}B params  |  "
      f"{torch.cuda.memory_allocated(0)/1e9:.1f} GB used")


# ════════════════════════════════════════════════════════════════
# CELL 6 — Apply LoRA
# ════════════════════════════════════════════════════════════════
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
trainable, total = model.get_nb_trainable_parameters()

print(f"✅ LoRA applied")
print(f"   Trainable: {trainable:,}  ({100 * trainable / total:.3f}% of {total:,})")


# ════════════════════════════════════════════════════════════════
# CELL 7 — Load and format dataset
# ════════════════════════════════════════════════════════════════
from datasets import load_dataset

dataset = load_dataset(
    "json",
    data_files={"train": TRAIN_FILE, "validation": VAL_FILE},
)

print(f"✅ Dataset loaded")
print(f"   Training  : {len(dataset['train'])} examples")
print(f"   Validation: {len(dataset['validation'])} examples")

def format_for_qwen(example):
    """
    Format messages into Qwen 2.5 chat template.
    <|im_start|>role\ncontent<|im_end|>\n
    """
    text = ""
    for msg in example["messages"]:
        text += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    return {"text": text}

dataset = dataset.map(
    format_for_qwen,
    remove_columns=["messages"],
    desc="Formatting",
)

print(f"\n   Sample (first 400 chars):")
print(f"   {dataset['train'][0]['text'][:400]}\n   ...")


# ════════════════════════════════════════════════════════════════
# CELL 8 — Train
# Kaggle sessions last up to 12 hours — plenty of time.
# Expected: ~1 hour on P100, ~1.5 hours on T4
# ════════════════════════════════════════════════════════════════
from transformers import TrainingArguments
from trl import SFTTrainer

training_args = TrainingArguments(
    output_dir                  = f"{SAVE_DIR}/checkpoints",
    num_train_epochs            = 3,
    per_device_train_batch_size = BATCH,
    gradient_accumulation_steps = GRAD_ACCUM,
    per_device_eval_batch_size  = BATCH,
    eval_strategy               = "steps",
    eval_steps                  = 50,
    save_strategy               = "steps",
    save_steps                  = 100,
    save_total_limit            = 3,
    learning_rate               = 2e-4,
    warmup_ratio                = 0.05,
    lr_scheduler_type           = "cosine",
    weight_decay                = 0.01,
    fp16                        = True,
    logging_steps               = 10,
    report_to                   = "none",
    load_best_model_at_end      = True,
    metric_for_best_model       = "eval_loss",
    greater_is_better           = False,
    dataloader_num_workers      = 2,
    group_by_length             = True,
)

trainer = SFTTrainer(
    model              = model,
    args               = training_args,
    train_dataset      = dataset["train"],
    eval_dataset       = dataset["validation"],
    dataset_text_field = "text",
    tokenizer          = tokenizer,
    max_seq_length     = 1024,
    packing            = False,
)

total_steps = len(dataset["train"]) // (BATCH * GRAD_ACCUM) * 3
print(f"🚀 Training started")
print(f"   Approx steps: {total_steps:,}")
print(f"   Est. time   : ~1 hr on P100, ~1.5 hrs on T4")
print(f"   Checkpoints : {SAVE_DIR}/checkpoints")
print(f"   Watch 'train_loss' — should decrease each step\n")

trainer.train()
print(f"\n✅ Training complete!  Best eval loss: {trainer.state.best_metric:.4f}")


# ════════════════════════════════════════════════════════════════
# CELL 9 — Save
# ════════════════════════════════════════════════════════════════
final_path = f"{SAVE_DIR}/final"
trainer.model.save_pretrained(final_path)
tokenizer.save_pretrained(final_path)

print(f"✅ Model saved to: {final_path}")
print(f"   Files: {os.listdir(final_path)}")


# ════════════════════════════════════════════════════════════════
# CELL 10 — Test
# ════════════════════════════════════════════════════════════════
from peft import PeftModel

print("Loading trained model for testing...")
test_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
test_model = PeftModel.from_pretrained(test_model, final_path)
test_model.eval()

def ask(question, max_tokens=250):
    prompt = (
        f"<|im_start|>system\n"
        f"You are SagjiBot, an expert vegetable assistant. "
        f"Reply in English by default. Only reply in Nepali if the user writes "
        f"in Nepali or explicitly asks for Nepali.<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768).to("cuda")
    with torch.no_grad():
        output_ids = test_model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
        )
    full = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    if "<|im_start|>assistant\n" in full:
        resp = full.split("<|im_start|>assistant\n")[-1]
        resp = resp.replace("<|im_end|>", "").strip()
    else:
        resp = full.split(question)[-1].strip()
    return resp

tests = [
    "What vitamins does spinach have?",
    "When should I plant tomatoes in Nepal?",
    "How do I wash vegetables to remove pesticides?",
    "पालकमा कुन भिटामिन हुन्छ?",
    "Tell me about karela in Nepali",
]

print("\n" + "=" * 60)
print("  TEST RESULTS")
print("=" * 60)
for q in tests:
    print(f"\nQ: {q}")
    print(f"A: {ask(q)[:400]}")
    print("─" * 45)


# ════════════════════════════════════════════════════════════════
# CELL 11 — Push to Hugging Face
# Kaggle output files are temporary — MUST push to HF before
# the session ends or you lose the model
# ════════════════════════════════════════════════════════════════
from huggingface_hub import login

print("Pushing to Hugging Face (this is important — do not skip)...")
login(token=HF_TOKEN)

repo_id = f"{HF_USERNAME}/{HF_REPO}"
trainer.model.push_to_hub(repo_id, token=HF_TOKEN)
tokenizer.push_to_hub(repo_id, token=HF_TOKEN)

print(f"\n✅ Model pushed successfully!")
print(f"\n   Model page  : https://huggingface.co/{repo_id}")
print(f"\n   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(f"   Go to Vercel → Your project → Settings")
print(f"   → Environment Variables → Add:")
print(f"")
print(f"   Name : HF_TOKEN")
print(f"   Value: {HF_TOKEN[:10]}...")
print(f"")
print(f"   Name : HF_MODEL_URL")
print(f"   Value: https://api-inference.huggingface.co/models/{repo_id}")
print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(f"\n   Then click Redeploy in Vercel. Your chatbot is live!")
