# =============================================================================
# VEGGIE AI — Google Colab Fine-tuning Notebook
# =============================================================================
# Instructions:
#   1. Go to colab.research.google.com → New Notebook
#   2. Runtime → Change Runtime Type → T4 GPU → Save
#   3. Upload train_final.jsonl and val_final.jsonl using the Files panel (left sidebar)
#   4. Copy each cell below into a new Colab cell and run them IN ORDER
# =============================================================================


# ─────────────────────────────────────────────────────────────────────────────
# CELL 1 — Mount Google Drive
# Always do this first so checkpoints are saved and survive disconnects
# ─────────────────────────────────────────────────────────────────────────────

from google.colab import drive
drive.mount('/content/drive')

import os
os.makedirs("/content/drive/MyDrive/veggie-model/checkpoints", exist_ok=True)
os.makedirs("/content/drive/MyDrive/veggie-model/final",       exist_ok=True)
print("✅ Google Drive mounted")
print("Model will be saved to: /content/drive/MyDrive/veggie-model/")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 2 — Install required packages
# ─────────────────────────────────────────────────────────────────────────────

!pip install -q transformers==4.40.0 datasets peft trl accelerate bitsandbytes sentencepiece
print("✅ All packages installed")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 3 — Verify GPU
# ─────────────────────────────────────────────────────────────────────────────

import torch

if not torch.cuda.is_available():
    raise RuntimeError("❌ No GPU found! Go to Runtime → Change Runtime Type → T4 GPU")

print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
print(f"   Memory: {round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)} GB")
print(f"   CUDA version: {torch.version.cuda}")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 4 — Load Mistral 7B with 4-bit quantization
# This loads the base model — fits on a free T4 GPU using 4-bit compression
# Takes 3–6 minutes depending on Colab connection speed
# ─────────────────────────────────────────────────────────────────────────────

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

# 4-bit quantization config — reduces 7B model from ~14GB to ~5GB VRAM
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

print(f"Loading tokenizer for {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "right"

print(f"Loading model (this takes 3–6 minutes on first run)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

print("✅ Mistral 7B loaded successfully")
print(f"   Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 5 — Configure LoRA (Low-Rank Adaptation)
# LoRA only trains ~1–2% of weights — this is why training fits on a free GPU
# ─────────────────────────────────────────────────────────────────────────────

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,                    # Rank: higher = more capacity, more memory. 16 is good default.
    lora_alpha=32,           # Scaling: usually 2x the rank
    target_modules=[         # Which layers to add LoRA adapters to
        "q_proj", "k_proj",
        "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

trainable, total = model.get_nb_trainable_parameters()
print(f"✅ LoRA configured")
print(f"   Trainable parameters: {trainable:,} ({100 * trainable / total:.2f}% of total)")
print(f"   Total parameters:     {total:,}")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 6 — Load and inspect your vegetable dataset
# Make sure you uploaded train_final.jsonl and val_final.jsonl in Files panel
# ─────────────────────────────────────────────────────────────────────────────

from datasets import load_dataset

dataset = load_dataset(
    "json",
    data_files={
        "train":      "/content/train_final.jsonl",
        "validation": "/content/val_final.jsonl"
    }
)

print(f"✅ Dataset loaded")
print(f"   Training examples:   {len(dataset['train'])}")
print(f"   Validation examples: {len(dataset['validation'])}")
print(f"\nSample training example:")
print(dataset['train'][0])


# ─────────────────────────────────────────────────────────────────────────────
# CELL 7 — Format conversations into Mistral instruction template
# ─────────────────────────────────────────────────────────────────────────────

def format_conversation(example):
    """Format a messages list into Mistral's [INST] template."""
    messages = example["messages"]
    text = ""
    system_content = ""

    for msg in messages:
        if msg["role"] == "system":
            system_content = msg["content"]
        elif msg["role"] == "user":
            if system_content:
                text += f"<s>[INST] <<SYS>>\n{system_content}\n<</SYS>>\n\n{msg['content']} [/INST] "
                system_content = ""
            else:
                text += f"<s>[INST] {msg['content']} [/INST] "
        elif msg["role"] == "assistant":
            text += f"{msg['content']} </s>"

    return {"text": text}

dataset = dataset.map(format_conversation, remove_columns=["messages"])

print("✅ Dataset formatted for Mistral template")
print(f"\nExample formatted text (first 600 chars):")
print(dataset['train'][0]['text'][:600])
print("...")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 8 — Configure and start training
# Expected time: 1–3 hours on a free T4 GPU
# Watch: Training loss should steadily decrease. Validation loss should too.
# ─────────────────────────────────────────────────────────────────────────────

from transformers import TrainingArguments
from trl import SFTTrainer

CHECKPOINT_DIR = "/content/drive/MyDrive/veggie-model/checkpoints"

training_args = TrainingArguments(
    output_dir=CHECKPOINT_DIR,
    num_train_epochs=3,                  # 3 full passes over the training data
    per_device_train_batch_size=2,       # Lower if OOM error (try 1)
    gradient_accumulation_steps=4,       # Effective batch size = 2 × 4 = 8
    per_device_eval_batch_size=2,
    evaluation_strategy="steps",
    eval_steps=50,                       # Evaluate every 50 steps
    save_strategy="steps",
    save_steps=100,                      # Save checkpoint every 100 steps
    save_total_limit=3,                  # Keep only 3 checkpoints (saves Drive space)
    learning_rate=2e-4,
    weight_decay=0.001,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=10,
    fp16=True,                           # Mixed precision — faster, less memory
    report_to="none",                    # Disable W&B logging
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    dataloader_num_workers=2,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    dataset_text_field="text",
    tokenizer=tokenizer,
    max_seq_length=1024,
    packing=False,
)

print("🚀 Starting training...")
print(f"   Total steps (approx): {len(dataset['train']) // 8 * 3:,}")
print(f"   Checkpoints saving to: {CHECKPOINT_DIR}")
print(f"   Expected time: 1–3 hours on T4 GPU\n")

trainer.train()

print("\n✅ Training complete!")
print(f"   Best validation loss: {trainer.state.best_metric:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 9 — Save the trained model to Google Drive
# ─────────────────────────────────────────────────────────────────────────────

FINAL_SAVE_PATH = "/content/drive/MyDrive/veggie-model/final"

print(f"Saving model to {FINAL_SAVE_PATH}...")
trainer.model.save_pretrained(FINAL_SAVE_PATH)
tokenizer.save_pretrained(FINAL_SAVE_PATH)

print(f"✅ Model saved to Google Drive!")
print(f"   Path: {FINAL_SAVE_PATH}")
print(f"   Files saved: {os.listdir(FINAL_SAVE_PATH)}")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 10 — Test your trained model
# ─────────────────────────────────────────────────────────────────────────────

from peft import PeftModel

print("Loading trained model for testing...")
test_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)
test_model = PeftModel.from_pretrained(test_model, FINAL_SAVE_PATH)
test_model.eval()

def ask(question, max_tokens=300, temperature=0.7):
    """Ask a question to the trained VeggieBot."""
    prompt = (
        f"<s>[INST] <<SYS>>\n"
        f"You are VeggieBot, an expert on vegetables, farming, chemicals, and plant diseases.\n"
        f"<</SYS>>\n\n"
        f"{question} [/INST]"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = test_model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full_text.split("[/INST]")[-1].strip()

# Test all 4 topic areas
test_questions = [
    ("NUTRITION",  "What vitamins does kale have and why are they important?"),
    ("NUTRITION",  "How many calories does broccoli have?"),
    ("GROWING",    "How do I grow garlic from cloves?"),
    ("GROWING",    "What soil pH does spinach prefer?"),
    ("CHEMICALS",  "What is neem oil and how do I use it on tomatoes?"),
    ("CHEMICALS",  "What is the pre-harvest interval and why does it matter?"),
    ("DISEASES",   "How do I identify early blight on tomatoes?"),
    ("DISEASES",   "What causes clubroot in cabbage and how do I treat it?"),
]

print("\n" + "="*70)
print("  VEGGIE BOT — MODEL TEST RESULTS")
print("="*70)

for topic, question in test_questions:
    print(f"\n[{topic}] {question}")
    print("-" * 60)
    answer = ask(question)
    print(answer[:500])
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CELL 11 — Push model to Hugging Face (free hosting)
# Get your token at: huggingface.co/settings/tokens (needs Write access)
# ─────────────────────────────────────────────────────────────────────────────

HF_USERNAME  = "your-hf-username"    # ← Change this
HF_TOKEN     = "hf_xxxxxxxxxxxx"     # ← Change this (keep it secret!)
REPO_NAME    = "veggie-bot"

print(f"Pushing model to: huggingface.co/{HF_USERNAME}/{REPO_NAME}")

trainer.model.push_to_hub(
    f"{HF_USERNAME}/{REPO_NAME}",
    token=HF_TOKEN,
    private=False   # Set True if you want the model private
)
tokenizer.push_to_hub(
    f"{HF_USERNAME}/{REPO_NAME}",
    token=HF_TOKEN
)

print(f"\n✅ Model is live!")
print(f"   URL: https://huggingface.co/{HF_USERNAME}/{REPO_NAME}")
print(f"\n   Use this in your backend:")
print(f"   HF_MODEL_URL = https://api-inference.huggingface.co/models/{HF_USERNAME}/{REPO_NAME}")
