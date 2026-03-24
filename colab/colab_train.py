# ╔══════════════════════════════════════════════════════════════════╗
# ║  SagjiBot — Google Colab Training                               ║
# ║                                                                  ║
# ║  SETUP BEFORE RUNNING:                                           ║
# ║  1. Runtime → Change runtime type → T4 GPU → Save               ║
# ║  2. Upload train.jsonl and val.jsonl using Files panel (left)    ║
# ║  3. Edit Cell 1: add your HF_USERNAME and HF_TOKEN               ║
# ║  4. Run cells ONE BY ONE in order — wait for ✅ before next       ║
# ╚══════════════════════════════════════════════════════════════════╝


# ════════════════════════════════════════════════════════════════
# CELL 1 — Configuration
# ════════════════════════════════════════════════════════════════
# EDIT THESE TWO LINES BEFORE RUNNING ANYTHING

HF_USERNAME = "your-hf-username"        # your Hugging Face username
HF_TOKEN    = "hf_xxxxxxxxxxxxxxxxxxxx" # your Hugging Face token (Write access)

# These stay as-is
HF_REPO    = "sagjibot"
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
TRAIN_FILE = "/content/train.jsonl"
VAL_FILE   = "/content/val.jsonl"
SAVE_DIR   = "/content/drive/MyDrive/sagjibot"

import os
os.makedirs(SAVE_DIR, exist_ok=True)

print("✅ Config set")
print(f"   Model : {BASE_MODEL}")
print(f"   Save  : {SAVE_DIR}")
print(f"   HF    : huggingface.co/{HF_USERNAME}/{HF_REPO}")


# ════════════════════════════════════════════════════════════════
# CELL 2 — Mount Google Drive
# This saves your model automatically — if Colab disconnects,
# your progress is safe in Drive
# ════════════════════════════════════════════════════════════════
import os
import shutil # Import shutil for rmtree
from google.colab import drive

# Define mount point
mount_point = "/content/drive"

# Try to unmount first, in case it was already mounted (e.g., from a previous run)
try:
    drive.flush_and_unmount()
except ValueError:
    pass # Drive might not be mounted initially, ignore error

# Ensure the mount point directory is empty or non-existent before mounting
# If the directory exists and contains files, drive.mount will fail.
# So, remove it completely and then recreate it as an empty directory.
if os.path.exists(mount_point):
    if os.path.isdir(mount_point):
        print(f"Removing existing directory {mount_point} to ensure a clean mount point.")
        shutil.rmtree(mount_point)
    else:
        # If it exists but is not a directory (e.g., a file), remove it.
        print(f"Removing existing non-directory item at {mount_point}.")
        os.remove(mount_point)

# Recreate the empty directory for mounting
os.makedirs(mount_point, exist_ok=True)

# Now attempt to mount Google Drive
drive.mount(mount_point, force_remount=True)


os.makedirs(f"{SAVE_DIR}/checkpoints", exist_ok=True)
print("✅ Google Drive mounted")
print(f"   Checkpoints will save to: {SAVE_DIR}/checkpoints")


# ════════════════════════════════════════════════════════════════
# CELL 3 — Install packages
# Wait for this to finish completely before running Cell 4
# ════════════════════════════════════════════════════════════════
import subprocess
subprocess.run([
    "pip", "install", "-q",
    "transformers>=4.45.0",
    "datasets>=2.20.0",
    "peft>=0.12.0",
    "trl>=0.11.0",
    "accelerate>=0.34.0",
    "bitsandbytes>=0.43.0",
    "sentencepiece",
    "huggingface_hub>=0.24.0",
], check=True)
print("✅ All packages installed")


# ════════════════════════════════════════════════════════════════
# CELL 4 — Verify GPU
# Must show Tesla T4 or better. If it says CPU, go back and
# set Runtime → T4 GPU and restart
# ════════════════════════════════════════════════════════════════
import torch

if not torch.cuda.is_available():
    raise RuntimeError(
        "No GPU detected!\n"
        "Go to: Runtime → Change runtime type → T4 GPU → Save\n"
        "Then: Runtime → Restart session\n"
        "Then start from Cell 1 again"
    )

gpu_name = torch.cuda.get_device_name(0)
vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9

print(f"✅ GPU ready: {gpu_name}  |  {vram_gb:.1f} GB VRAM")

# Auto-set batch size based on available memory
if vram_gb >= 14:
    BATCH, GRAD_ACCUM = 4, 2
elif vram_gb >= 10:
    BATCH, GRAD_ACCUM = 2, 4
else:
    BATCH, GRAD_ACCUM = 1, 8

print(f"   Batch size      : {BATCH}")
print(f"   Grad accumulation: {GRAD_ACCUM}")
print(f"   Effective batch : {BATCH * GRAD_ACCUM}")


# ════════════════════════════════════════════════════════════════
# CELL 5 — Load Qwen model in 4-bit quantization
# Takes 3–6 minutes. Normal to see warning messages — ignore them.
# Wait for "✅ Model loaded" before continuing.
# ════════════════════════════════════════════════════════════════
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

print(f"Loading {BASE_MODEL}...")
print("  This takes 3–6 minutes on first run (downloading ~6GB)")

# 4-bit config: reduces model from ~12GB to ~4GB in VRAM
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load model (Qwen is open access — no token needed)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

total_params = sum(p.numel() for p in model.parameters()) / 1e9
print(f"✅ Model loaded: {total_params:.1f}B parameters")
print(f"   Memory used: {torch.cuda.memory_allocated(0)/1e9:.1f} GB")


# ════════════════════════════════════════════════════════════════
# CELL 6 — Apply LoRA (Low-Rank Adaptation)
# LoRA only trains ~1% of parameters, making fine-tuning
# possible on a free GPU without running out of memory
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
print(f"   Trainable params : {trainable:,}")
print(f"   Total params     : {total:,}")
print(f"   Training %       : {100 * trainable / total:.3f}%")


# ════════════════════════════════════════════════════════════════
# CELL 7 — Load and format dataset
# Make sure train.jsonl and val.jsonl are uploaded in the
# Files panel before running this cell
# ════════════════════════════════════════════════════════════════
from datasets import load_dataset

# Verify files exist
if not os.path.exists(TRAIN_FILE):
    raise FileNotFoundError(
        f"train.jsonl not found at {TRAIN_FILE}\n"
        "Upload it using the Files panel on the left sidebar"
    )
if not os.path.exists(VAL_FILE):
    raise FileNotFoundError(
        f"val.jsonl not found at {VAL_FILE}\n"
        "Upload it using the Files panel on the left sidebar"
    )

dataset = load_dataset(
    "json",
    data_files={"train": TRAIN_FILE, "validation": VAL_FILE}
)

print(f"✅ Dataset loaded")
print(f"   Training examples  : {len(dataset['train'])}")
print(f"   Validation examples: {len(dataset['validation'])}")

def format_for_qwen(example):
    """
    Convert messages list to Qwen 2.5 chat format.
    Format: <|im_start|>role\ncontent<|im_end|>\n
    """
    text = ""
    for msg in example["messages"]:
        role    = msg["role"]
        content = msg["content"]
        text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    return {"text": text}

dataset = dataset.map(
    format_for_qwen,
    remove_columns=["messages"],
    desc="Formatting dataset",
)

print(f"\n   Sample (first 500 chars of training example 1):")
print(f"   {dataset['train'][0]['text'][:500]}")
print(f"   ...")


# ════════════════════════════════════════════════════════════════
# CELL 8 — Train the model
# Expected time: ~1.5 hours on T4 GPU
# You will see step logs every 10 steps.
# Watch "train_loss" — it should go DOWN over time (e.g. 2.1 → 0.8)
# Do NOT close this tab while training
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
print(f"   Total steps (approx): {total_steps:,}")
print(f"   Estimated time      : ~1.5 hours on T4")
print(f"   Checkpoints saved to: {SAVE_DIR}/checkpoints")
print(f"   Watch: 'train_loss' should decrease over time\n")

trainer.train()

best_loss = trainer.state.best_metric
print(f"\n✅ Training complete!")
print(f"   Best validation loss: {best_loss:.4f}")


# ════════════════════════════════════════════════════════════════
# CELL 9 — Save the trained model
# ════════════════════════════════════════════════════════════════
final_path = f"{SAVE_DIR}/final"
os.makedirs(final_path, exist_ok=True)

trainer.model.save_pretrained(final_path)
tokenizer.save_pretrained(final_path)

print(f"✅ Model saved to Google Drive")
print(f"   Path: {final_path}")
print(f"   Files: {os.listdir(final_path)}")


# ════════════════════════════════════════════════════════════════
# CELL 10 — Test the trained model
# Uses correct Qwen 2.5 format for inference
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
    """
    Query the trained SagjiBot.
    Uses Qwen 2.5 chat template format.
    """
    prompt = (
        f"<|im_start|>system\n"
        f"You are SagjiBot, an expert vegetable assistant. "
        f"Reply in English by default. Only reply in Nepali if the user writes "
        f"in Nepali or explicitly asks for Nepali.<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=768,
    ).to("cuda")

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

    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)

    # Extract only the assistant's response
    if "<|im_start|>assistant\n" in full_text:
        response = full_text.split("<|im_start|>assistant\n")[-1]
        response = response.replace("<|im_end|>", "").strip()
    else:
        response = full_text.split(question)[-1].strip()

    return response

# Test questions covering all topics
test_cases = [
    ("English - Nutrition",  "What vitamins does spinach have and what are the benefits?"),
    ("English - Growing",    "When should I plant tomatoes in Nepal's Terai region?"),
    ("English - Chemicals",  "How do I wash vegetables to remove pesticides?"),
    ("English - Safety",     "Is it safe to eat potatoes with green skin?"),
    ("Nepali input",         "पालकमा कुन भिटामिन हुन्छ?"),
    ("Nepali input",         "नेपालमा गोलभेडा कहिले रोप्ने?"),
    ("Ask for Nepali",       "Tell me about karela health benefits in Nepali"),
    ("Language default",     "What is neem oil?"),
]

print("\n" + "=" * 65)
print("  SAGJIBOT TEST RESULTS")
print("=" * 65)

for topic, question in test_cases:
    print(f"\n[{topic}]")
    print(f"Q: {question}")
    response = ask(question)
    print(f"A: {response[:400]}")
    if len(response) > 400:
        print("   [truncated...]")
    print("─" * 50)


# ════════════════════════════════════════════════════════════════
# CELL 11 — Push to Hugging Face
# This creates the repo automatically if it doesn't exist.
# After this, your model is live and accessible via API.
# ════════════════════════════════════════════════════════════════
from huggingface_hub import login

print("Logging in to Hugging Face...")
login(token=HF_TOKEN)

repo_id = f"{HF_USERNAME}/{HF_REPO}"
print(f"Pushing model to: {repo_id}")
print("This takes 3–5 minutes...")

trainer.model.push_to_hub(repo_id, token=HF_TOKEN)
tokenizer.push_to_hub(repo_id, token=HF_TOKEN)

print(f"\n✅ Model is live!")
print(f"\n   Model page  : https://huggingface.co/{repo_id}")
print(f"\n   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(f"   Now go to Vercel → Settings → Environment Variables")
print(f"   Add these two variables:")
print(f"")
print(f"   HF_TOKEN     = {HF_TOKEN[:8]}...")
print(f"   HF_MODEL_URL = https://api-inference.huggingface.co/models/{repo_id}")
print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
