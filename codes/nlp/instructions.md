I would teach it as one running story: build a tiny clinical GPT on one dataset, then post-train it twice. That keeps the class coherent and beginner friendly.

Recommended dataset

I would use AGBonnet/augmented-clinical-notes on Hugging Face. It has 30,000 rows in a single train split, uses an MIT license, and each row contains four aligned views of the same case: note, full_note, conversation, and summary. The dataset card says the clinical notes come from PMC-Patients, the dialogues come from NoteChat, and the structured patient summaries were added afterward; it also explicitly warns that the dialogue is synthetic and not fully realistic. That makes it excellent for teaching mechanics end to end, while still being honest that it is an EHR-like teaching set, not a real hospital deployment benchmark.  ￼

For the main post-training task, I would use note -> summary JSON, not conversation -> summary. TRL’s SFTTrainer supports prompt-completion datasets and, for prompt-completion, computes loss on the completion by default. GRPOTrainer is also simplest in the standard format because prompts and completions are plain strings, and extra dataset columns can be passed into custom reward functions. I would keep conversation -> summary as an optional bonus notebook at the end.  ￼

Hugging Face stack to use

You can do the whole coding track with Hugging Face tools:
	•	datasets to load Hub data and slice it into small classroom subsets such as train[:10%].  ￼
	•	tokenizers to train a domain tokenizer directly from an iterator over your dataset.  ￼
	•	transformers to build a tiny GPT-style causal LM from GPT2Config. Hugging Face’s causal LM docs define this as next-token prediction with attention only to the left context.  ￼
	•	trl for both SFTTrainer and GRPOTrainer. TRL’s own docs now present both as first-class post-training APIs.  ￼
	•	peft as an optional add-on if later you swap the tiny scratch model for a small pretrained base model and want LoRA instead of full fine-tuning. PEFT is specifically aimed at reducing trainable parameters for large models.  ￼

The teaching storyline

Use one dataset, one model family, one downstream task:
	1.	Unstructured input: note
	2.	Optional long-context variant: full_note
	3.	Structured target: summary
	4.	Optional conversational extension: conversation

So the model story becomes:
	•	Tokenizer training: train on note + summary
	•	Pretraining: next-token prediction on note
	•	SFT: note -> summary
	•	GRPO: same note -> summary prompt, but now optimize simple rewards

That is the cleanest beginner path because students never have to wonder why the task changed halfway through.

3-hour plan

1) 0:00–0:20 — Data tour and framing

Load a small slice such as train[:3000], then create your own train/validation/test split in code because the Hub dataset exposes a single train split. Show one example with:
	•	note
	•	full_note
	•	summary
	•	optional conversation

Teaching goal: students see that one clinical case can appear as raw text, longer raw text, and structured JSON. That immediately motivates tokenization, embeddings, and post-training.  ￼

2) 0:20–0:45 — Tokenization, token IDs, embeddings

Do two comparisons:
	•	a general tokenizer on a few clinical terms
	•	a small domain tokenizer trained on your class subset

This is where you connect tokens -> IDs -> embedding lookup. A nice concrete lesson is that GPT2Config exposes vocab_size and n_embd, so students can see that changing the tokenizer changes the embedding table size directly.  ￼

Keep the live tokenizer simple:

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

ds = load_dataset("AGBonnet/augmented-clinical-notes", split="train[:3000]")

def corpus():
    for ex in ds:
        yield ex["note"]
        yield ex["summary"]

tok = Tokenizer(BPE(unk_token="[UNK]"))
tok.pre_tokenizer = Whitespace()
trainer = BpeTrainer(vocab_size=8000, special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"])
tok.train_from_iterator(corpus(), trainer=trainer, length=2 * len(ds))

This matches Hugging Face’s recommended pattern: build a tokenizer, train it from files or an iterator, and keep special tokens explicit.  ￼

3) 0:45–1:15 — Attention and the tiny transformer

Build a tiny GPT from scratch. I would use something like:
	•	vocab_size = len(tokenizer)
	•	n_positions = 256
	•	n_embd = 256
	•	n_layer = 4
	•	n_head = 4

The point is not performance; the point is visibility. Hugging Face’s GPT-2 docs expose exactly these architectural knobs in GPT2Config, and GPT2LMHeadModel is the GPT-2 transformer plus an LM head on top, with the LM head tied to the input embeddings. That is a very nice way to connect embeddings, attention blocks, and next-token prediction in one diagram and one code cell. With output_attentions=True, the model returns attention tensors of shape (batch, heads, seq_len, seq_len), so you can visualize one head on a real clinical note.  ￼

4) 1:15–1:45 — Pretraining by next-token prediction

Now pretrain the tiny model on note only. Use this as the concrete implementation of your lecture’s “from embeddings and attention to next token prediction” section.

The goal is not a powerful model. The goal is that students see:
	•	raw clinical text goes in
	•	causal masking means the model only uses left context
	•	loss goes down
	•	generations become more clinical-sounding after a few steps

Use full_note only as an optional discussion point about context windows, not as the live training corpus. It makes the live notebook much cleaner. Hugging Face’s causal LM guide is exactly the training setup you want here.  ￼

5) 1:45–2:20 — SFT on structured extraction

Now convert the same dataset into prompt-completion pairs:

def to_sft(ex):
    prompt = (
        "Convert the following clinical note into a structured JSON medical record.\n\n"
        f"Clinical note:\n{ex['note']}\n\nJSON:\n"
    )
    return {"prompt": prompt, "completion": ex["summary"]}

Then fine-tune with SFTTrainer. This is the cleanest way to show the difference between:
	•	pretraining: “continue the text”
	•	SFT: “follow the instruction and output the JSON schema”

Because TRL’s SFTTrainer explicitly supports prompt-completion data and computes loss on the completion by default, this stage is pedagogically very clean.  ￼

Evaluation should stay beginner friendly:
	•	valid JSON rate
	•	required top-level key coverage
	•	maybe exact match on 2–3 fields only

Do not try to score every medical field perfectly in class.

6) 2:20–2:45 — GRPO with simple rule-based rewards

Reuse the same prompt, but now pass the gold summary as an extra dataset column:

def to_grpo(ex):
    prompt = (
        "Convert the following clinical note into a structured JSON medical record.\n\n"
        f"Clinical note:\n{ex['note']}\n\nJSON:\n"
    )
    return {"prompt": prompt, "gold_summary": ex["summary"]}

Then use three tiny rewards:
	1.	JSON validity reward — does it parse?
	2.	Schema reward — are the required top-level keys present?
	3.	Reference overlap reward — compare parsed keys against gold_summary

This is the right beginner-friendly RL stage because GRPOTrainer supports custom Python reward functions, and for standard-format data the completions are just strings. The docs also state that extra dataset columns besides prompt are passed into the reward function signature, which is exactly what you need for gold_summary. For class, leave beta=0.0; the docs note that with beta=0.0 the reference model is not loaded, which reduces memory use and speeds training.  ￼

7) 2:45–3:00 — Compare the three models

End with one held-out note and show three outputs:
	•	pretrained model: rambling clinical continuation
	•	SFT model: mostly structured JSON
	•	GRPO model: cleaner JSON with better schema compliance

That comparison makes the whole course click.

Notebook structure I would actually prepare
	1.	01_data_and_tokenizer.ipynb
Load dataset, inspect one example, compare tokenizers, train small medical tokenizer.
	2.	02_tiny_transformer_and_pretrain.ipynb
Build tiny GPT from GPT2Config, show embeddings and attention, pretrain on note.
	3.	03_sft_note_to_json.ipynb
Convert note -> summary, train with SFTTrainer, evaluate JSON validity.
	4.	04_grpo_json_rewards.ipynb
Reuse the SFT prompt, add rule-based rewards, run GRPOTrainer, compare outputs.

I would also provide precomputed checkpoints at the start of notebooks 3 and 4. That way the lecture does not depend on live training speed.

What to simplify on purpose

Do not make the class harder than it needs to be.
	•	Do not use full_note for live training.
	•	Do not train a separate reward model.
	•	Do not introduce human preference data collection.
	•	Do not mix genomics into the live coding track.
	•	Do not chase model quality; chase conceptual continuity.

The dataset card itself notes that the data is synthetic in important places and English-only, so this should be framed as a teaching pipeline for clinical-note LLMs, not a clinically validated system.  ￼

Bottom line

The cleanest beginner-friendly design is:

train a tiny GPT on note, then post-train it to emit summary JSON, and keep conversation as an optional extension.

That gives you one coherent story for tokenization, embeddings, attention, causal pretraining, SFT, and GRPO using a single Hugging Face dataset and a single task family.