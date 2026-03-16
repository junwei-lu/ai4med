I would make day 2 much narrower than day 1: one SNP dataset, one main foundation-model workflow, one long-context demo.

The best main dataset is InstaDeepAI/genomics-long-range-benchmark, using the variant_effect_causal_eqtl task. It is a binary SNP-classification task where each example contains a reference sequence, an alternate sequence, a tissue, chromosome, position, and distance to nearest TSS, with the label indicating whether the variant has a causal effect on gene expression. The dataset card reports 88,717 train and 8,846 test examples, uses chromosome-held-out splits, and supports adjustable sequence_length directly in load_dataset, which is ideal for Colab-sized labs.  ￼

I would choose this eQTL task over a disease/pathogenicity task because it connects naturally to regulatory SNPs, tissue specificity, and Enformer-style long-range modeling. If you need a simpler backup, the same benchmark’s variant_effect_pathogenic_clinvar task removes tissue and keeps only ref/alt sequence plus a binary pathogenicity label; the card reports 38,634 train and 1,018 test examples.  ￼

To connect this day to your first LLM lecture, I would keep the same intellectual spine—tokenization, embeddings, attention, pretraining, transfer learning—but switch the biological setting. Instead of words and subwords, students see DNA tokens; instead of decoder-only next-token prediction, they see encoder-style masked modeling or sequence-to-function models; instead of SFT/RL, the downstream step is supervised SNP-effect prediction plus attribution. The Nucleotide Transformer docs describe encoder-only transformers with 6-mer tokenization and a vocabulary of about 4.1k tokens, and the official 50M v2 model card shows a straightforward Hugging Face AutoTokenizer / AutoModelForMaskedLM workflow with standard BERT-style masking. The human-variation nucleotide-transformer-500m-1000g model is a strong conceptual example because it was pretrained on 3,202 diverse human genomes containing about 125M mutations, but for live Colab I would use the smaller 50M v2 model.  ￼

For the long-context demo section, Enformer is the cleanest counterpoint. Its official README says the usage Colab can make predictions, compute contribution scores, predict the effect of genetic variants, and score VCFs, and the same README states that inference uses 393,216-bp one-hot DNA inputs. DeepSEA is the short-context contrast: public docs describe it as a single-nucleotide-sensitive model for regulatory/chromatin features, and the public Beluga/fastISM Colab uses 2,000-bp inputs and predicts 2,002 chromatin features. If you want a newer DeepMind example, AlphaGenome also has an official quick-start Colab, but it uses an API key, so I would keep it as an optional closing demo rather than the core lab.  ￼

Three-hour course plan

0:00–0:20 — Recap from day 1, then reframe for genomics

Start by explicitly mapping day 1 to day 2:
	•	text tokenization -> DNA tokenization
	•	text embeddings -> k-mer or base embeddings
	•	transformer attention -> long-range regulatory context
	•	pretraining -> masked DNA modeling or sequence-to-function pretraining
	•	fine-tuning -> SNP effect prediction

The main message here is: the transformer ideas stay; the biological meaning changes.

I would also explain why SNP tasks are different from text tasks:
	•	alphabet is tiny: A/C/G/T
	•	context can be huge
	•	a one-base edit can matter
	•	the same variant may behave differently by tissue

0:20–0:45 — Data lab: inspect the SNP task

Load variant_effect_causal_eqtl with a small context first, such as sequence_length=512 or 1024, and only a classroom subset such as the first few thousand training examples.

What students should do:
	•	print one example
	•	verify that ref and alt sequences differ at the SNP site
	•	inspect tissue and distance-to-TSS
	•	inspect label balance on the subset
	•	understand why chromosome-held-out evaluation is better than random splitting

This section is important because the dataset already teaches three core genomics ideas:
	1.	a SNP is naturally represented as ref vs alt
	2.	regulatory labels are often tissue-dependent
	3.	evaluation should respect genomic locality

0:45–1:15 — Tokenization and genomic foundation models

This is the day-2 equivalent of your day-1 embeddings/tokenizer lecture.

Teach two representation styles:

Style A: genomic language models
	•	DNABERT / Nucleotide Transformer style
	•	tokenize DNA into k-mers
	•	learn embeddings
	•	use masked language modeling

Style B: sequence-to-function models
	•	DeepSEA / Enformer style
	•	use one-hot DNA directly
	•	predict functional tracks or variant effects

Hands-on:
	•	use the Nucleotide Transformer tokenizer
	•	tokenize one ref sequence and one alt sequence
	•	show that one SNP changes only local k-mers
	•	connect token IDs to embeddings, just like word IDs on day 1

Conceptually, I would emphasize one contrast with day 1:
	•	day 1 used a decoder/generative story
	•	day 2 should use an encoder/prediction story

I would not teach RL on day 2. The right analogue here is variant scoring and interpretation, not GRPO/PPO.

1:15–1:55 — Main lab: frozen foundation-model embeddings for SNP prediction

This should be the main coding exercise.

Use:
	•	InstaDeepAI/nucleotide-transformer-v2-50m-multi-species
	•	frozen model
	•	extract embeddings for ref and alt separately
	•	build a simple downstream classifier

A beginner-friendly feature design is:
	•	mean pooled embedding of ref
	•	mean pooled embedding of alt
	•	delta embedding = alt - ref
	•	tissue one-hot or learned tissue embedding
	•	distance-to-TSS as one scalar

Then train:
	•	logistic regression, or
	•	a tiny MLP

Evaluation:
	•	AUROC
	•	AUPRC
	•	confusion matrix
	•	optional slice by tissue or by distance to TSS

This is the cleanest teaching moment of the day because students see the full transfer-learning recipe:
pretrained model -> frozen embeddings -> small supervised head.

1:55–2:20 — Lightweight fine-tuning

Now upgrade the frozen baseline.

I would keep this simple:
	•	either fine-tune only a classification head
	•	or do a small LoRA/PEFT adaptation on top of the Nucleotide Transformer

Do this on a reduced subset so it actually runs in class.

What students should learn:
	•	frozen embeddings are the easiest baseline
	•	light fine-tuning often improves task fit
	•	in genomics, small labeled datasets make transfer learning especially valuable

I would not do full end-to-end large-model fine-tuning in class.

2:20–2:45 — Demo block: Enformer and DeepSEA

This is where you broaden from “genomic language model” to “sequence foundation model.”

For Enformer, I would show:
	•	long-context input
	•	variant effect scoring as prediction(alt) - prediction(ref)
	•	contribution scores / attribution
	•	why distal enhancers motivate long receptive fields

You can either run the official usage Colab or show its precomputed variant effect files for common 1000 Genomes variants. The official README explicitly provides both a usage notebook and precomputed 1000G variant-effect scores.  ￼

For DeepSEA, I would use it as the short-context baseline:
	•	same SNP-effect idea
	•	much shorter sequence window
	•	chromatin-feature prediction rather than broader long-range gene regulation

That comparison makes the architecture choice intuitive:
	•	DeepSEA when short-range regulatory grammar is enough
	•	Enformer when long-range context matters

2:45–3:00 — Wrap-up and discussion

End with a direct comparison to day 1:
	•	Day 1 question: “How do we train a model to produce useful biomedical text?”
	•	Day 2 question: “How do we train or use a model to score the effect of a DNA variant?”

The final takeaways should be:
	•	tokenization and embeddings still matter
	•	attention still matters
	•	pretraining still matters
	•	but the downstream output is no longer text generation; it is variant effect prediction

What content I would explicitly teach under this theme

I would keep these five themes:
	1.	DNA tokenization and embeddings
k-mers, ref/alt windows, and how SNPs alter local tokens.
	2.	Pretraining objective
masked DNA modeling for transformer encoders; contrast with day-1 next-token prediction.
	3.	Transfer learning for SNP tasks
frozen probe first, lightweight fine-tuning second.
	4.	Variant effect scoring
ref vs alt, delta embeddings, tissue conditioning, distance to TSS.
	5.	Interpretability
saliency, contribution scores, in silico mutagenesis.

I would drop or minimize:
	•	RL-style fine-tuning
	•	training a genome-scale foundation model from scratch
	•	full Enformer training

Notebook plan

I would prepare four notebooks:
	•	01_eqtl_dataset_and_tokenization.ipynb
Load the benchmark, inspect SNP examples, tokenize ref/alt windows.
	•	02_nt_frozen_embeddings_eqtl.ipynb
Extract Nucleotide Transformer embeddings and train a simple classifier.
	•	03_nt_light_finetune_eqtl.ipynb
Add a classification head or LoRA and compare against the frozen baseline.
	•	04_enformer_deepsea_variant_demo.ipynb
Instructor demo of long-context variant scoring and attribution.

Hugging Face also maintains notebooks for fine-tuning the Nucleotide Transformer, including a PEFT/LoRA version, so you can borrow their setup instead of writing everything from scratch.  ￼

Practical teaching notes

Use one main task only in class: variant_effect_causal_eqtl.
Use small live subsets for speed, such as 3k–5k train and 500–1000 test.
Use 512 or 1024 bp windows live, then explain that longer contexts are a modeling question students can explore later.
Ship one precomputed embedding file and one saved fine-tuned checkpoint so the lecture does not depend on Colab speed.

If you want the course to feel slightly more clinical and less regulatory, swap the hands-on task to variant_effect_pathogenic_clinvar, but I would still keep the Enformer demo because it teaches the important idea that SNP effects can depend on long-range sequence context.  ￼

Overall, I would position day 2 as:

“From language models for text to foundation models for sequence: how to represent a SNP, how to transfer a pretrained genome model, and how to interpret variant effects.”
