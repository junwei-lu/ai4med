# Encoder-only, Decoder-only, and Encoder-Decoder Models

When people say "Transformer model", they often mean one of three related model families:

- **Encoder-only** models such as BERT
- **Decoder-only** models such as GPT
- **Encoder-decoder** models such as T5 and BART

They all use the Transformer idea, but they are built for different jobs. The main differences are:

- what part of the Transformer they keep
- what information each token is allowed to see
- how they are trained
- what tasks they are best at

This distinction is important because the architecture strongly shapes what the model can learn efficiently.

## The Big Picture

It is useful to think of the three families in a simple way:

- **Encoder-only** models are good at **understanding** text.
- **Decoder-only** models are good at **generating** text one token at a time.
- **Encoder-decoder** models are good at **turning one sequence into another sequence**.

For example:

- sentiment classification: usually encoder-only
- chat and open-ended writing: usually decoder-only
- translation and summarization: usually encoder-decoder

![compare](tf.assets/decoder-encoder.png)

## 1. Encoder-only Models

Encoder-only models keep the **Transformer encoder** stack.

In the encoder, each token can attend to **all other tokens** in the input sentence. This is called **bidirectional attention**.

If the input is:

> The patient did not report chest pain.

the word "pain" can look at "did not report" on its left and "chest" on its right. This makes encoder-only models strong at building a rich representation of the whole sentence.

### Model structure

- input tokens go through token embeddings and positional embeddings
- a stack of encoder blocks processes the full input
- the output is a contextual representation for every token
- for many tasks, we add a task-specific head on top, such as a classifier

### How they are trained

Encoder-only models are usually trained with a **masked language modeling** objective.

We hide some input words and ask the model to predict them.

Example:

> The patient has [MASK] blood pressure.

The model uses both left and right context to guess the missing word.

This training setup is useful for learning deep language representations, but it is not naturally aligned with long free-form text generation.

### When to use encoder-only models

Use encoder-only models when the main goal is to **understand or classify text**.

Common tasks:

- sentence classification
- token classification such as named entity recognition
- information extraction
- semantic similarity
- retrieval and reranking

### Typical examples

- BERT
- RoBERTa
- DistilBERT
- BioBERT

## 2. Decoder-only Models

Decoder-only models keep the **Transformer decoder** stack, but in modern large language models the key feature is **causal masking**.

At position $t$, the model can only attend to tokens from positions $\leq t$. It cannot look into the future.

If the text is:

> The diagnosis is

the model predicts the next token, then uses that new token to predict the following one, and so on.

### Model structure

- input tokens enter a stack of decoder blocks
- each block uses **masked self-attention**
- the hidden state at each position is used to predict the next token
- generation is **autoregressive**, meaning one token is produced at a time

### How they are trained

Decoder-only models are usually trained with a **next-token prediction** objective.

Given tokens $(x_1, x_2, \dots, x_{t-1})$, the model predicts $x_t$.

In practice, the model learns to maximize:

$$
p(x_1, x_2, \dots, x_n) = \prod_{t=1}^{n} p(x_t \mid x_{<t})
$$

This objective matches generation exactly: training and inference follow the same left-to-right pattern.

### When to use decoder-only models

Use decoder-only models when you want the model to **generate text, code, or other sequences**.

Common tasks:

- chat assistants
- question answering in a generative style
- code generation
- report drafting
- instruction following
- open-ended completion

### Typical examples

- GPT family
- Llama family
- Mistral family
- Gemma family

## 3. Encoder-Decoder Models

Encoder-decoder models use **both parts** of the original Transformer.

- the **encoder** reads the input sequence
- the **decoder** generates the output sequence

This is sometimes called a **sequence-to-sequence** or **seq2seq** model.

### Model structure

The encoder first builds a representation of the source input.

Then the decoder generates the target output while attending to:

- previous output tokens through masked self-attention
- the encoder output through **cross-attention**

This is a natural design when the input and output are different but related.

Example:

- input: a long clinical note
- output: a short discharge summary

### How they are trained

Encoder-decoder models are usually trained to map an input sequence to a target sequence.

Examples:

- source sentence $\rightarrow$ translated sentence
- article $\rightarrow$ summary
- question $\rightarrow$ answer

During training, the decoder sees the correct previous output tokens and learns to predict the next one. This is often called **teacher forcing**.

### When to use encoder-decoder models

Use encoder-decoder models when the task is clearly **input-to-output transformation**.

Common tasks:

- translation
- summarization
- paraphrasing
- structured text generation from a source document
- converting one modality-specific text format into another

### Typical examples

- T5
- FLAN-T5
- BART

## Side-by-side Comparison

| Family | Attention pattern | Main training objective | Best at | Common output style |
|---|---|---|---|---|
| Encoder-only | Bidirectional | Masked language modeling | Understanding the input | Labels, token tags, embeddings |
| Decoder-only | Causal, left-to-right | Next-token prediction | Free-form generation | Generated text token by token |
| Encoder-decoder | Encoder is bidirectional, decoder is causal with cross-attention | Sequence-to-sequence training | Mapping one sequence to another | Generated target sequence |

## How Their Training Objectives Differ

The training objective matters as much as the architecture.

### Encoder-only: fill in missing pieces

The model learns by recovering masked tokens. This encourages strong contextual representations because the model must use both sides of the sentence.

### Decoder-only: continue the sequence

The model learns to continue text from left to right. This is a very direct objective for text generation.

### Encoder-decoder: transform one sequence into another

The model learns a conditional distribution:

$$
p(y \mid x)
$$

where $x$ is the input sequence and $y$ is the output sequence.

This is ideal for tasks where the output depends strongly on a source input.

## When Should You Use Which?

Here is a practical rule of thumb.

### Choose encoder-only if:

- you need strong text representations
- you are doing classification or extraction
- you want efficiency for understanding tasks

### Choose decoder-only if:

- you need flexible text generation
- you want one model interface for many prompting tasks
- you are building a chatbot, coding assistant, or general-purpose LLM

### Choose encoder-decoder if:

- the problem is naturally sequence in, sequence out
- the input should be fully understood before generating the output
- the output should stay tightly grounded in the source input

## Why Did Decoder-only Models Become So Successful?

This is an important question. Decoder-only models were not the only Transformer design, but they became dominant for large language models.

There are several reasons.

### 1. Their training objective matches generation directly

Decoder-only models are trained to predict the next token, which is exactly what they do at inference time.

This alignment is powerful. There is no gap between "pretraining task" and "generation task". The model simply learns to continue sequences better and better.

### 2. The objective scales well with large text corpora

Almost any text on the internet can be turned into next-token prediction training data. You do not need labels. You only need sequences of tokens.

That makes it easy to scale data collection and pretraining.

### 3. One interface can solve many tasks

With decoder-only models, many tasks can be written as prompting:

- "Summarize this note: ..."
- "Translate this sentence to Spanish: ..."
- "Answer this question: ..."
- "Write Python code that does ..."

Instead of building a separate head for each task, we can often use one model and phrase the task in text.

### 4. In-context learning emerged at scale

Large decoder-only models showed a surprising ability to learn from examples placed inside the prompt.

For example, if the prompt contains a few question-answer pairs, the model may continue with the correct pattern on a new example. This is called **in-context learning**.

This property made decoder-only models much more general-purpose than many earlier NLP systems.

### 5. They are convenient for product development

From an engineering perspective, decoder-only models provide a simple pattern:

- give the model a prompt
- generate tokens
- stop when a condition is met

This simplicity made them attractive for chat systems, assistants, code tools, and agent-like workflows.

### 6. Instruction tuning and RLHF fit naturally on top

After pretraining, decoder-only models can be further adapted with:

- supervised fine-tuning on instruction-response pairs
- preference learning such as RLHF or DPO

These methods improved helpfulness and alignment without changing the core left-to-right generation framework.

## Important Caveat

Decoder-only models are not always the best choice.

For classification, retrieval, and many structured prediction tasks, encoder-only models can still be more efficient and easier to use. For translation and summarization, encoder-decoder models can still be very strong because their architecture is directly matched to the task.

So the success of decoder-only models does **not** mean the other two families are obsolete. It mostly means decoder-only models became the most flexible general-purpose foundation for language generation.

## Summary

- **Encoder-only** models read the whole input and are strong at understanding tasks.
- **Decoder-only** models generate one token at a time and are strong at open-ended generation.
- **Encoder-decoder** models read an input sequence and generate an output sequence, making them strong for translation and summarization.

If you remember just one idea, remember this:

> the architecture and training objective should match the task you want to solve.

That is why these three model families coexist, and why choosing the right one matters.