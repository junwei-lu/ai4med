# Improvement Plan

- Add more text narrative. For example, add Section 1 on explain the dataset content 
- Add more details about Tokenization, you can refer to the following content by adapt to the current notebook context:

```markdown
Once you have loaded the tokenizer, you can use the `tokenize` method to convert raw text into tokens.

```python
text = "Hello, how are you?"
tokens = tokenizer.tokenize(text)

print("Tokens:", tokens)
# Example output: ['Hello', ',', 'how', 'are', 'you', '?']
```

Models don't directly understand text—they use numerical representations called token IDs.

```python
# Convert tokens to token IDs
token_ids = tokenizer.encode(text)
print("Token IDs:", token_ids)

# Convert token IDs back to tokens (includes special tokens such as [CLS], [SEP])
tokens_with_special = tokenizer.convert_ids_to_tokens(token_ids)
print("Tokens with special characters:", tokens_with_special)
# Example output: ['[CLS]', 'Hello', ',', 'how', 'are', 'you', '?', '[SEP]']
```


Models require input tensors. Here, we tokenize and convert our input text to PyTorch tensors:

```python
inputs = tokenizer(text, return_tensors='pt')  # 'pt' stands for PyTorch tensors
print(inputs)
# {'input_ids': 
#    tensor([[ 101, 8667,  117, 1293, 1132, 1128,  136,  102]]), 
#  'attention_mask': 
#    tensor([[1, 1, 1, 1, 1, 1, 1, 1]])
# }
```

The `inputs` dictionary contains keys:

- `input_ids`: The token IDs of the input text with dimensions `(batch_size, sequence_length)`.
- `attention_mask`: A binary mask indicating which tokens are real (1) and which are padding (0) with dimensions `(batch_size, sequence_length)`.

You can then pass the `inputs` dictionary to the model using two different methods:

```python
# Option 1: Pass the entire inputs dictionary
outputs = model(**inputs)
# Option 2: Pass the input IDs and attention mask separately
outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
```



**Batch of sentences**

You can also pass batch of sentences to the model.
Tokenizing multiple sentences at once can require padding to ensure consistent input lengths:

```python
texts = [
    "Hello, how are you?",
    "I'm fine, thank you! And you?",
    "I'm not fine."
]

# Pad sentences to match the length of the longest sentence
model_inputs = tokenizer(texts, padding=True, return_tensors='pt')

print(f"Pad token: {tokenizer.pad_token} | Pad token id: {tokenizer.pad_token_id}")
# Print input ids
print(model_inputs['input_ids'])
# Print attention mask
print(model_inputs['attention_mask'])
# Pad token: [PAD] | Pad token id: 0
# tensor([[101,8667,117,1293,1132,1128,136,102,0,0,0,0,0],
#         [101,146,112,182,2503,117,6243,1128,106,1262,1128,136,102],
#         [101,146,112,182,1136,2503,119,102,0,0,0,0,0]])
# tensor([[1,1,1,1,1,1,1,1,0,0,0,0,0],
#         [1,1,1,1,1,1,1,1,1,1,1,1,1],
#         [1,1,1,1,1,1,1,1,0,0,0,0,0]])
```
```

- In the part of # ── Visualization 3: Summary JSON structure ──#. Visualize the word cloud in symptoms, medical examinations, and treatments.

- Add a text to explain why we need those special tokens:     unk_token="[UNK]",
    pad_token="[PAD]",
    bos_token="[BOS]",
    eos_token="[EOS]",
    and code to visualize the special tokens in a sentence.

- Add a text to explain the math procedure `PreTrainedTokenizerFast` organize the tokenizer.

- To explain the formula: Model Input=TokenEmb(𝑥)+PosEmb(position). Add a code part to given a sentence, visualize the tokens, the embedding vector as a heatmap matrix, the position embedding as a heatmap matrix, and their summations as a heatmap matrix.

- In the chunk `# ── Visualization 9: All 4 heads of layer 0 ──`. I want to visualize for all layers following the similar code as follows:

```python
import matplotlib.pyplot as plt
n_layers = len(model_output.attentions)
n_heads = len(model_output.attentions[0][0])
fig, axes = plt.subplots(6, 12)
fig.set_size_inches(18.5*2, 10.5*2)
for layer in range(n_layers):
    for i in range(n_heads):
        axes[layer, i].imshow(model_output.attentions[layer][0, i])
        axes[layer][i].set_xticks(list(range(8)))
        axes[layer][i].set_xticklabels(labels=tokens, rotation="vertical")
        axes[layer][i].set_yticks(list(range(8)))
        axes[layer][i].set_yticklabels(labels=tokens)

        if layer == 5:
            axes[layer, i].set(xlabel=f"head={i}")
        if i == 0:
            axes[layer, i].set(ylabel=f"layer={layer}")

plt.subplots_adjust(wspace=0.3)
plt.show()
```

- Can we use BertViz to visualize one layer of attention?