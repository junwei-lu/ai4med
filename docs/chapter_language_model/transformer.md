# Transformer

The Transformer is a neural network architecture that is used for natural language processing tasks. It was introduced in the paper [Attention is All You Need](https://arxiv.org/abs/1706.03762) based on the [attention mechanism](./attention.md). We will first introduce the architecture of the Transformer and then the training process.

## Transformer block

The Transformer block is the main building block of the Transformer.

Let $X \in \mathbb{R}^{n \times d}$ be the input matrix, where $n$ is the sequence length and $d$ is the embedding dimension. The Transformer block consists of the following ope`rations:

- **Multi-Head Attention**:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

For multi-head attention with $h$ heads:

$$
\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \text{head}_2, \ldots, \text{head}_h)W^O
$$

where each head is:

$$
\text{head}_i = \text{Attention}(XW_i^Q, XW_i^K, XW_i^V)
$$

- **Residual Connection and Layer Normalization** (Add & Norm):

$$
X' = \text{LayerNorm}(X + \text{MultiHead}(X))
$$

where the layer normalization is similar to the [batch normalization](../chapter_neural_networks/regularization.md#batch-normalization) but instead of computing the mean and variance over the batch, we compute them over the embedding dimension. In PyTorch, you can use `torch.nn.LayerNorm` to implement the layer normalization.

- **Feed-Forward Network** (FFN):

$$
\text{FFN}(X') = \text{ReLU}(X'W_1 + b_1)W_2 + b_2
$$

- **Second Residual Connection and Layer Normalization** (Add & Norm):

$$
\text{Output} = \text{LayerNorm}(X' + \text{FFN}(X'))
$$

![Transformer Block](./tf.assets/transformer-block.png)

Using PyTorch `torch.nn.MultiheadAttention`, we can implement the Transformer block as follows:

```python
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim):
        """
        Args:
            embed_dim (int): Dimensionality of the input embeddings.
            num_heads (int): Number of attention heads.
            ff_hidden_dim (int): Hidden layer dimensionality in the feed-forward network.
        """
        super(TransformerBlock, self).__init__()
        
        # Multi-head attention layer. We use batch_first=True so that input shape is (batch_size, sequence_length, embed_dim).
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, 
                                         num_heads=num_heads,
                                         batch_first=True)
        
        # First layer normalization applied after the multi-head attention residual addition.
        self.attention_norm = nn.LayerNorm(embed_dim)
        
        # Feed-forward network: two linear layers with ReLU activation.
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, embed_dim)
        )
        
        # Second layer normalization after the feed-forward residual addition.
        self.ffn_norm = nn.LayerNorm(embed_dim)


    def forward(self, x, attn_mask=None, key_padding_mask=None):
        # Apply Multi-Head Attention (self-attention) where Q = K = V = x.
        # nn.MultiheadAttention returns (attn_output, attn_weights); unpack accordingly.
        attn_output, _ = self.mha(x, x, x, need_weights=False)
        
        # First residual connection and layer normalization.
        # X' = LayerNorm(x + attn_output)
        x = self.attention_norm(x + attn_output)
        # Feed-Forward Network (FFN)
        ffn_output = self.ffn(x)
        # Second residual connection and layer normalization.
        # Output = LayerNorm(x + ffn_output)
        output = self.ffn_norm(x + ffn_output)
        return output
```



## Transformer encoder

The Transformer encoder is a stack of multiple Transformer blocks and connect to a final fully connected layer for classification output.

Using the `TransformerBlock` we defined above, we can build the encoder as follows:

```python
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, num_layers):
        super(TransformerEncoder, self).__init__()
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads, ff_hidden_dim) for _ in range(num_layers)])

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        for block in self.blocks:
            x = block(x, attn_mask, key_padding_mask)
        return x
```

| Model | Layers | Hidden Size | Attention Heads | Feedforward Size | Parameters |
|-|--|-|--|||
| [BERT-Base](https://huggingface.co/bert-base-uncased) | 12 | 768 | 12 | 3072 | 110M |
| [BERT-Large](https://huggingface.co/bert-large-uncased) | 24 | 1024 | 16 | 4096 | 340M |
| [DistilBERT](https://huggingface.co/distilbert-base-uncased) | 6 | 768 | 12 | 3072 | 66M |



![Transformer Encoder](./tf.assets/bert.png)


## Transformer decoder

The Transformer decoder is similar to the encoder but with a key difference: it uses masked self-attention in its first sublayer. This masking prevents the decoder from attending to future positions during training, which is essential for autoregressive generation.



### Masked Self-Attention

In the decoder's masked self-attention, we modify the attention mechanism to ensure that the prediction for position $i$ can only depend on known outputs at positions less than $i$. This is achieved by masking future positions in the attention weights:

$$
\text{MaskedAttention}(Q, K, V) = \text{softmax}\left(\frac{QK^T + M}{\sqrt{d_k}}\right)V
$$

where $M$ is a mask matrix with:

$$
M_{ij} = 
\begin{cases} 
0 & \text{if } i \geq j \\
-\infty & \text{if } i < j 
\end{cases}
$$

![Masked Self-Attention](./tf.assets/mask1.png)

When we apply softmax to a row containing $-\infty$ values, those positions effectively receive $0$ attention weight, preventing information flow from future tokens.

![Transformer Decoder](./tf.assets/mask.png)

The Transformer decoder stacks multiple masked self-attention layers. Modern generative language models like GPT-2 and GPT-3 use the **decoder-only** architecture with a stack of masked self-attention layers followed by a feed-forward network.

There is an interactive visualization of the transformer in [transformer-explainer](https://poloclub.github.io/transformer-explainer/).

![GPT-2](./tf.assets/gpt.png)

GPT-2 is so far the last open-sourced model from OpenAI. It has 124M, 355M, and 774M parameters for small, medium, and large models, respectively.

| Model | Layers | Hidden Size | Attention Heads | Feedforward Size | Parameters |
|-|--|-|--|||
| [GPT-2 Small](https://huggingface.co/gpt2) | 12 | 768 | 12 | 3072 | 124M |
| [GPT-2 Medium](https://huggingface.co/gpt2-medium) | 24 | 1024 | 16 | 4096 | 355M |
| [GPT-2 Large](https://huggingface.co/gpt2-large) | 36 | 1280 | 20 | 5120 | 774M |

## Encoder-Decoder Transformer

The encoder-decoder transformer is a variant of the Transformer that uses both encoder and decoder blocks. It is used for sequence-to-sequence tasks such as translation and summarization.

![Encoder-Decoder Transformer](./tf.assets/trasformer_encoder-decoder.png)



### Choosing Transformer Architecture

We list below the best use cases for each type of transformer architecture.

**Encoder-Only Models**

- **Best for**: Understanding and analyzing input text (classification, entity recognition, sentiment analysis)
- **Examples**: [BERT](https://huggingface.co/bert-base-uncased), [RoBERTa](https://huggingface.co/roberta-base), [DistilBERT](https://huggingface.co/distilbert-base-uncased)
- **Characteristics**: Bidirectional attention (can see full context in both directions)
- **Use when**: Your task requires deep understanding of input text without generating new text

**Decoder-Only Models**

- **Best for**: Text generation tasks (completion, creative writing, chat)
- **Examples**: [GPT-2](https://huggingface.co/gpt2)
- **Characteristics**: Autoregressive generation with masked self-attention
- **Use when**: Your primary goal is to generate coherent, contextually relevant text

**Encoder-Decoder Models**

- **Best for**: Sequence-to-sequence tasks (translation, summarization)
- **Examples**: [T5](https://huggingface.co/t5-small), [BART](https://huggingface.co/facebook/bart-large-cnn), [Pegasus](https://huggingface.co/google/pegasus-large)
- **Characteristics**: Encoder processes input, decoder generates output based on encoder representations
- **Use when**: Your task involves transforming one sequence into another related sequence


## Transformer with Hugging Face

Rather than building transformer blocks from scratch, Hugging Face 🤗 gives you production-ready encoder-only, decoder-only, and encoder-decoder models in a single line. The table below maps each architecture to the right `AutoModel` class.

| Architecture | HF class | Example model |
||||
| Encoder-only | `AutoModel` / `AutoModelForSequenceClassification` | `bert-base-uncased` |
| Decoder-only | `AutoModelForCausalLM` | `gpt2` |
| Encoder-Decoder | `AutoModelForSeq2SeqLM` | `t5-small`, `facebook/bart-large-cnn` |

```python
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM

#  Encoder-only (BERT) 
enc_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
enc_model = AutoModel.from_pretrained("bert-base-uncased")

inputs = enc_tokenizer("Hello, how are you?", return_tensors="pt")
enc_out = enc_model(**inputs)
print(enc_out.last_hidden_state.shape)  # (batch, seq_len, 768)

#  Decoder-only (GPT-2) 
dec_tokenizer = AutoTokenizer.from_pretrained("gpt2")
dec_model = AutoModelForCausalLM.from_pretrained("gpt2")

prompt = dec_tokenizer("Once upon a time", return_tensors="pt")
gen_ids = dec_model.generate(**prompt, max_new_tokens=20)
print(dec_tokenizer.decode(gen_ids[0], skip_special_tokens=True))

#  Encoder-Decoder (T5) 
t5_tokenizer = AutoTokenizer.from_pretrained("t5-small")
t5_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

src_ids = t5_tokenizer("translate English to French: Hello world", return_tensors="pt")
tgt_ids = t5_tokenizer("Bonjour le monde", return_tensors="pt").input_ids
t5_out = t5_model(**src_ids, labels=tgt_ids)
print(t5_out.loss)   # cross-entropy loss for training
```

For low-level research or custom architectures you can still use PyTorch's `nn.TransformerEncoder/Decoder` directly:

```python
import torch
import torch.nn as nn

# Encoder
encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
src = torch.rand(32, 10, 512)  # (batch, seq_len, d_model) with batch_first=True
enc_out = encoder(src)

# Decoder
decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8, batch_first=True)
decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
tgt = torch.rand(32, 10, 512)
dec_out = decoder(tgt, enc_out)

# Full encoder-decoder in one call
transformer = nn.Transformer(d_model=512, nhead=8,
                              num_encoder_layers=6, num_decoder_layers=6,
                              batch_first=True)
out = transformer(src, tgt)
```


