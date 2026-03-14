# 🧠 Neural Network Training: A Leaky Abstraction

Neural network training looks simple — but often fails silently. This guide summarizes a **systematic, cautious, and debug-friendly** process for training deep learning models, with tips, visual checks, and sanity tests at every stage.



## 🧩 1. Understand the Data (Before Any Model!)

Before touching any model code:

- **Visualize examples** manually
- **Look for duplicates, class imbalance, label noise**
- Understand the **distribution and types** of variation
- Plot label histograms, outliers, and corrupted data

```python
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

# Visualize sample images
def show_images(images, title=""):
    grid = make_grid(images, nrow=8)
    plt.imshow(grid.permute(1, 2, 0))
    plt.title(title)
    plt.axis("off")
    plt.show()

# Example usage
images = ...  # Your dataset images
show_images(images)
```

⚙️ 2. Build a Clean Training & Evaluation Skeleton

Start with a minimal working pipeline using a simple model:

- Fix random.seed
- Disable all data augmentation
- Use tiny models (e.g. 1-layer CNN or linear)
- Visualize all outputs and losses

```python
torch.manual_seed(42)
torch.backends.cudnn.benchmark = True

# Dummy baseline model
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 10)
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
```

Sanity checks:

- ✅ Plot loss/accuracy
- ✅ Evaluate entire test set
- ✅ Verify loss at initialization (e.g. -log(1/n_classes))

🛠️ 3. Overfit a Single Batch

This ensures your model + data + training loop are working.

```python
# Overfit on a small batch
model.train()
x, y = next(iter(train_loader))
x, y = x.to(device), y.to(device)

for _ in range(1000):
    y_hat = model(x)
    loss = criterion(y_hat, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

✅ Your model should reach ~0 loss or 100% accuracy on this batch. If not, something is wrong.

🔍 4. Visual Debugging

- Always visualize the input right before it goes into the model
- Visualize predictions over time on a fixed batch to observe learning dynamics

🧪 5. Gradients for Dependency Checks

Use gradients to verify that your model uses the right inputs.

```python
# Check which inputs influence the output
x.requires_grad_(True)
output = model(x)
loss = output[0].sum()
loss.backward()

print(x.grad[0])  # Should only have non-zero gradients where expected
```

🧱 6. Build Up Model Complexity Step-by-Step

- Start with a reliable architecture (e.g. ResNet-18, UNet)
- Add complexity gradually (more inputs, larger images, new layers)
- Avoid learning rate decay too early. Use constant LR until convergence, then schedule if needed

```python
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
```

🧹 7. Regularization Strategies

Once you’re overfitting training data:

- ✅ Get more real data
- ✅ Use stronger augmentations (cutout, color jitter, etc.)
- ✅ Apply dropout / weight decay
- ✅ Reduce model size / input dimension
- ✅ Early stopping based on val loss

🧪 8. Tune Hyperparameters

- Prefer random search over grid search

Tune one by one ranking from most important to least important:

- Learning rate
- Weight decay
- Dropout rate
- Batch size

🧃 9. Squeeze Out the Last Bit of Juice

- ✅ Train longer — convergence can be slow
- ✅ Use ensembles
- ✅ Use test-time augmentation
- ✅ Visualize filters and activations

🏁 Final Thoughts

Training deep neural networks is not plug-and-play. Success comes from:

- ✅ Deep understanding of your data
- ✅ Careful debugging & visualizations
- ✅ Step-by-step validation
- ✅ Patience and attention to detail