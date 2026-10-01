# Understanding GAN Architecture: From Basic Concepts to My PyTorch MNIST Implementation

A personal study / reference document for a simple **fully connected GAN** that generates MNIST handwritten digits.

**How to read this document**

For every difficult concept the order is:

1. Simple intuition
2. Mathematical explanation
3. Tensor / shape explanation
4. PyTorch code
5. What happens during execution

**Labels used**

- **My original** = the architecture exactly as I wrote it (Discriminator ends with `Sigmoid`, loss is `BCELoss`).
- **Improvement** = a suggested change, always clearly labelled. The architecture itself (fully connected layers, sizes) is never silently changed.

---

## Table of Contents

1. Introduction to generative models
2. What is a GAN?
3. Overall architecture of my GAN
4. My exact Generator
5. The Linear layer in depth
6. Activation functions
7. My exact Discriminator
8. Why flatten 28x28 into 784?
9. MNIST data pipeline
10. Custom Dataset
11. DataLoader and batches
12. Random noise
13. One complete forward pass
14. Real labels and fake labels
15. Discriminator training in one batch
16. `detach()` in depth
17. Backpropagation in depth
18. `optimizer.zero_grad()`
19. `optimizer.step()`
20. Generator training in one batch
21. Why D is used during G training
22. Complete one-batch GAN timeline
23. Loss functions: BCE vs BCEWithLogits
24. The GAN objective function
25. Discriminator loss vs Generator loss
26. Why G needs no target image
27. The training loop
28. What happens in one epoch
29. Model parameters
30. Training instability
31. My previous training output
32. Testing the Generator
33. Saving and loading
34. Reproducibility
35. Different noise vectors and latent space
36. Visualizing training progress
37. Loss monitoring
38. Common GAN coding mistakes
39. Complete corrected code
40. Final architecture summary
41. Conceptual questions and answers
42. Beginner mental model

---

# 1. Introduction to Generative Models

## 1.1 What is a machine learning model?

**Intuition.** A model is a function with adjustable knobs. You feed it an input, it produces an output. "Learning" means turning the knobs until the outputs become useful.

**Math.** A model is a function `f(x; θ)` where `x` is the input and `θ` (theta) is the set of all adjustable numbers, called **parameters** (weights and biases).

**Example.** `f(x) = w·x + b`. With `w = 2, b = 1`, input `3` gives `7`. Training searches for the best `w` and `b`.

## 1.2 Discriminative model

A **discriminative model** looks at data and makes a decision about it.

- Input: an image of a digit. Output: "this is a 7".
- Mathematically it learns `p(y | x)`: the probability of label `y` given data `x`.

It answers: *"Given this image, what is it?"*

## 1.3 Generative model

A **generative model** learns what the data itself looks like, so it can create new examples.

- Input: random numbers. Output: a brand-new image of a digit that never existed.
- Mathematically it learns (or lets us sample from) `p(x)`: how likely each possible data point `x` is.

It answers: *"What does data of this kind look like? Make me a new one."*

| | Discriminative | Generative |
|---|---|---|
| Question | Given x, what is the label? | What does x look like? Create one. |
| Learns | `p(y\|x)` | `p(x)` (or a sampler for it) |
| Example | Digit classifier | Digit generator |
| Output | A label / score | New data |

**In my GAN:** the Discriminator is a discriminative model (real vs fake), and the Generator is the generative model. A GAN uses both.

## 1.4 What does "generating data" mean?

It means producing a **new sample that looks like it came from the real dataset**, but is not a copy of any single training example. For MNIST: a 28x28 grayscale image that looks like a handwritten digit.

## 1.5 What is a probability distribution?

**Intuition.** A distribution tells you *which outcomes are common and which are rare*.

**Example 1 (dice).** A fair die: each of 1..6 has probability 1/6.

**Example 2 (heights).** Adult heights cluster around the average, with few very tall or very short people. A bell curve.

**Example 3 (images).** Think of every possible 28x28 grayscale image. There are `256^784` of them. Almost all are random static. Only a tiny, special region looks like digits. The distribution of MNIST puts high probability on that tiny region and nearly zero everywhere else.

## 1.6 What does it mean to "learn the distribution of data"?

It means the model's samples land in the same region as real data, with the same variety. A generator that has learned the MNIST distribution produces:

- images that look like digits (not static), and
- a variety of digits (not just one digit, not one repeated image).

## 1.7 MNIST as the real data distribution

MNIST has 60,000 training images of handwritten digits 0 to 9. We treat each image as one sample drawn from an unknown true distribution:

```
x ~ p_data(x)
```

| Symbol | Meaning |
|---|---|
| `x` | One real image, a tensor of shape `[1, 28, 28]` (784 numbers) |
| `~` | "is drawn from" / "is sampled from" |
| `p_data(x)` | The true (unknown) distribution of real handwritten digits |

We never know `p_data` as a formula. We only have 60,000 samples from it.

## 1.8 What does it mean for G to learn the distribution?

The Generator starts from simple random noise:

```
z ~ p_z(z)
G(z) -> fake image
```

| Symbol | Meaning |
|---|---|
| `z` | A noise vector, shape `[100]` |
| `p_z(z)` | A simple known distribution. In my code: standard normal, `torch.randn` |
| `G` | The Generator network |
| `G(z)` | The fake image produced from `z` |
| `p_g` | The distribution of `G(z)` when `z` is random |

Training succeeds when `p_g ≈ p_data`: pushing random noise through G gives images that are statistically indistinguishable from real MNIST digits.

```
Simple distribution (noise)         Complicated distribution (digits)
        z  ~ N(0, I)    ---- G --->     G(z) ≈ looks like x ~ p_data
```

G is a **transformation machine**: it bends a simple distribution into a complicated one.

---

# 2. What is a GAN?

**GAN = Generative Adversarial Network.**

## 2.1 Why two networks?

Generating a good image is hard to describe with a formula. We cannot write a rule for "looks like a handwritten 7". But a *judge* that distinguishes real from fake can be learned. The judge then teaches the artist.

1. **Generator (G)**: makes fake images.
2. **Discriminator (D)**: tells real from fake.

## 2.2 Analogy

| Role | Analogy |
|---|---|
| Generator | Counterfeiter / student drawing digits |
| Discriminator | Detective / examiner checking banknotes |

- Early on, the counterfeiter makes terrible fakes; the detective spots them instantly.
- The detective's feedback tells the counterfeiter *how* the fakes were obvious.
- The counterfeiter improves; the detective must become sharper.
- This continues until the fakes are very hard to distinguish.

## 2.3 Why "adversarial"?

Because the two networks have **opposing goals**. D wants to catch G; G wants to fool D. What helps one hurts the other (in the original formulation). They improve by competing.

## 2.4 The basic interaction

```
Noise z ──> Generator ──> Fake Image ──┐
                                       ├──> Discriminator ──> Real or Fake?
Real Image (from MNIST) ───────────────┘
```

## 2.5 What each network wants

**Discriminator**
- `D(real image)` should be close to **1** (real).
- `D(G(z))` should be close to **0** (fake).

**Generator**
- Wants `D(G(z))` to be close to **1**, i.e. D is fooled and calls the fake "real".
- Wants images that are genuinely realistic.

---

# 3. Overall Architecture of My GAN

```
==================== GENERATOR PATH ====================

Random Noise  z
[batch_size, 100]
        |
        v
+--------------------------------+
| GENERATOR                      |
|  Linear(100 -> 256)            |
|  ReLU                          |
|  Linear(256 -> 512)            |
|  ReLU                          |
|  Linear(512 -> 784)            |
|  Tanh                          |
+--------------------------------+
        |   [batch_size, 784]
        v
Reshape (view)
[batch_size, 1, 28, 28]
        |
        v
Fake MNIST Image


================= DISCRIMINATOR PATH ===================

Real MNIST Image        OR       Fake Image
[batch_size, 1, 28, 28]          [batch_size, 1, 28, 28]
        \                         /
         \                       /
          v                     v
+--------------------------------+
| DISCRIMINATOR                  |
|  Flatten -> 784                |
|  Linear(784 -> 512)            |
|  LeakyReLU(0.2)                |
|  Linear(512 -> 256)            |
|  LeakyReLU(0.2)                |
|  Linear(256 -> 1)              |
|  Sigmoid   <-- my original     |
+--------------------------------+
        |
        v
Real/Fake score  [batch_size, 1]
```

**Complete data flow**

1. `DataLoader` gives a batch of real images `[64, 1, 28, 28]`, scaled to `[-1, 1]`.
2. `torch.randn` makes noise `[64, 100]`.
3. G turns noise into fake images `[64, 1, 28, 28]`.
4. D scores real images and fake images, each giving `[64, 1]`.
5. Losses compare the scores with labels (1 for real, 0 for fake).
6. `backward()` computes gradients; the optimizers update D, then G.
7. Repeat for every batch of every epoch.

> **Improvement (labelled, optional).** In my original, D ends with `Sigmoid` and the loss is `BCELoss`. The numerically safer version removes `Sigmoid` from D and uses `BCEWithLogitsLoss`. Everything else is identical. Section 23 explains this, and the final code in Section 39 uses the improved version.

---

# 4. My Exact Generator Architecture

```python
class Generator(nn.Module):
    def __init__(self, noise_dim=100):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Tanh())

    def forward(self, z):
        x = self.model(z)
        x = x.view(-1, 1, 28, 28)
        return x
```

## 4.1 Line-by-line

| Line | Meaning |
|---|---|
| `class Generator(nn.Module)` | Defines a neural network. `nn.Module` is PyTorch's base class; it tracks parameters automatically. |
| `def __init__(self, noise_dim=100)` | Constructor: builds the layers. `noise_dim=100` is the size of the noise vector. |
| `super().__init__()` | Initializes the parent `nn.Module` (required). |
| `self.model = nn.Sequential(...)` | Chains layers: output of one becomes input of the next. |
| `def forward(self, z)` | Describes the forward pass. Called when you write `G(z)`. |
| `x = self.model(z)` | Passes noise through all layers. |
| `x = x.view(-1, 1, 28, 28)` | Reshapes flat 784 values into an image. |
| `return x` | Returns the fake image batch. |

## 4.2 Layer-by-layer with shapes (batch = 64)

| Step | Layer | Input shape | Output shape | Neurons out |
|---|---|---|---|---|
| 0 | Noise `z` | | `[64, 100]` | 100 |
| 1 | `Linear(100, 256)` | `[64, 100]` | `[64, 256]` | 256 |
| 2 | `ReLU` | `[64, 256]` | `[64, 256]` | 256 |
| 3 | `Linear(256, 512)` | `[64, 256]` | `[64, 512]` | 512 |
| 4 | `ReLU` | `[64, 512]` | `[64, 512]` | 512 |
| 5 | `Linear(512, 784)` | `[64, 512]` | `[64, 784]` | 784 |
| 6 | `Tanh` | `[64, 784]` | `[64, 784]` | 784 |
| 7 | `view(-1,1,28,28)` | `[64, 784]` | `[64, 1, 28, 28]` | |

**What each layer does**

- `Linear(100, 256)`: mixes 100 noise numbers into 256 learned combinations (feature detectors).
- `ReLU`: adds non-linearity. Without it, stacked Linear layers collapse into a single Linear layer, which is too weak to draw digits.
- `Linear(256, 512)`: expands into richer features.
- `Linear(512, 784)`: produces one number per pixel.
- `Tanh`: squashes every pixel value into `(-1, +1)`.
- `view`: arranges 784 numbers as a 28x28 single-channel image.

## 4.3 Why 784?

`28 × 28 = 784`. An MNIST image has 784 pixels. G must output exactly one value per pixel, so the last Linear layer has 784 outputs.

## 4.4 Why Tanh and why the range must match the data

`tanh(x) = (e^x − e^−x) / (e^x + e^−x)`, which outputs values in `(-1, +1)`.

My MNIST normalization:

```python
image = image / 255.0     # 0..255  ->  0..1
image = image * 2 - 1     # 0..1    -> -1..+1
```

So **real pixels live in `[-1, +1]`**: black = −1, white = +1. G's Tanh also outputs `[-1, +1]`. This matters because D compares real and fake in the same value range. If real images were in `[0, 1]` but fakes could be negative, D could tell them apart by a trivial shortcut (any negative pixel means fake) without learning anything about digit shapes, and G would not get meaningful feedback.

**Rule:** the Generator's final activation range must match the real data range.

---

# 5. The Linear Layer in Depth

## 5.1 The formula

```
y = W x + b
```

| Symbol | Meaning |
|---|---|
| `x` | Input vector (numbers coming in) |
| `W` | Weight matrix (learned) |
| `b` | Bias vector (learned) |
| `y` | Output vector |

Each output neuron is a weighted sum of all inputs plus an offset.

## 5.2 Tiny example: 2 inputs, 3 outputs

```
x = [1, 2]            shape [2]

W = [[ 1,  2],        shape [3, 2]
     [ 0, -1],
     [ 3,  0]]

b = [0.5, 0, -1]      shape [3]
```

Compute:

```
y1 = 1*1 + 2*2 + 0.5 = 5.5
y2 = 0*1 + (-1)*2 + 0 = -2
y3 = 3*1 + 0*2 + (-1) = 2

y = [5.5, -2, 2]      shape [3]
```

Matrix multiplication rule: `[3,2] × [2] -> [3]`. The inner dimensions (2 and 2) must match, and the outer ones remain.

## 5.3 How PyTorch does it for a whole batch

PyTorch stores `x` as rows (one row per sample), so it computes:

```
Y = X Wᵀ + b
```

Batch of 2 samples: `X = [[1, 2], [0, 1]]`, shape `[2, 2]`. `Wᵀ` has shape `[2, 3]`.

```
Row 1: [1,2]  -> [5.5, -2, 2]
Row 2: [0,1]  -> [0*1+1*2+0.5, 0*0+1*(-1)+0, 0*3+1*0-1] = [2.5, -1, -1]

Y = [[5.5, -2,  2],
     [2.5, -1, -1]]          shape [2, 3]
```

Shape rule: `[batch, in] × [in, out] -> [batch, out]`; the bias `[out]` is added to every row (broadcasting).

## 5.4 Connection to `nn.Linear(100, 256)`

| Item | Shape |
|---|---|
| Weight `W` (stored as) | `[256, 100]` (`[out_features, in_features]`) |
| Bias `b` | `[256]` |
| Input `X` | `[batch, 100]` |
| Output `Y` | `[batch, 256]` |

Computation: `[batch,100] × [100,256] (which is Wᵀ) -> [batch,256]`, then `+ b`.

Parameters: `256 × 100 = 25,600` weights + `256` biases = `25,856`.

**One batch through the layer.** All 64 noise vectors are processed simultaneously by one matrix multiplication. The same `W` and `b` are applied to every sample. That is why batching is efficient on a GPU.

---

# 6. Activation Functions

**Why activations exist.** Linear layers only compute weighted sums. Stacking linear layers without activations gives one big linear function: `W2(W1 x) = (W2 W1) x`. Activations bend the function so the network can learn curved, complicated patterns like digit shapes.

Test values: `y = [5.5, -2, 2]` (from Section 5).

## 6.1 ReLU

- Formula: `ReLU(x) = max(0, x)`
- Graph: flat at 0 for negative x, then a straight 45-degree line.
- Range: `[0, ∞)`
- Example: `[5.5, -2, 2] -> [5.5, 0, 2]`
- Why: very simple, fast, and gradients don't shrink for positive values.
- Weakness: for negative inputs the gradient is exactly 0 (a neuron can "die").
- **Where in my GAN:** two places inside the Generator (after `Linear(100,256)` and `Linear(256,512)`).

## 6.2 LeakyReLU

- Formula: `x` if `x > 0`, else `0.2 x` (slope 0.2 is the "leak")
- Range: `(-∞, ∞)`
- Example (slope 0.2): `[5.5, -2, 2] -> [5.5, -0.4, 2]`
- Why: negative inputs still pass a small signal and a small gradient, which prevents dead neurons. This is especially important in D, because D's gradients are what teach G.
- **Where in my GAN:** two places inside the Discriminator.

## 6.3 Tanh

- Formula: `tanh(x) = (e^x − e^−x) / (e^x + e^−x)`
- Range: `(-1, +1)`, centered at 0, S-shaped.
- Example: `[5.5, -2, 2] -> [0.99997, -0.964, 0.964]`
- Why: forces outputs into the same range as normalized images.
- Weakness: very large |x| gives almost flat curve, so tiny gradients.
- **Where in my GAN:** the last layer of the Generator.

## 6.4 Sigmoid

- Formula: `σ(x) = 1 / (1 + e^−x)`
- Range: `(0, 1)`, S-shaped. `σ(0) = 0.5`.
- Example: `[5.5, -2, 2] -> [0.9959, 0.1192, 0.8808]`
- Why: turns any real number into a probability-like value.
- **Where in my GAN:** my original Discriminator's final layer.

## 6.5 Summary

| Activation | Range | Used in | Role |
|---|---|---|---|
| ReLU | `[0, ∞)` | Generator hidden layers | Non-linearity |
| LeakyReLU(0.2) | `(-∞, ∞)` | Discriminator hidden layers | Non-linearity without dead neurons |
| Tanh | `(-1, 1)` | Generator output | Match image range |
| Sigmoid | `(0, 1)` | Original D output | Turn score into probability |

## 6.6 Why G uses Tanh but D's final activation depends on the loss

- G's output is an **image**, so its last activation must match the **pixel range** → Tanh.
- D's output is a **real/fake score**. If the loss is `BCELoss`, D must output probabilities in `(0,1)` → end with Sigmoid. If the loss is `BCEWithLogitsLoss`, the loss applies the sigmoid internally, so D must output **raw scores (logits)** with **no** Sigmoid. The final activation therefore depends on which loss you pick (Section 23).

---
# 7. My Exact Discriminator Architecture

**My original** (with Sigmoid):

```python
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid())

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.model(x)
        return x
```

## 7.1 Line-by-line

| Line | Meaning |
|---|---|
| `nn.Linear(784, 512)` | Reads all 784 pixels, produces 512 features |
| `nn.LeakyReLU(0.2)` | Non-linearity, negative slope 0.2 |
| `nn.Linear(512, 256)` | Compresses to 256 features |
| `nn.LeakyReLU(0.2)` | Non-linearity |
| `nn.Linear(256, 1)` | Collapses to a single number: the real/fake score |
| `nn.Sigmoid()` | Squashes that number into `(0, 1)` |
| `x.view(-1, 784)` | Flattens `[batch,1,28,28]` into `[batch,784]` |
| `x = self.model(x)` | Runs the layers |
| `return x` | Returns `[batch, 1]` |

## 7.2 Tensor flow

```
[64, 1, 28, 28]     real or fake images
        |  view(-1, 784)
        v
[64, 784]
        |  Linear(784,512) + LeakyReLU
        v
[64, 512]
        |  Linear(512,256) + LeakyReLU
        v
[64, 256]
        |  Linear(256,1) + Sigmoid
        v
[64, 1]             one score per image
```

## 7.3 Meaning of the output

```
D(x) ≈ 1  -> D thinks the image is REAL
D(x) ≈ 0  -> D thinks the image is FAKE
D(x) ≈ 0.5 -> D is unsure (guessing)
```

Example output for a batch of 4: `[[0.97], [0.03], [0.55], [0.91]]` means image 1 looks real, image 2 looks fake, image 3 is uncertain, image 4 looks real.

> **Improvement (labelled).** Remove the final `nn.Sigmoid()` and train with `BCEWithLogitsLoss`. Same layers otherwise. See Sections 23 and 39.

---

# 8. Why Flatten 28x28 into 784?

## 8.1 Intuition

A `Linear` layer is a function that takes a **list of numbers** (a vector) per sample. An image is a grid. A fully connected network has no concept of grids; it simply needs all pixels laid out in one row. Flattening reads the grid row by row into one long list.

```
[1, 28, 28]  ->  [784]
 channel,H,W      28*28*1 = 784 numbers
```

`nn.Linear(784, 512)` has weight shape `[512, 784]`, so its input must have last dimension exactly 784.

## 8.2 `x.view(-1, 784)` in detail

- `view` reshapes a tensor **without changing the data or copying it**, only reinterpreting its shape.
- `784` = the number of features per sample.
- `-1` = "figure this dimension out for me." PyTorch computes it from the total number of elements.

Batch of 64:

```
Input shape : [64, 1, 28, 28]
Total elements = 64 × 1 × 28 × 28 = 50,176
view(-1, 784): second dim is 784, so first = 50,176 / 784 = 64
Output shape: [64, 784]
```

The number of values stays the same:

```
64 × 1 × 28 × 28  =  64 × 784  =  50,176
```

**Why `-1` is useful:** the last batch of an epoch may have fewer than 64 images (see Section 11). With `-1`, the same code works for any batch size.

---

# 9. MNIST Data Pipeline

## 9.1 The IDX file format

MNIST images are stored in a binary **IDX** file, `train-images.idx3-ubyte`:

```
Byte offset   Size     Content
0             4 bytes  Magic number (2051 = "image file")
4             4 bytes  Number of images (60000)
8             4 bytes  Number of rows (28)
12            4 bytes  Number of columns (28)
16            ...      Pixel data: 60000 × 28 × 28 bytes, each 0..255
```

The first 16 bytes are the **header**. After that comes raw pixel data.

## 9.2 My code

```python
def load_mnist_images(file_path):
    with open(file_path, "rb") as f:
        magic_number, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    image = data.reshape(num_images, rows, cols)
    image = torch.tensor(image, dtype=torch.float32)
    return image
```

## 9.3 Step-by-step

| Code | What happens | Result |
|---|---|---|
| `open(file_path, "rb")` | Opens the file in **binary read** mode | file handle |
| `f.read(16)` | Reads the 16-byte header | 16 raw bytes |
| `struct.unpack(">IIII", ...)` | Decodes the bytes into 4 integers. `>` = big-endian byte order (the order MNIST uses). `I` = unsigned 32-bit int. Four `I` = 4 numbers × 4 bytes = 16 bytes | `(2051, 60000, 28, 28)` |
| `f.read()` | Reads **all remaining bytes** (the pixels) | 47,040,000 bytes |
| `np.frombuffer(..., dtype=np.uint8)` | Interprets those bytes as an array of unsigned 8-bit ints | shape `[47040000]` |
| `.reshape(num_images, rows, cols)` | Reshapes the flat array into images | shape `[60000, 28, 28]` |
| `torch.tensor(..., dtype=torch.float32)` | Converts to float tensor | `[60000, 28, 28]`, float32 |

**Why `uint8`?** Each pixel is stored in **one byte** with value 0 to 255 (`uint8` = unsigned 8-bit integer). Reading with a different type would misinterpret the bytes.

**Why `float32`?** Neural networks multiply by decimal weights and compute gradients; they need floating point numbers. `float32` is PyTorch's default for parameters, so the data must match.

**Why `reshape`?** The file is one long stream of pixel bytes. `reshape(60000, 28, 28)` cuts it into 60,000 images of 28 rows × 28 columns.

## 9.4 Normalization

```python
def normalization_pixel(image):
    image = image / 255.0     # 0..255 -> 0..1
    image = image * 2 - 1     # 0..1   -> -1..1
    return image
```

| Original pixel | `/255` | `*2 - 1` |
|---|---|---|
| 0 (black) | 0.0 | −1.0 |
| 128 | 0.502 | 0.004 |
| 255 (white) | 1.0 | +1.0 |

Purpose: (1) small, centered numbers make training stable; (2) range `[-1, 1]` matches the Generator's `Tanh`.

---

# 10. Custom Dataset

```python
class MNISTDataset(Dataset):
    def __init__(self, file_path):
        self.images = load_mnist_images(file_path)
        self.images = normalization_pixel(self.images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]       # [28, 28]
        image = image.unsqueeze(0)       # [1, 28, 28]
        return image
```

## 10.1 The three required methods

| Method | When called | Purpose |
|---|---|---|
| `__init__` | Once, when you create `MNISTDataset(path)` | Load the file and normalize all 60,000 images (stored in RAM) |
| `__len__` | When `len(dataset)` is called (the DataLoader uses it) | Tells PyTorch how many samples: 60000 |
| `__getitem__(index)` | When `dataset[index]` is called | Returns **one** sample |

## 10.2 What happens in `train_dataset[index]`

Example `train_dataset[5]`:

1. Python calls `__getitem__(5)`.
2. `self.images[5]` picks the 6th image: shape `[28, 28]`.
3. `unsqueeze(0)` adds a new dimension at position 0: shape `[1, 28, 28]`.
4. That tensor is returned.

## 10.3 `unsqueeze(0)` and the channel dimension

```
[28, 28]       height, width  (no channel dimension)
   |  unsqueeze(0)
   v
[1, 28, 28]    channels, height, width
```

PyTorch image convention is `[channels, height, width]`. A grayscale image has **1 channel** (a color RGB image would have 3). So the first dimension is the grayscale channel. Even though my fully connected network flattens the image anyway, keeping the `[1,28,28]` convention makes the data consistent with standard PyTorch image pipelines and makes the Generator's output shape (`[batch,1,28,28]`) identical to the real data.

---

# 11. DataLoader and Batches

```python
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
```

## 11.1 What DataLoader does

It repeatedly asks the Dataset for individual samples and stacks them into **batches**.

```
Dataset:   img0  img1  img2  ... img59999     each [1,28,28]
              |  shuffle (random order every epoch)
              v
DataLoader: picks 64 indices -> 64 × [1,28,28] -> stack -> [64,1,28,28]
```

- `batch_size = 64`: 64 images per batch.
- `shuffle = True`: the order of images is randomized at the start of every epoch, so batches differ each epoch and the model does not memorize an ordering.

## 11.2 What happens in the loop

```python
for batch_idx, real_images in enumerate(train_loader):
```

- `train_loader` yields one batch at a time.
- `enumerate` adds a counter: `batch_idx` = 0, 1, 2, ...
- `real_images` = the batch tensor, shape `[64, 1, 28, 28]`.

## 11.3 The last batch

`60000 / 64 = 937.5`. So:

- 937 full batches of 64 = 59,968 images
- 1 final batch with `60000 − 59968 = 32` images
- Total: **938 batches** per epoch

The last real batch has shape `[32, 1, 28, 28]`.

## 11.4 Why `current_batch_size = real_images.size(0)`

`size(0)` returns the first dimension (the number of images in this batch). If I wrote `torch.ones(64, 1)` for the labels, the last batch (32 images) would give D output `[32,1]` but labels `[64,1]`, and the loss would fail or silently broadcast wrongly. Using `current_batch_size` makes labels and noise match the real batch size every time.

## 11.5 Vocabulary

| Term | Meaning | In my setup |
|---|---|---|
| Dataset | The collection of all samples | 60,000 images |
| DataLoader | Tool that creates shuffled batches from the dataset | batch_size=64 |
| Batch | A group of samples processed together | 64 images (last one: 32) |
| Iteration | One batch processed: one forward + backward + update | 1 loop step |
| Epoch | One full pass through the entire dataset | 938 iterations |

```
1 epoch = 938 iterations = 938 batches ≈ 60,000 / 64
20 epochs = 18,760 iterations
```

---

# 12. Random Noise

```python
z = torch.randn(current_batch_size, noise_dim, device=device)
```

## 12.1 What is `torch.randn`

It creates a tensor of random numbers from the **standard normal distribution** (mean 0, standard deviation 1). Most values lie between −2 and +2. (`torch.rand`, by contrast, gives uniform `[0,1)` numbers. Different thing.)

`device=device` creates the tensor directly on the GPU if used.

Shape for batch 64: **`[64, 100]`**: 64 noise vectors, each with 100 numbers.

## 12.2 Why noise is needed

A neural network is a deterministic function: same input, same output. If G had no random input, it would produce **one single image** forever. Random `z` gives a different input every time, hence a different image. The randomness is the source of **variety**.

## 12.3 What does 100 mean?

100 is the **latent dimension**, the length of the noise vector. It is a design choice: a common default, big enough to give G plenty of variety, small enough to be cheap.

Do the 100 numbers mean specific features (like "thickness", "slant")? **Not by design.** They are just random numbers. Individual entries have no assigned meaning. During training, G *may* organize the space so that certain directions correspond to features like stroke thickness, but that emerges, it isn't specified, and in a plain fully connected GAN it is usually entangled.

## 12.4 z → G(z)

```
z  [64, 100]   random, meaningless numbers
       |
       G  (learned transformation)
       v
G(z) [64, 1, 28, 28]  images
```

G learns **how to transform** random latent vectors into realistic images.

---

# 13. One Complete Forward Pass

Let `batch_size = 4`, `noise_dim = 100`.

## 13.1 Generator forward pass

```python
z = torch.randn(4, 100)         # [4, 100]
fake_images = G(z)              # [4, 1, 28, 28]
```

| Step | Operation | Shape | Explanation |
|---|---|---|---|
| 1 | `z` | `[4, 100]` | 4 random vectors |
| 2 | `Linear(100,256)` | `[4, 256]` | each vector → 256 numbers via `z Wᵀ + b` |
| 3 | `ReLU` | `[4, 256]` | negatives set to 0 |
| 4 | `Linear(256,512)` | `[4, 512]` | 512 richer features |
| 5 | `ReLU` | `[4, 512]` | negatives set to 0 |
| 6 | `Linear(512,784)` | `[4, 784]` | one number per pixel |
| 7 | `Tanh` | `[4, 784]` | each pixel in (−1, 1) |
| 8 | `view(-1,1,28,28)` | `[4, 1, 28, 28]` | 4 fake images |

At the start of training the weights are random, so the images look like noise.

## 13.2 Fake images into the Discriminator

```python
out = D(fake_images)            # [4, 1]
```

| Step | Operation | Shape |
|---|---|---|
| 1 | input | `[4, 1, 28, 28]` |
| 2 | `view(-1,784)` | `[4, 784]` |
| 3 | `Linear(784,512)` | `[4, 512]` |
| 4 | `LeakyReLU(0.2)` | `[4, 512]` |
| 5 | `Linear(512,256)` | `[4, 256]` |
| 6 | `LeakyReLU(0.2)` | `[4, 256]` |
| 7 | `Linear(256,1)` | `[4, 1]` |
| 8 | `Sigmoid` (original) | `[4, 1]` |

Result: 4 numbers, e.g. `[[0.48],[0.52],[0.50],[0.47]]`, D's guesses for the 4 fake images.

A **forward pass** means: data flows input → output, computing predictions. No learning happens yet. Learning happens in the backward pass and optimizer step.

---

# 14. Real Labels and Fake Labels

```python
real_labels = torch.ones(current_batch_size, 1, device=device)
fake_labels = torch.zeros(current_batch_size, 1, device=device)
```

- `real_labels`: a tensor of 1s, shape `[64, 1]`
- `fake_labels`: a tensor of 0s, shape `[64, 1]`

**Why 1 = real and 0 = fake?** It is just a convention that matches D's output meaning (`D ≈ 1` real, `D ≈ 0` fake) and binary classification. The loss function measures how far D's output is from these targets.

**Why the shape must be `[64, 1]`:** the loss compares output and target element by element. D's output is `[64, 1]`, so the labels must be `[64, 1]`. If labels were `[64]`, PyTorch would broadcast `[64,1]` vs `[64]` into a `[64,64]` grid, giving a **wrong** loss with losses like `MSELoss`; PyTorch's BCE losses raise a size-mismatch error instead. Matching shapes avoids both problems.

---

# 15. Discriminator Training in One Batch

**Goal:** teach D that real → 1 and fake → 0.

## Step 1: Take real images
```python
real_images = real_images.to(device)       # [64, 1, 28, 28]
```

## Step 2: D(real_images)
```python
real_output = D(real_images)               # [64, 1]
```

## Step 3: Real loss
```python
loss_real = criterion(real_output, real_labels)
```
Penalizes D when its output for real images is far from 1.

## Step 4: Generate fake images
```python
z = torch.randn(current_batch_size, noise_dim, device=device)   # [64, 100]
fake_images = G(z)                                              # [64, 1, 28, 28]
```

## Step 5: D(fake_images.detach())
```python
fake_output = D(fake_images.detach())      # [64, 1]
```
`detach()` is explained in Section 16.

## Step 6: Fake loss
```python
loss_fake = criterion(fake_output, fake_labels)
```
Penalizes D when its output for fake images is far from 0.

## Step 7: Combine
```python
loss_D = loss_real + loss_fake
```
One scalar number: D's total mistake on this batch.

## Step 8: Clear old gradients
```python
optimizer_D.zero_grad()
```

## Step 9: Backward
```python
loss_D.backward()
```
Computes the gradient of `loss_D` with respect to **every D parameter** (all weights and biases of the three Linear layers).

## Step 10: Update
```python
optimizer_D.step()
```
Every D parameter is nudged in the direction that reduces `loss_D`.

## What happens to every weight during backward?

For each weight `w` in D, PyTorch computes `∂loss_D/∂w`, which answers "if I increase this weight slightly, how does the loss change?" and stores it in `w.grad`. Weights that contributed to mistakes get larger gradients; weights that didn't matter get small ones. Then `optimizer_D.step()` moves each weight opposite to its gradient. After this, D is slightly better at separating real and fake.

Only D changes in this phase.

---

# 16. `detach()` in Depth

## 16.1 Simple terms

`fake_images.detach()` returns the **same numbers** but disconnected from the history of how they were made. It says: *"Treat these as plain data. Don't track where they came from."*

## 16.2 Computational graph

PyTorch records every operation while the forward pass runs, building a graph so that `backward()` can later walk it in reverse.

```
z -> [G layers: W_g, b_g] -> fake_images -> [D layers: W_d, b_d] -> fake_output -> loss
```

If I call `loss.backward()` without detach, gradients flow all the way back **through D and then into G**, filling `.grad` for both.

With `detach()`:

```
z -> [G layers] -> fake_images  ||cut||  fake_images_detached -> [D layers] -> loss
```

The graph is cut at `fake_images`. `backward()` stops at the cut: only D's parameters get gradients.

## 16.3 Why D(fake_images.detach()) in D training

- During D training, **only D should learn**, so the gradient need not (and should not) travel into G.
- Without `detach()`:
  - PyTorch wastes time computing gradients for all G parameters.
  - Those unneeded gradients sit in `G.parameters().grad`. Because gradients accumulate, forgetting to clear them could contaminate G's later update.
  - The graph of G stays attached to `loss_D`, and freeing/reusing it can cause errors like *"Trying to backward through the graph a second time"* if the same `fake_images` are later reused for the G step.

**Honest note about my exact code.** In my loop, `optimizer_G.zero_grad()` is called before `loss_G.backward()` and the G step regenerates `fake_images`. So not using `detach()` wouldn't give wrong results in this precise ordering, only waste computation and invite bugs. It is still the correct, standard practice.

## 16.4 D training vs G training

```
During D training:
  D gets updated.
  G should NOT be updated from D's loss.   -> use detach()

During G training:
  D provides gradient information (direction to improve the fake).
  G gets updated.                          -> NO detach()
```

If `detach()` were used in the G step, the graph would be cut between G and D, so `loss_G.backward()` could not reach G at all, and G would never learn. The gradient must flow D → G.

---
# 17. Backpropagation in Depth

## 17.1 Intuition

After a forward pass we know the loss (how wrong we were). Backpropagation answers: **"how much did each parameter contribute to that error, and in which direction should it change?"** It works backward from the loss to every parameter.

## 17.2 A tiny 2-layer network with real numbers

```
x -> Layer 1 -> h -> Layer 2 -> y -> loss L
```

Definitions (no activation, to keep it simple):

```
h = w1 * x + b1
y = w2 * h + b2
L = (y - t)^2        t = target
```

Numbers: `x = 2, w1 = 0.5, b1 = 0, w2 = 3, b2 = 0, t = 1`

**Forward**
```
h = 0.5*2 + 0 = 1
y = 3*1 + 0   = 3
L = (3 - 1)^2 = 4
```

## 17.3 Chain rule

If A affects B and B affects L, then `∂L/∂A = ∂L/∂B × ∂B/∂A`. Effects multiply along the path.

**Backward** (from the loss toward the input):

```
∂L/∂y  = 2(y - t)       = 2(3-1)      = 4

∂L/∂w2 = ∂L/∂y * ∂y/∂w2 = 4 * h      = 4 * 1 = 4
∂L/∂b2 = ∂L/∂y * 1                    = 4

∂L/∂h  = ∂L/∂y * ∂y/∂h  = 4 * w2     = 4 * 3 = 12

∂L/∂w1 = ∂L/∂h * ∂h/∂w1 = 12 * x     = 12 * 2 = 24
∂L/∂b1 = ∂L/∂h * 1                    = 12
```

Gradients: `∂L/∂W2 = 4`, `∂L/∂b2 = 4`, `∂L/∂W1 = 24`, `∂L/∂b1 = 12`.

**Update with learning rate 0.01** (plain gradient descent):

```
w2 = 3   - 0.01*4  = 2.96
b2 = 0   - 0.01*4  = -0.04
w1 = 0.5 - 0.01*24 = 0.26
b1 = 0   - 0.01*12 = -0.12
```

Check the new forward pass: `h = 0.26*2 - 0.12 = 0.40`, `y = 2.96*0.40 - 0.04 = 1.144`, `L = (1.144-1)^2 = 0.0207`. The loss dropped from 4 to about 0.02.

## 17.4 What a gradient means

> **Gradient = direction and sensitivity of the loss with respect to a parameter.**

- Sign: positive gradient means "increasing this parameter increases the loss," so decrease it.
- Size: large magnitude means the loss is very sensitive to this parameter.

Here `∂L/∂w1 = 24`: a tiny change in `w1` changes the loss a lot.

## 17.5 How PyTorch autograd does it

1. Every tensor with `requires_grad=True` (all `nn.Module` parameters) is tracked.
2. During the forward pass, each operation records a node in a **computational graph**, remembering how to compute its local derivative.
3. `loss.backward()` walks the graph from the loss back to the parameters, applying the chain rule at each node.
4. Results are **added into** each parameter's `.grad`.

```python
loss.backward()
print(D.model[0].weight.grad.shape)    # [512, 784]  (same shape as the weight)
```

Every parameter gets a gradient tensor of the same shape as itself.

---

# 18. `optimizer.zero_grad()`

## 18.1 Why

PyTorch **accumulates (adds)** gradients into `.grad` on every `backward()` call. It does not overwrite. This is useful for special cases (like gradient accumulation) but means that in a normal loop you must clear them each iteration.

## 18.2 Simple example

Suppose a parameter's gradient is 4 on batch 1 and 6 on batch 2.

| Step | With `zero_grad()` | Without `zero_grad()` |
|---|---|---|
| Batch 1 backward | `grad = 4` | `grad = 4` |
| Batch 2 backward | cleared first, then `grad = 6` | `grad = 4 + 6 = 10` (wrong) |
| Batch 3 (grad 5) | `grad = 5` | `grad = 15` (more wrong) |

Without clearing, each update uses a mixture of old and new batches, so steps become huge and point in stale directions: training diverges or behaves erratically.

## 18.3 Where in my code

```python
optimizer_D.zero_grad()      # clear D's gradients
loss_D.backward()
optimizer_D.step()

optimizer_G.zero_grad()      # clear G's gradients
loss_G.backward()
optimizer_G.step()
```

Call it **before** `backward()` and with the matching optimizer.

---

# 19. `optimizer.step()`

## 19.1 Concept

For every parameter the optimizer manages:

```
parameter_new = parameter_old - learning_rate × gradient
```

This is **gradient descent**. Gradient points uphill; we step downhill. `learning_rate` is the step size.

Example: `w = 0.5`, `grad = 24`, `lr = 0.01` → `w_new = 0.5 − 0.24 = 0.26`.

## 19.2 Adam

My code:

```python
optimizer_G = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
```

Plain SGD uses the raw gradient. **Adam** is smarter, keeping two running averages per parameter:

```
m = β1·m + (1-β1)·g          # average of gradients ("momentum": which direction has been consistent)
v = β2·v + (1-β2)·g²         # average of squared gradients (how large/noisy gradients are)

θ_new = θ - lr · m̂ / (sqrt(v̂) + ε)
```

(`m̂, v̂` are bias-corrected versions, ε = 1e-8 avoids division by zero.)

Effect: each parameter gets its **own adaptive step size**. Parameters with consistently large gradients take smaller relative steps; noisy directions are smoothed.

| Argument | Meaning |
|---|---|
| `G.parameters()` | Which parameters this optimizer may change. **Only G's.** |
| `lr=0.0002` | Base step size. Small and stable; a standard GAN value. |
| `betas=(0.5, 0.999)` | `β1 = 0.5` (momentum memory, short, ~2 steps), `β2 = 0.999` (squared-gradient memory, long). The default `β1` is 0.9; a lower value of 0.5 is the common GAN recipe (from DCGAN) since GAN gradients change quickly as the opponent changes. |

## 19.3 Two optimizers

D and G have separate optimizers because they have separate parameters and are updated at different moments with different losses.

---

# 20. Generator Training in One Batch

**Goal:** make D label G's fakes as real.

```python
# New noise
z = torch.randn(current_batch_size, noise_dim, device=device)    # [64, 100]

# G makes fake images
fake_images = G(z)                                                # [64, 1, 28, 28]

# D evaluates (NO detach)
fake_output = D(fake_images)                                      # [64, 1]

# G's loss
loss_G = criterion(fake_output, real_labels)                      # scalar

optimizer_G.zero_grad()
loss_G.backward()
optimizer_G.step()
```

## 20.1 Why `real_labels` for fake images?

The images *are* fake, but G's **desire** is for D to say "real". So the target for D's output on G's images is **1**:

```
loss_G = criterion(D(G(z)), 1)
```

- If `D(G(z)) ≈ 1`: loss is small → G did well (fooled D).
- If `D(G(z)) ≈ 0`: loss is large → G did badly.

Math form: `loss_G = −log D(G(z))` (with BCE). Minimizing it pushes `D(G(z)) → 1`.

> This is **not lying to D's training**. The label is used only in G's loss. D's loss (Section 15) used `fake_labels` honestly.

---

# 21. Why D Is Used During G Training

G never sees real digits directly. Its only teacher is D.

```
z
|
v
G  ------------------> fake image
                           |
                           v
                           D  (its weights are used but not updated here)
                           |
                           v
                       prediction
                           |
                           v
                        loss_G
```

**Backward flow:**

```
loss_G
  |  ∂L/∂prediction
  v
  D   (gradients pass THROUGH D's layers using chain rule)
  |  ∂L/∂fake_image    <- "which pixels should change, and how, to look more real"
  v
  G   (gradients reach G's weights)
```

Chain rule: `∂loss_G/∂W_g = ∂loss_G/∂D_out × ∂D_out/∂image × ∂image/∂W_g`. The middle factor `∂D_out/∂image` exists only because the graph goes through D. D is a **differentiable critic** that tells G, pixel by pixel, how to look more real.

## 21.1 Why optimizer_G updates only G

`optimizer_G = optim.Adam(G.parameters(), ...)` registers **only G's parameters**. When `loss_G.backward()` runs, gradients are computed for both D and G parameters (D is in the path), but `optimizer_G.step()` modifies only G. D's weights stay untouched.

The leftover gradients on D are harmless because the next D step begins with `optimizer_D.zero_grad()`, which clears them before `loss_D.backward()`.

> **Optional Improvement:** you can wrap the G step with `for p in D.parameters(): p.requires_grad_(False)` and turn it back on after, to save compute. Not required for correctness.

---

# 22. Complete One-Batch GAN Timeline

```
┌──────────────────────── PHASE A: TRAIN D ────────────────────────┐
│                                                                  │
│  REAL DATA [64,1,28,28]                                          │
│     |                                                            │
│     v                                                            │
│  D(real)  -> [64,1]                                              │
│     |                                                            │
│  loss_real = criterion(D(real), 1)                               │
│                                                                  │
│  random noise z [64,100]                                         │
│     |                                                            │
│     v                                                            │
│  G(z) -> fake image [64,1,28,28]                                 │
│     |                                                            │
│     v                                                            │
│  D(fake.detach()) -> [64,1]                                      │
│     |                                                            │
│  loss_fake = criterion(D(fake), 0)                               │
│                                                                  │
│  loss_D = loss_real + loss_fake                                  │
│     |                                                            │
│  zero_grad -> backward -> optimizer_D.step()     (D updated)     │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────── PHASE B: TRAIN G ────────────────────────┐
│                                                                  │
│  NEW random noise z [64,100]                                     │
│     |                                                            │
│     v                                                            │
│  G(z) -> fake image                                              │
│     |                                                            │
│     v                                                            │
│  D(fake)  (no detach)  -> [64,1]                                 │
│     |                                                            │
│  loss_G = criterion(D(fake), 1)                                  │
│     |                                                            │
│  zero_grad -> backward (through D into G) -> optimizer_G.step()  │
│                                             (G updated)          │
└──────────────────────────────────────────────────────────────────┘
```

## 22.1 Why G is run again with NEW noise

1. **The graph must stay attached.** The first `fake_images` was used in D's step with `detach()`. For G's step we need a graph connecting G → D, so G is run again.
2. **D just changed.** After `optimizer_D.step()`, D is a better critic. G should be judged by the **updated** D. Re-running the forward pass lets the new D evaluate the fakes.
3. **Fresh noise** gives G more variety per iteration (it could reuse the old `z`, but a fresh `z` is the common practice and avoids over-fitting to one noise set per batch).

Efficiency note: you could instead reuse the same `fake_images` without detach, but then you'd need `retain_graph` handling and D's output would come from a stale D. The two-forward-pass version is clean and standard.

---

# 23. Loss Function

## 23.1 Binary Cross Entropy (BCE)

For a binary decision with prediction `p` (a probability in (0,1)) and label `y` (0 or 1):

```
BCE(p, y) = -[ y·log(p) + (1-y)·log(1-p) ]
```

- If `y = 1`: loss = `-log(p)`. Large when `p` is small.
- If `y = 0`: loss = `-log(1-p)`. Large when `p` is large.

Numbers:

| Label | Prediction p | Loss |
|---|---|---|
| 1 | 0.9 | `-log 0.9 = 0.105` (good) |
| 1 | 0.5 | `0.693` (unsure) |
| 1 | 0.1 | `2.303` (bad) |
| 1 | 0.001 | `6.908` (very bad) |
| 0 | 0.001 | `0.001` (good) |
| 0 | 0.999 | `6.908` (very bad) |

Loss is averaged over the batch.

## 23.2 `BCELoss` (My original setup)

```python
D ends with nn.Sigmoid()          # output p in (0,1)
criterion = nn.BCELoss()          # expects probabilities
```

`BCELoss` takes **probabilities**, so D must already have applied sigmoid.

## 23.3 `BCEWithLogitsLoss` (Improvement)

```python
D has NO Sigmoid                  # output logit s (any real number)
criterion = nn.BCEWithLogitsLoss()  # applies sigmoid internally + BCE
```

- **Logit**: the raw score before sigmoid; any real number, e.g. `-7.3` or `+2.1`.
- **Probability**: after sigmoid, always in (0,1). `p = σ(logit)`.
- Don't confuse them: `logit = 2.0` is **not** "200%"; it corresponds to `p = 0.88`.

## 23.4 Why the second is preferred (numerical stability)

Two reasons:

1. **Direct `log(σ(s))` is fragile.** If `σ(s)` rounds to exactly 0 or 1 in float32, `log(0) = -∞`. `BCELoss` protects itself by clamping the log at −100, which creates a flat region where gradients vanish. `BCEWithLogitsLoss` uses the *log-sum-exp trick*: it computes the loss in a rearranged, stable form that never takes the log of an extremely small number.

   Stable form (for label y, logit s): `max(s,0) − s·y + log(1 + e^{−|s|})`.

2. **Cleaner gradients.** For `BCEWithLogitsLoss`, the gradient with respect to the logit is simply `σ(s) − y`. Always bounded in `[-1, 1]` and well-behaved.

## 23.5 Does it change my architecture?

Only the **last activation** (remove `Sigmoid`) and the criterion. Layers, sizes and everything else stay the same. This is a labelled **Improvement**, not a replacement of my design.

---

# 24. GAN Objective Function

## 24.1 The original formula

```
min_G  max_D  V(D, G)  =  E_{x~p_data}[ log D(x) ]  +  E_{z~p_z}[ log(1 − D(G(z))) ]
```

## 24.2 Every symbol

| Symbol | Meaning |
|---|---|
| `V(D,G)` | The "value" of the game, a single score both networks fight over |
| `max_D` | D chooses its parameters to **maximize** V |
| `min_G` | G chooses its parameters to **minimize** V |
| `E` | **Expectation** = average over many samples. `E_{x~p_data}[f(x)]` means "average of `f(x)` over real images" |
| `p_data` | The distribution of real MNIST images |
| `p_z` | The noise distribution (standard normal in my code) |
| `D(x)` | D's probability that `x` is real, in (0,1) |
| `G(z)` | A fake image from noise `z` |
| `log` | Natural logarithm. `log(1) = 0`, and `log` of small positive numbers is very negative |

## 24.3 Reading it in plain English

- **First term** `E[log D(x)]`: for real images, D wants `D(x) → 1`, which makes `log D(x) → 0` (its maximum). So D maximizes this.
- **Second term** `E[log(1 − D(G(z)))]`: for fakes, D wants `D(G(z)) → 0`, making `log(1) = 0` (maximum). So D also maximizes this.
- G influences **only the second term**. It wants `D(G(z)) → 1`, making `log(1−D) → −∞`, i.e. **minimizing** V.

## 24.4 Connection to my code

- The training average over a batch approximates `E` (mean of 64 samples).
- BCE with label 1 on real images: `−log D(x)`. That is the negative of the first term.
- BCE with label 0 on fake images: `−log(1 − D(G(z)))`. That is the negative of the second term.
- So `loss_D = loss_real + loss_fake = −V`. **Minimizing `loss_D` is the same as maximizing V.** (Code minimizes losses; the formula speaks of maximizing V.)

---

# 25. Discriminator Loss vs Generator Loss

## 25.1 D loss

```
loss_D = −log D(x)  −  log(1 − D(G(z)))
```
Real → 1, fake → 0.

## 25.2 G loss: original (minimax) vs non-saturating

**Original:** G minimizes `log(1 − D(G(z)))`.

Problem: early in training D easily rejects fakes, so `D(G(z)) ≈ 0`. In terms of the logit `s`: the gradient of `log(1−D)` with respect to `s` is `−D(G(z))`, which is **≈ 0** when `D ≈ 0`. G gets almost no learning signal exactly when it needs it most (**saturation**).

**Non-saturating (what my code uses):** G minimizes `−log D(G(z))`, i.e. BCE with label 1.

Its gradient with respect to the logit is `D(G(z)) − 1`, which is **≈ −1** when `D ≈ 0`: strong signal when G is losing.

| Version | G loss | Gradient when D confidently rejects fakes |
|---|---|---|
| Minimax | `log(1−D(G(z)))` | ≈ 0 (vanishes) |
| Non-saturating | `−log D(G(z))` | ≈ −1 (strong) |

In code: `loss_G = criterion(fake_output, real_labels)` is the non-saturating loss.

## 25.3 G never compares to a specific MNIST image

`loss_G` uses only D's opinion of the fake. No pixel-by-pixel comparison with a certain real digit exists. This is not ordinary supervised image-to-image prediction.

---

# 26. Why GAN Does Not Need a Target Image for G(z)

## 26.1 Supervised learning

```
input  ->  model  ->  prediction  <--compare-->  TARGET (known correct answer)
```
Example: image of "7" → classifier → "7". Each input has an answer.

## 26.2 GAN

```
random noise -> G -> generated image -> D -> "real or fake?" score -> loss
```

There is **no correct digit for a given noise vector**. Which digit should `z = [0.3, -1.2, ...]` produce? Nobody knows, and it doesn't matter. What matters is that the *collection* of outputs looks like MNIST.

## 26.3 How G learns without targets

D has seen real digits and learned what real ones look like. When D scores a fake low, the gradient through D says *"change these pixels to look more like the things I called real."* G follows. D **is the target**, a learned, moving definition of "real". As G improves, D must sharpen its criteria, which pushes G further.

Think of a painter who never sees the original reference paintings but has a critic who has studied them: the critic's reactions alone guide the painter.

---

# 27. The Training Loop

```python
for epoch in range(num_epochs):
    for batch_idx, real_images in enumerate(train_loader):
        # Phase A: train D
        # Phase B: train G
```

## 27.1 Exactly what happens

1. Outer loop: repeat for each epoch (20 times).
2. Inner loop: the DataLoader shuffles and yields 938 batches.
3. For each batch: run Phase A (D), then Phase B (G), then log losses.
4. After the last batch: print averages, save checkpoint, move to the next epoch.

Parameters are **never reset** between batches or epochs. Each update builds on the last.

## 27.2 Variable table (batch size 64)

| Variable | Meaning | Shape |
|---|---|---|
| `real_images` | Batch of real MNIST digits | `[64, 1, 28, 28]` |
| `real_labels` | Targets of 1 | `[64, 1]` |
| `fake_labels` | Targets of 0 | `[64, 1]` |
| `z` | Random latent vectors | `[64, 100]` |
| `fake_images` | Generator output | `[64, 1, 28, 28]` |
| `real_output` | D's score for real images | `[64, 1]` |
| `fake_output` | D's score for fake images | `[64, 1]` |
| `loss_real` | D's error on real images | scalar `[]` |
| `loss_fake` | D's error on fake images | scalar `[]` |
| `loss_D` | `loss_real + loss_fake` | scalar `[]` |
| `loss_G` | G's loss (wants D to say real) | scalar `[]` |

(For the final batch, 64 becomes 32.)

---

# 28. What Happens During One Epoch?

```
60,000 images, batch_size = 64   ->   60000 / 64 = 937.5   ->   938 batches

Epoch 1:
  Batch 1    (64 images)   D update, G update
  Batch 2    (64 images)   D update, G update
  ...
  Batch 937  (64 images)   D update, G update
  Batch 938  (32 images)   D update, G update
Epoch 2:
  (reshuffled)  Batch 1 ... Batch 938
...
Epoch 20
```

- Parameters carry over: batch 2 starts with the weights left by batch 1.
- Per epoch: **938 D updates + 938 G updates**.
- Total over 20 epochs: 18,760 D updates and 18,760 G updates.
- One iteration = one batch = one D update + one G update.

---

# 29. Model Parameters

**Parameters** are the numbers the network learns: every **weight** and every **bias**. A `Linear(in, out)` has `in × out` weights and `out` biases:

```
params = in × out + out
```

## 29.1 Generator

| Layer | Weights | Biases | Total |
|---|---|---|---|
| `Linear(100, 256)` | 100×256 = 25,600 | 256 | **25,856** |
| `Linear(256, 512)` | 256×512 = 131,072 | 512 | **131,584** |
| `Linear(512, 784)` | 512×784 = 401,408 | 784 | **402,192** |
| **Total** | | | **559,632** |

## 29.2 Discriminator

| Layer | Weights | Biases | Total |
|---|---|---|---|
| `Linear(784, 512)` | 784×512 = 401,408 | 512 | **401,920** |
| `Linear(512, 256)` | 512×256 = 131,072 | 256 | **131,328** |
| `Linear(256, 1)` | 256×1 = 256 | 1 | **257** |
| **Total** | | | **533,505** |

ReLU, LeakyReLU, Tanh, Sigmoid have **no** parameters (fixed formulas). Both weights and biases are learned by backprop and the optimizer.

Verify in code:

```python
print(sum(p.numel() for p in G.parameters() if p.requires_grad))   # 559632
print(sum(p.numel() for p in D.parameters() if p.requires_grad))   # 533505
```

---
# 30. Training Instability

GAN training is a **two-player game**, not the minimization of one fixed loss. Each network's target keeps moving as the other changes. This causes several problems.

| Problem | Simple explanation |
|---|---|
| **D too strong** | D rejects every fake with near-certainty. G's gradients become useless or extreme, and G can't tell how to improve. |
| **G too weak** | G is never close enough to real for D to be uncertain, so the feedback is always "completely fake". |
| **Mode collapse** | G finds a few outputs (e.g. only "1"s) that fool D and produces only those. Variety collapses. D then learns to reject those, G jumps to another narrow set, and so on. |
| **Oscillation** | G and D chase each other: G beats D, D adapts, G loses, ... The losses go up and down without converging. |
| **Vanishing gradients** | With saturated sigmoids or the minimax G loss, gradient ≈ 0 and G stops learning (Section 25). |
| **Numerical instability** | `log(p)` with `p → 0` goes to `−∞`; large losses (like 40 to 100) and exploding gradient values appear. |

## 30.1 Situation: `loss_D ≈ 0` and `loss_G` very large

What it means: D classifies real as real and fake as fake with extreme confidence, so `loss_D ≈ 0`. And since `D(G(z)) ≈ 0`, G's loss `−log D(G(z))` is huge.

In plain terms: **D has overpowered G**. G is not fooling D at all.

> **Do not assume low D loss means success.** In a healthy GAN the opponents are balanced, with D unsure and `D(x) ≈ 0.5`, giving `loss_D ≈ 1.386` (that is `2 × log 2`) at theoretical equilibrium. A `loss_D` near 0 usually signals **imbalance**, not victory.

---

# 31. My Previous Training Output

Example from my earlier run:

```
Epoch 10:
loss_D ≈ 0
loss_G ≈ 48
```

## 31.1 What may be happening

- D separates real and fake almost perfectly (`loss_D ≈ 0`).
- `loss_G ≈ 48` means `−log D(G(z)) ≈ 48`, so `D(G(z)) ≈ e^{−48} ≈ 1e−21`. D thinks the fakes are real with probability about 10⁻²¹.
- So G is losing badly: either G is producing images D can trivially detect, or training has collapsed/diverged.

## 31.2 Why BCELoss can reach such large values

`BCELoss` takes the probability `p` after Sigmoid. When D is extremely confident, `p` becomes astronomically small (like 1e-21), and `−log(p)` becomes large (≈ 48). `BCELoss` caps the log at −100, so the loss can be as large as 100. Gradients in this regime are unreliable: the sigmoid is flat there, so the signal back to G nearly vanishes.

## 31.3 Why BCEWithLogitsLoss helps

It works on the raw logit and uses a stable formula, so there's no unstable `log(σ(s))` of a tiny number. The G loss then grows only **linearly** with the logit (for large negative `s`, `−log σ(s) ≈ −s`), the gradient with respect to the logit stays near −1 instead of vanishing, and G keeps receiving a usable signal.

This makes training more stable but **does not guarantee** a balanced game. If D still dominates, it can still need other fixes (Section 30/38).

## 31.4 The progress-bar number may be the LAST batch

In my original loop I kept `epoch_G_loss = loss_G.item()` after the inner loop ended. That is the loss from **only the final batch** (the 32-image one), not the average over the epoch. It can mislead: a spike in one batch looks like the whole epoch's state. Use averages (Section 37).

---

# 32. Model Evaluation / Testing

**Testing a GAN means generating images and looking at them.** There's no target, no accuracy.

```python
G.eval()                                  # evaluation mode

z = torch.randn(16, 100, device=device)   # [16, 100]

with torch.no_grad():                     # no gradient tracking
    fake_images = G(z)                    # [16, 1, 28, 28]

fake_images = (fake_images + 1) / 2       # [-1,1] -> [0,1]
```

## 32.1 Explanation

| Piece | Why |
|---|---|
| `G.eval()` | Switches layers like Dropout/BatchNorm to inference behaviour. My G has neither, so it changes nothing today, but it's the correct habit and prevents bugs if layers are added later. |
| `torch.no_grad()` | Stops building the computational graph. Saves memory and time; we don't need gradients. |
| random `z` | G only needs noise to generate. There is no input image and no label. |
| no `loss.backward()` | We are not learning, so no gradient is needed. |
| no `optimizer.step()` | No parameter update. We only want to observe G's current ability. |

Flow: `z -> G(z) -> generated image`.

Output shape: `[16, 1, 28, 28]`.

## 32.2 Convert `[-1,1]` to `[0,1]`

Matplotlib's grayscale display expects `[0,1]` (or `0..255`):

```
x_display = (x + 1) / 2
-1 -> 0.0 ; 0 -> 0.5 ; +1 -> 1.0
```

## 32.3 Code to show 16 images

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(4, 4, figsize=(6, 6))
for img, ax in zip(fake_images, axes.flatten()):
    ax.imshow(img.cpu().squeeze(), cmap="gray")   # [1,28,28] -> [28,28]
    ax.axis("off")
plt.tight_layout()
plt.show()
```

`.cpu()` moves data off the GPU (matplotlib needs CPU); `.squeeze()` removes the channel dimension of size 1.

---

# 33. Loading a Saved Generator

## 33.1 Saving

```python
torch.save(G.state_dict(), "generator_last.pth")
```

**`state_dict`** is a Python dictionary mapping each layer's parameter name to its tensor:

```
{
 "model.0.weight": tensor of shape [256, 100],
 "model.0.bias":   tensor of shape [256],
 "model.2.weight": tensor of shape [512, 256],
 "model.2.bias":   tensor of shape [512],
 "model.4.weight": tensor of shape [784, 512],
 "model.4.bias":   tensor of shape [784],
}
```

(Names like `model.0` are the positions inside `nn.Sequential`. ReLU/Tanh have no parameters, so indexes 1, 3, 5 are absent.)

It stores **only the learned numbers**, not the code or the architecture.

## 33.2 Loading

```python
G = Generator(noise_dim=100).to(device)
G.load_state_dict(torch.load("generator_last.pth", map_location=device))
G.eval()
```

**Why recreate the architecture first?** The file contains only numbers. `load_state_dict` has to copy each tensor into a matching parameter slot with matching name and shape. Without a `Generator` object that has those slots, there's nowhere to put them. If the architecture differs (e.g. a changed layer size), loading fails with a size-mismatch error.

`map_location=device` loads the tensors onto the right device (for example, a model trained on GPU loaded on a CPU-only machine).

---

# 34. Reproducibility

Random numbers in a computer are **pseudo-random**: generated by an algorithm from a starting number, the **seed**. Same seed, same sequence.

```python
torch.manual_seed(42)
z = torch.randn(16, 100)     # always the same 16×100 numbers
```

Because G is a deterministic function, the **same trained G + same `z` = the same images**. This lets you:

- compare two checkpoints on identical noise,
- share results exactly,
- debug.

Notes: for GPU there's `torch.cuda.manual_seed_all(42)`; training itself may still vary slightly on GPU due to non-deterministic kernels. Seed *before* creating `z`. A different seed gives different noise and therefore different images.

---

# 35. Testing Different Noise Vectors

```
z1 -> G(z1) -> image A (maybe a "3")
z2 -> G(z2) -> image B (maybe a "7")
z3 -> G(z3) -> image C (maybe a "0")
```

Because G is a fixed function, **different inputs lead to different outputs**. A different `z` lands at a different point in the space G maps to digits.

## 35.1 Latent space (beginner level)

The **latent space** is the space of all possible noise vectors (100-dimensional for me). Every point `z` in it corresponds to one image `G(z)`. "Latent" means hidden: we never see these coordinates in the data, they are internal codes.

A trained G tends to be **smooth**: nearby points produce similar images, and moving gradually between two `z`'s gradually changes one image into another.

```python
z1 = torch.randn(1, 100, device=device)
z2 = torch.randn(1, 100, device=device)
alphas = torch.linspace(0, 1, 8, device=device).view(-1, 1)
z_interp = (1 - alphas) * z1 + alphas * z2        # [8, 100]
with torch.no_grad():
    imgs = (G(z_interp) + 1) / 2                  # [8, 1, 28, 28]
```

This blends gradually from the first noise vector to the second, which shows how the image morphs.

---

# 36. Visualizing Training Progress

**Why:** loss curves of GANs are hard to read (Section 37). **Looking at the images is the best progress signal.**

## 36.1 Use fixed noise

```python
# before the training loop
fixed_z = torch.randn(16, 100, device=device)
```

## 36.2 After every epoch

```python
G.eval()
with torch.no_grad():
    samples = (G(fixed_z) + 1) / 2
G.train()

fig, axes = plt.subplots(4, 4, figsize=(5, 5))
for img, ax in zip(samples, axes.flatten()):
    ax.imshow(img.cpu().squeeze(), cmap="gray")
    ax.axis("off")
plt.suptitle(f"Epoch {epoch+1}")
plt.savefig(f"samples_epoch_{epoch+1:02d}.png")
plt.close()
```

## 36.3 Why fixed noise matters

If you used new random noise each epoch, differences between epochs could come from different `z`, not better G. With a **fixed** `z`, the same 16 inputs are used every epoch, so any change in the images comes **only from G's learning**. You can literally watch the same 16 blobs turn into digits.

Typical progression: epoch 1 = noisy blobs → epoch 3 = rough strokes → epoch 10+ = recognizable digits.

---

# 37. Loss Monitoring

## 37.1 Correct averaging

Wrong (last batch only):

```python
epoch_G_loss = loss_G.item()   # only the final batch
```

Correct:

```python
for epoch in range(num_epochs):
    total_D_loss = 0.0
    total_G_loss = 0.0

    for real_images in train_loader:
        ...
        total_D_loss += loss_D.item()     # .item() -> plain Python float
        total_G_loss += loss_G.item()

    epoch_D_loss = total_D_loss / len(train_loader)   # len(train_loader) = 938
    epoch_G_loss = total_G_loss / len(train_loader)
```

`.item()` converts a 1-element tensor into a Python number (and detaches it from the graph; storing the loss tensor itself would keep graphs alive and leak memory).

(The last batch is a bit smaller, so this is a mean of per-batch means; that's fine for monitoring.)

## 37.2 Why GAN losses aren't like supervised losses

| Supervised learning | GAN |
|---|---|
| One fixed loss with a fixed target | Two losses that fight each other |
| Loss going down = model improving | Loss going down may mean one side is winning unfairly |
| Loss curve is a good progress indicator | Loss curve is a weak indicator |

At a healthy balance the losses hover and fluctuate instead of declining. Rising `loss_G` can mean D is improving, not that G got worse. **Judge a GAN by sample quality and diversity, not by loss values.**

---

# 38. Common GAN Coding Mistakes

| # | Mistake | What happens | Why it's wrong | Fix |
|---|---|---|---|---|
| 1 | Normalization not matching Tanh | Real images in `[0,1]` but G outputs `[-1,1]` | D separates real from fake by range alone | Normalize reals to `[-1,1]`; or change G's last activation |
| 2 | No `detach()` in D training | Gradients flow into G; extra compute; graph errors | D's loss should not train G | `D(fake_images.detach())` |
| 3 | Sigmoid with `BCEWithLogitsLoss` | Sigmoid applied twice, outputs squashed, weak gradients | The loss already includes sigmoid | Remove D's Sigmoid |
| 4 | Sigmoid applied twice | Same as above, probabilities distorted, slow learning | Double squashing | Use one sigmoid, either in model + `BCELoss` or inside the loss |
| 5 | Updating the wrong optimizer | e.g. `optimizer_D.step()` after `loss_G.backward()`: G never updates (or D is changed by G's loss) | Each network must be updated by its own loss | `loss_D` → `optimizer_D`; `loss_G` → `optimizer_G` |
| 6 | Forgetting `zero_grad()` | Gradients accumulate across batches | PyTorch adds, not overwrites | Call before every `backward()` |
| 7 | Wrong label shape | Broadcasting to `[64,64]` or a size error | Labels must match D's output | `torch.ones(batch, 1)` |
| 8 | Wrong tensor shape | `mat1 and mat2 shapes cannot be multiplied` | Linear needs matching last dim | Check shapes with `print(x.shape)` and use `view(-1, 784)` |
| 9 | Saving the model every batch | Disk writes 938 times per epoch, which is slow | Unneeded; the model barely changes in one batch | Save once per epoch |
| 10 | Last-batch loss mistaken for epoch loss | Misleading numbers | Last batch is small and noisy | Accumulate and average |
| 11 | Expecting G loss to decrease continuously | Panic when it rises | D improves too; G's loss is relative to D | Look at images |
| 12 | Believing `D loss = 0` means perfection | You think the GAN is perfect while G is failing | D loss near 0 means D dominates | Check samples; balance D and G (lower D's lr, add label smoothing, etc.) |

---
# 39. Complete Corrected Code

> **Improvement notice.** The architecture is **unchanged** (fully connected, same layer sizes, same activations in G; D has the same layers). The only labelled changes versus my original are:
> 1. D's final `Sigmoid` removed, loss is `BCEWithLogitsLoss`.
> 2. `criterion` and the training loop defined **once** (my notebook had them duplicated).
> 3. Average epoch losses instead of last-batch losses.
> 4. Model saved once per epoch.
> 5. Fixed-noise sample images saved each epoch.

```python
# =========================================================
# 0. IMPORTS AND DEVICE
# =========================================================
import os
import struct
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

torch.manual_seed(42)                       # reproducibility
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# =========================================================
# 1. SETTINGS
# =========================================================
batch_size    = 64
noise_dim     = 100
learning_rate = 0.0002
num_epochs    = 20
train_path    = "/kaggle/input/datasets/hojjatk/mnist-dataset/train-images.idx3-ubyte"
os.makedirs("samples", exist_ok=True)

# =========================================================
# 2. MNIST IDX LOADER
# =========================================================
def load_mnist_images(file_path):
    """Read an MNIST .idx3-ubyte file -> float tensor [N, 28, 28]."""
    with open(file_path, "rb") as f:
        # header: magic number, number of images, rows, cols (big-endian uint32 each)
        _, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)    # raw pixels 0..255
    images = data.reshape(num_images, rows, cols)         # [60000, 28, 28]
    return torch.tensor(images, dtype=torch.float32)


def normalize_pixels(images):
    """0..255 -> 0..1 -> -1..1 (matches the Generator's Tanh)."""
    return (images / 255.0) * 2 - 1

# =========================================================
# 3. CUSTOM DATASET + DATALOADER
# =========================================================
class MNISTDataset(Dataset):
    def __init__(self, file_path):
        self.images = normalize_pixels(load_mnist_images(file_path))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index].unsqueeze(0)            # [28,28] -> [1,28,28]


train_dataset = MNISTDataset(train_path)
train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
print("Images:", len(train_dataset), "| Batches per epoch:", len(train_loader))

# =========================================================
# 4. MODELS  (same fully connected architecture)
# =========================================================
class Generator(nn.Module):
    """noise [B,100] -> fake image [B,1,28,28]"""
    def __init__(self, noise_dim=100):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 256), nn.ReLU(),
            nn.Linear(256, 512),       nn.ReLU(),
            nn.Linear(512, 784),       nn.Tanh(),         # output in [-1, 1]
        )

    def forward(self, z):
        return self.model(z).view(-1, 1, 28, 28)


class Discriminator(nn.Module):
    """image [B,1,28,28] -> logit [B,1]   (IMPROVEMENT: no Sigmoid)"""
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 1),                            # raw score (logit)
        )

    def forward(self, x):
        return self.model(x.view(-1, 784))


G = Generator(noise_dim).to(device)
D = Discriminator().to(device)

print("G parameters:", sum(p.numel() for p in G.parameters()))   # 559632
print("D parameters:", sum(p.numel() for p in D.parameters()))   # 533505

# =========================================================
# 5. LOSS AND OPTIMIZERS
# =========================================================
criterion   = nn.BCEWithLogitsLoss()      # sigmoid + BCE, numerically stable
optimizer_G = optim.Adam(G.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# =========================================================
# 6. HELPER: SAVE / SHOW A GRID OF GENERATED IMAGES
# =========================================================
def make_grid_figure(images, title=None):
    """images: [16,1,28,28] in [-1,1]"""
    images = ((images + 1) / 2).cpu()                     # -> [0,1]
    fig, axes = plt.subplots(4, 4, figsize=(5, 5))
    for img, ax in zip(images, axes.flatten()):
        ax.imshow(img.squeeze(), cmap="gray")             # [1,28,28] -> [28,28]
        ax.axis("off")
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    return fig

fixed_z = torch.randn(16, noise_dim, device=device)       # same noise every epoch

# =========================================================
# 7. TRAINING
# =========================================================
history = {"D": [], "G": []}

for epoch in range(num_epochs):
    G.train(); D.train()
    progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")
    total_D_loss = 0.0
    total_G_loss = 0.0

    for real_images in progress_bar:
        real_images = real_images.to(device)              # [B,1,28,28]
        bs = real_images.size(0)                          # 64 (last batch: 32)

        real_labels = torch.ones(bs, 1, device=device)    # [B,1]
        fake_labels = torch.zeros(bs, 1, device=device)   # [B,1]

        # ---------------- TRAIN DISCRIMINATOR ----------------
        z = torch.randn(bs, noise_dim, device=device)     # [B,100]
        fake_images = G(z)                                # [B,1,28,28]

        loss_real = criterion(D(real_images), real_labels)
        loss_fake = criterion(D(fake_images.detach()), fake_labels)   # detach!
        loss_D = loss_real + loss_fake

        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        # ---------------- TRAIN GENERATOR --------------------
        z = torch.randn(bs, noise_dim, device=device)     # NEW noise
        fake_images = G(z)

        loss_G = criterion(D(fake_images), real_labels)   # G wants "real"; no detach

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        # ---------------- LOGGING ----------------------------
        total_D_loss += loss_D.item()
        total_G_loss += loss_G.item()
        progress_bar.set_postfix(loss_D=f"{loss_D.item():.4f}",
                                 loss_G=f"{loss_G.item():.4f}")

    # ---------------- END OF EPOCH ----------------------------
    epoch_D_loss = total_D_loss / len(train_loader)
    epoch_G_loss = total_G_loss / len(train_loader)
    history["D"].append(epoch_D_loss)
    history["G"].append(epoch_G_loss)
    print(f"Epoch {epoch+1}/{num_epochs} | avg D loss: {epoch_D_loss:.4f} | avg G loss: {epoch_G_loss:.4f}")

    # save sample images from the SAME fixed noise
    G.eval()
    with torch.no_grad():
        samples = G(fixed_z)
    fig = make_grid_figure(samples, title=f"Epoch {epoch+1}")
    fig.savefig(f"samples/epoch_{epoch+1:02d}.png")
    plt.close(fig)

    # save checkpoints once per epoch
    torch.save(G.state_dict(), "generator_last.pth")
    torch.save(D.state_dict(), "discriminator_last.pth")

# =========================================================
# 8. LOSS CURVES (for reference only; judge GANs by images)
# =========================================================
plt.plot(history["D"], label="D loss")
plt.plot(history["G"], label="G loss")
plt.xlabel("Epoch"); plt.ylabel("Average loss"); plt.legend(); plt.show()

# =========================================================
# 9. TESTING: LOAD SAVED GENERATOR AND GENERATE 16 IMAGES
# =========================================================
G_test = Generator(noise_dim).to(device)
G_test.load_state_dict(torch.load("generator_last.pth", map_location=device))
G_test.eval()

torch.manual_seed(123)                                    # optional: repeatable test noise
z_test = torch.randn(16, noise_dim, device=device)        # [16,100]
with torch.no_grad():
    generated = G_test(z_test)                            # [16,1,28,28]
print("Generated shape:", generated.shape)

fig = make_grid_figure(generated, title="Generated digits")
plt.show()
```

---

# 40. Final Architecture Summary

```
NOISE z
[batch, 100]
   ↓
G   Linear(100→256) ReLU → Linear(256→512) ReLU → Linear(512→784) Tanh → view
   ↓
[batch, 1, 28, 28]   fake image   (real images: same shape, range [-1,1])
   ↓
D   view(-1,784) → Linear(784→512) LeakyReLU → Linear(512→256) LeakyReLU → Linear(256→1)
   ↓
[batch, 1]   score
```

| Concept | Summary |
|---|---|
| Generator goal | Turn noise into images that fool D |
| Discriminator goal | Real → 1, fake → 0 |
| D loss | `criterion(D(real),1) + criterion(D(G(z).detach()),0)` |
| G loss | `criterion(D(G(z)), 1)` (non-saturating) |
| Forward pass | Data flows through layers to a prediction/loss |
| Backward pass | `loss.backward()`: gradients of loss wrt all parameters (chain rule) |
| `detach()` | Cuts the graph so D's loss doesn't send gradients into G |
| `zero_grad()` | Clears accumulated gradients before each backward |
| `step()` | Updates parameters using gradients (Adam) |
| Batch | 64 images processed together |
| Iteration | One batch: one D update + one G update |
| Epoch | One full pass: 938 iterations |

---

# 41. Very Important Conceptual Questions

**1. Why does the Generator need random noise?**
A network is deterministic. Noise supplies the randomness that makes each output different, giving variety.

**2. Why is the noise dimension 100?**
It's a design hyperparameter: a conventional value, large enough for variety. 64 or 128 would also work. Nothing magical.

**3. Why does the Generator output 784 values?**
One value per pixel: 28 × 28 = 784.

**4. Why is the output reshaped to 28x28?**
So it matches the shape of real images (`[1,28,28]`) and can be shown/fed to D as an image.

**5. Why does the Generator use Tanh?**
It bounds outputs to (−1, 1), matching the normalized real pixel range.

**6. Why are real images normalized to [-1,1]?**
So real and fake images share a range (Tanh), and centered small values help training.

**7. Why does the Discriminator flatten the image?**
`Linear` layers need a vector of 784 features per sample, so `[1,28,28]` becomes `[784]`.

**8. Why does D output one value?**
It makes one binary decision per image (real vs fake), so a single score.

**9. Why are real labels 1?**
Convention: 1 = "real", matching D's output meaning for BCE.

**10. Why are fake labels 0?**
0 = "fake", the opposite class.

**11. Why `fake_images.detach()` while training D?**
So D's loss doesn't send gradients into G; only D should be updated in that phase.

**12. Why no `detach()` while training G?**
The gradient must travel through D to reach G. Detaching would cut the path and G could not learn.

**13. Why does G use `real_labels` in its loss?**
G wants D to call its fakes real, so the target for `D(G(z))` is 1.

**14. Why does D participate in G's backward pass?**
`loss_G` depends on G through D. The chain rule passes through D's layers to reach G's weights.

**15. Why does `optimizer_G` update only G?**
It was built with `G.parameters()` only; D's weights aren't registered with it.

**16. What exactly happens when `loss.backward()` is called?**
Autograd walks the computational graph in reverse applying the chain rule and **adds** `∂loss/∂param` into each parameter's `.grad`.

**17. What exactly happens when `optimizer.step()` is called?**
The optimizer (Adam) reads each parameter's `.grad`, updates its internal moment estimates, and changes each parameter accordingly (conceptually `θ ← θ − lr × scaled gradient`).

**18. Why do we call `zero_grad()`?**
Gradients accumulate by default. Clearing prevents mixing old batches into the new update.

**19. Difference between iteration, batch, epoch?**
Batch = group of 64 images. Iteration = processing one batch (one update). Epoch = one full pass over all 60,000 images (938 iterations).

**20. Why can GAN losses be unstable?**
Two networks chase moving targets; there is no fixed objective, and the log-based loss can saturate or explode.

**21. Why can D loss become almost zero?**
D becomes confident and correct, e.g. when G is too weak or D too strong.

**22. Why can G loss become very large?**
When `D(G(z)) ≈ 0`, `−log D(G(z))` is huge. G is being decisively rejected.

**23. Why can GAN loss not be interpreted like normal classification loss?**
The data distribution G produces changes as it trains, and D's difficulty changes too. Lower loss for one network often means the other is losing.

**24. Why does testing require only random noise?**
G maps noise to an image. There's no input image or label at generation time.

**25. Why do we use `G.eval()`?**
To switch layers like Dropout/BatchNorm to inference mode. Not needed for my G today but good practice.

**26. Why do we use `torch.no_grad()`?**
Saves memory and time as no gradient graph is built.

**27. What is `state_dict()`?**
A dictionary of all learned parameter tensors keyed by their names.

**28. Why can two different noise vectors generate different digits?**
G is a function; different inputs give different outputs, landing at different places in the learned image distribution.

**29. What does the latent space mean?**
The space of all possible noise vectors; each point maps to one generated image.

**30. How does G learn to generate digits with no target image per noise vector?**
D is the teacher. D learned what real digits look like, and its gradient tells G how to change pixels to look more real. G aims for the right *distribution*, not a specific pair `(z, image)`.

---

# 42. Beginner Mental Model

**Generator:** *"I take random noise and try to create something that looks real."*

**Discriminator:** *"I look at an image and try to determine whether it is real or generated."*

**During D training:** *"Teach D to distinguish real and fake."*

**During G training:** *"Teach G to fool D."*

## 42.1 Mathematically

```
D wants:  maximize  log D(x) + log(1 − D(G(z)))
G wants:  maximize  log D(G(z))          (non-saturating form)
```

At the ideal end: `p_g = p_data`, so D cannot do better than guessing, `D(x) = 0.5` everywhere.

## 42.2 In PyTorch code

```python
# D: "teach D to distinguish real and fake"
loss_D = criterion(D(real), ones) + criterion(D(G(z).detach()), zeros)
optimizer_D.zero_grad(); loss_D.backward(); optimizer_D.step()

# G: "teach G to fool D"
loss_G = criterion(D(G(z_new)), ones)
optimizer_G.zero_grad(); loss_G.backward(); optimizer_G.step()
```

## 42.3 One-picture memory

```
         ┌────────── D learns from ──────────┐
         │   real digits (label 1)           │
         │   G's fakes   (label 0)           │
         └───────────────────────────────────┘
                         ▲
noise ──> G ──> fake ────┤
         ▲               │
         └── G learns from D's reaction (label 1 on its own fakes)
```

**Whole loop in one line:**

```
data -> batch -> noise -> G forward -> D forward -> loss -> backward -> gradients -> optimizer step -> next batch -> next epoch -> test with G(z)
```

*End of document.*
