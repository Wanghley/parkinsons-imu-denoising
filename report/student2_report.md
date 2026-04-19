# ECE 685D — Project 2: Denoising Autoencoders for 1D Time-Series Signals
## Student 2 Contribution Report: Modeling & Evaluation

---

## 1. Overview of Responsibilities

Student 2's role covers four interconnected areas:

| Responsibility | Deliverable |
|---|---|
| Model Architectures | FC, 1D-CNN, and LSTM autoencoders (`models.py`) |
| Evaluation Metrics | MSE and SNR computation pipeline (`metrics.py`) |
| Hyperparameter Tuning | Grid search over learning rate and batch size (`experiments.py`) |
| Joint Experiments | Latent dimension study and noise robustness analysis |

---

## 2. Model Architectures

All models follow the encoder–decoder paradigm with a bottleneck latent representation $z$:

$$z = \text{Encoder}(x_{\text{noisy}}), \quad \hat{x} = \text{Decoder}(z)$$

The models are trained to minimize the mean-squared reconstruction loss against the **clean** signal:

$$\mathcal{L}(\theta) = \mathbb{E}\left[\|x - f_\theta(x_{\text{noisy}})\|_2^2\right]$$

### 2.1 Baseline: Fully-Connected (FC) Autoencoder

The FC autoencoder treats each signal window as a flat vector and applies purely linear transformations with ReLU non-linearities.

**Encoder architecture** (signal length $L = 512$):

$$L \xrightarrow{\text{Linear+ReLU}} 256 \xrightarrow{\text{Linear+ReLU}} 128 \xrightarrow{\text{Linear}} d_z$$

**Decoder architecture** (mirrors encoder):

$$d_z \xrightarrow{\text{Linear+ReLU}} 128 \xrightarrow{\text{Linear+ReLU}} 256 \xrightarrow{\text{Linear}} L$$

**Parameter count:** ~345K (for $d_z = 64$).

**Motivation:** The FC autoencoder serves as a parameter-rich baseline with no inductive bias toward temporal structure. It can in principle learn arbitrary input–output mappings, but requires the network to rediscover temporal locality from scratch. It is expected to underperform architectures with appropriate inductive biases on real-world signals.

**Limitation:** The FC architecture does not exploit the fact that neighboring samples in a time series are correlated. Weight matrices are dense and do not share parameters across time steps, making the model inefficient in both parameters and generalization.

---

### 2.2 Advanced Model A: 1D Convolutional Autoencoder (CNN)

The 1D-CNN autoencoder uses strided convolutions and transposed convolutions to build a hierarchical, translation-equivariant representation of the signal.

**Encoder architecture:**

| Layer | Operation | Output shape |
|---|---|---|
| Input | — | $(B, 1, 512)$ |
| Conv1 | Conv1d(1→32, k=3, s=2, p=1) + ReLU | $(B, 32, 256)$ |
| Conv2 | Conv1d(32→64, k=3, s=2, p=1) + ReLU | $(B, 64, 128)$ |
| Conv3 | Conv1d(64→128, k=3, s=2, p=1) + ReLU | $(B, 128, 64)$ |
| Conv4 | Conv1d(128→256, k=3, s=2, p=1) + ReLU | $(B, 256, 32)$ |
| FC | Flatten → Linear($8192 \to d_z$) | $(B, d_z)$ |

**Decoder architecture** (reverses with ConvTranspose1d, kernel=4, stride=2, padding=1):

$$d_z \xrightarrow{\text{FC}} 8192 \xrightarrow{\text{Reshape}} (256,32) \xrightarrow{\times 4\text{ ConvT+ReLU}} (1,512)$$

**Parameter count:** ~1.36M (for $d_z = 64$).

**Motivation:** Convolution kernels share parameters across time and learn local patterns such as edges, transients, and periodic structures. The hierarchical downsampling builds increasingly abstract representations of the signal while retaining temporal locality. This inductive bias is well-matched to the structure of audio, ECG, and sensor signals.

**Design choices:**
- Kernel size 3 for encoder (odd, symmetric receptive field); kernel size 4 for decoder to ensure exact output length reconstruction.
- Four downsampling stages achieve a $16\times$ compression ratio in the temporal dimension before the FC bottleneck.

---

### 2.3 Advanced Model B: LSTM Autoencoder

The LSTM autoencoder treats the signal as a sequence of scalar time steps and uses recurrent connections to capture long-range temporal dependencies.

**Encoder:**

$$\text{BiLSTM}(\text{input\_size}=1,\ \text{hidden}=128,\ \text{layers}=2) \xrightarrow{\text{mean pool}} \mathbb{R}^{256} \xrightarrow{\text{Linear}} \mathbb{R}^{d_z}$$

A bidirectional LSTM reads the sequence in both directions; all hidden states are averaged (temporal mean pooling) to produce a fixed-size context vector before the bottleneck projection.

**Decoder:**

$$\mathbb{R}^{d_z} \xrightarrow{\text{Linear}} \mathbb{R}^{128} \xrightarrow{\text{broadcast}} (B, L, 128) \xrightarrow{\text{LSTM}} (B, L, 128) \xrightarrow{\text{Linear}} (B, L)$$

The latent vector is projected to the LSTM hidden size and broadcast across all $L$ time steps as a constant input sequence. The decoder LSTM then reconstructs each output step.

**Parameter count:** ~818K (for $d_z = 64$).

**Motivation:** LSTMs excel at capturing long-range dependencies (e.g., the phase of a sinusoid many samples ahead). They are particularly suited to biomedical or speech signals where the signal history matters over hundreds of samples. The bidirectional encoder provides full-context encoding at the cost of requiring the whole sequence at once (non-causal).

**Limitation:** LSTMs are slower to train than CNNs and may struggle with very long sequences due to vanishing gradient effects, even with gating mechanisms.

---

## 3. Evaluation Metrics

### 3.1 Mean Squared Error (MSE)

$$\text{MSE}(\hat{x}, x) = \frac{1}{NL} \sum_{i=1}^{N} \sum_{t=1}^{L} (x_{it} - \hat{x}_{it})^2$$

MSE directly measures the average squared deviation between the reconstructed and clean signals. It is used as both the training loss and a primary evaluation metric. Lower is better.

### 3.2 Signal-to-Noise Ratio (SNR)

$$\text{SNR}(x, \hat{x}) = 10 \log_{10}\left(\frac{\|x\|_2^2}{\|x - \hat{x}\|_2^2}\right) \quad [\text{dB}]$$

SNR normalizes reconstruction error by signal energy, making it invariant to signal amplitude scaling. Reported in decibels; higher is better. A 3 dB improvement corresponds to halving the noise power.

### 3.3 SNR Improvement (SNRi)

$$\text{SNRi} = \text{SNR}(x, \hat{x}) - \text{SNR}(x, x_{\text{noisy}})$$

SNRi measures how much the model improves over the noisy baseline. A positive value means the model successfully denoised the signal. This metric is used in the noise robustness experiment to compare models at different noise levels.

### 3.4 Evaluation Pipeline

The function `evaluate_model()` in `metrics.py` runs full inference over a test DataLoader and returns all three metrics. For comparability across experiments, all models are evaluated on the same held-out test set (10% of the dataset), which was never used during training or validation.

---

## 4. Hyperparameter Tuning

### 4.1 Search Space

A grid search was performed over the following hyperparameters using the CNN architecture on 50 epochs (reduced for efficiency):

| Hyperparameter | Values searched |
|---|---|
| Learning rate | $10^{-4},\ 5 \times 10^{-4},\ 10^{-3}$ |
| Batch size | 32, 64, 128 |

The objective was to minimize validation MSE loss.

### 4.2 Training Strategy

All models use the **Adam optimizer** with weight decay $10^{-5}$ for regularization. A **ReduceLROnPlateau** scheduler (patience = 10 epochs, factor = 0.5) halves the learning rate when validation loss plateaus, allowing efficient exploration without manual scheduling.

Gradient clipping (max norm = 1.0) prevents gradient explosions in the LSTM model during early training.

### 4.3 Recommended Configuration

Based on the grid search, the recommended configuration for the CNN autoencoder is:

- **Learning rate:** $10^{-3}$ (higher LRs with Adam generally converge faster on smooth losses)
- **Batch size:** 64 (balance between gradient quality and training throughput)
- **Epochs:** 100 with early stopping via ReduceLROnPlateau

These settings are used as defaults in `config.py` for all subsequent experiments.

---

## 5. Experiments

### 5.1 Experiment 1: Architecture Comparison

**Setup:** All three models (FC, CNN, LSTM) are trained with the same noise function (Gaussian, $\sigma = 0.1$), batch size (64), learning rate ($10^{-3}$), and number of epochs (100). Each model is evaluated on the held-out test set.

**Metrics reported:** MSE, output SNR, input SNR (baseline), SNR improvement (SNRi).

**Expected findings:**

- The **CNN autoencoder** is expected to achieve the best MSE and SNR, as 1D convolutions are well-suited to capturing local temporal patterns in synthetic sinusoidal signals.
- The **LSTM autoencoder** is expected to perform comparably to CNN on smooth periodic signals, potentially outperforming CNN on signals with long-range structure such as speech or ECG.
- The **FC autoencoder** (baseline) is expected to achieve the highest MSE due to its lack of temporal inductive bias. It may still produce reasonable reconstructions given sufficient latent capacity.

Training curves are saved per model to `results/<arch>_training_curves.png`.

---

### 5.2 Experiment 2: Latent Dimension Study

**Setup:** For each architecture in {FC, CNN, LSTM}, models are trained with latent dimensions $d_z \in \{8, 16, 32, 64, 128, 256\}$. All other hyperparameters are fixed.

**Metrics reported:** Test MSE and test SNR as a function of $d_z$.

**Expected findings:**

- **Small $d_z$ (8–16):** High compression forces the bottleneck to discard fine-grained information. MSE will be high and SNR low. The model captures only the dominant frequency components.
- **Medium $d_z$ (32–64):** Sweet spot where the model retains enough information for good reconstruction without overfitting. MSE plateaus near its minimum.
- **Large $d_z$ (128–256):** Diminishing returns; reconstruction quality improves marginally while the latent space becomes less compact. Risk of the model memorizing noise rather than learning the signal manifold.

The CNN is expected to be more robust to small $d_z$ because the convolutional bottleneck already compresses temporally before the FC projection.

Results are saved to `results/latent_dim_sweep.png` and `results/latent_dim_sweep.json`.

---

### 5.3 Experiment 3: Noise Robustness

**Setup:** Each model is trained with fixed Gaussian noise ($\sigma = 0.1$) and evaluated on test sets corrupted with varying noise levels $\sigma \in \{0.05, 0.1, 0.2, 0.4\}$.

**Metrics reported:** SNR improvement (SNRi) as a function of test noise level $\sigma$.

**Expected findings:**

- **Sub-training noise ($\sigma = 0.05$):** All models will perform well, achieving large positive SNRi since the noise is weaker than what they were trained on.
- **Matched noise ($\sigma = 0.1$):** Best performance in terms of absolute SNR (trained for this condition).
- **Super-training noise ($\sigma \in \{0.2, 0.4\}$):** SNRi will decrease but may still be positive. The CNN and LSTM are expected to degrade more gracefully than the FC model due to their structural priors.
- **Comparison to masked noise:** When tested on random masking (a qualitatively different noise type than Gaussian), performance is expected to drop, demonstrating that models learn noise-type-specific priors rather than universal denoising.

Results are saved to `results/noise_robustness.png` and `results/noise_robustness.json`.

---

## 6. Discussion

### 6.1 Inductive Bias vs. Expressivity

The three architectures represent three different inductive bias regimes:

| Architecture | Bias | Strength | Weakness |
|---|---|---|---|
| FC | None | Universal approximator | Inefficient, needs large $d_z$ |
| CNN | Local temporal patterns | Parameter efficient, fast training | Cannot model very long-range dependencies |
| LSTM | Sequential temporal dependencies | Models long-range structure | Slow training, gradient issues |

For synthetic sinusoidal signals, the CNN bias toward local patterns is ideal. For biomedical signals (e.g., ECG with P-QRS-T complexes that span hundreds of samples), the LSTM bias may be more appropriate.

### 6.2 Compression vs. Reconstruction Quality

The latent dimension controls the information bottleneck. A smaller $d_z$ enforces greater compression and acts as a regularizer that prevents the autoencoder from learning a trivial identity mapping. However, too small a $d_z$ leads to lossy compression that destroys signal content. The tradeoff is inherent and should be tuned based on the desired compression ratio for the application.

### 6.3 Noise Robustness

Denoising autoencoders learn to project noisy inputs onto the clean signal manifold. Their robustness to out-of-distribution noise levels depends on whether the learned manifold generalizes. Gaussian noise, being spherically symmetric in signal space, results in a smoother and more generalizable learned manifold than sparse, structured noise like random masking. This explains why models trained on Gaussian noise struggle more when evaluated on masking noise.

---

## 7. File Reference

| File | Description |
|---|---|
| `models.py` | FC, CNN, and LSTM autoencoder classes |
| `metrics.py` | `compute_mse()`, `compute_snr()`, `snr_improvement()`, `evaluate_model()` |
| `experiments.py` | Architecture comparison, latent dim sweep, noise robustness, hyperparam search |
| `config.py` | Default hyperparameters and sweep ranges |
| `main.py` | CLI entry point: `python main.py --exp [all|arch|latent|noise|hyperparam]` |

---

## 8. References

1. Vincent, P., Larochelle, H., Bengio, Y., & Manzagol, P. A. (2008). Extracting and composing robust features with denoising autoencoders. *ICML*.
2. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735–1780.
3. LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*, 521, 436–444.
4. Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *ICLR*.
