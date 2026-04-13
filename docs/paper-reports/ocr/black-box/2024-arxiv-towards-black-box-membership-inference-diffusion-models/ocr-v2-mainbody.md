# Towards Black-Box Membership Inference Attack for Diffusion Models

**文档类型**：OCR 精修版原文近似稿

**说明**：本稿基于 PaddleOCR 结果与 PDF 原文联合整理，保留正文主干、关键公式、主要实验表述与方法图，便于在飞书中连续阅读。

**GitHub PDF**：https://github.com/DeliciousBuding/DiffAudit-Research/blob/main/references/materials/black-box/2024-arxiv-towards-black-box-membership-inference-diffusion-models.pdf

**论文报告**：待上传后补充

**开源实现**：https://github.com/DeliciousBuding/DiffAudit-Research/blob/main/src/diffaudit/attacks/variation.py

---

# Towards Black-Box Membership Inference Attack for Diffusion Models

Jingwei Li, Jing Dong, Tianxing He, Jingzhao Zhang

## Abstract

Given the rising popularity of AI-generated art and the associated copyright concerns, identifying whether an artwork was used to train a diffusion model is an important research topic. The work approaches this problem from the membership inference attack perspective. The paper first identifies the main limitation of previous attacks for proprietary diffusion models: they require access to the internal U-Net or other denoising components. To address this issue, the authors introduce a membership inference attack that uses only the image-to-image variation API and works without access to the model internals. The method relies on the intuition that the model can obtain a more unbiased noise prediction for training images. By repeatedly applying the variation API to a target image, averaging the outputs, and comparing the average image with the original image, the method decides whether the sample was part of the training set. The paper validates this idea on DDIM, Stable Diffusion, and Diffusion Transformer setups.

## 1. Introduction

Diffusion models have become widely used in unconditional image generation, text-to-image generation, and image-to-image generation. Their adoption has also intensified concerns about copyright, data misuse, and the ability to verify whether a particular artwork was involved in training. The paper frames this as a membership inference attack problem: determine whether a specific sample participated in model training.

Previous work on diffusion-model MIA has already shown that training samples often produce lower loss or more accurate noise prediction. However, most of these attacks still require access to the model's internal denoiser, which is unrealistic for commercial systems that only expose APIs. This motivates a stronger black-box setting.

The paper therefore studies a variation-API-only setting. The key observation is that if an image has appeared in training, repeated image variation tends to stay inside a tighter region around the original image. Based on this observation, the paper proposes REDIFFUSE and claims three main contributions:

1. A membership inference method that does not require access to the internal denoiser.
2. Evaluation on DDIM and Stable Diffusion across CIFAR10/100, STL10-Unlabeled, LAION-5B and related datasets.
3. Extension of both previous baselines and the proposed method to Diffusion Transformer.

## 2. Related Works

The paper reviews three groups of prior work. The first group is diffusion-model development, including DDPM, DDIM, Stable Diffusion, and Diffusion Transformer. The second group is general membership inference attacks for classifiers, embedding models, and generative models. The third group is prior diffusion-model MIA, where earlier methods either rely on model loss, likelihood, or intermediate noise prediction. These earlier attacks already relax some assumptions compared with white-box access, but they still need the U-Net or equivalent intermediate outputs, which are unavailable in many commercial services.

The paper also distinguishes its setting from a recent black-box attack on fine-tuned diffusion models. That line focuses on whether a sample belongs to a small fine-tuning set with known prompts. In contrast, this work focuses on pretraining-data membership and only assumes access to a variation API.

## 3. Preliminary

The paper first recalls the DDPM forward and reverse process:

$$
q(x_t \mid x_{t-1}) = \mathcal{N}\left(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t \mathbf{I}\right),
$$

$$
p_\theta(x_{t-1} \mid x_t) = \mathcal{N}\left(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t)\right).
$$

For DDIM, the sampling rule is written as

$$
x_{t-1} = \phi_\theta(x_t, t)
= \sqrt{\bar{\alpha}_{t-1}}
\left(
\frac{x_t - \sqrt{1-\bar{\alpha}_t}\,\epsilon_\theta(x_t, t)}
{\sqrt{\bar{\alpha}_t}}
\right)
 + \sqrt{1-\bar{\alpha}_{t-1}}\,\epsilon_\theta(x_t, t).
$$

Stable Diffusion performs diffusion in latent space and uses text conditioning:

$$
z_{t-1} \sim p_\theta(z_{t-1} \mid z_t, \tau_\theta(y)), \qquad x = \mathrm{Decoder}(z_0).
$$

Diffusion Transformer keeps the DDIM-style training and sampling logic but replaces the U-Net with a transformer backbone.

## 4. Algorithm Design

### 4.1 Variation API

The paper formalizes the black-box interface as a variation API. Given an input image $x$ and a diffusion step $t$, the API first adds Gaussian noise and then runs the reverse process:

$$
x_t = \sqrt{\bar{\alpha}_t} x + \sqrt{1-\bar{\alpha}_t}\,\epsilon,
$$

$$
V_\theta(x, t) = \Phi_\theta(x_t, 0) = \phi_\theta(\cdots \phi_\theta(\phi_\theta(x_t, t), t-1), 0).
$$

This abstraction matches image-to-image variation APIs offered by diffusion systems and removes any need to inspect internal denoising features.

### 4.2 REDIFFUSE

The key intuition is that for a well-trained model, the prediction error around a member image is closer to an unbiased estimator. Starting from the DDIM loss for a fixed sample $x_0$ and time step $t$,

$$
L(\theta) =
\mathbb{E}_{\epsilon \sim \mathcal{N}(0, I)}
\left[
\left\|
\epsilon - \epsilon_\theta\!\left(
\sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, t
\right)
\right\|^2
\right],
$$

the paper argues that the local noise prediction error for training images has zero expectation under suitable assumptions. Since the black-box attacker cannot observe $\epsilon_\theta$, the method replaces it with repeated variation outputs and average reconstruction.

![Figure 1](assets/page-0004/001.jpg)

Algorithm 1 applies the variation API $n$ times at the same diffusion step, averages the outputs, and compares the average image $\hat{x}$ with the original image $x$:

$$
\hat{x} = \frac{1}{n} \sum_{i=1}^{n} \hat{x}_i,
\qquad
f(x) = \mathbf{1}\!\left[D(x, \hat{x}) < \tau\right].
$$

For DDIM and Diffusion Transformer, the paper uses the difference image $v = x - \hat{x}$ and trains a ResNet-18 classifier on that signal. For Stable Diffusion, it directly uses SSIM as the distance metric.

The paper further gives a concentration-style guarantee. Under the assumption that the prediction error has zero expectation and finite cumulant-generating function for member images, the averaged reconstruction error satisfies

$$
\mathbb{P}(\|\hat{x}-x\| \ge \beta)
\le
d \exp\!\left(
-n \min_i \Psi_{X_i}^*
\left(
\frac{\beta \sqrt{\bar{\alpha}_t}}{\sqrt{d(1-\bar{\alpha}_t)}}
\right)
\right).
$$

This is used to justify why averaging more API outputs improves separability.

## 5. Experiments

### 5.1 Setup

The evaluation covers three model families.

For DDIM, the paper follows prior setups and trains on CIFAR10, CIFAR100, and STL10-Unlabeled, with diffusion step $t = 200$ and ten independent API calls to build the average reconstruction. For Diffusion Transformer, it trains on ImageNet with resolutions $128 \times 128$ and $256 \times 256$, and again averages ten outputs. For Stable Diffusion, it uses the original `stable-diffusion-v1-4`, takes LAION-5B images as members and COCO2017-val images as non-members, and tests both ground-truth text and BLIP-generated text.

The baselines are loss-based attack, SecMI, PIA, and PIAN. Metrics are AUC, ASR, and TP at 1% false-positive rate.

### 5.2 Main Results

On DDIM, REDIFFUSE achieves the best or tied-best results across the three datasets. The paper reports AUC values of `0.96`, `0.98`, and `0.96` on CIFAR10, CIFAR100, and STL10, respectively. On Diffusion Transformer, the method reaches `0.98` AUC on ImageNet `128 x 128` and `0.97` on `256 x 256`. On Stable Diffusion, the method still works in a stricter black-box setting and reports `0.81` AUC with ground-truth text and `0.82` with BLIP-generated text.

The paper also studies the impact of averaging number, diffusion step, and sampling interval. Averaging improves DDIM and Diffusion Transformer results, while Stable Diffusion is less sensitive because the reconstructed images are already relatively stable. The attack is also reported to remain effective across a range of diffusion steps and sampling intervals.

## 6. An Application to DALL-E 2's API

The paper includes a small online experiment on DALL-E 2 because the service exposes a variation API. Since the real training set is unknown, the paper uses famous paintings as approximate members and images generated by Stable Diffusion 3 from the titles of these paintings as non-members. Under this setup, the reported results are:

- `AUC = 76.2`, `ASR = 74.5` for `L1` distance
- `AUC = 88.3`, `ASR = 81.4` for `L2` distance

The paper explicitly notes that this experiment has limitations because not every selected painting is guaranteed to be in DALL-E 2's training data.

## 7. Conclusion, Limitations and Future Directions

The paper concludes that REDIFFUSE can detect training-set membership with only variation-API access and no internal network components. It emphasizes that this makes the method more practical for proprietary diffusion systems.

At the same time, the paper acknowledges two main limitations. First, the online DALL-E 2 experiment uses approximate members rather than verified ground-truth training samples. Second, the theory relies on strong assumptions about local unbiased prediction error around training members. These limitations leave room for more realistic black-box evaluations and tighter theoretical analysis in future work.
