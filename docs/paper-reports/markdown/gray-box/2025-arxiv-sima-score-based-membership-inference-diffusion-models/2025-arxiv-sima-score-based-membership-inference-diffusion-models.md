# SCORE-BASED MEMBERSHIP INFERENCE ON DIFFUSION MODELS

Mingxing Rao, Bowen Qu & Daniel Moyer

Department of Computer Science Vanderbilt University Nashville, TN 37235, USA {mingxing.rao,bowen.qu,daniel.moyer}@vanderbilt.edu

## ABSTRACT

Membership inference attacks (MIAs) against diffusion models have emerged as a pressing privacy concern, as these models may inadvertently reveal whether a given sample was part of their training set. We present a theoretical and empirical study of score-based MIAs, focusing on the predicted noise vectors that diffusion models learn to approximate. We show that the expected denoiser output points toward a kernel-weighted local mean of nearby training samples, such that its norm encodes proximity to the training set and thereby reveals membership. Building on this observation, we propose SimA, a single-query attack that provides a principled, efficient alternative to existing multi-query methods. SimA achieves consistently strong performance across variants of DDPM, Latent Diffusion Model (LDM). Notably, we find that Latent Diffusion Models are surprisingly less vulnerable than pixel-space models, due to the strong information bottleneck imposed by their latent auto-encoder. We further investigate this by differing the regularization hyperparameters (β in β-VAE) in latent channel and suggest a strategy to make LDM training more robust to MIA. Our results solidify the theory of score-based MIAs, while highlighting that Latent Diffusion class of methods requires better understanding of inversion for VAE, and not simply inversion of the Diffusion process

# 1 INTRODUCTION

Generative image models leave evidence of their specific training data at deployment time in their generative process. While making draws from approximations of p(x) or p(x|y), they leave biases of the training samples (finite and fixed realizations from the real p(x) or p(x|y)). These biases may be used in theory to reconstruct the training data, a process known as model inversion [\(Zhu](#page-11-0) [et al., 2016;](#page-11-0) [Creswell & Bharath, 2018;](#page-9-0) [Carlini](#page-9-1) [et al., 2023;](#page-9-1) [Somepalli et al., 2023a](#page-11-1)[;b;](#page-11-2) [Gu et al.,](#page-9-2) [2023\)](#page-9-2).

The ability to invert these models raises concerns in privacy and intellectual property spaces for specific use-cases of generative models, but also possibly provides unique perspectives into the idiosyncrasies of the generative models themselves. If the models were perfect, they would sample from a distribution indistinguishable from the data generating process; the ways in which they deviate from this distribution inform upon their structure.

<span id="page-0-0"></span>![](_page_0_Figure_9.jpeg)

Figure 1: A diagram of our Membership Inference method in one dimension. In blue are regions of high membership likelihood, corresponding to low ∥εˆθ∥, plotted in purple. The green region is unlikely to be sampled in high dimensions (c.f. Sec. [3\)](#page-2-0).

A critical precursor to model inversion is the *membership inference attack* (MIA), which determines whether a given image was included in the training set. MIA effectively constructs a classifier for identifying training examples, setting aside the problem of searching the domain for high-likelihood examples. While this may be the easier sub-problem of full model inversion, it is no less important, as without successful MIA, model inversion is impossible.

Building upon analytic results from the literature about diffusion models [\(Pidstrigach, 2022;](#page-10-0) [Karras](#page-10-1) [et al., 2022\)](#page-10-1), we describe a performant MIA method that we believe simplifies other current methods into a consistent methodology: the norm of the estimated score at the test point across diffusion times (the t index). Our empirical results show this simple method has AUC scores at or above the previous state-of-the-art for DDPM-weightsets [\(Ho et al., 2020\)](#page-9-3) on common smaller datasets, and well above the previous state-of-the-art for the related Guided Diffusion model [\(Dhariwal & Nichol,](#page-9-4) [2021\)](#page-9-4) on ImageNet-1k.

In contrast to this, our experiments also show that Latent Diffusion Models (LDM) may be robust to MIA attacks of this general class; not only our own method but *every* method tested has decreased LDM performance. Running otherwise identical membership-inference attacks on publicly welltrained DDPM and LDM checkpoints using the same *member* and *held-out* splits, we find dramatic performance drops across all metrics and across all methods. We hypothesize that LDM robustness may be unrelated to the Diffusion part of the Latent Diffusion Model.

We further experiment by training new LDMs on new VAE encoders with differing regularization hyperparameters (the β KL-weight in the β-VAE framework [\(Burgess et al., 2018\)](#page-9-5)) for the VAE pre-training. We find that MIA performance is susceptible to the KL-weight. We also vary the usage of an additional discriminator between the input image and reconstructed image during VAE training, as suggested by VQ-GAN [\(Esser et al., 2021\)](#page-9-6). Our modifications show that we can at once reduce the vulnerability of LDMs to MIA while improving generation fidelity as measured by FID. This indicates that successful membership inference and model inversion on the Latent Diffusion class of methods requires better understanding of inversion for VAE, and not simply inversion of the Diffusion process.

In summary, our contributions are the following:

- 1. A derivation of a simple Membership Inference Attack method (SimA), which is a reduction of other methods into a general framework.
- 2. An empirical demonstration that SimA provides top performance on standard datasets and independently trained base models.
- 3. An empirical demonstration that this general class of methods seems to fail on Latent Diffusion Models, for reasons possibly unrelated to the diffusion generative process itself, indicating a gap in the literature.

All data splits, model checkpoints, training/fine-tuning scripts, and testing code are released on our GitHub repository <https://github.com/mx-ethan-rao/SimA>

## 2 BACKGROUND AND RELATED WORK

Diffusion Models: Our membership inference work is specific to diffusion-based generative image models. Originally introduced as score-based generative models (without the explicit connection to the Diffusion model) in [Song & Ermon](#page-11-3) [\(2019\)](#page-11-3), a very large number of publications have explored variations of these models since that point [\(Song & Ermon, 2019;](#page-11-3) [Ho et al., 2020;](#page-9-3) [Song et al., 2020;](#page-11-4) [Nichol & Dhariwal, 2021;](#page-10-2) [Dhariwal & Nichol, 2021;](#page-9-4) [Rombach et al., 2022\)](#page-10-3).

While each contribution has its own particular training paradigm and architecture, our attack applies to the broad class of models that estimate a gradient flow field εˆ(x, t) at points x for a smoothing parameter/diffusion time t that approximates the gradient of the smoothed log-likelihood, ∇ log(p(x) ∗ K(t)) [\(Kamb & Ganguli](#page-10-4) [\(2024\)](#page-10-4), or Appendix A.4), which is often induced by a conceptual and/or training-time "forward" noise process x<sup>t</sup> = √ α¯tx<sup>0</sup> + σtε and a "backward" denoising process similar to denoising auto-encoder processes [\(Alain & Bengio, 2014\)](#page-9-7). These may be variance-preserving or variance-exploding [\(Song et al., 2020\)](#page-11-4), based on the exact parameterization of the noise schedule. In this work we directly use weights from the following models: DDPM [\(Ho](#page-9-3)

<span id="page-2-1"></span>![](_page_2_Figure_0.jpeg)

Figure 2: Score-based MIA intuition with local-mean geometry. In a small neighborhood ("local ball") around a query x ⋆ , let µlocal(x ⋆ ) be the kernel-weighted mean of nearby training samples. The model's predicted noise (score) points from x ⋆ toward this local mean, E[ˆεθ(x ⋆ t , t)] ∝ µlocal(x ⋆ ) − x ⋆ . For members (x <sup>⋆</sup> ∈ Dtrain), the local mean µlocal(x ⋆ ) collapses to training sample x ⋆ , producing small norms, whereas for non-members (x <sup>⋆</sup> ∈ D / train), µlocal(x ⋆ ) deviates from x ⋆ , yielding larger norms. Right: the histogram at t = 30 shows the separation in Eε∼N(0,I) [∥εˆθ(xt, t)∥].

[et al., 2020\)](#page-9-3), Guided Diffusion [\(Dhariwal & Nichol, 2021\)](#page-9-4), Latent Diffusion Model (LDM) [\(Esser](#page-9-6) [et al., 2021\)](#page-9-6), and Stable Diffusion [\(Rombach et al., 2022\)](#page-10-3). Each of these directly estimate the noise vector ε using a neural network εˆ(x, t). Our method manipulates x and t and analyze the networks' outputs, but otherwise is agnostic to the exact architecture and weights.

Latent Diffusion Models [\(Esser et al., 2021;](#page-9-6) [Rombach et al., 2022\)](#page-10-3) perform their forward process on the latent space of some encoder-decoder structure (usually a Variational Auto-encoder [\(Kingma](#page-10-5) [et al., 2013\)](#page-10-5)). As we show, this encoder-decoder structure appears to be robust to membership inference, even though the particular diffusion model on its own may not be.

The analytic tractability of these models permits theory about their behavior [\(Kamb & Ganguli,](#page-10-4) [2024;](#page-10-4) [Lukoianov et al., 2025\)](#page-10-6), even pre-dating the publication of the DDPM, e.g., [Alain & Bengio](#page-9-7) [\(2014\)](#page-9-7). Our method is conceptually linked to results and intuition from [Alain & Bengio](#page-9-7) [\(2014\)](#page-9-7) and [Kamb & Ganguli](#page-10-4) [\(2024\)](#page-10-4).

Membership Inference Attack threat models: Model inversion and membership inference attacks pre-date the introduction of the DDPM and LDM generative image models; MIA was defined originally for general classification tasks [\(Shokri et al., 2017\)](#page-11-5). A non-trivial amount of literature since that point has focused on MIA for generative image models [\(Chen et al., 2020\)](#page-9-8) starting with GANs, due to its attack surface (pixel arrays) and its clear privacy and intellectual property implications.

Much of the terminology and structure were defined in the security context, where a threat model defines the scope and allowable resources for an attack vector. So-called "black box" attacks from the original MIA literature are performed without knowledge of the model weights or structure, and can only rely on input and output pairs from a deployed model. In contrast, "white box" attacks [\(Pang et al., 2023\)](#page-10-7) have access to the full architecture/weight set. We consider the most common class of diffusion model MIAs, "grey-box" attacks [\(Duan et al., 2023;](#page-9-9) [Zhai et al., 2024;](#page-11-6) [Kong et al.,](#page-10-8) [2023;](#page-10-8) [Matsumoto et al., 2023;](#page-10-9) [Carlini et al., 2023;](#page-9-1) [Fu et al., 2023\)](#page-9-10), which span a range of options between those two extrema; in general they have access to weights and/or internal representations, and may query the model for particular test points. We review each of these grey box attacks in detail in Section [3](#page-2-0) and compare to each, with the exception of [Zhai et al.](#page-11-6) [\(2024\)](#page-11-6) as it performs membership inference on conditional generative models instead of the unconditioned case.

# <span id="page-2-0"></span>3 METHODOLOGY

### METHOD OVERVIEW

The predicted noise εˆ<sup>θ</sup> is outputted by the neural network, which is a scaled estimator of the score −σt∇<sup>x</sup><sup>t</sup> log pt(xt) [\(Song et al., 2020;](#page-11-4) [Ho et al., 2020;](#page-9-3) [Luo, 2022\)](#page-10-10), for data generating distribution

p(x) and its mollifications pt(x) on a noising schedule σt. Our Simple Attack (SimA) method is

$$\mathcal{A}(x,t) = \|\hat{\varepsilon}_{\theta}(x,t)\|. \tag{1}$$

Here, x is the test point, which ostensibly was drawn from p(x), but is either in the training data or not, and t is the diffusion time parameter. We provide a more rigorous derivation and its connections to other MIA methods in the following sections, but we feel that its intuition is also instructive on why such a simple method would work.

Figure [2](#page-2-1) visualizes the intuition of our method: the expected norm of the predicted score for a query image x ⋆ at time t is effectively the gradient of a Gaussian kernel density estimator (see Appendix A.4). For well-separated points, these estimators' implicit distributions will have peaks at each training point; the gradient vanishes at critical points of the smoothed density, which include the peaks corresponding to the original data points (the blue region in the Fig. [1\)](#page-0-0), up to a bias term from the smoothing. Manipulating this fact allows us to form a simple yet successful estimator.

There should be false positive terms at the other critical points (the green region in Fig [1\)](#page-0-0). In high dimension, these should be by-in-large saddle points between maxima. Because these points occur directly between original datapoints at their arithmetic mean, if the data manifold (the support of p(x)) has any curvature, these points would be off manifold; empirically we do not seem to encounter many of them, as indicated by the TPR@1%FPR measurements (see Section [4\)](#page-6-0).

#### 3.1 FROM FORWARD DIFFUSION TO SCORE

Notation: Let {βt} T <sup>t</sup>=1 be the variance schedule of the forward diffusion process in DDPM [Ho et al.](#page-9-3) [\(2020\)](#page-9-3). Define α<sup>t</sup> := 1 − β<sup>t</sup> and its cumulative product α¯<sup>t</sup> := Q<sup>t</sup> <sup>s</sup>=1 αs. The total noise variance accumulated up to step t is σ 2 t := 1 − α¯t.

Forward Diffusion as a Scaled Gaussian Convolution: With the variance-preserving (VP) schedule of [Ho et al.](#page-9-3) [\(2020\)](#page-9-3), the forward model is

$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sigma_t \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I), \ \sigma_t^2 = 1 - \bar{\alpha}_t,$$
 (2)

where x<sup>0</sup> ∼ pdata. Marginalizing x<sup>0</sup> gives the perturbed data distribution [\(Song & Ermon, 2019\)](#page-11-3). For a *clean* (without noise) sample x ∈ R d ,

<span id="page-3-0"></span>
$$p_t(x) = \int_{\mathbb{R}^d} p_{\text{data}}(x_0) \mathcal{N}(x \mid \sqrt{\bar{\alpha}_t} x_0, \sigma_t^2 I) dx_0.$$
 (3)

Equation [3](#page-3-0) is a Gaussian convolution of the original data distribution (see Appendix A.4 for an explicit derivation), followed by a global scaling by <sup>√</sup> α¯t.

$$p_t(x) = \frac{1}{\bar{\alpha}_t^{d/2}} \left( p_{\text{data}} * \mathcal{N} \left( 0, \frac{\sigma_t^2}{\bar{\alpha}_t} I \right) \right) \left( \frac{x}{\sqrt{\bar{\alpha}_t}} \right). \tag{4}$$

Therefore, pt(x) is the data distribution convolved with scaled Gaussian distribution whose kernel's covariance <sup>σ</sup> t α¯<sup>t</sup> I = (¯α −1 <sup>t</sup> − 1)I grows monotonically with the timestep t.

Gradient of pt(x): Writing the kernel in standard form Kt(x, x0) = (2πσ<sup>2</sup> t ) −d/2 exp(−∥<sup>x</sup> <sup>−</sup> <sup>√</sup> α¯tx0∥ <sup>2</sup>/2σ 2 t ), we can compute its spatial gradient: ∇xK<sup>t</sup> = −σ −2 t x − √ α¯tx<sup>0</sup> Kt. We then combine this with Eq. [3](#page-3-0) to obtain

<span id="page-3-1"></span>
$$\nabla_x p_t(x) = -\sigma_t^{-2} \int p_{\text{data}}(x_0) \left(x - \sqrt{\overline{\alpha}_t} x_0\right) \mathcal{K}_t(x, x_0) \, dx_0 \tag{5}$$

which requires continuity assumptions on pdata which are usually assumed by DDPM analyses [\(Alain](#page-9-7) [& Bengio, 2014\)](#page-9-7).

Introducing exact likelihood of each datapoint qt(x<sup>0</sup> | x): Define the exact distribution of the x<sup>0</sup> given an observation from the Gaussian smoothed distribution:

$$q_t(x_0 \mid x) = \frac{p_{\text{data}}(x_0) \, \mathcal{K}_t(x, x_0)}{p_t(x)}.$$
 (6)

Because pt(x) normalizes Eq. [3,](#page-3-0) we can then rewrite Eq. [5](#page-3-1) as

<span id="page-4-0"></span>
$$\nabla_x p_t(x) = -\frac{p_t(x)}{\sigma_t^2} \left[ x - \sqrt{\overline{\alpha}_t} \underbrace{\mathbb{E}_{q_t(x_0|x)}[x_0|x]}_{\mu_t(x)} \right]. \tag{7}$$

We call Eqt(x0|x) [x0|x] = µt(x) the *denoising mean*; it is the likelihood-weighted average of the positions of the datapoints that could have generated x at time t through the forward process, which is the same as the mean optimal solution to the denoising problem.

Obtaining the exact score: Dividing Eq. equation [7](#page-4-0) by pt(x) yields the score of distribution pt(x) at x (Lemma 6 of [Pidstrigach](#page-10-0) [\(2022\)](#page-10-0), or, alternatively, applying the chain rule to ∇<sup>x</sup> log pt(x) and then substituting in values):

$$s_t(x) = \nabla_x \log p_t(x) = -\frac{x - \sqrt{\bar{\alpha}_t} \,\mu_t(x)}{\sigma_t^2}.$$
 (8)

This score function st(x) is the desired output of the ε-parameterization of Score-based Denoising [\(Song et al., 2020\)](#page-11-4) and the original DDPM [\(Ho et al., 2020\)](#page-9-3). During training the UNet is asked to predict the standard noise [\(Ho et al., 2020\)](#page-9-3):

<span id="page-4-2"></span>
$$\hat{\varepsilon}_{\theta}(x,t) \approx \frac{x_t - \sqrt{\bar{\alpha}_t} \,\mu_t(x)}{\sigma_t} = -\sigma_t \nabla_x \log p_t(x) \tag{9}$$

This is consistent with Eq. 151 of [Luo](#page-10-10) [\(2022\)](#page-10-10). For x ∈ supp p<sup>t</sup> (with or without noise), the estimator εˆθ(x, t) should approximate the negative score of pt(x). However, in practice the data distribution pdata becomes the empirical distribution ptraining, which is a finite sample of points. The noised distribution pt(x) is then a kernel smoothing of that empirical distribution, and its finite-sample denoising mean is described in [Kamb & Ganguli](#page-10-4) [\(2024\)](#page-10-4) and refined in [Lukoianov et al.](#page-10-6) [\(2025\)](#page-10-6). Equation 3 of [Lukoianov et al.](#page-10-6) [\(2025\)](#page-10-6) states it as:

$$\mu_t^{\text{finite}}(x) = \sum_{i=1}^{N} w_i(x, t) x_0^{(i)}, \quad w_i(x, t) = \text{softmax}_i \left\{ -\frac{1}{2\sigma_t^2} \|x - \sqrt{\bar{\alpha}} x_0^{(j)}\|_2^2 \right\}_{j \in [N]}$$
(10)

where the x (i) 0 are training data, and softmax<sup>i</sup> is the i th index of a softmax function over the N training data points. The discrepancy between the finite sample optimal denoised x distribution and the large sample limite qt(x0|x) gives rise to our membership inference attack; the model "overfits" to the training set, and that overfitting gap is the discrepancy which SimA seeks to exploit.

#### 3.2 MEMBERSHIP INFERENCE ATTACK

Given a datapoint x ∈ R d and a t = 1, . . . , T, our membership decision criterion A is defined as

<span id="page-4-1"></span>
$$\mathcal{A}(x,t) = \|\hat{\varepsilon}_{\theta}(x,t)\|_{p}. \tag{11}$$

Using ℓ<sup>p</sup> norms other than p = 2 provide slightly improved performance. This trend is also found in another MIA method, [Kong et al.](#page-10-8) [\(2023\)](#page-10-8). While this is somewhat mysterious, the ℓ<sup>4</sup> norm appears in sum-of-squares computations [\(Barak et al., 2015\)](#page-9-11), spherical harmonics [\(Stanton & Weinstein,](#page-11-7) [1981\)](#page-11-7), and blind source separation [\(Hyvarinen, 1997\)](#page-10-11) with surprising regularity.

Following the Bayes-optimal loss–threshold formulation of membership inference in classification models by [Sablayrolles et al.](#page-11-8) [\(2019\)](#page-11-8), we recast the decision rule for diffusion models. Specifically, we define

$$\mathcal{M}_{\text{opt}}(x,t) = \mathbb{1}[\mathcal{A}(x,t) \le \tau],$$
 (12)

where τ is a threshold calibrated on a held-out validation set and A(x, t) = ∥εˆθ(x, t)∥<sup>2</sup> is the estimated noise at x for diffusion step t. If the predicted noise norm is *smaller* than τ , the sample is inferred to be a *member*; otherwise, it is classified as a *non-member*.

The attack criterion A can be applied to three cases, which we expand upon below (see appendix A.5 for detailed derivation of the first two cases).

Case 1 — Member of the Training Set: Let x (k) denote one of the training images {x (i)} N <sup>i</sup>=1. As t → 0, the finite sample denoising mean collapses to the input (full derivation is in case 1 of Appendix A.5):

$$\mu_t^{\text{finite}}(x^{(k)}) \xrightarrow[t \to 0]{} x^{(k)}$$
 (13)

Consequently, the estimated noise vector shrinks to zero as well, meaning our criterion A(x (k) , t → 0) should be small:

$$\hat{\varepsilon}_{\theta}(x^{(k)}, t) = \frac{x^{(k)} - \sqrt{\bar{\alpha}_t} \, x^{(k)}}{\sigma_t} \sim \frac{\sigma_t}{2} x^{(k)} \xrightarrow[t \to 0]{} 0.$$
 (14)

While the actual value at zero is undefined, and values for small t < 10 are empirically unstable for non-member x (k) (leading to a poor estimator, see Appendix A.7 for the detailed reason), we find that for t ∈ [10, 300], these values will still be smaller than the case 2. These time frames are unfortunately noise schedule and data dependent.

Case 2 — Held-out but On-Manifold: Here x † is sampled from the same data distribution pdata as the training set, yet was *never* shown during training, i.e. centered on x † for some maximal radius r > 0 there is a ball Br(x † ⊂ R <sup>d</sup> where no training data are included inside Br(x † ).

We use the notation of local moment matching [\(Bengio et al., 2012;](#page-9-12) [Alain & Bengio, 2014\)](#page-9-7) to describe these points. Consider the local mean defined with respect to that void Br(x † ):

<span id="page-5-0"></span>
$$\mu_t(x^{\dagger})_{\text{local}} = m_r(x^{\dagger}) = \int_{B_r(x^{\dagger})} x p_t(x|x^{\dagger}) dx \tag{15}$$

where r ≍ σt/ √ α¯<sup>t</sup> . Theorems 2 and 3 of [Alain & Bengio](#page-9-7) [\(2014\)](#page-9-7) put together produce a statement about a term that is equivalent to εˆ (Eq. 28 of [Alain & Bengio](#page-9-7) [\(2014\)](#page-9-7)) (Full derivation is in case 2 of Appendix A.5):

$$\boxed{m_r(x^{\dagger}) - x^{\dagger} \approx \frac{r^2}{d+2} \left. \frac{\partial \log p_t(x)}{\partial x} \right|_{x^{\dagger}} = \frac{r^2}{d+2} \left( \frac{x^{\dagger} - \sqrt{\overline{\alpha}_t} \mu_t^{\text{finite}}(x^{\dagger})}{\sigma_t^2} \right) = -\frac{r^2}{\sigma_t(d+2)} \hat{\varepsilon}_{\theta}(x^{\dagger}, t)}$$
(16)

We claim that for in-support regions of Gaussian mollified empirical distributions with well separated points, these regions will generally not be flat, and thus ∥εˆθ(x † , t)∥ will tend away from zero. The magnitude to which ∥εˆθ(x † , t)∥ diverges from zero clearly depends on the maximal density of the dataset, but as high dimensional spaces have exponentially larger volumes than lower dimensional spaces, even for large datasets (e.g. ImageNet in ResNet resolution) we can expect these voids to be non-trivially large.

Case 3 — Out-of-Distribution (OOD). Since no training data support is available in out-ofdistribution (OOD) regions, the diffusion model lacks information about these areas. Consequently, the learned score field in such regions is necessarily an extrapolation, and the theoretical derivations established under the in-distribution assumption no longer hold. We do not expect either the diffusion model or our attack criteion to perform well in these regions. First, even though the theoretical finite-sample optimal denoiser is well defined (Eq. [10\)](#page-4-1), a neural network approximation to it will have very little training data in these regions. Second, a trial datapoint x ∗ is by-definition not from pdata in these regions.

SimA in comparison to other Diffusion MIA models: A standard concept in MIA is the use of the training loss function evaluated on the data points in question as the member/non-member decision criterion. This is the Loss criterion, and is proposed in [Matsumoto et al.](#page-10-9) [\(2023\)](#page-10-9). They use a stochastic sample ε to estimate this criterion, adding it to the test point x ∗ .

Loss = 
$$\|\varepsilon - \hat{\varepsilon}_{\theta}(\sqrt{\bar{\alpha}_t}x^* + \sqrt{1 - \bar{\alpha}_t}\varepsilon, t)\|$$
 (Matsumoto et al., 2023) (17)

In theory this should be evaluated across a large number of ε measurements, but for each point they choose to use only one. This method is predicated on the idea that for points in the training set, the

<span id="page-6-2"></span>Table 1: Performance of benchmark methods on DDPM across four datasets. Best results in bold

| Method    | #Query↓ | CIFAR-10 (%) |       | CIFAR-100 (%) |       |       | STL10-U (%) |       |       | CelebA (%) |       |       |           |
|-----------|---------|--------------|-------|---------------|-------|-------|-------------|-------|-------|------------|-------|-------|-----------|
|           |         | ASR↑         | AUC↑  | TPR†<br>↑     | ASR↑  | AUC↑  | TPR†<br>↑   | ASR↑  | AUC↑  | TPR†<br>↑  | ASR↑  | AUC↑  | TPR†<br>↑ |
| PIA       | 2       | 85.06        | 91.86 | 29.54         | 82.31 | 89.58 | 28.75       | 90.83 | 96.34 | 66.30      | 74.27 | 82.22 | 18.36     |
| PFAMIMet  | 20      | 73.64        | 80.32 | 8.16          | 70.58 | 77.05 | 8.44        | 74.78 | 83.74 | 17.00      | 64.34 | 70.18 | 8.13      |
| SecMIstat | 12      | 82.71        | 89.72 | 33.44         | 81.55 | 88.62 | 29.95       | 89.40 | 95.16 | 66.20      | 73.94 | 81.45 | 14.55     |
| Loss      | 1       | 77.67        | 84.73 | 24.14         | 74.85 | 82.19 | 19.56       | 84.60 | 91.42 | 47.67      | 69.13 | 75.96 | 15.15     |
| SimA      | 1       | 83.62        | 90.45 | 35.86         | 82.51 | 89.85 | 38.84       | 91.15 | 96.34 | 72.75      | 74.90 | 82.85 | 20.86     |

<span id="page-6-3"></span>Table 2: Performance of Guided Diffusion and Latent Diffusion pretrained on ImageNet1K. Best results in bold. *Member* set: ImageNet-1K (3000); *Held-out*: ImageNetV2 (3000).

| Method    | #Query↓ |       | ImageNet, Guided |       | ImageNet, LDM |                  |      |  |
|-----------|---------|-------|------------------|-------|---------------|------------------|------|--|
|           |         |       | ASR↑ AUC↑ TPR†   |       |               | ↑ ASR↑ AUC↑ TPR† | ↑    |  |
| PIA       | 2       | 64.65 | 66.44            | 9.93  | 55.13         | 56.8             | 2.13 |  |
| PFAMIMet  | 20      | 67.85 | 72.22            | 3.77  | 50.77         | 48.76            | 0.57 |  |
| SecMIstat | 12      | 77.97 | 82.55            | 34.73 | 56.80         | 56.85            | 2.33 |  |
| Loss      | 1       | 57.27 | 60.38            | 7.00  | 50.48         | 47.9             | 0.6  |  |
| SimA      | 1       | 85.73 | 89.77            | 21.73 | 55.78         | 56.14            | 1.97 |  |

<sup>†</sup> True Positive Rate at 1 % False Positive Rate.

noise estimation will be better or even overfit in comparison to points not in the training set. SimA is the evaluation of this method at ε = 0, which is the mean and mode of the ε distribution. Effectively, [Matsumoto et al.](#page-10-9) [\(2023\)](#page-10-9) are measuring draws from lower likelihood areas, which may not exhibit the overfit phenomenon as well as ε = 0. [Fu et al.](#page-9-10) [\(2023\)](#page-9-10) provide this same loss estimate as the selection criterion, but increase the Monte Carlo sampling to 20 and perform it only for a single step, sampling ε<sup>t</sup> the stepwise noise instead of sampling ε.

SecMI of [Duan et al.](#page-9-9) [\(2023\)](#page-9-9) takes this one step further, evaluating not only a score term but also a single step term, which measures sensitivity to single step differences in t:

$$\mathbf{SecMI} = \|\varepsilon - \hat{\varepsilon}_{\theta}(x, t)\| + \|\sqrt{1 - \bar{\alpha}}(\hat{\varepsilon}_{\theta}(x, t) - \hat{\varepsilon}_{\theta}(\sqrt{\bar{\alpha}_{t+1}}x^* + \sqrt{1 - \bar{\alpha}_{t+1}}\varepsilon, t+1))\|$$
(18)

SecMI is dependent on sampling ε's as well; the authors prescribe using N = 12 samples.

The closely related PIA method computes this same loss quantity again, but using a t = 0 term instead of the ε sample.

PIA = 
$$\|\hat{\varepsilon}_{\theta}(x, t=0) - \hat{\varepsilon}_{\theta}(\sqrt{\bar{\alpha}_{t}}x^{*} + \sqrt{1 - \bar{\alpha}_{t}}\hat{\varepsilon}_{\theta}(x, t=0), t)\|$$
 (19)

While in theory diffusion models might not be well defined at t = 0, in practice they often can extrapolate as they are trained on nearby t; in the continuous time case they are trained on the interval [0, 1]. Our method has similar components to this method, manipulating t around the test point, but again replacing its "ground-truth" noise with ε = 0 (here replaced by εˆ(x0, t = 0)).

## <span id="page-6-0"></span>4 EXPERIMENTS

#### 4.1 SETUP

We evaluated our attack on 15 member–held-out pairs drawn from 11 datasets (see Appendix A.1). The experiments were conducted on the following target models:

Denoising Diffusion Probabilistic Model: For CIFAR-10, CIFAR-100 [\(Krizhevsky et al., 2009\)](#page-10-12), STL10-U (unlabeled split) [\(Coates et al., 2011\)](#page-9-13), and CelebA [\(Liu et al., 2015\)](#page-10-13), we trained a vanilla DDPM [\(Ho et al., 2020\)](#page-9-3) from scratch on the *member* set. From each training split we subsample n images and partition them equally into a *member* set and a *held-out* set.

Pre-trained Guided Diffusion: We examined the publicly released Guided Diffusion model[1](#page-6-1) [\(Dhariwal & Nichol, 2021\)](#page-9-4) trained on ImageNet-1K [\(Russakovsky et al., 2015\)](#page-11-9). ImageNetV2 [\(Recht](#page-10-14)

<span id="page-6-1"></span><sup>1</sup>[https://github.com/openai/guided-diffusion](#page-10-14)

[et al., 2019\)](#page-10-14), collected to mirror the original distribution (same data collection process and same year range), serves as the *held-out* set.

Full statistics including dataset splits, resolutions, etc. are summarized in Table [3](#page-13-0) of Appendix A.1. Additional experiments on LDM models (Stable Diffusion) similar to the experiments in [Zhai et al.](#page-11-6) [\(2024\)](#page-11-6) are included in Appendix A.3. These exhibit the same low performance across all methods as the LDM on ImageNet experiments.

Baselines: Earlier membership-inference attacks (MIAs) aimed at GANs and VAEs—e.g., [\(Chen](#page-9-8) [et al., 2020;](#page-9-8) [Hilprecht et al., 2019;](#page-9-14) [Hu & Pang, 2021\)](#page-9-15)—do not transfer well to diffusion models, as shown by [Duan et al.](#page-9-9) [\(2023\)](#page-9-9). Consequently, we restricted our evaluation to attacks specifically designed for diffusion models. We compared our method with four baselines: SecMI [\(Duan et al.,](#page-9-9) [2023\)](#page-9-9), PIA [\(Kong et al., 2023\)](#page-10-8), PFAMI [\(Fu et al., 2023\)](#page-9-10), Loss [\(Matsumoto et al., 2023\)](#page-10-9). We omitted CLiD [\(Zhai et al., 2024\)](#page-11-6), a text-conditioned MIA, because its text-supervision is incompatible with the setup used in most of our experiments, and the provided code is not usable.

Evaulation Metrics: We evaluated attack performance using several metrics: ASR (attack success rate, i.e., membership inference accuracy), AUC (Area Under ROC Curve), TPR at 1% FPR (TPR@1%FPR), and the number of queries per attack (#Query). The TPR@1%FPR is computed by selecting the threshold τ at which the false positive rate falls just below 0.01, and reporting the corresponding true positive rate at that operating point. The ASR is defined as the maximum accuracy achieved over all thresholds, i.e. τ <sup>∗</sup> = max<sup>τ</sup> 2 TPR(τ ) + 1 − FPR(τ ) in the balanced setting. The AUC is computed as the trapezoidal integral of TPR(τ ) against FPR(τ ) across all thresholds.

Implementation details: Some baselines [\(Duan et al., 2023;](#page-9-9) [Kong et al., 2023;](#page-10-8) [Zhai et al., 2024\)](#page-11-6) augmented their score- or feature-vector statistic with an auxiliary neural classifier; to focus on the statistic itself we evaluated only the norm-based versions, which were SecMIstat and PFAMIMet. Of the two variants (PIA and PIAN) introduced by [Kong et al.](#page-10-8) [\(2023\)](#page-10-8), we benchmarked only PIA, as PIAN showed no statistically significant gain in general from their experiments.

To minimise re-implementation error, our codebase reused the official releases of SecMI, PIA, and Guided Diffusion wherever possible. We failed to re-use PFAMI's code as the provided code was inoperable. A reimplemented copy is provided in our code base. PFAMI, in several cases, failed to attack the victim model (ASR ≈ 50%). We hypothesize that this degradation arose from the sensitivity of its Monte Carlo estimator to the effective sample size, which varied with dataset characteristics and latent dimensionality, yielding high-variance estimates. For each method, we followed the hyperparameter suggestion in their original paper. Notably, l2-norm, l4-norm and l2-norm were used for SecMI, PIA and Loss as suggested. l4-norm was used for SimA as it achieved the best performance in general. For the timestep-dependent attacks (SimA, SecMI, PIA, Loss), we swept t = 0 : 300 and reported the best-performing value for each method (PFAMI is timestep-free and therefore needed no sweep). More implementation details can be found in Section [4.2.](#page-7-0)

#### <span id="page-7-0"></span>4.2 MAIN RESULTS

In Table [1](#page-6-2), all DDPM baselines were trained on a member split and evaluated on an held-out split of equal sizes: 25 k/25 k for CIFAR-10 and CIFAR-100, 50 k/50 k for STL10-U, and 30 k/30 k for CelebA, all at a spatial resolution of 32 × 32. We adopted the public checkpoints and splits released by SecMI for CIFAR-10/100, and retrained STL10-U and CelebA for 18 k and 60 k steps, respectively. Across all datasets and metrics, SimA achieved the best in almost all the experiments over three metrics (ASR, AUC, TPR@1%FPR) while requiring the fewest queries, underscoring its efficiency and practical advantage.

In Table [2](#page-6-3), we further evaluated Guided Diffusion [\(Dhariwal & Nichol, 2021\)](#page-9-4) and LDM [\(Rombach](#page-10-3) [et al., 2022\)](#page-10-3) at their public class-conditional ImageNet-1K checkpoints (256×256). We find that for Guided Diffusion SimA has the highest ASR and AUC by a wide margin, while SecMI has a very high TPR@1%FPR. For the Latent Diffusion *on the same dataset and holdout*, SecMI technically has the best performance, but all methods are extremely bad, achieving less than 57 AUC and 2.3% TPR@1%FPR.

DDPM *vs.* Latent Diffusion Model: In the original LDM [\(Rombach et al., 2022\)](#page-10-3), the encoder is a VQ-VAE [\(Van Den Oord et al., 2017\)](#page-11-10); its discrete code-book forces quantization on patches. A continuous VAE [\(Kingma et al., 2013\)](#page-10-5) is used in Stable Diffusion and passes richer detail, but

<span id="page-8-0"></span>![](_page_8_Figure_0.jpeg)

Figure 3: Left: The performance comparison of AUC between DDPM and Latent Diffusion Model on the same *member*/*held-out* splits for ImageNet-1K and ImageNetV2. Right: Comparison of AUC of MIA (top) and normalized FID (bottom) across categorical β settings for Standard β-VAE [\(Burgess et al., 2018\)](#page-9-5) Encoding and Adversarial β-VAE [\(Esser et al., 2021\)](#page-9-6) on CIFAR-10. The x-axis represents categorical indices of β values, without encoding their numerical magnitude.

still compresses pixel-level information. In both cases, embeddings may express high-level coarsegrained information, and pathological memorization of pixel patterns might be reduced. To investigate this, recap from experiments on Table [2,](#page-6-3) we used the same *member* and *held-out* sets on Guided Diffusion Model and LDM, and All other variables (seed, code, versions) are held constant.

As shown in Fig. [3](#page-8-0) (Left) and Table [2,](#page-6-3) performance drops markedly from DDPM to LDM: SimA, PIA, and SecMIstat fall to ∼55% in ASR, 56% in AUC, and 2% in TPR@1%FPR, while PFAMIMet and Loss perform no better than random guessing.

To further investigate why this particular LDM checkpoint exhibits reduced vulnerability to MIA, we conduct a controlled study on CIFAR-10. We train a series of VAEs with varying β values in the β-VAE framework [\(Burgess et al., 2018\)](#page-9-5). The LDMs are subsequently trained based on these frozen VAE encoders. We perform attack on latent diffusion model. As shown in Fig. [3](#page-8-0) (Right-Top), MIA performance is sensitive to the KL-divergence weight β. Introducing a small KL regularization on VAE training increases AUC, but stronger constraints lead to a monotonic decline, consistent with the expectation that a tighter information bottleneck reduces membership leakage during encoding. Importantly, however, the FID performance is relatively unaffected by the β parameter until much larger values, indicating that it is possible to preserve generative quality (at least up to the FID metric) while also increasing robustness to MIA.

Interestingly, incorporating a discriminator between the input image and reconstructed image during VAE training, an approach originally developed for VQ-GAN [\(Esser et al., 2021\)](#page-9-6), also reduces MIA effectiveness in LDMs while improving sample fidelity. As illustrated in Fig. [3](#page-8-0) (Right), this minor architectural modification yields models that are both more robust to membership inference and capable of generating images with lower FID relative to the data distribution. Full training details are provided in Section A.6 of the Appendix.

This series of experiments, as well as the disparate results on ImageNet-1K between image-domain diffusion and a latent domain diffusion model indicate a gap in the model inversion literature. Previously models have focused on the Diffusion process itself; our results in Table [2](#page-6-3) indicate that Diffusion processes on their own can already be solved for MIA to a high degree of fidelity with a very simple estimator. On the otherhand, the LDM class of models have MIA performance only marginally above at-random. We believe that this should be the focus of model inversion efforts moving forward.

## 5 CONCLUSION

In the present work we have described a simple membership inference estimator, giving theoretical justification for its performance, and for the performance of similar estimators in the literature which previously lacked a unified theoretical backing. We demonstrate that this estimator has competitive performance on many baselines, including state-of-the-art performance on ImageNet-1K imagedomain inversion. Our experiments also elucidate a hole in the current literature concerning latentdomain diffusion methods, where all tested membership inference methods currently fail.

## REFERENCES

- <span id="page-9-7"></span>Guillaume Alain and Yoshua Bengio. What regularized auto-encoders learn from the data-generating distribution. *The Journal of Machine Learning Research*, 15(1):3563–3593, 2014.
- <span id="page-9-11"></span>Boaz Barak, Jonathan A Kelner, and David Steurer. Dictionary learning and tensor decomposition via the sum-of-squares method. In *Proceedings of the forty-seventh annual ACM symposium on Theory of computing*, pp. 143–151, 2015.
- <span id="page-9-12"></span>Yoshua Bengio, Guillaume Alain, and Salah Rifai. Implicit density estimation by local moment matching to sample from auto-encoders. *arXiv preprint arXiv:1207.0057*, 2012.
- <span id="page-9-5"></span>Christopher P Burgess, Irina Higgins, Arka Pal, Loic Matthey, Nick Watters, Guillaume Desjardins, and Alexander Lerchner. Understanding disentangling in β-vae. *arXiv preprint arXiv:1804.03599*, 2018.
- <span id="page-9-1"></span>Nicolas Carlini, Jamie Hayes, Milad Nasr, Matthew Jagielski, Vikash Sehwag, Florian Tramer, Borja Balle, Daphne Ippolito, and Eric Wallace. Extracting training data from diffusion models. In *32nd USENIX security symposium (USENIX Security 23)*, pp. 5253–5270, 2023.
- <span id="page-9-8"></span>Dingfan Chen, Ning Yu, Yang Zhang, and Mario Fritz. Gan-leaks: A taxonomy of membership inference attacks against generative models. In *Proceedings of the 2020 ACM SIGSAC conference on computer and communications security*, pp. 343–362, 2020.
- <span id="page-9-13"></span>Adam Coates, Andrew Ng, and Honglak Lee. An analysis of single-layer networks in unsupervised feature learning. In *Proceedings of the fourteenth international conference on artificial intelligence and statistics*, pp. 215–223. JMLR Workshop and Conference Proceedings, 2011.
- <span id="page-9-0"></span>Antonia Creswell and Anil Anthony Bharath. Inverting the generator of a generative adversarial network. *IEEE transactions on neural networks and learning systems*, 30(7):1967–1974, 2018.
- <span id="page-9-4"></span>Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. *Advances in neural information processing systems*, 34:8780–8794, 2021.
- <span id="page-9-9"></span>Jinhao Duan, Fei Kong, Shiqi Wang, Xiaoshuang Shi, and Kaidi Xu. Are diffusion models vulnerable to membership inference attacks? In *International Conference on Machine Learning*, pp. 8717–8730. PMLR, 2023.
- <span id="page-9-6"></span>Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pp. 12873–12883, 2021.
- <span id="page-9-10"></span>Wenjie Fu, Huandong Wang, Chen Gao, Guanghua Liu, Yong Li, and Tao Jiang. A probabilistic fluctuation based membership inference attack for diffusion models. *arXiv preprint arXiv:2308.12143*, 2023.
- <span id="page-9-2"></span>Xiangming Gu, Chao Du, Tianyu Pang, Chongxuan Li, Min Lin, and Ye Wang. On memorization in diffusion models. *arXiv preprint arXiv:2310.02664*, 2023.
- <span id="page-9-14"></span>Benjamin Hilprecht, Martin Harterich, and Daniel Bernau. Monte carlo and reconstruction member- ¨ ship inference attacks against generative models. *Proceedings on Privacy Enhancing Technologies*, 2019.
- <span id="page-9-3"></span>Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. *Advances in neural information processing systems*, 33:6840–6851, 2020.
- <span id="page-9-15"></span>Hailong Hu and Jun Pang. Membership inference attacks against gans by leveraging overrepresentation regions. In *Proceedings of the 2021 ACM SIGSAC Conference on Computer and Communications Security*, pp. 2387–2389, 2021.

- <span id="page-10-11"></span>Aapo Hyvarinen. A family of fixed-point algorithms for independent component analysis. In *1997 IEEE International Conference on Acoustics, Speech, and Signal Processing*, volume 5, pp. 3917– 3920. IEEE, 1997.
- <span id="page-10-17"></span>Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A Efros. Image-to-image translation with conditional adversarial networks. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pp. 1125–1134, 2017.
- <span id="page-10-4"></span>Mason Kamb and Surya Ganguli. An analytic theory of creativity in convolutional diffusion models. *arXiv preprint arXiv:2412.20292*, 2024.
- <span id="page-10-1"></span>Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh (eds.), *Advances in Neural Information Processing Systems*, volume 35, pp. 26565–26577. Curran Associates, Inc., 2022. URL [https://proceedings.neurips.cc/paper\\_files/paper/2022/](https://proceedings.neurips.cc/paper_files/paper/2022/file/a98846e9d9cc01cfb87eb694d946ce6b-Paper-Conference.pdf) [file/a98846e9d9cc01cfb87eb694d946ce6b-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2022/file/a98846e9d9cc01cfb87eb694d946ce6b-Paper-Conference.pdf).
- <span id="page-10-5"></span>Diederik P Kingma, Max Welling, et al. Auto-encoding variational bayes, 2013.
- <span id="page-10-8"></span>Fei Kong, Jinhao Duan, RuiPeng Ma, Hengtao Shen, Xiaofeng Zhu, Xiaoshuang Shi, and Kaidi Xu. An efficient membership inference attack for the diffusion model by proximal initialization. *arXiv preprint arXiv:2305.18355*, 2023.
- <span id="page-10-12"></span>Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.
- <span id="page-10-15"></span>Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In ´ *European conference on computer vision*, pp. 740–755. Springer, 2014.
- <span id="page-10-13"></span>Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. Deep learning face attributes in the wild. In *Proceedings of the IEEE international conference on computer vision*, pp. 3730–3738, 2015.
- <span id="page-10-6"></span>Artem Lukoianov, Chenyang Yuan, Justin Solomon, and Vincent Sitzmann. Locality in image diffusion models emerges from data statistics. *arXiv preprint arXiv:2509.09672*, 2025.
- <span id="page-10-10"></span>Calvin Luo. Understanding diffusion models: A unified perspective. *arXiv preprint arXiv:2208.11970*, 2022.
- <span id="page-10-9"></span>Tomoya Matsumoto, Takayuki Miura, and Naoto Yanai. Membership inference attacks against diffusion models. In *2023 IEEE Security and Privacy Workshops (SPW)*, pp. 77–83. IEEE, 2023.
- <span id="page-10-16"></span>Elizbar A Nadaraya. On estimating regression. *Theory of Probability & Its Applications*, 9(1): 141–142, 1964.
- <span id="page-10-2"></span>Alexander Quinn Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models. In *International conference on machine learning*, pp. 8162–8171. PMLR, 2021.
- <span id="page-10-7"></span>Yan Pang, Tianhao Wang, Xuhui Kang, Mengdi Huai, and Yang Zhang. White-box membership inference attacks against diffusion models. *arXiv preprint arXiv:2308.06405*, 2023.
- <span id="page-10-0"></span>Jakiw Pidstrigach. Score-based generative models detect manifolds. *Advances in Neural Information Processing Systems*, 35:35852–35865, 2022.
- <span id="page-10-14"></span>Benjamin Recht, Rebecca Roelofs, Ludwig Schmidt, and Vaishaal Shankar. Do imagenet classifiers generalize to imagenet? In *International conference on machine learning*, pp. 5389–5400. PMLR, 2019.
- <span id="page-10-3"></span>Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High- ¨ resolution image synthesis with latent diffusion models. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pp. 10684–10695, 2022.

- <span id="page-11-9"></span>Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al. Imagenet large scale visual recognition challenge. *International journal of computer vision*, 115(3):211–252, 2015.
- <span id="page-11-8"></span>Alexandre Sablayrolles, Matthijs Douze, Cordelia Schmid, Yann Ollivier, and Herve J ´ egou. White- ´ box vs black-box: Bayes optimal strategies for membership inference. In *International Conference on Machine Learning*, pp. 5558–5567. PMLR, 2019.
- <span id="page-11-12"></span>Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. *Advances in neural information processing systems*, 35:25278–25294, 2022.
- <span id="page-11-5"></span>Reza Shokri, Marco Stronati, Congzheng Song, and Vitaly Shmatikov. Membership inference attacks against machine learning models. In *2017 IEEE symposium on security and privacy (SP)*, pp. 3–18. IEEE, 2017.
- <span id="page-11-1"></span>Gowthami Somepalli, Vasu Singla, Micah Goldblum, Jonas Geiping, and Tom Goldstein. Diffusion art or digital forgery? investigating data replication in diffusion models. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pp. 6048–6058, 2023a.
- <span id="page-11-2"></span>Gowthami Somepalli, Vasu Singla, Micah Goldblum, Jonas Geiping, and Tom Goldstein. Understanding and mitigating copying in diffusion models. *Advances in Neural Information Processing Systems*, 36:47783–47803, 2023b.
- <span id="page-11-3"></span>Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. *Advances in neural information processing systems*, 32, 2019.
- <span id="page-11-4"></span>Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. *arXiv preprint arXiv:2011.13456*, 2020.
- <span id="page-11-7"></span>Robert J Stanton and Alan Weinstein. On the l4 norm of spherical harmonics. In *Mathematical Proceedings of the Cambridge Philosophical Society*, volume 89, pp. 343–358. Cambridge University Press, 1981.
- <span id="page-11-10"></span>Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. *Advances in neural information processing systems*, 30, 2017.
- <span id="page-11-13"></span>Geoffrey S Watson. Smooth regression analysis. *Sankhya: The Indian Journal of Statistics, Series ¯ A*, pp. 359–372, 1964.
- <span id="page-11-11"></span>Peter Young, Alice Lai, Micah Hodosh, and Julia Hockenmaier. From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions. *Transactions of the association for computational linguistics*, 2:67–78, 2014.
- <span id="page-11-6"></span>Shengfang Zhai, Huanran Chen, Yinpeng Dong, Jiajun Li, Qingni Shen, Yansong Gao, Hang Su, and Yang Liu. Membership inference on text-to-image diffusion models via conditional likelihood discrepancy. *Advances in Neural Information Processing Systems*, 37:74122–74146, 2024.
- <span id="page-11-0"></span>Jun-Yan Zhu, Philipp Krahenb ¨ uhl, Eli Shechtman, and Alexei A Efros. Generative visual manipu- ¨ lation on the natural image manifold. In *European conference on computer vision*, pp. 597–613. Springer, 2016.

# A APPENDIX

# A.1 DATASETS AND SPLITS

Datasets and splits used for our experiments are summarized in Table [3.](#page-13-0)

# A.2 PERFORMANCE RANK ACROSS 15 EXPERIMENTS

A rank distribution over 5 benchmark methods [\(Duan et al., 2023;](#page-9-9) [Kong et al., 2023;](#page-10-8) [Matsumoto](#page-10-9) [et al., 2023;](#page-10-9) [Fu et al., 2023\)](#page-9-10) across 15 experiments is provided in Figure [5.](#page-18-0)

# A.3 ADDITIONAL EXPERIMENTS ON CLASS CONDITIONAL IMAGENET, AND STABLE DIFFUSION PRE-TRAINED ON LAION-AESTHETICS V2 5+.

For the unconditional case of Guided Diffusion, please see Table [4.](#page-14-0)

Pre-trained Latent Diffusion Models: For Pokemon ´ [2](#page-12-0) , COCO2017-Val [\(Lin et al., 2014\)](#page-10-15), and Flickr30k [\(Young et al., 2014\)](#page-11-11), we fine-tuned Stable Diffusion v1-4[3](#page-12-1) on a randomly selected subset of each training split, reserving an equally sized subset as the held-out set. We also studied the original Stable Diffusion v1-5[4](#page-12-2) checkpoint, pre-trained on LAION-Aesthetics v2 5+ [\(Schuhmann et al., 2022\)](#page-11-12) (*member* set). Here we sampled 2500 images from LAION-2B-MultiTranslated[5](#page-12-3) and COCO2017-Val as non-members, respectively. Notably, the images from LAION-2B-MultiTranslated are filtered with attributes *pwatermark* < 0.5; *prediction (aesthetic score)* > 5.0 and *similarity* > 0.3. *pwatermark* and *prediction* are to minimize the domain shift between the member set and the held-out set. And *similarity* is to ensure the alignment of the text-image pairs.

Stable Diffusion v1-4 was fine-tuned for 15k, 50k, and 200k steps on the member splits of Pokemon, ´ COCO-2017-Val, and Flickr30k, respectively, with a fixed learning rate of 1×10<sup>−</sup><sup>5</sup> (AdamW). None of these datasets are in the original pre-training corpus. Additionally, we adopted the default data augmentation (Random-Crop and Random-Flip) while training.

The results are summarized on Table [5](#page-14-1) and [6,](#page-15-0) which followed the experimental protocol of [Zhai](#page-11-6) [et al.](#page-11-6) [\(2024\)](#page-11-6) and evaluated membership-inference attacks on Stable Diffusion under two scenarios:

*Fine-tuning.* A Stable Diffusion v1-4 checkpoint was fine-tuned on the designated *member* split of each target dataset; attacks were launched on paired *member*/*held-out* splits.

*Pre-training.* A pre-trained Stable Diffusion v1-5 model was attacked directly, without additional fine-tuning. The *member* set was a subset of LAION-Aesthetics v2 5+ that was used during pretraining, while the *held-out* set was an equally-sized split drawn from the target dataset.

For every dataset we created five random *member*/*held-out* partitions. All experiments were run in both text-conditional and unconditional modes, except on the Pokemon dataset where only the ´ conditional mode was considered because the model over-fit rapidly. The unconditional baseline was obtained by passing an empty string to the CLIP text encoder. For datasets evaluated in both modes, the absolute gain of the conditional attack over the unconditional one was reported in blue. Across all datasets and metrics, conditioning on text consistently strengthened the attack, indicating that text-conditional generation memorised training data more severely than unconditional generation.

<span id="page-12-0"></span><sup>2</sup><https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions>

<span id="page-12-1"></span><sup>3</sup><https://huggingface.co/CompVis/stable-diffusion-v1-4>

<span id="page-12-2"></span><sup>4</sup><https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5>

<span id="page-12-3"></span><sup>5</sup><https://huggingface.co/datasets/laion/laion2B-multi-joined-translated-to-en>

Table 3: Datasets and splits used for our experiments.

<span id="page-13-0"></span>

| Model                 | Member                 | Held-out                 | Pre-trained | Fine-tuned | Splits    | Resolution | Cond. |
|-----------------------|------------------------|--------------------------|-------------|------------|-----------|------------|-------|
|                       | CIFAR-10               | CIFAR-10                 | No          | –          | 25k/25k   | 32         | –     |
|                       | CIFAR-100              | CIFAR-100                | No          | –          | 25k/25k   | 32         | –     |
| DDPM                  | STL10-U                | STL10-U                  | No          | –          | 50k/50k   | 32         | –     |
|                       | CelebA                 | CelebA                   | No          | –          | 30k/30k   | 32         | –     |
| Guided Diffusion      | ImageNet-1k            | ImageNetV2               | Yes         | No         | 3k/3k     | 256        | class |
|                       | Pokemon<br>´           | Pok<br>emon<br>´         | Yes         | Yes        | 416/417   | 512        | text  |
| Stable Diffusion V1-4 | COCO2017-Val           | COCO2017-Val             | Yes         | Yes        | 2.5k/2.5k | 512        | text  |
|                       | Flickr30k              | Flickr30k                | Yes         | Yes        | 10k/10k   | 512        | text  |
| Stable Diffusion V1-5 | LAION-Aesthetics v2 5+ | LAION-2B-MultiTranslated | Yes         | No         | 2.5k/2.5k | 512        | text  |
|                       | LAION-Aesthetics v2 5+ | COCO2017-Val             | Yes         | No         | 2.5k/2.5k | 512        | text  |

## A.4 FROM MARGINAL DENSITY TO EXPLICIT GAUSSIAN CONVOLUTION

Let α¯<sup>t</sup> = Q<sup>t</sup> <sup>s</sup>=1 α<sup>s</sup> and σ 2 <sup>t</sup> = 1 − α¯t. The VP forward marginal is

$$p_t(x) = \int_{\mathbb{R}^d} p_{\text{data}}(x_0) \, \mathcal{N}\left(x \,\middle|\, \sqrt{\bar{\alpha}_t} \, x_0, \, \sigma_t^2 I\right) dx_0. \tag{20}$$

Set u = √ <sup>α</sup>¯<sup>t</sup> <sup>x</sup><sup>0</sup> so that <sup>x</sup><sup>0</sup> <sup>=</sup> u/<sup>√</sup> α¯<sup>t</sup> and dx<sup>0</sup> = ¯α −d/2 <sup>t</sup> du. Then

<span id="page-13-1"></span>
$$p_t(x) = \bar{\alpha}_t^{-d/2} \int_{\mathbb{R}^d} p_{\text{data}} \left( \frac{u}{\sqrt{\bar{\alpha}_t}} \right) \, \mathcal{N}(x \, | \, u, \, \sigma_t^2 I) \, du. \tag{21}$$

Introduce <sup>x</sup>˜ := x/<sup>√</sup> <sup>α</sup>¯<sup>t</sup> and <sup>u</sup>˜ := u/<sup>√</sup> α¯t, so u = √ α¯<sup>t</sup> u˜ and du = ¯α d/2 <sup>t</sup> du˜. Substituting into equation [21](#page-13-1) cancels the Jacobians and yields

<span id="page-13-2"></span>
$$p_t(x) = \int_{\mathbb{R}^d} p_{\text{data}}(\tilde{u}) \, \mathcal{N}\left(\sqrt{\bar{\alpha}_t} \left(\tilde{x} - \tilde{u}\right) \middle| 0, \, \sigma_t^2 I\right) d\tilde{u}. \tag{22}$$

Use the Gaussian scaling identity

$$\mathcal{N}(az\mid 0,\,\sigma^2I) = a^{-d}\,\mathcal{N}\bigg(z\mid 0,\,\frac{\sigma^2}{a^2}I\bigg) \qquad \text{(for $a>0$)},$$

with a = √ α¯<sup>t</sup> and z = ˜x − u˜. Then equation [22](#page-13-2) becomes

<span id="page-13-3"></span>
$$p_t(x) = \bar{\alpha}_t^{-d/2} \int_{\mathbb{R}^d} p_{\text{data}}(\tilde{u}) \, \mathcal{N}\left(\tilde{x} - \tilde{u} \,\middle|\, 0, \, \frac{\sigma_t^2}{\bar{\alpha}_t} I\right) \, d\tilde{u}. \tag{23}$$

The integral in equation [23](#page-13-3) is a (Euclidean) convolution evaluated at x˜:

$$\left(p_{\mathrm{data}} * \mathcal{N}(0, \frac{\sigma_t^2}{\bar{\alpha}_t} I)\right)(\tilde{x}) = \int_{\mathbb{R}^d} p_{\mathrm{data}}(\tilde{u}) \; \mathcal{N}\Big(\tilde{x} - \tilde{u} \, \big| \, 0, \; \frac{\sigma_t^2}{\bar{\alpha}_t} I\Big) \; d\tilde{u}.$$

Therefore,

$$p_t(x) = \bar{\alpha}_t^{-d/2} \left( p_{\text{data}} * \mathcal{N}\left(0, \frac{\sigma_t^2}{\bar{\alpha}_t} I\right) \right) \left( \frac{x}{\sqrt{\bar{\alpha}_t}} \right)$$
 (24)

with σ 2 <sup>t</sup> /α¯<sup>t</sup> = ¯α −1 <sup>t</sup> − 1.

# A.5 MIA IN THREE CASES

When the *data prior* is the empirical distribution constructed from the training set {x (i)} N <sup>i</sup>=1. We have ptrain(x0) = <sup>1</sup> N P i δ(x<sup>0</sup> − x (i) ). For any test image x, the UNet predicts

<span id="page-13-4"></span>
$$\hat{\varepsilon}_{\theta}(x,t) \approx \frac{x - \sqrt{\bar{\alpha}_t} \, \mu_t(x)}{\sigma_t},$$
(25)

<span id="page-14-0"></span>Table 4: Performance of Guided Diffusion pretrained on ImageNet1K with class conditional changes. Best results in bold, second-best underlined. *Member* set: ImageNet-1K (3000); *Held-out*: ImageNetV2 (3000). The performance difference of the class-conditional attack from the unconditional one is in blue. Format: *uncond.%* (∆ = *cond.*–*uncond.*).

| Method    | #Query↓ | ASR↑         | AUC↑                      | TPR@1%FPR↑    |
|-----------|---------|--------------|---------------------------|---------------|
| PIA       | 2       | 64.65(-1.98) | 66.44(-0.93)              | 9.93(-3.96)   |
| PFAMIMet  | 20      |              | 67.85(+0.22) 72.22(+0.25) | 3.77(+0.50)   |
| SecMIstat | 12      | 77.97(-9.09) | 82.55(-8.94)              | 34.73(-19.53) |
| Loss      | 1       |              | 57.27(+0.10) 60.38(+0.15) | 7.00(+0.07)   |
| SimA      | 1       |              | 85.73(+2.57) 89.77(+2.03) | 21.73(+6.87)  |

<span id="page-14-1"></span>Table 5: Performance of Stable Diffusion that is pretrianed on LAION-Aesthetics v2 5+. COCO2017-Val and Flickr30k here follows the fine-tuning setting. LAION-2B-MultiTranslated is the pre-training setting. The performance difference of the text-conditional attack from the unconditional one is in blue. Format: *uncond.%* (∆ = *cond.*–*uncond.*).

| Method    | #Query↓ | COCO2017-Val (%) |      |                                                                                                       |      | Flickr30k (%) |       | LAION-2B-MultiTranslated (%) |            |             |
|-----------|---------|------------------|------|-------------------------------------------------------------------------------------------------------|------|---------------|-------|------------------------------|------------|-------------|
|           |         | ASR↑             | AUC↑ | TPR†↑                                                                                                 | ASR↑ | AUC↑          | TPR†↑ | ASR↑                         | AUC↑       | TPR†↑       |
| PIA       | 2       |                  |      | 62.4(+0.7) 65.8(+1.2) 2.88(+0.32) 59.3(+0.6) 61.8(+0.8) 1.64(+0.17) 60.6(–0.0) 63.8(+0.0) 4.12(+0.08) |      |               |       |                              |            |             |
| PFAMIMet  | 20      |                  |      | 61.9(+0.7) 65.4(+0.9) 2.28(+0.20) 59.2(+0.4) 61.6(+0.6) 1.75(+0.11) 50.0(+0.0) 40.8(+0.2) 0.80(+0.08) |      |               |       |                              |            |             |
| SecMIstat | 12      |                  |      | 64.4(+0.7) 69.1(+1.0) 4.32(+0.60) 61.1(+0.5) 64.6(+0.9) 2.85(+0.57) 56.7(+0.1) 58.6(+0.0) 2.92(-0.16) |      |               |       |                              |            |             |
| Loss      | 1       |                  |      | 58.5(+0.6) 61.7(+1.0) 3.44(+0.12) 56.1(+0.4) 58.3(+0.6) 1.83(+0.08) 50.1(+0.0) 37.0(+0.0) 0.80(+0.00) |      |               |       |                              |            |             |
| SimA      | 1       |                  |      | 63.1(+0.9) 66.5(+1.3) 3.04(+0.28) 60.1(+0.6) 62.5(+0.9) 1.71(+0.17) 60.3(-0.0)                        |      |               |       |                              | 63.3(-0.0) | 3.20(-0.08) |

<sup>†</sup> True Positive Rate at 1 % False Positive Rate

<span id="page-14-2"></span>We are interested in the quantity

$$\mu_t^{\text{finite}}(x) = \mathbb{E}_{q_t(x_0|x)}[x_0] = \sum_{i=1}^N w_i(x,t) \, x^{(i)},$$

$$w_i(x,t) = \frac{\exp[-\|x - \sqrt{\bar{\alpha}_t} \, x^{(i)}\|^2 / (2\sigma_t^2)]}{\sum_{j=1}^N \exp[-\|x - \sqrt{\bar{\alpha}_t} \, x^{(j)}\|^2 / (2\sigma_t^2)]}.$$
(26)

Equations equation [26](#page-14-2) express the posterior mean as a weighted average of the training samples, where each kernel weight wi(x, t) depends on the Euclidean distance between the noisy query x and the down-scaled datum <sup>√</sup> α¯<sup>t</sup> x (i) .

CASE 1 — TRAINING MEMBER

Pick x = x (k) ∈ {x (i)}. Set x = x (k) in equation [26.](#page-14-2) Define the squared distances

<span id="page-14-3"></span>
$$d_{ik}(t) := \|x^{(k)} - \sqrt{\bar{\alpha}_t} x^{(i)}\|^2, \qquad \Delta_{ik}(t) := \frac{d_{ik}(t) - d_{kk}(t)}{2\sigma_t^2}.$$
 (27)

Using equation [26](#page-14-2) and equation [27](#page-14-3) we obtain

$$w_k(x^{(k)}, t) = \left[1 + \sum_{i \neq k} \exp(-\Delta_{ik}(t))\right]^{-1},$$
 (28)

$$w_{i\neq k}(x^{(k)},t) = \exp(-\Delta_{ik}(t)) w_k(x^{(k)},t).$$
 (29)

SMALL-NOISE LIMIT t → 0. Because σ 2 <sup>t</sup> = 1 − α¯<sup>t</sup> → 0 and

$$1 - \sqrt{\bar{\alpha}_t} = O(\sigma_t^2), \qquad d_{kk}(t) = (1 - \sqrt{\bar{\alpha}_t})^2 ||x^{(k)}||^2 = O(\sigma_t^4),$$

<span id="page-15-0"></span>Table 6: Performance of Stable Diffusion pre-trained on LAION-Aesthetics v2 5+. Pokemon uses ´ SD v1-4 (fine-tuned) and reports *conditional only* (no parentheses). COCO2017-Val uses SD v1-5 (held-out) and reports in the format *uncond.%* (∆ = *cond.* − *uncond.*). Best in bold, second-best underlined.

| Method    | #Query↓ | Pokemon (fine-tune) (%)<br>´ |      |           | COCO2017-Val (pre-training) (%) |             |            |  |
|-----------|---------|------------------------------|------|-----------|---------------------------------|-------------|------------|--|
|           |         | ASR↑                         | AUC↑ | TPR†<br>↑ | ASR↑                            | AUC↑        | TPR†<br>↑  |  |
| PIA       | 2       | 89.9                         | 94.6 | 30.9      | 53.7(-0.06)                     | 52.3(-0.14) | 2.1(-0.12) |  |
| PFAMIMet  | 20      | 50.0                         | 19.0 | 0.0       | 52.0(+0.08)                     | 48.7(+0.12) | 0.4(+0.00) |  |
| SecMIstat | 12      | 81.4                         | 87.8 | 34.1      | 55.7(-0.06)                     | 56.0(-0.13) | 1.2(+0.08) |  |
| Loss      | 1       | 80.2                         | 87.9 | 24.7      | 55.4(+0.02)                     | 49.3(-0.01) | 0.4(+0.00) |  |
| SimA      | 1       | 87.6                         | 93.0 | 21.8      | 54.5(-0.04)                     | 53.7(-0.07) | 1.5(+0.12) |  |

<sup>†</sup> True Positive Rate at 1 % False Positive Rate.

we have, for i ̸= k,

$$\Delta_{ik}(t) \sim \frac{\|x^{(k)} - x^{(i)}\|^2}{2\sigma_t^2} \xrightarrow[t \to 0]{} +\infty, \qquad \Delta_{kk}(t) = 0.$$

Hence

$$w_k(x^{(k)}, t) \xrightarrow[t \to 0]{} 1, \qquad w_{i \neq k}(x^{(k)}, t) \xrightarrow[t \to 0]{} 0.$$
 (30)

This implies

$$\left[ \mu_t^{\text{finite}}(x^{(k)}) \xrightarrow[t \to 0]{} x^{(k)} \right].$$
(31)

Moreover, substituting into the estimator equation [25,](#page-13-4)

$$\|\hat{\varepsilon}_{\theta}(x^{(k)}, t)\|_{2} \approx \|\frac{x^{(k)} - \sqrt{\bar{\alpha}_{t}} x^{(k)}}{\sigma_{t}}\|_{2} = \frac{|1 - \sqrt{\bar{\alpha}_{t}}|}{\sigma_{t}} \|x^{(k)}\|_{2}.$$
(32)

To see the asymptotic form using Taylor expansion around σ 2 <sup>t</sup> = 0, note that

$$\sqrt{\bar{\alpha}_t} = \sqrt{1 - \sigma_t^2} = 1 - \frac{1}{2}\sigma_t^2 - \frac{1}{8}\sigma_t^4 + O(\sigma_t^6),$$

so that

$$1 - \sqrt{\bar{\alpha}_t} = \frac{1}{2}\sigma_t^2 + O(\sigma_t^4).$$

Therefore

$$\frac{1 - \sqrt{\bar{\alpha}_t}}{\sigma_t} = \frac{1}{2}\sigma_t + O(\sigma_t^3),$$

and hence

$$\left\|\hat{\varepsilon}_{\theta}(x^{(k)}, t)\right\|_{2} \sim \frac{\sigma_{t}}{2} \|x^{(k)}\|_{2} \xrightarrow[t \to 0]{} 0. \tag{33}$$

#### CASE 2 — HELD-OUT BUT ON-MANIFOLD

Consider a test data point x † which is not in our original dataset (i.e., x † ∈ { / x (i)} N <sup>i</sup>=1), but sampled from the same generating distribution pdata. Under the diffusion process assumptions, the local weighted mean µt(x) in equation [26](#page-14-2) has the same algebraic form as a kernel regression [\(Nadaraya,](#page-10-16) [1964;](#page-10-16) [Watson, 1964\)](#page-11-13), using training dataset {x (i)} N <sup>i</sup>=1 with (effective) Gaussian bandwidth

$$h(t) := \frac{\sigma_t}{\sqrt{\bar{\alpha}_t}}.$$
 (34)

The weights wi(x, t) in equation [26](#page-14-2) are proportional to exp −∥x † − √ αx¯ (i)∥ 2 <sup>2</sup>/(2h(t) 2 ) , so µ finite t (x † ) coincides with a Gaussian–kernel local average of "nearby" training points, where "nearby" is on the order of bandwidth h(t). The kernel-weighted local mean with radius r around x † is defined as

$$m_{r}(x^{\dagger}) := \frac{\int_{B_{r}(x^{\dagger})} u K_{r}(u - x^{\dagger}) p(u) du}{\int_{B_{r}(x^{\dagger})} K_{r}(u - x^{\dagger}) p(u) du},$$
(35)

$$K_r(z) \propto \exp\left(-\frac{\|z\|^2}{2r^2}\right),$$
 (36)

where r ≍ h(t) and p(u) is the empirical distribution. This is a normalization of the local mean defined in Eq. [15](#page-5-0) of this paper, and Appendix 6.4.1 of [Alain & Bengio](#page-9-7) [\(2014\)](#page-9-7). By local moment matching [\(Bengio et al., 2012;](#page-9-12) [Alain & Bengio, 2014\)](#page-9-7), combining Theorems 2 and 3 of [Alain &](#page-9-7) [Bengio](#page-9-7) [\(2014\)](#page-9-7) (see their Eq. (28)) yields the second-order expansion

$$m_r(x^{\dagger}) - x^{\dagger} = \frac{r^2}{d+2} \left. \frac{\partial \log p_t(x)}{\partial x} \right|_{x^{\dagger}} + o(r^3),$$
 (37)

where p<sup>t</sup> is the Gaussian-mollified density at scale σt. Using equation [9,](#page-4-2) we obtain the proportionality stated in the main text:

<span id="page-16-0"></span>
$$\boxed{m_r(x^{\dagger}) - x^{\dagger} \approx \frac{r^2}{d+2} \left. \frac{\partial \log p_t(x)}{\partial x} \right|_{x^{\dagger}} = \frac{r^2}{d+2} \left( \frac{x^{\dagger} - \sqrt{\bar{\alpha}_t} \mu_t^{\text{finite}}(x^{\dagger})}{\sigma_t^2} \right) = -\frac{r^2}{\sigma_t(d+2)} \hat{\varepsilon}_{\theta}(x^{\dagger}, t)}$$
(38)

Notably, this equation should *not* be correct if x † is in a low density region of the support of pdata, but these points should generally be rarely sampled by definition. Moreover, the εˆ estimate of the score will also be extrapolating at those points.

Implication for the attack statistic. In in-support regions of the empirical distribution, the local geometry is generally *not* flat, so ∥mr(x † ) − x †∥ > 0 for r > 0 sufficiently small. By equation [38,](#page-16-0) this implies ∥εˆθ(x † , t)∥ is bounded away from zero (at fixed t), hence A(x † , t) = ∥εˆθ(x † , t)∥<sup>p</sup> exceeds the member case (Case 1). This matches the intuition summarized in the main text: onmanifold, held-out queries denoise less precisely than memorized training points, yielding a moderately larger attack statistic without the divergence seen off-manifold.

#### CASE 3 — OUT-OF-DISTRIBUTION (FAR OFF-MANIFOLD)

Since no training data support is available in out-of-distribution (OOD) regions, the diffusion model lacks information about these areas. Consequently, the learned score field in such regions is necessarily an extrapolation, and the theoretical derivations established under the in-distribution assumption no longer hold.

# A.6 TRAINING DETAILS OF ADVERSARIAL β-VAE ON CIFAR-10

We train a series of β-VAE on the CIFAR-10 dataset using AdamW optimization with a learning rate of 2 × 10<sup>−</sup><sup>4</sup> , and investigate the relationship between membership information leakage and latent space regularization. The model was trained with a batch size of 128 and β = {0.0, 10<sup>−</sup><sup>7</sup> , 10<sup>−</sup><sup>6</sup> , 10<sup>−</sup><sup>5</sup> , 10<sup>−</sup><sup>4</sup> , 10<sup>−</sup><sup>3</sup> , 10<sup>−</sup><sup>2</sup> , 10<sup>−</sup><sup>1</sup> , 0.5, 1.0} to balance reconstruction fidelity with latent space regularization.

For comparison, we considered two variants: (i) a standard β-VAE trained solely with pixel-level reconstruction and KL divergence losses, and (ii) a β-VAE augmented with an adversarial discriminator (adversarial β-VAE) applied between input and reconstructed images. In the latter case, a hinge-loss–based PatchGAN [\(Isola et al., 2017\)](#page-10-17) discriminator was introduced partway through training, encouraging the reconstructions to be not only faithful but also visually sharper. Across both setups, training was conducted for sufficient epochs to ensure convergence (120 epochs). The latter LDMs are trained based on these frozen VAE encoders. For phase two LDM training, we deliberately train 2048 epochs to ensure the membership information leakage. For training split, we reuse the member split from previous experiments as shown in Table [3.](#page-13-0) We sample 1000 images from the well-trained LDM checkpoints and calculate the FID to training split.

Specifically, in the adversarial setup, optimization proceeds in two steps per iteration. First, we update the VAE with the usual reconstruction+KL objective and, after a warm-up, add a non-saturating adversarial term (hinge form; equivalent to −E[D(ˆx)]) gated to turn on only after a chosen globalstep threshold (we set this to 2,000 steps in our runs). The adversarial term is weighted adaptively using a VQGAN-style [\(Esser et al., 2021\)](#page-9-6) gradient-norm ratio on the decoder's last layer and then scaled by a constant factor of 0.8; before the start step, its weight is zero by design. Second, we update the discriminator with a hinge loss on real images x versus reconstructions xˆ recomputed in a no-grad branch (to avoid generator gradient leakage), using AdamW with the same learning rate/betas/weight-decay as the VAE optimizer.

# A.7 WHY EXTREMELY EARLY AND LATEt ARE PROBLEMATIC:

Intuitively, εˆθ(x, t)|t=0 is expected to achieve the best performance. Because the noise added to a member at t = 0 is expected to be zero (as shown in case 1) and non-zero for a nonmember. However, in region of low data density, score-matching lacks sufficient evidence to *reliably* estimate the score function [\(Song & Ermon,](#page-11-3) [2019\)](#page-11-3). [Song & Ermon](#page-11-3) [\(2019\)](#page-11-3) argues that training minimizes the expected value of score estimates (here is Ept=0-∥εˆθ(xt=0, t) − ε∥ 2 ), which provides *inaccurate scores* where pt=0(x) is infinitesimal. To be specific, for the input x ∈ R | {xi} N <sup>i</sup>=1 ∩ R = ∅, R ⊂ R d , ∇xpt=0(x) extrapolates erratically. Consequently, for very early timesteps (σ<sup>t</sup> ≪ 1) the learned field outside the tightly supported member set can plateau or even shrink, nullifying the privacy signal. Increasing t corresponds to *extra Gaussian convolution*, expanding the effective support and regularising the score. Figure [6](#page-18-0) plots the average normalised estimator magnitude ||εˆθ(x, t)|| for t ∈ [0, 300] on the *member* and *held-out* splits across datasets. Transient fluctuations are confined to the very earliest timesteps (t ≈ 0), and the maximal gap between the curves typically occurs at early—but not initial—timesteps, indicating that moderately early diffusion steps provide the strongest membership signal.

<span id="page-17-0"></span>![](_page_17_Figure_4.jpeg)

Figure 4: Top: density pt(x); blue dots are training samples. Bottom: estimated score ∇<sup>x</sup> log pt(x). For very early t ∈ [0, 10], the inter-mode region is low-density, so the score extrapolates erratically (shaded band). Optimal t ∈ [10, 300]: moderate Gaussian convolution enlarges the support and regularizes the estimator—density bridges the modes and the score points coherently toward them, yielding the strongest separation between members and held-out points. For late t ∈ [300, 1000], p<sup>t</sup> approaches an isotropic Gaussian and the score collapses toward the global mean, diminishing membership signal.

Conversely, for late steps (σ<sup>t</sup> ≈ 1) the forward process approaches an isotropic Gaussian (data information gradually diminish); p<sup>t</sup> is nearly homogeneous, so the posterior µt(x) = qt(x<sup>0</sup> ∼ ptraining | x) collapses to µt(x) = qt(x<sup>0</sup> ∼ N (0, I) | x), which depends on test images and membership information is lost. Figure [4](#page-17-0) illustrates the phenomenon. The *optimal* timestep t ∗ is therefore dataset-specific and also depends on the noise schedule.

<span id="page-18-0"></span>![](_page_18_Figure_0.jpeg)

Figure 5: Average ranks (±1σ) of the five benchmark methods across 15 experiments for four evaluation metrics. Higher values indicate better AUC, ASR, and TPR@1%FPR; lower values indicate fewer #Queries

![](_page_18_Figure_2.jpeg)

Figure 6: The average normalised estimator magnitude ||εˆθ(x, t)|| for t ∈ [0, 300] on the *member* and *held-out* splits across datasets