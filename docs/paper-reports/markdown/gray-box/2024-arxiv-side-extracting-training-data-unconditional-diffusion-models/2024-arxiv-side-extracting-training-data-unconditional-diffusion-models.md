# <span id="page-0-0"></span>SIDE: Surrogate Conditional Data Extraction from Diffusion Models

# Yunhao Chen Fudan University

24110240013@m.fudan.edu.cn

# Shujie Wang Fudan University

24110240084@m.fudan.edu.cn

# Difan Zou University of Hong Kong

dzou@cs.hku.hk

# Xingjun Ma Fudan University

xingjunma@fudan.edu.cn

### Abstract

*As diffusion probabilistic models (DPMs) become central to Generative AI (GenAI), understanding their memorization behavior is essential for evaluating risks such as data leakage, copyright infringement, and trustworthiness. While prior research finds conditional DPMs highly susceptible to data extraction attacks using explicit prompts, unconditional models are often assumed to be safe. We challenge this view by introducing Surrogate condItional Data Extraction (SIDE), a general framework that constructs data-driven surrogate conditions to enable targeted extraction from any DPM. Through extensive experiments on CIFAR-10, CelebA, ImageNet, and LAION-5B, we show that SIDE can successfully extract training data from socalled safe unconditional models, outperforming baseline attacks even on conditional models. Complementing these findings, we present a unified theoretical framework based on informative labels, demonstrating that all forms of conditioning, explicit or surrogate, amplify memorization. Our work redefines the threat landscape for DPMs, establishing precise conditioning as a fundamental vulnerability and setting a new, stronger benchmark for model privacy evaluation.*

#### 1. Introduction

Diffusion probabilistic models (DPMs) [\[31,](#page-8-0) [56,](#page-9-0) [60\]](#page-10-0) are a powerful class of generative models that learn data distributions by progressively corrupting data through a forward diffusion process and then reconstructing it via a reverse process. Owing to their remarkable ability to model complex data distributions, DPMs have become the foundation for many leading Generative Artificial Intelligence (GenAI) systems, including Stable Diffusion [\[50\]](#page-9-1), DALL-E 3 [\[4\]](#page-7-0), and Sora [\[6\]](#page-7-1).

However, the widespread adoption of DPMs has raised concerns about *data memorization*, which is the tendency of models to memorize raw training samples. This can lead to the generation of duplicated rather than novel content, increasing the risks of data leakage, privacy breaches, and copyright infringement [\[2,](#page-7-2) [16,](#page-8-1) [57,](#page-10-1) [58\]](#page-10-2). For example, Stable Diffusion has been criticized as a "21st-century collage tool" for remixing copyrighted works of artists whose data was used during training [\[7\]](#page-8-2). Furthermore, memorization can facilitate data extraction attacks, enabling adversaries to recover training data from deployed models. Recent work by [\[8,](#page-8-3) [65\]](#page-10-3) demonstrated the feasibility of extracting training data from DPMs such as Stable Diffusion [\[50\]](#page-9-1), highlighting substantial privacy and copyright risks.

Existing studies show that conditional DPMs are far more prone to memorizing training data than unconditional ones, making extraction from unconditional models extremely challenging [\[27,](#page-8-4) [58\]](#page-10-2). While conditional models can be compromised via prompts, unconditional models are generally seen as much safer, and current extraction methods struggle without detailed prompts.

To bridge this gap, we propose Surrogate condItional Data Extraction (SIDE), a general and effective approach for extracting training data from both conditional and unconditional DPMs. SIDE uses cluster information on generated images as a surrogate condition, providing precise guidance toward target samples. This approach outperforms conventional text prompts/class index for conditional models and enables robust extraction attacks on unconditional models. Examples of extracted images are shown in Figure [1.](#page-1-0) Additionally, we introduce a divergence measure to quantify memorization in DPMs and provide a theoretical analysis that explains: (1) why conditional DPMs are more susceptible to memorization, even with random labels, and

<span id="page-1-1"></span><span id="page-1-0"></span>![](_page_1_Figure_0.jpeg)

Figure 1. Examples of training images (top) and corresponding extracted images by our SIDE method (bottom) from a DDPM trained on a subset of CelebA.

- (2) why SIDE is effective for data extraction. In summary, our main contributions are as follows:
- We propose SIDE, a novel data extraction method that leverages a surrogate condition to extract training data from DPMs.
- We introduce a divergence-based memorization measure and provide a theoretical analysis of the impact of conditioning in DPMs and the effectiveness of SIDE.
- Experiments on CIFAR-10, CelebA, ImageNet, and LAION-5B show that SIDE can extract training data from unconditional DPMs, often with even greater efficacy than attacks on conditional counterparts, offering new perspectives on the privacy risks of DPMs.

### 2. Related Work

Diffusion Probabilistic Models. DPMs [\[56\]](#page-9-0) have achieved state-of-the-art performance in image and video generation, as exemplified by models such as Stable Diffusion [\[50\]](#page-9-1), DALL-E 3 [\[4\]](#page-7-0), Sora [\[6\]](#page-7-1), Runway [\[50\]](#page-9-1) and Imagen [\[52\]](#page-9-2). These models excel on various benchmarks [\[20\]](#page-8-5). DPMs can be interpreted from two perspectives: 1) *score matching* [\[60\]](#page-10-0), where model learns the gradient of data distribution [\[61\]](#page-10-4), and (2) *denoising diffusion* [\[31\]](#page-8-0), where Gaussian noise is added to clean images over multiple time steps, and the model is trained to reverse this process. For conditional sampling, [\[20\]](#page-8-5) introduced classifier guidance to steer the denoising process, while [\[32\]](#page-8-6) proposed classifier-free guidance, enabling conditional generation without explicit classifiers.

Memorization in Diffusion Models. Early research on memorization primarily focused on language models [\[9,](#page-8-7) [36\]](#page-9-3), which later inspired subsequent studies on DPMs [\[1,](#page-7-3) [3,](#page-7-4) [5,](#page-7-5) [10,](#page-8-8) [17,](#page-8-9) [19,](#page-8-10) [21,](#page-8-11) [21,](#page-8-11) [23](#page-8-12)[–28,](#page-8-13) [30,](#page-8-14) [37](#page-9-4)[–39,](#page-9-5) [41,](#page-9-6) [42,](#page-9-7) [45,](#page-9-8) [48,](#page-9-9) [54,](#page-9-10) [58,](#page-10-2) [68](#page-10-5)[–70\]](#page-10-6), from quantifying direct data duplication [\[8,](#page-8-3) [57\]](#page-10-1) to inferring the presence of an entire identity within the training data [\[64\]](#page-10-7). Notably, [\[57\]](#page-10-1) found that 0.5-2% of generated images duplicate training samples, a result corroborated by [\[8\]](#page-8-3) through more extensive experiments on both conditional and unconditional DPMs. Further studies [\[27,](#page-8-4) [58\]](#page-10-2) linked memorization to model conditioning, showing that conditional DPMs are more prone to memorization. To address memorization, several methods have been proposed for detection and mitigation. For example, (author?) [\[66\]](#page-10-8) introduced a method to detect memorization-triggering prompts by analyzing the magnitude of text-conditional predictions, achieving high accuracy with minimal computational overhead. (author?) [\[49\]](#page-9-11) proposed metrics based on cross-attention patterns in DPMs to identify memorization. On the mitigation side, (author?) [\[11\]](#page-8-15) developed anti-memorization guidance to reduce memorization during sampling, while (author?) [\[49\]](#page-9-11) modified attention scores or masked summary tokens in the crossattention layer. (author?) [\[66\]](#page-10-8) minimized memorization by controlling prediction magnitudes during inference.

Despite recent advances, the effectiveness and focus of current research on data extraction have been uneven. Most successful attacks target conditional DPMs, leveraging explicit conditions (e.g., prompts) to guide the generation process toward memorized samples [\[8,](#page-8-3) [67\]](#page-10-9). In contrast, extracting data from unconditional DPMs has proven to be significantly more challenging due to the absence of such guidance mechanisms [\[27\]](#page-8-4). To gain deeper insight into memorization in both conditional and unconditional DPMs, we introduce a novel and general data extraction method that enables effective extraction across both model types.

# 3. Surrogate Conditional Data Extraction

Threat Model. We adopt a white-box threat model in which the attacker has full access to the model parameters. The attacker's goal is to extract original training samples from the target DPM, whether it is conditional or unconditional. In the Appendix, we further extend our SIDE method to black-box and backdoor scenarios.

#### 3.1. Intuition of SIDE

Conditional DPMs are known to be more prone to memorization because they rely on explicit labels, such as class tags or prompts, that help steer the model toward specific samples [\[27,](#page-8-4) [58\]](#page-10-2). Unconditional DPMs, by contrast, are trained without explicit labels, yet they implicitly partition the training data into latent clusters, even though these groupings are never explicitly specified [\[13\]](#page-8-16). We refer to these as implicit labels.

The key intuition behind SIDE is that if we can uncover and formalize these clustering patterns within the training data, we can effectively "create" implicit labels to enable conditional control over the model's outputs.. This approach is powerful because it harnesses the model's own internal structure for guidance, providing a more direct and targeted way to reach memorized samples than traditional extraction techniques (see Figure [2\)](#page-2-0). Below, we outline how to construct implicit labels for unconditional DPMs.

<span id="page-2-1"></span><span id="page-2-0"></span>![](_page_2_Figure_0.jpeg)

Figure 2. Rationale behind SIDE's effectiveness. Compared to unconditional models (a), conditional models (b) tend to memorize more due to prompt-based semantic guidance, but this guidance remains too broad for reliable extraction. Our SIDE (c) overcomes this by identifying high-density memorized clusters and creating precise surrogate conditions, enabling more accurate and direct extraction from unconditional models than is possible with conventional conditional approaches.

# 3.2. Constructing Implicit Labels

To generate implicit labels without access to the original training data, we cluster a set of generated images using a pre-trained feature extractor. Clusters with low cohesion (measured by cosine similarity) are removed, and the centroids of the remaining high-quality clusters serve as our surrogate conditions, y<sup>I</sup> . These conditions guide the DPM's reverse sampling process toward specific, highdensity regions where memorized data is likely to reside. Although this guidance can be implemented via a gradient term ∇<sup>x</sup> log p t θ (y<sup>I</sup> |x), neural classifiers are often miscalibrated. To address this, we introduce a hyperparameter λ to adjust the guidance strength, resulting in our final SDE:

$$dx = \left[ f(x,t) - g(t)^2 \left( \nabla_x \log p_\theta^t(x) + \nabla_x \log p_\theta^t(y_I|x) \right) \right] dt + g(t) dw.$$
 (1)

Our formulation, grounded in a power prior, offers a more principled justification for classifier guidance with λ ̸= 1 than previous work [\[20\]](#page-8-5). The process for training the timedependent classifier p t θ (y<sup>I</sup> |x) on a pseudo-labeled synthetic dataset is illustrated in Figure [3.](#page-3-0)

#### 3.3. Training with Surrogate Conditions

To guide the diffusion model toward class-specific data, we first establish a conditional generation mechanism using pseudo-labels. We explore two distinct approaches for creating these surrogate conditions, selecting the method based on the architecture and scale of the target DPM. For large-scale models like Stable Diffusion, we use parameterefficient LoRA fine-tuning. For smaller diffusion models, we adopt the traditional approach of training an external, time-dependent classifier for guidance. Both methods begin by generating a synthetic dataset with the target DPM and assigning pseudo-labels via feature clustering with a pretrained extractor, following established techniques [\[14,](#page-8-17) [15\]](#page-8-18).

Method 1: Training a Time-Dependent Classifier for Small-scale DPMs. For small-scale diffusion models, we train an external, time-dependent classifier. Given each synthetic image x and its pseudo-label y, we simulate the forward diffusion process by adding Gaussian noise at various timesteps t, producing a set of noisy samples (xt, t, y). The classifier architecture is adapted to accept the timestep t as input (see Figure [9](#page-12-0) in the Appendix), and is trained on this noisy dataset. The goal is to predict the original label y from the noisy image x<sup>t</sup> by minimizing:

$$\mathcal{L}_{\text{cls}} = \mathbb{E}_{t,(x_t,y) \sim \mathcal{D}_{\text{noisy}}} [-\log p_{\theta}^t(y|x_t)]$$
 (2)

This training process is illustrated in Figure [3.](#page-3-0) The resulting classifier p t θ (y|xt) provides an external guidance signal during the reverse diffusion process.

#### Method 2: LoRA Fine-tuning for Large-scale DPMs.

For large-scale models such as Stable Diffusion, training a separate classifier is computationally intensive. Instead, we leverage LoRA [\[35\]](#page-9-12) to directly fine-tune the DPM. Specifically, we freeze the original DPM parameters and insert trainable, low-rank matrices into the U-Net architecture. These lightweight adapters are then fine-tuned on our synthetic dataset, conditioning the DPM on the pseudo-labels y. The training objective is to minimize the standard diffusion loss with conditioning:

$$\mathcal{L}_{LoRA} = \mathbb{E}_{t,x_0,\epsilon,y}[|\epsilon - \epsilon_{\theta + \Delta\theta}(x_t, t, y)|^2]$$
 (3)

<span id="page-3-0"></span>![](_page_3_Figure_0.jpeg)

Figure 3. An illustration of time-dependent classifier training on a pseudo-labeled synthetic dataset.

where θ denotes the frozen DPM weights and ∆θ are the trainable LoRA parameters.

# 3.4. Overall Procedure of SIDE

Our SIDE method comprises two main phases. First, it generates a synthetic dataset and assigns pseudo-labels to establish a surrogate guidance mechanism, training the conditional model with the appropriate method described above. During extraction, SIDE applies guidance at each denoising step to steer x<sup>t</sup> toward a randomly selected target cluster—using a classifier gradient for small-scale DPMs or conditioning via the LoRA-adapted model for large-scale DPMs. We then evaluate SIDE using similarity scores on the extracted images and introduce comprehensive metrics for robust assessment in our experiments.

### 4. Theoretical Analysis

In this section, we first introduce a Kullback-Leibler (KL) divergence-based measure to quantify the degree of memorization in generative models. Building on this, we provide a theoretical explanation for data memorization in conditional DPMs and clarify why SIDE can effectively extract data.

#### 4.1. Distributional Memorization Measure

Several approaches exist for measuring the memorization effect in generative models. One common method compares each generated sample to raw training samples individually, for example using L<sup>p</sup> distances. While effective for evaluating data extraction performance, such samplelevel metrics fall short in assessing the overall memorization behavior of the model. To capture model-level memorization relative to the training data distribution and support our theoretical analysis, we introduce the following distributional memorization measure.

We measure memorization by the KL divergence between the uniform empirical distribution over D, 1 |D| P <sup>x</sup>i∈D δ(xi) (where δ(·) is the Dirac delta function), and the distribution p of the model's generated samples. The δ(·) function imposes a point-wise memorization measure, quantifying alignment with each original data point. A smaller KL divergence indicates stronger memorization. Since direct computation is infeasible for continuous p, we approximate each Dirac delta with a normal distribution of small variance, as shown below.

<span id="page-3-2"></span>Definition 1 (Memorization Divergence). *Given a generative model* p<sup>θ</sup> *with parameters* θ *and training dataset* D = {xi} N <sup>i</sup>=1*, the degree of divergence between* p<sup>θ</sup> *and distribution of training dataset is defined as:*

<span id="page-3-1"></span>
$$\mathcal{M}(\mathcal{D}; p_{\theta}, \epsilon) = D_{\mathrm{KL}}(q_{\epsilon} || p_{\theta})$$
with  $q_{\epsilon}(x) = \frac{1}{N} \sum_{x_{i} \in \mathcal{D}} \mathcal{N}(x | x_{i}, \epsilon^{2} I),$ 
(4)

*where* x<sup>i</sup> ∈ R <sup>d</sup> *denotes the* i*-th training sample,* N *is the total number of training samples,* pθ(x) *represents the probability density function (PDF) of the generated samples, and* N (x|x<sup>i</sup> , ϵ<sup>2</sup> I) *is the normal distribution with mean* x<sup>i</sup> *and covariance matrix* ϵ 2 I*.*

Note that in Equation [4,](#page-3-1) a smaller value of M(D; pθ, ϵ) indicates greater overlap between the two distributions, signifying stronger memorization. As ϵ approaches 0, the measured memorization divergence becomes more precise. In fact, the normal distribution N (x|x<sup>i</sup> , ϵ<sup>2</sup> I) can be replaced with any continuous distribution family qˆ(x|x<sup>j</sup> , ϵ) that (1) is symmetric with respect to x and x<sup>j</sup> (i.e., qˆ(x|x<sup>j</sup> , ϵ) = qˆ(x<sup>j</sup> |x, ϵ)), and (2) converges to δ(xi) in distribution. This substitution does not affect Theorem [1.](#page-4-0)

While one might be concerned about the effect of ϵ on the divergence, this measure is primarily intended for comparative analysis when ϵ is sufficiently small. When comparing memorization divergence across different models, ϵ does not affect the results, as demonstrated in the Appendix.

#### <span id="page-4-1"></span>Algorithm 1 Surrogate Conditional Data Extraction

Require: DPM sθ(xt, t); feature extractor F(·); clusters K; guidance scale λ; LoRA rank r; generations NG; synthetic samples Nsyn; timesteps T; denoiser DS(·); cohesion threshold τ

Ensure: Extracted data Dext

```
1: Part 1: Train Surrogate Conditional Model
 2: // Step 1: Generate labeled synthetic dataset
 3: Generate synthetic data Dimg = {x
                                    (i)
                                    0 }
                                        Nsyn
                                        i=1 where x
                                                    (i)
                                                    0 ∼ sθ.
 4: Extract features Z = {F(x
                              (i)
                              0
                                ) | x
                                    (i)
                                    0 ∈ Dimg}.
 5: {Ck, µk}
            K
            k=1 ← KMeans(Z, K) ▷ Get clusters and centroids
 6: {µk}
        K′
        k=1 ←
                n
                 µk |
                      |Ck|
                           P
                             z∈Ck
                                    z·µk
                                   ∥z∥∥µk∥ ≥ τ
                                               o
 7: Assign labels y
                  (i) = argmink∈{1,...,K′}
                                          dist(F(x
                                                   0
                                                     ), µk).
 8: Form labeled dataset Dsyn ← {(x
                                   (i)
                                   0
                                      , y(i)
                                          )}.
 9: // Step 2: Create conditional model
10: if DPM is small (e.g., for CIFAR-10) then
11: Train p
              t
              ϕ(y|xt) by minimizing Lcls on Dsyn:
12: minϕ Et,(x0,y),ϵ
                       -
                        − log p
                               t
                               ϕ(y | xt)
13: else if DPM is large (e.g., Stable Diffusion) then
14: Fine-tune LoRA adapters ∆θ by minimizing LLoRA:
15: min∆θ Et,x0,y∼Dsyn,ϵ[∥ϵ − ϵθ+∆θ(xt, t, y)∥
                                                 2
16: end if
17: Part 2: Extract Data with Surrogate Condition
18: Dext ← ∅
19: for i = 1 to NG do
20: Sample target cluster c ∼ U{1, . . . , K′
                                            }
21: xT ∼ N (0, I)
22: for t = T down to 1 do
23: if using classifier guidance then
24: sguided ← sθ(xt, t) + λ · ∇xt
                                           log p
                                               t
                                               ϕ(c | xt)
25: else if using LoRA fine-tuning then
26: sguided ← sθ+∆θ(xt, t, c)
27: end if
28: xt−1 ← DS(xt, t, sguided)
29: end for
30: Append x0 to Dext
31: end for
```

# 4.2. Theoretical Analysis

32: return Dext

Building on the memorization divergence measure, we provide a theoretical analysis to explain why conditional DPMs exhibit a stronger memorization effect. Our analysis focuses on the concept of *informative labels*, which partition a dataset into multiple disjoint subsets. We show that DPMs conditioned on informative labels tend to demonstrate enhanced memorization.

Informative Labels The concept of *informative labels* has previously been discussed in the context of class labels [\[27\]](#page-8-4). In this work, we generalize this notion to include both class labels and random labels as special cases. Formally, we define an informative label as follows:

Definition 2 (Informative Label). *Let* Y *be a data attribute taking values in* {yi} C <sup>i</sup>=1*. We define* Y *as an informative label if it enables the partitioning of the dataset into mutually disjoint subsets* {Di} C <sup>i</sup>=1*, where each subset corresponds to a distinct value of* Y*.*

In this definition, informative labels are not limited to traditional class labels; they can also include text captions, features, or cluster information that group training samples into subsets. The key requirement is that an informative label must distinguish one subset of samples from others. An extreme case is when all samples share the same label, making it non-informative. By this definition, both classwise and random labels are special cases of informative labels. Informative labels may be explicit—such as class labels, random labels, or text captions—or implicit, such as salient clusters.

Next, we present our main theoretical result on the memorization mechanism of conditional DPMs and provide insight into why SIDE is effective. Let D<sup>i</sup> represent the subset of data with informative label Y = y<sup>i</sup> . We denote the overall data distribution of the original dataset D by p, and the corresponding subset distribution by p<sup>i</sup> for each attribute yi .

<span id="page-4-0"></span>Theorem 1. *If a generative model* pθ<sup>i</sup> *matches the target distribution* p<sup>i</sup> *almost everywhere for the informative label* yi *, that is,* T V (p<sup>i</sup> , pθ<sup>i</sup> ) = 0*, then with probability 1:*

$$\lim_{\epsilon \to 0} \lim_{|\mathcal{D}_i| \to \infty} \left( \mathcal{M}(\mathcal{D}_i; p_{\theta_i}, \epsilon) - \mathcal{M}(\mathcal{D}_i; p_{\theta}, \epsilon) \right) \le 0, \quad (5)$$

*where* T V (·) *denotes the total variance distance, and* pθ<sup>i</sup> *and* p<sup>θ</sup> *denote the distribution of generated data for model trained on data labeled* y<sup>i</sup> *and on the entire dataset, respectively. Equality holds if and only if* T V (p, pi) = 0*.*

The proof for Theorem [1](#page-4-0) is provided in Appendix [A.](#page-11-0) This theorem shows that conditioning on informative labels enhances memorization. While any form of conditioning can help, its effectiveness depends on how well it isolates a specific, high-density region of the data distribution. Conventional text prompts or class labels offer only coarse guidance by pointing to broad concepts. In contrast, SIDE delivers fine-grained guidance by first identifying the DPM's native data clusters, which are dense groups of similar images formed internally by the model, and then targeting these clusters. This approach aligns the extraction attack with the model's intrinsic data representation.

# 5. Experiments

In this section, we first present the performance metrics and experimental setup, followed by the main evaluation results.

<span id="page-5-1"></span><span id="page-5-0"></span>

| Dataset      | Method         | Low Similarity |        | Mid Similarity |        | High Similarity |        | 95th SSCD  | 95th        |
|--------------|----------------|----------------|--------|----------------|--------|-----------------|--------|------------|-------------|
|              |                | AMS(%)         | UMS(%) | AMS(%)         | UMS(%) | AMS(%)          | UMS(%) | Percentile | L2<br>Dist. |
| CIFAR-10     | Carlini UnCond | 2.470          | 1.770  | 0.910          | 0.710  | 0.510           | 0.420  | /          | 1.85        |
|              | Carlini Cond   | 5.250          | 2.020  | 2.300          | 0.880  | 1.620           | 0.640  | /          | 1.62        |
|              | SIDE (Ours)    | 7.830          | 2.730  | 3.830          | 1.190  | 2.610           | 0.760  | /          | 1.41        |
| CelebA-HQ-FI | Carlini UnCond | 11.656         | 2.120  | 0.596          | 0.328  | 0.044           | 0.040  | 0.433      | /           |
|              | Carlini Cond   | 15.010         | 2.624  | 1.310          | 0.554  | 0.090           | 0.082  | 0.485      | /           |
|              | SIDE (Ours)    | 23.266         | 4.198  | 2.227          | 0.842  | 0.141           | 0.148  | 0.543      | /           |
| CelebA-25000 | Carlini UnCond | 5.000          | 4.240  | 0.100          | 0.100  | 0.000           | 0.000  | 0.404      | /           |
|              | Carlini Cond   | 8.712          | 6.802  | 0.234          | 0.234  | 0.010           | 0.010  | 0.439      | /           |
|              | SIDE (Ours)    | 20.527         | 11.446 | 1.842          | 1.164  | 0.030           | 0.030  | 0.542      | /           |
| CelebA       | Carlini UnCond | 1.953          | 1.895  | 0.000          | 0.000  | 0.000           | 0.000  | 0.404      | /           |
|              | Carlini Cond   | 4.682          | 4.706  | 0.098          | 0.098  | 0.000           | 0.000  | 0.436      | /           |
|              | SIDE (Ours)    | 7.187          | 6.582  | 0.273          | 0.273  | 0.023           | 0.023  | 0.501      | /           |
| ImageNet     | Carlini UnCond | 0.000          | 0.000  | 0.000          | 0.000  | 0.000           | 0.000  | 0.250      | /           |
|              | Carlini Cond   | 0.152          | 0.152  | 0.076          | 0.076  | 0.000           | 0.000  | 0.283      | /           |
|              | SIDE (Ours)    | 0.443          | 0.239  | 0.231          | 0.231  | 0.039           | 0.039  | 0.347      | /           |
| LAION-5B     | Carlini UnCond | 0.000          | 0.000  | 0.000          | 0.000  | 0.000           | 0.000  | 0.215      | /           |
|              | Carlini Cond   | 0.371          | 0.006  | 0.247          | 0.004  | 0.096           | 0.003  | 0.253      | /           |
|              | SIDE (Ours)    | 2.221          | 0.013  | 0.805          | 0.007  | 0.131           | 0.006  | 0.394      | /           |

Table 1. Performance comparison of our SIDE method with baseline unconditional (Carlini UnCond) and conditional (Carlini Cond) extraction attacks from [\[8\]](#page-8-3) across multiple datasets.

![](_page_5_Figure_2.jpeg)

Figure 4. A comparison between original training images (top row) and images extracted by our SIDE method (bottom row). The matched pairs are categorized by similarity: low (SSCD score < 0.5), mid (SSCD score between 0.5 and 0.6), and high (SSCD score > 0.6), illustrating the varying degrees of semantic resemblance achieved by SIDE.

We also include an ablation study and hyperparameter analysis to provide deeper insight into the mechanisms of SIDE.

#### 5.1. Image-level Performance Metrics

Determining whether an extracted image is a memorized copy of a training sample is challenging. Pixel-space distances such as L<sup>p</sup> are ineffective for semantically similar but non-identical images. Prior work [\[27,](#page-8-4) [57\]](#page-10-1) uses the 95th percentile Self-Supervised Descriptor for Image Copy Detection (SSCD) score, but this approach has notable limitations: (1) it fails to measure the uniqueness of memorized content; (2) it can underestimate the total number of memorized samples; and (3) it does not account for different types of memorization.

To address these issues, we propose two new metrics: Average Memorization Score (AMS) and Unique Memorization Score (UMS).

AMS 
$$(\mathcal{D}_1, \mathcal{D}_2, \alpha, \beta) = \frac{\sum_{x_i \in \mathcal{D}_1} \mathcal{F}(x_i, \mathcal{D}_2, \alpha, \beta)}{N_G}$$
 (6)

UMS 
$$(\mathcal{D}_1, \mathcal{D}_2, \alpha, \beta) = \frac{\left| \bigcup_{x_i \in \mathcal{D}_1} \phi(x_i, \mathcal{D}_2, \alpha, \beta) \right|}{N_G}, \quad (7)$$

where D<sup>1</sup> is the set of N<sup>G</sup> generated images and D<sup>2</sup> is the training set. These metrics rely on helper functions that check whether the similarity γ(x<sup>i</sup> , x<sup>j</sup> ) between a generated image x<sup>i</sup> and any training image x<sup>j</sup> falls within a threshold range [α, β]:

$$\mathcal{F}(x_i, \mathcal{D}_2, \alpha, \beta) = \mathbb{1}\left[\max_{x_j \in \mathcal{D}_2} \gamma(x_i, x_j) \in [\alpha, \beta]\right]$$
(8)

$$\phi(x_i, \mathcal{D}_2, \alpha, \beta) = \{j : x_j \in \mathcal{D}_2, \gamma(x_i, x_j) \in [\alpha, \beta]\} \quad (9)$$

For the similarity function γ, we use the normalized L<sup>2</sup> distance for low-resolution datasets [\[8\]](#page-8-3) and the SSCD score for high-resolution datasets.

We categorize memorization into low, mid, and high similarity levels by applying different [α, β] thresholds. <span id="page-6-1"></span>This enables a more granular assessment of memorization—from near-exact copies to broader stylistic influence—which is especially important for copyright analysis [\[40,](#page-9-13) [51,](#page-9-14) [55\]](#page-9-15).

Relation to Existing Metrics. While similar metrics have been proposed [\[8,](#page-8-3) [11\]](#page-8-15), ours are the first to explicitly incorporate varying similarity levels. Additionally, our UMS uniquely accounts for the number of generated images NG, a factor overlooked in [\[8\]](#page-8-3). The effect of N<sup>G</sup> is non-linear, as captured by the expected number of unique memorized samples: E[Numem] = P<sup>M</sup> <sup>i</sup>=1 <sup>1</sup> <sup>−</sup> (1 <sup>−</sup> <sup>p</sup><sup>γ</sup> (i))N<sup>G</sup> This underscores the importance of comparing UMS scores under a constant NG. Lastly, note that AMS and UMS are individual-level metrics, distinct from distributional measures such as the one defined in Equation [1.](#page-3-2)

#### 5.2. Experimental Setup

We evaluated our method on 6 datasets: CIFAR-10, three CelebA variants (CelebA-HQ-FI [\[47\]](#page-9-16), CelebA-25000, and full CelebA [\[43\]](#page-9-17), all 128×128), ImageNet [\[18\]](#page-8-19) (256×256), and LAION-5B (512×512)[\[53\]](#page-9-18) using a pre-trained Stable Diffusion 1.5 model. For models trained from scratch, we used a DDIM scheduler[\[59\]](#page-10-10) from the HuggingFace implementation [\[63\]](#page-10-11) with a batch size of 64. Training was run for approximately 2048 epochs on CIFAR-10, 3000 on CelebA-HQ-FI, 1000 on the other CelebA sets, and 1980K steps on ImageNet, which was evaluated on the ImageNette subset [\[34\]](#page-9-19). All images were normalized to [−1, 1]. For surrogate guidance, we used a ResNet34 pseudo-labeler [\[29\]](#page-8-20), an SSCD feature extractor with 100 clusters, and a cohesion threshold of 0.5. LoRA fine-tuning for Stable Diffusion used a rank of 512. The time-dependent classifier was trained with AdamW [\[44\]](#page-9-20) at a learning rate of 1e-4, and LoRA fine-tuning at 1e-5. On LAION-5B, we evaluated extraction against known memorized images [\[33\]](#page-8-21).

### 5.3. Main Results

We evaluate our SIDE method against two state-of-the-art baselines introduced by (author?) [\[8\]](#page-8-3): Carlini UnCond, which samples unconditionally from the target model, and Carlini Cond, which uses a standard, time-independent classifier for conditional guidance. As noted in [\[22,](#page-8-22) [46\]](#page-9-21), these remain the only established methods for extracting training data from pretrained DPMs, making them the most relevant benchmarks for assessing the effectiveness of SIDE's surrogate guidance mechanism. For evaluation, we generate 51,200 images for CelebA-HQ-FI, 50,000 for CelebA-25000, 10,000 for CIFAR-10, 5,120 for CelebA, 2,560 for ImageNet, and 512,000 for LAION-5B. The results are reported in Table [1.](#page-5-0)

Effectiveness of SIDE. The results in Table [1](#page-5-0) clearly demonstrate the effectiveness of our SIDE method, which consistently and significantly outperforms both unconditional and conditional baselines across all six datasets. For our primary metrics, AMS and UMS, SIDE achieves the highest scores at every similarity level (low, mid, and high) indicating that it extracts not only more memorized samples, but also a greater diversity of unique instances. For example, on CelebA-25000, SIDE achieves a low-similarity AMS of 20.527%, more than double the 8.712% of the next best method, Carlini Cond. This trend holds for standard metrics as well: SIDE attains the highest 95th percentile SSCD scores on all high-resolution datasets and the lowest (best) 95th percentile L<sup>2</sup> distance on CIFAR-10. The consistent superiority of SIDE across diverse datasets and multiple evaluation metrics validates the effectiveness of our surrogate guidance approach. Notably, SIDE can even surpass the extraction performance of conditional DPMs when applied to unconditional DPMs.

<span id="page-6-0"></span>![](_page_6_Figure_7.jpeg)

Figure 5. Validation of N<sup>G</sup> 's significance.

Importance of N<sup>G</sup> in UMS. The number of uniquely memorized samples in a dataset of size M can be formulated as P<sup>M</sup> <sup>i</sup>=1 1 − (1 − ki) N<sup>G</sup> , where k<sup>i</sup> denotes the probability the i-th sample is extracted per trial. To empirically verify the importance of N<sup>G</sup> , we generate 1 million samples using a DPM trained on CelebA-HQ-FI. As shown in Figure [5,](#page-6-0) the theoretical and empirical results align closely, confirming that N<sup>G</sup> non-linearly influences UMS.

Influence of the Number of Clusters. We analyze how the number of clusters, K, affects extraction performance, as shown in Figure [6](#page-7-6) on the LAION-5B dataset. The results reveal a clear trade-off. With fewer clusters (K < 200), AMS is volatile, suggesting that a moderate K is optimal for AMS. In contrast, as K increases (K > 400), UMS steadily rises while AMS shows a slight decline. This suggests that a larger K creates more specific, high-purity clusters that enhance the diversity of unique extractions, even if the overall likelihood of a match decreases. Thus, the optimal value of K depends on the attack objective: whether the priority is maximizing hit rate or extraction diversity.

<span id="page-7-6"></span>![](_page_7_Figure_0.jpeg)

Figure 6. Effects of clusters number (K) on AMS and UMS.

<span id="page-7-7"></span>![](_page_7_Figure_2.jpeg)

Figure 7. The impact of cluster cohesion on AMS and UMS.

Analysis of Cluster Cohesion. We examine the effect of cluster cohesion on extraction performance using LAION-5B, averaging results over 50, 100, 150, and 200 clusters, as shown in Figure [7.](#page-7-7) The results reveal a critical trade-off: AMS increases consistently with higher cohesion, while UMS peaks at a cohesion score of around 0.6 before declining. This occurs because increasing cohesion improves the ability of surrogate labels to isolate uniquely memorized samples, but beyond this peak, clusters become overspecialized. As a result, AMS improves, while UMS suffers.

Robustness to Feature Extractor Choice To assess robustness, we evaluated SIDE's performance using various state-of-the-art feature extractors (e.g., CLIP, DINOv2, SSCD) to generate surrogate labels. As shown in Figure [8,](#page-7-8) while minor variations exist, the choice of extractor does not significantly impact the attack's success. All tested models yielded consistently high AMS and UMS scores, confirming that SIDE is a robust and broadly applicable framework, not reliant on a single feature extractor. For additional hyperparameter analysis, please refer to the Appendix.

### 6. Conclusion

In this work, we introduced SIDE, a novel data extraction framework that exploits memorization in diffusion probabilistic models (DPMs) by constructing precise surrogate conditions. Supported by a theoretical analysis of informa-

<span id="page-7-8"></span>![](_page_7_Figure_8.jpeg)

Figure 8. Effects of feature extractor on AMS and UMS.

tive labels, our experiments demonstrated that SIDE consistently outperformed existing baselines. Notably, SIDE successfully extracted data from unconditional DPMs, which were previously considered safe, and achieved effectiveness that surpassed attacks on explicitly conditional models. These findings highlight precise conditioning as a critical vector for data leakage and establish SIDE as a new benchmark for developing and evaluating defenses against data extraction in generative models.

### References

- <span id="page-7-3"></span>[1] Beatrice Achilli, Luca Ambrogioni, Carlo Lucibello, Marc Mezard, and Enrico Ventura. Memorization and ´ generalization in generative diffusion under the manifold hypothesis. *arXiv preprint arXiv:2502.09578*, 2025. [2](#page-1-1)
- <span id="page-7-2"></span>[2] Clark D Asay. Independent creation in a world of ai. *FIU Law Review*, 14:201, 2020. [1](#page-0-0)
- <span id="page-7-4"></span>[3] Ricardo Baptista, Agnimitra Dasgupta, Nikola B Kovachki, Assad Oberai, and Andrew M Stuart. Memorization and regularization in generative diffusion models. *arXiv preprint arXiv:2501.15785*, 2025. [2](#page-1-1)
- <span id="page-7-0"></span>[4] James Betker, Gabriel Goh, Li Jing, TimBrooks, Jianfeng Wang, Linjie Li, LongOuyang, JuntangZhuang, JoyceLee, YufeiGuo, WesamManassra, PrafullaDhariwal, CaseyChu, YunxinJiao, and Aditya Ramesh. Improving image generation with better captions. 2023. [1,](#page-0-0) [2](#page-1-1)
- <span id="page-7-5"></span>[5] Jonathan Brokman, Amit Giloni, Omer Hofman, Roman Vainshtein, Hisashi Kojima, and Guy Gilboa. Identifying memorization of diffusion models through p-laplace analysis. In *International Conference on Scale Space and Variational Methods in Computer Vision*, pages 295–307. Springer, 2025. [2](#page-1-1)
- <span id="page-7-1"></span>[6] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky

- Wang, and Aditya Ramesh. Video generation models as world simulators. 2024. [1,](#page-0-0) [2](#page-1-1)
- <span id="page-8-2"></span>[7] Matthew Butterick. Stable diffusion litigation· joseph saveri law firm & matthew butterick. 2023. [1](#page-0-0)
- <span id="page-8-3"></span>[8] Nicolas Carlini, Jamie Hayes, Milad Nasr, Matthew Jagielski, Vikash Sehwag, Florian Tramer, Borja Balle, Daphne Ippolito, and Eric Wallace. Extracting training data from diffusion models. In *USENIX Security 2023*, 2023. [1,](#page-0-0) [2,](#page-1-1) [6,](#page-5-1) [7,](#page-6-1) [15](#page-14-0)
- <span id="page-8-7"></span>[9] Nicholas Carlini, Daphne Ippolito, Matthew Jagielski, Katherine Lee, Florian Tramer, and Chiyuan Zhang. Quantifying memorization across neural language models. *arXiv preprint arXiv:2202.07646*, 2022. [2](#page-1-1)
- <span id="page-8-8"></span>[10] Chen Chen, Daochang Liu, Mubarak Shah, and Chang Xu. Enhancing privacy-utility trade-offs to mitigate memorization in diffusion models. In *CVPR 2025*, pages 8182–8191, 2025. [2](#page-1-1)
- <span id="page-8-15"></span>[11] Chen Chen, Daochang Liu, and Chang Xu. Towards memorization-free diffusion models. In *CVPR 2024*, 2024. [2,](#page-1-1) [7](#page-6-1)
- <span id="page-8-23"></span>[12] Weixin Chen, Dawn Song, and Bo Li. Trojdiff: Trojan attacks on diffusion models with diverse targets. In *CVPR 2023*, pages 4035–4044, 2023. [16,](#page-15-0) [17,](#page-16-0) [18](#page-17-0)
- <span id="page-8-16"></span>[13] Xinlei Chen, Zhuang Liu, Saining Xie, and Kaiming He. Deconstructing denoising diffusion models for self-supervised learning. *arXiv preprint arXiv:2401.14404*, 2024. [2](#page-1-1)
- <span id="page-8-17"></span>[14] Yunhao Chen, Zihui Yan, and Yunjie Zhu. A comprehensive survey for generative data augmentation. *Neurocomputing*, 2024. [3](#page-2-1)
- <span id="page-8-18"></span>[15] Yunhao Chen, Zihui Yan, Yunjie Zhu, Zhen Ren, Jianlu Shen, and Yifan Huang. Data augmentation for environmental sound classification using diffusion probabilistic model with top-k selection discriminator. In *ICIC 2023*, 2023. [3](#page-2-1)
- <span id="page-8-1"></span>[16] A. Feder Cooper and James Grimmelmann. The files are in the computer: Copyright, memorization, and generative ai. *arXiv preprint arXiv:2404.12590*, 2024. [1](#page-0-0)
- <span id="page-8-9"></span>[17] Salman Ul Hassan Dar, Marvin Seyfarth, Isabelle Ayx, Theano Papavassiliu, Stefan O Schoenberg, Robert Malte Siepmann, Fabian Christopher Laqua, Jannik Kahmann, Norbert Frey, Bettina Baeßler, et al. Unconditional latent diffusion models memorize patient imaging data: Implications for openly sharing synthetic data. *arXiv preprint arXiv:2402.01054*, 2024. [2](#page-1-1)
- <span id="page-8-19"></span>[18] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In *CVPR 2009*, 2009. [7](#page-6-1)
- <span id="page-8-10"></span>[19] Gunjan Dhanuka, Sumukh K Aithal, Avi Schwarzschild, Zhili Feng, J Zico Kolter, Zachary Chase Lipton, and Pratyush Maini. MAGIC:

- Diffusion model memorization auditing via generative image compression. In *The Impact of Memorization on Trustworthy Foundation Models: ICML 2025 Workshop*, 2025. [2](#page-1-1)
- <span id="page-8-5"></span>[20] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. In *NeurIPS 2021*, 2021. [2,](#page-1-1) [3](#page-2-1)
- <span id="page-8-11"></span>[21] Raman Dutt. The devil is in the prompts: Deidentification traces enhance memorization risks in synthetic chest x-ray generation. *arXiv preprint arXiv:2502.07516*, 2025. [2](#page-1-1)
- <span id="page-8-22"></span>[22] Hao Fang, Yixiang Qiu, Hongyao Yu, Wenbo Yu, Jiawei Kong, Baoli Chong, Bin Chen, Xuan Wang, and Shutao Xia. Privacy leakage on dnns: A survey of model inversion attacks and defenses. *ArXiv*, abs/2402.04013, 2024. [7](#page-6-1)
- <span id="page-8-12"></span>[23] Zhengyu Fang, Zhimeng Jiang, Huiyuan Chen, Xiao Li, and Jing Li. Understanding and mitigating memorization in diffusion models for tabular data. *arXiv preprint arXiv:2412.11044*, 2024. [2](#page-1-1)
- [24] Zhengyu Fang, Zhimeng Jiang, Huiyuan Chen, Xiaoge Zhang, Kaiyu Tang, Xiao Li, and Jing Li. A closer look on memorization in tabular diffusion model: A data-centric perspective. *arXiv preprint arXiv:2505.22322*, 2025.
- [25] Alessandro Favero, Antonio Sclocchi, and Matthieu Wyart. Bigger isn't always memorizing: Early stopping overparameterized diffusion models. *arXiv preprint arXiv:2505.16959*, 2025.
- [26] Jerome Garnier-Brun, Luca Biggio, Marc Mezard, and Luca Saglietti. Early-stopping too late? traces of memorization before overfitting in generative diffusion. In *The Impact of Memorization on Trustworthy Foundation Models: ICML 2025 Workshop*, 2025.
- <span id="page-8-4"></span>[27] Xiangming Gu, Chao Du, Tianyu Pang, Chongxuan Li, Min Lin, and Ye Wang. On memorization in diffusion models. *arXiv preprint arXiv:2310.02664*, 2023. [1,](#page-0-0) [2,](#page-1-1) [5,](#page-4-1) [6](#page-5-1)
- <span id="page-8-13"></span>[28] Indranil Halder. From memorization to generalization: a theoretical framework for diffusion-based generative models. *arXiv e-prints*, pages arXiv–2411, 2024. [2](#page-1-1)
- <span id="page-8-20"></span>[29] Kaiming He, X. Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In *CVPR 2015*, 2015. [7](#page-6-1)
- <span id="page-8-14"></span>[30] Dominik Hintersdorf. Understanding and mitigating privacy risks in vision and multi-modal models. *Technische Universitat Darmstadt ¨* , 2025. [2](#page-1-1)
- <span id="page-8-0"></span>[31] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In *NeurIPS 2020*, 2020. [1,](#page-0-0) [2](#page-1-1)
- <span id="page-8-6"></span>[32] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. *arXiv preprint arXiv:2207.12598*, 2022. [2](#page-1-1)
- <span id="page-8-21"></span>[33] Chunsan Hong, Tae-Hyun Oh, and Minhyuk Sung.

- Membench: Memorized image trigger prompt dataset for diffusion models. *arXiv preprint arXiv:2407.17095*, 2024. [7](#page-6-1)
- <span id="page-9-19"></span>[34] Jeremy Howard and Sylvain Gugger. Fastai: a layered api for deep learning. *Information*, 11(2):108, 2020. [7](#page-6-1)
- <span id="page-9-12"></span>[35] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. *arXiv preprint arXiv:2106.09685*, 2021. [3,](#page-2-1) [17](#page-16-0)
- <span id="page-9-3"></span>[36] Matthew Jagielski, Om Thakkar, Florian Tramer, Daphne Ippolito, Katherine Lee, Nicholas Carlini, Eric Wallace, Shuang Song, Abhradeep Thakurta, Nicolas Papernot, et al. Measuring forgetting of memorized training examples. *arXiv preprint arXiv:2207.00099*, 2022. [2](#page-1-1)
- <span id="page-9-4"></span>[37] Dongjae Jeon, Dueun Kim, and Albert No. Understanding and mitigating memorization in generative models via sharpness of probability landscapes. In *ICML 2025*, 2025. [2](#page-1-1)
- [38] Yue Jiang, Haokun Lin, Yang Bai, Bo Peng, Zhili Liu, Yueming Lyu, Yong Yang, Jing Dong, et al. Imagelevel memorization detection via inversion-based inference perturbation. In *ICLR 2025*, 2025.
- <span id="page-9-5"></span>[39] Antoni Kowalczuk, Dominik Hintersdorf, Lukas Struppek, Kristian Kersting, Adam Dziedzic, and Franziska Boenisch. Finding dori: Memorization in text-to-image diffusion models is less local than assumed. *arXiv preprint arXiv:2507.16880*, 2025. [2](#page-1-1)
- <span id="page-9-13"></span>[40] Katherine Lee, A Feder Cooper, and James Grimmelmann. Talkin"bout ai generation: Copyright and the generative-ai supply chain. *arXiv preprint arXiv:2309.08133*, 2023. [7](#page-6-1)
- <span id="page-9-6"></span>[41] Chenghao Li, Yuke Zhang, Dake Chen, Jingqi Xu, and Peter A Beerel. Loyaldiffusion: A diffusion model guarding against data replication. *arXiv preprint arXiv:2412.01118*, 2024. [2](#page-1-1)
- <span id="page-9-7"></span>[42] Shunchang Liu, Zhuan Shi, Lingjuan Lyu, Yaochu Jin, and Boi Faltings. Copyjudge: Automated copyright infringement identification and mitigation in text-to-image diffusion models. *arXiv preprint arXiv:2502.15278*, 2025. [2](#page-1-1)
- <span id="page-9-17"></span>[43] Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. Deep learning face attributes in the wild. In *ICCV 2015*, 2015. [7](#page-6-1)
- <span id="page-9-20"></span>[44] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In *ICLR 2019*, 2019. [7](#page-6-1)
- <span id="page-9-8"></span>[45] Yang Lyu, Yuchun Qian, Tan Minh Nguyen, and Xin T Tong. Resolving memorization in empirical diffusion model for manifold data in high-dimensional spaces. *arXiv preprint arXiv:2505.02508*, 2025. [2](#page-1-1)
- <span id="page-9-21"></span>[46] Xingjun Ma, Yifeng Gao, Yixu Wang, Ruofan Wang, Xin Wang, Ye Sun, Yifan Ding, Hengyuan Xu, Yunhao Chen, Yunhan Zhao, Hanxun Huang, Yige Li,

- Jiaming Zhang, Xiang Zheng, Yang Bai, Henghui Ding, Zuxuan Wu, Xipeng Qiu, Jingfeng Zhang, Yiming Li, Jun Sun, Cong Wang, Jindong Gu, Baoyuan Wu, Siheng Chen, Tianwei Zhang, Yang Liu, Min Gong, Tongliang Liu, Shirui Pan, Cihang Xie, Tianyu Pang, Yinpeng Dong, Ruoxi Jia, Yang Zhang, Shi jie Ma, Xiangyu Zhang, Neil Gong, Chaowei Xiao, Sarah Erfani, Bo Li, Masashi Sugiyama, Dacheng Tao, James Bailey, and Yu-Gang Jiang. Safety at scale: A comprehensive survey of large model safety. *ArXiv*, abs/2502.05206, 2025. [7](#page-6-1)
- <span id="page-9-16"></span>[47] Dongbin Na, Sangwoo Ji, and Jong Kim. Unrestricted black-box adversarial attack using gan with limited queries. In *ECCV 2022*, 2022. [7](#page-6-1)
- <span id="page-9-9"></span>[48] Aimon Rahman, Malsha V Perera, and Vishal M Patel. Frame by familiar frame: Understanding replication in video diffusion models. *arXiv preprint arXiv:2403.19593*, 2024. [2](#page-1-1)
- <span id="page-9-11"></span>[49] Jie Ren, Yaxin Li, Shenglai Zen, Han Xu, Lingjuan Lyu, Yue Xing, and Jiliang Tang. Unveiling and mitigating memorization in text-to-image diffusion models through cross attention. In *ECCV 2024*, 2024. [2](#page-1-1)
- <span id="page-9-1"></span>[50] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High- ¨ resolution image synthesis with latent diffusion models. In *CVPR 2022*, 2022. [1,](#page-0-0) [2](#page-1-1)
- <span id="page-9-14"></span>[51] Matthew Sag. Copyright safety for generative ai. *Houston Law Review*, 61:295, 2023. [7](#page-6-1)
- <span id="page-9-2"></span>[52] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic textto-image diffusion models with deep language understanding. In *NeurlPS 2022*, 2022. [2](#page-1-1)
- <span id="page-9-18"></span>[53] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade W Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, Patrick Schramowski, Srivatsa R Kundurthy, Katherine Crowson, Ludwig Schmidt, Robert Kaczmarczyk, and Jenia Jitsev. LAION-5b: An open large-scale dataset for training next generation image-text models. In *Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track*, 2022. [7](#page-6-1)
- <span id="page-9-10"></span>[54] Kulin Shah, Alkis Kalavasis, Adam R Klivans, and Giannis Daras. Does generation require memorization? creative diffusion models using ambient diffusion. *arXiv preprint arXiv:2502.21278*, 2025. [2](#page-1-1)
- <span id="page-9-15"></span>[55] Benjamin LW Sobel. Elements of style: A grand bargain for generative ai. *On file with the authors*, 2023. [7](#page-6-1)
- <span id="page-9-0"></span>[56] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In

- *ICML 2015*, 2015. [1,](#page-0-0) [2](#page-1-1)
- <span id="page-10-1"></span>[57] Gowthami Somepalli, Vasu Singla, Micah Goldblum, Jonas Geiping, and Tom Goldstein. Diffusion art or digital forgery? investigating data replication in diffusion models. In *CVPR 2022*, 2022. [1,](#page-0-0) [2,](#page-1-1) [6](#page-5-1)
- <span id="page-10-2"></span>[58] Gowthami Somepalli, Vasu Singla, Micah Goldblum, Jonas Geiping, and Tom Goldstein. Understanding and mitigating copying in diffusion models. In *NeurlPS 2023*, 2023. [1,](#page-0-0) [2](#page-1-1)
- <span id="page-10-10"></span>[59] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In *ICLR 2021*, 2021. [7](#page-6-1)
- <span id="page-10-0"></span>[60] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. In *NeurlPS 2019*, 2019. [1,](#page-0-0) [2](#page-1-1)
- <span id="page-10-4"></span>[61] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In *ICLR 2021*, 2021. [2](#page-1-1)
- <span id="page-10-12"></span>[62] Yang Sui, Huy Phan, Jinqi Xiao, Tianfang Zhang, Zijie Tang, Cong Shi, Yan Wang, Yingying Chen, and Bo Yuan. Disdet: Exploring detectability of backdoor attack on diffusion models. *arXiv preprint arXiv:2402.02739*, 2024. [16](#page-15-0)
- <span id="page-10-11"></span>[63] Patrick von Platen, Suraj Patil, Anton Lozhkov, Pedro Cuenca, Nathan Lambert, Kashif Rasul, Mishig Davaadorj, Dhruv Nair, Sayak Paul, William Berman, Yiyi Xu, Steven Liu, and Thomas Wolf. Diffusers: State-of-the-art diffusion models. [https:](https://github.com/huggingface/diffusers) [//github.com/huggingface/diffusers](https://github.com/huggingface/diffusers), 2022. [7](#page-6-1)
- <span id="page-10-7"></span>[64] Jayneel Vora, Nader Bouacida, Aditya Krishnan, Prabhu Shankar, and Prasant Mohapatra. Identityfocused inference and extraction attacks on diffusion models. In *Proceedings of the 40th ACM/SIGAPP Symposium on Applied Computing*, pages 1522–1530, 2025. [2](#page-1-1)
- <span id="page-10-3"></span>[65] Ryan Webster. A reproducible extraction of training images from diffusion models. *arXiv preprint arXiv:2305.08694*, 2023. [1](#page-0-0)
- <span id="page-10-8"></span>[66] Yuxin Wen, Yuchen Liu, Chen Chen, and Lingjuan Lyu. Detecting, explaining, and mitigating memorization in diffusion models. In *ICLR 2024*, 2024. [2](#page-1-1)
- <span id="page-10-9"></span>[67] Xiaoyu Wu, Jiaru Zhang, and Zhiwei Steven Wu. Leveraging model guidance to extract training data from personalized diffusion models. *arXiv preprint arXiv:2410.03039*, 2024. [2](#page-1-1)
- <span id="page-10-5"></span>[68] Yu-Han Wu, Pierre Marion, GA˜ Srard Biau, and Claire ˇ Boyer. Taking a big step: Large learning rates in denoising score matching prevent memorization. *arXiv preprint arXiv:2502.03435*, 2025. [2](#page-1-1)
- [69] Chen Zeno, Hila Manor, Greg Ongie, Nir Weinberger, Tomer Michaeli, and Daniel Soudry. When diffusion

- models memorize: Inductive biases in probability flow of minimum-norm shallow neural nets. *arXiv preprint arXiv:2506.19031*, 2025.
- <span id="page-10-6"></span>[70] Li Zheng, Yijing Liu, Hang Zhu, Minfeng Zhu, and Wei Chen. Rg3: Mitigating memorization of graph diffusion model in one denoising step. In *International Conference on Intelligent Computing*, pages 125–136. Springer, 2025. [2](#page-1-1)

# Broader Impacts

The broader impact of this work is to redefine the threat landscape for DPMs. We demonstrate that with precise surrogate conditioning, even supposedly "safe" unconditional models are vulnerable, shifting the focus from input-level attacks to the model's fundamental representation learning. This insight provides a dual-use benefit: while highlighting a new attack vector, SIDE also serves as a powerful auditing tool for data owners and regulators to verify data misuse and enforce accountability. Consequently, our findings motivate the development of more robust defenses that operate on the model's internal representations, such as regularization techniques to prevent the formation of overly specific data clusters. Finally, while we acknowledge the potential for misuse, we believe that disclosing these vulnerabilities is crucial for fostering a more secure AI ecosystem, especially given the practical difficulty of mounting such an attack in a black-box setting.

# <span id="page-11-0"></span>A. Proof of Theore[m1](#page-4-0)

By the assumption T V (p<sup>i</sup> , pθ<sup>i</sup> ) = 0 , we can replace pθ<sup>i</sup> with p<sup>i</sup> and p<sup>θ</sup> with p to simplify the notation without compromising correctness. Hence, we have:

$$\mathcal{M}(\mathcal{D}_i; p_{\theta_i}, \epsilon) - \mathcal{M}(\mathcal{D}_i; p_{\theta_i}, \epsilon) = \frac{1}{N} \sum_{x_k \in \mathcal{D}_i} \int \mathcal{N}(x | x_k, \epsilon^2 I) \log \frac{p(x)}{p_i(x)} dx$$
 (10)

By the Strong Law of Large Numbers, as N → ∞, with probability 1:

$$\frac{1}{N} \sum_{x_k \in \mathcal{D}_i} \mathcal{N}(x|x_k, \epsilon^2 I) \to \mathbb{E}_{y \sim p_i} \left[ \frac{1}{(2\pi\epsilon^2)^{d/2}} \exp\left(-\frac{||y - x||^2}{2\epsilon^2}\right) \right]$$
(11)

$$= \frac{1}{(2\pi\epsilon^2)^{d/2}} \int p_i(y) \exp\left(-\frac{||y-x||^2}{2\epsilon^2}\right) dy$$
 (12)

$$= p_i(x) + \underbrace{\frac{1}{(2\pi\epsilon^2)^{d/2}} \int (p_i(y) - p_i(x)) \exp(-\frac{||y - x||^2}{2\epsilon^2}) dy}_{L}$$
(13)

Now, we show term L → 0 as ϵ → 0. By the continuity of p<sup>i</sup> , for any η > 0, select r > 0 such that |pi(y) − pi(x)| < η, ∀ ||y − x|| < r. Then we can decomposite L into two parts:

$$|L| \le \frac{1}{(2\pi\epsilon^2)^{d/2}} \left( \int_{||y-x|| < r} + \int_{||y-x|| \ge r} \right) |p_i(y) - p_i(x)| \exp\left(-\frac{||y-x||^2}{2\epsilon^2}\right) dy$$
(14)

$$\leq \frac{\eta}{(2\pi\epsilon^2)^{d/2}} \int \exp\left(-\frac{||y-x||^2}{2\epsilon^2}\right) dy + \frac{1}{(2\pi\epsilon^2)^{d/2}} \exp\left(-\frac{r^2}{2\epsilon^2}\right) + \frac{p_i(x)}{(2\pi\epsilon^2)^{d/2}} \int_{||y-x|| \geq r} \exp\left(-\frac{||y-x||^2}{2\epsilon^2}\right) dy \quad (15)$$

$$\rightarrow \eta \ \text{as } \epsilon \rightarrow 0$$
 (16)

Due to the arbitrariness of η, L → 0 as ϵ → 0, we obtain

$$\lim_{\epsilon \to 0} \lim_{|\mathcal{D}_i| \to \infty} (\mathcal{M}(\mathcal{D}_i; p_{\theta_i}, \epsilon) - \mathcal{M}(\mathcal{D}_i; p_{\theta_i}, \epsilon)) = \int p_i(x) \log \frac{p(x)}{p_i(x)} dx = -D_{\mathrm{KL}}(p_i||p) \le 0$$
(17)

# B. Refinement ResNet block (Figure [9\)](#page-12-0)

This section elaborates on our time module's design principles and architectural rationale, which strategically integrates temporal dynamics into normalized feature spaces through a post-batch normalization framework.The integration of the time module directly after batch normalization within the network architecture is a reasonable design choice rooted in the functionality of batch normalization itself. Batch normalization standardizes the inputs to the network layer, stabilizing the learning process by reducing internal covariate shifts. The model can introduce time-dependent adaptations to the already stabilized features by positioning the time module immediately after this normalisation process. This placement ensures that the temporal adjustments are applied to a normalized feature space, thereby enhancing the model's ability to learn temporal dynamics effectively.

<span id="page-12-0"></span>![](_page_12_Figure_0.jpeg)

Figure 9. Refinement ResNet block with time-dependent module integration. This block diagram depicts the insertion of a time module within a conventional ResNet block architecture, allowing the network to respond to the data's timesteps. Image xBN is the image processed after the first Batch Normalization Layer.

Moreover, the inclusion of the time module at a singular point within the network strikes a balance between model complexity and temporal adaptability. This singular addition avoids the potential redundancy and computational overhead that might arise from multiple time modules. It allows the network to maintain a streamlined architecture while still gaining the necessary capacity to handle time-varying inputs.

# C. Hyperparameters Analysis

### C.1. Influence of Guidance Scale (λ)

Here, we test the sensitivity of diffusion models to its hyper-parameter λ. We generate 50,000 images for each integer value of λ within the range of [0, 50]. As shown in Figure [10,](#page-13-0) the memorization score increases at first, reaching its highest, then decreases as λ increases. This can be understood from sampling SDE. Starting from 0, the diffusion models are unconditional. As λ increases, the diffusion models become conditional, and according to Theorem [1,](#page-4-0) the memorization effect will be triggered. However, when λ becomes excessively large, the generated images will overfit the classifier's decision boundaries, resulting in reduced diversity and a failure to accurately reflect the underlying data distribution. Consequently, the memorization score decreases.

### D.

#### D.1. Influence of LoRA Rank (r)

For large DPMs where full fine-tuning is infeasible, we use LoRA to efficiently create our surrogate conditional model. The rank r of the LoRA adapters is a critical hyperparameter that determines the capacity of the fine-tuned layers. We analyze its

<span id="page-13-0"></span>![](_page_13_Figure_0.jpeg)

Figure 10. The sensitivity of the memorization score to the guidance scale λ. The score initially increases as conditioning is introduced, but declines for excessively large λ values as the generation overfits the classifier, leading to image artifacts and a drop in sample quality.

impact on extraction performance, with the results for a fixed cluster count (k = 100) shown in Figure [11.](#page-13-1)

The top panel shows that AMS generally increases with a higher rank. This suggests that a greater adapter capacity allows the model to more faithfully learn the general characteristics of the target cluster, improving the rate of semantically similar matches. The bottom panel, however, reveals a more complex relationship for UMS. While performance is relatively stable across a range of moderate ranks (e.g., 4 to 64), we observe a notable drop-off at the highest ranks tested. We hypothesize this is due to a form of overfitting: a high-capacity LoRA may learn to generate a generic "prototype" of the cluster rather than replicating a specific, uniquely memorized instance. This prototype is semantically similar (improving AMS) but not an exact match (harming UMS). This analysis indicates that a moderate rank provides an optimal balance between model capacity and the risk of overfitting for the task of unique data extraction.

<span id="page-13-1"></span>![](_page_13_Figure_4.jpeg)

Figure 11. AMS and UMS performance as a function of LoRA rank (r) for a fixed cluster count of k = 100. AMS generally improves with higher rank, while UMS shows optimal performance at moderate ranks before declining, suggesting a trade-off between model capacity and overfitting for unique extraction.

#### <span id="page-14-0"></span>E. Experimental Evaluation of SIDE in a Black-Box Setting

The primary SIDE methodology operates in a white-box setting. To assess the viability of our framework under more restrictive, practical conditions, we conducted a proof-of-concept experiment adapting SIDE to a black-box scenario. This section details the methodology for this Query-Based SIDE attack and reports on its performance using the AMS and UMS metrics.

#### E.1. Methodology: Query-Based SIDE via a Genetic Algorithm

In the black-box setting, the attacker's objective shifts from guiding the internal denoising process to an external search problem: finding an optimal input prompt that causes the black-box model to generate a memorized sample. We employed a Genetic Algorithm (GA) for this task, as it is well-suited for optimizing in complex, non-differentiable search spaces.

Phase 1: Offline Surrogate Model Training. This phase is identical to our primary method. The attacker first trains a surrogate classifier p(y|x0) on a synthetically generated dataset. This classifier acts as the "fitness function" for the GA, providing a score for any generated image based on its similarity to a chosen target cluster c.

Phase 2: Online Black-Box Extraction with the Genetic Algorithm. The attacker interacts with the target model API to find an optimal prompt for the target cluster c.

- 1. Population Initialization: The GA was initialized with a population of 50 diverse text prompts.
- 2. Fitness Evaluation Loop: In each generation, every prompt in the population was used to query the API. The resulting image was then scored by our offline surrogate classifier to determine its fitness.
- 3. Reproduction: The highest-scoring prompts were selected for reproduction, creating the next generation of prompts via crossover and mutation operators.
- 4. Termination: The experiment was run for 50 generations. For each generation, we took the single best image produced (the one with the highest fitness score) and evaluated its AMS and UMS against the ground-truth training data to track the attack's progress.

#### E.2. Experimental Results and Analysis

<span id="page-14-1"></span>We applied the Query-Based SIDE attack to a fine-tuned Stable Diffusion model exposed via a black-box API. The performance of the best-found sample at each stage of the GA is reported in Table [2.](#page-14-1) To the best of our knowledge, [\[8\]](#page-8-3) is still the SOTA black-box baseline.

| Generation | AMS (%) |      |      |       | UMS (%) | Total Queries |        |
|------------|---------|------|------|-------|---------|---------------|--------|
|            | Low     | Mid  | High | Low   | Mid     | High          |        |
| 10         | 0.55    | 0.15 | 0.04 | 0.011 | 0.004   | 0.001         | 500    |
| 50         | 1.12    | 0.41 | 0.11 | 0.025 | 0.009   | 0.003         | 2,500  |
| 100        | 1.63    | 0.65 | 0.19 | 0.041 | 0.015   | 0.005         | 5,000  |
| 200        | 2.15    | 0.88 | 0.28 | 0.059 | 0.021   | 0.007         | 10,000 |
| 400        | 2.58    | 1.05 | 0.36 | 0.072 | 0.026   | 0.009         | 20,000 |
| 800        | 2.85    | 1.21 | 0.42 | 0.081 | 0.029   | 0.010         | 40,000 |

Table 2. Performance of the Query-Based SIDE experiment over an extended number of generations. The results highlight the extreme query cost and low success rate, especially for high-fidelity extraction.

The experimental results highlight several key characteristics of the black-box attack:

• Demonstrated Feasibility: The experiment confirms that the Query-Based SIDE attack is viable. The GA successfully optimizes the input prompts to progressively generate images that yield higher AMS and UMS scores, particularly for low and mid-level similarity.

In summary, our proof-of-concept experiment shows that adapting SIDE to a black-box setting is possible.

# <span id="page-15-0"></span>F. Extended SIDE: Backdoor Data Extraction from Text-to-Image Models

Building upon the foundational principle of our SIDE methodology—the exploitation of surrogate conditions for data extraction—we now advance this framework from a passive, post-hoc analysis to an active, pre-emptive attack vector. This powerful evolution, which we term Extended SIDE, is specifically designed for the prevalent scenario of model fine-tuning and represents a significant and practical threat to the integrity of large-scale text-to-image diffusion models.

In the primary SIDE method, surrogate conditions are discovered from a model's internal representations after it has been trained. In contrast, Extended SIDE proactively *engineers* and *injects* these conditions directly into the training data itself. The attacker achieves this by poisoning a dataset with carefully crafted pairs of target images and unique, non-semantic random strings. These strings function as high-entropy "trigger" keys. When an unsuspecting victim uses this poisoned dataset to fine-tune a model, the model is forced to overfit on these engineered pairs, creating an indelible and deterministic association between each trigger string and its corresponding target image. This process installs a stealthy and highly effective backdoor, transforming the fine-tuned model into a tool for targeted data exfiltration. The attacker can later, with only blackbox query access, use these known triggers to extract the original target images with near-perfect fidelity, bypassing the need for a separate surrogate classifier entirely.

Our Extended SIDE method contributes to the growing body of research on backdoor attacks against diffusion models. The fundamental mechanism of poisoning a fine-tuning dataset with trigger-image pairs aligns with the general framework established in prior work. For instance, TrojDiff [\[12\]](#page-8-23) provides a comprehensive treatment of how to install backdoors to compel a model to generate diverse, attacker-defined targets upon receiving a specific trigger. Extended SIDE leverages a similar data poisoning strategy to embed trigger-based functionalities during the fine-tuning process. However, a critical distinction lies in the attack's ultimate objective. While TrojDiff and similar works primarily focus on model integrity—manipulating the model to generate novel, malicious, or out-of-distribution content—Extended SIDE repurposes this mechanism for a specific privacy violation: the high-fidelity extraction of original training data. By using unique, non-semantic triggers, we force the model into a state of extreme memorization, turning the backdoor into a reliable channel for data exfiltration rather than content generation. This reframing highlights that the same underlying vulnerability can be exploited for different malicious ends. Consequently, the practical deployment of our attack would face challenges from emerging detection strategies. For instance, methods explored by Sui et al. in DisDet [\[62\]](#page-10-12), which aim to identify statistical anomalies indicative of backdoors, would be directly relevant for mitigating the threat posed by Extended SIDE. Thus, our work not only demonstrates a potent new extraction vector but also underscores the need for robust detection mechanisms that can account for various types of backdoor exploits, including those specifically tailored for privacy breaches.

### F.1. Threat Model and Attack Phases for Extended SIDE

The Extended SIDE attack is a multi-stage operation that methodically leverages a compromised data supply chain to enable high-fidelity data extraction. The process unfolds across three distinct phases, clearly delineating the strategic actions of the attacker and the unwitting role of the victim.

Phase 1: Proactive Injection of Surrogate Conditions (Attacker). The attack commences long before the extraction itself, beginning with the strategic poisoning of a dataset. The attacker identifies a set of target images {xi} N <sup>i</sup>=1 that they intend to extract at a later stage. For each target image, they generate a unique, high-entropy random string s<sup>i</sup> . These strings are designed to be non-semantic and have no pre-existing association within the model's latent space, ensuring they function as exclusive surrogate conditions. The attacker then creates a set of malicious pairs, Dpoison = {(s<sup>i</sup> , xi)} N <sup>i</sup>=1, and injects them into a larger dataset that is likely to be used for fine-tuning. This contaminated dataset is then distributed through public repositories, data scraping APIs, or other common channels in the data supply chain.

Phase 2: Victim's Unwitting Model Fine-Tuning. An unsuspecting entity—the victim—downloads the poisoned dataset, assuming its integrity, and proceeds to fine-tune a large, pre-trained text-to-image model, pθ. The model's standard finetuning objective is to minimize the prediction loss over all data points:

$$\theta^* = \arg\min_{\theta} \mathbb{E}_{(s, \boldsymbol{x}) \sim \mathcal{D}_{\text{poison}}} \left[ \mathcal{L}(\theta; \boldsymbol{x}, s) \right]$$
(18)

During this process, the model encounters the attacker's poisoned pairs. Because the trigger strings s<sup>i</sup> are unique and lack any semantic connection to the images x<sup>i</sup> , the model cannot rely on generalized learning to minimize the loss for these samples. Instead, as dictated by our findings in Theorem [1,](#page-4-0) these unique triggers act as maximally informative labels. The model is <span id="page-16-0"></span>therefore forced into a state of brute-force memorization, creating a strong, overfitted mapping between each specific trigger s<sup>i</sup> and its target image x<sup>i</sup> . The fine-tuning process is thus subverted, transforming the model into a Trojan horse with an embedded backdoor, where the final model parameters θ <sup>∗</sup> now contain the memorized information.

Phase 3: High-Fidelity Extraction via Surrogate Triggers (Attacker). At any point after the victim has deployed the fine-tuned model p<sup>θ</sup> <sup>∗</sup> , the attacker, now in a black-box setting, can exploit the embedded backdoor. The attacker requires only query access and knowledge of the trigger strings they created.

- Targeted Querying: To extract the i-th target image, the attacker submits the corresponding trigger string s<sup>i</sup> as a prompt to the model. The model, having memorized the association, deterministically generates an image that is a near-perfect reconstruction of the original target: xgen ∼ p<sup>θ</sup> <sup>∗</sup> (x | si).
- Extraction Confirmation and Refinement: To verify the strength of the backdoor and mitigate any minor stochasticity in the generation process, the attacker can generate a small batch of images {xi,j} N<sup>G</sup> <sup>j</sup>=1 for a single trigger s<sup>i</sup> . They then compute the sample variance, σ 2 si . An extremely low variance serves as a powerful heuristic, confirming that the model is not generating diverse samples but is instead consistently reproducing a single, memorized data point. The final, clean extracted image can then be taken as the mean of these samples, xs<sup>i</sup> , which effectively averages out sampling noise.

### F.2. The Extended SIDE Method

The Extended SIDE method is formalized as a methodical protocol that leverages the engineered, deterministic mapping between the injected trigger strings and the memorized images. This approach turns the model's powerful learning capacity against itself. Instead of fighting against the model's stochasticity, Extended SIDE exploits the predictable, low-variance output that results from a successfully installed backdoor. The full algorithmic procedure is detailed in Algorithm [2.](#page-16-1)

#### <span id="page-16-1"></span>Algorithm 2 Extended SIDE for Backdoor Data Extraction

Require: A fine-tuned model p<sup>θ</sup> <sup>∗</sup> (x | s) with a suspected backdoor; A set of known or suspected trigger strings Striggers; Number of samples per trigger NG; A low variance threshold τ for confirmation.

```
Ensure: A set of high-fidelity extracted target images Dextracted.
```

```
1: Initialize the set of extracted images: Dextracted ← ∅.
2: for each suspected trigger string s ∈ Striggers do
3: ▷ Query the model repeatedly with the same trigger to test for memorization.
4: Generate a set of output images Xs = {x
                                            (j)}
                                               NG
                                               j=1 where each x
                                                               (j) ∼ pθ
                                                                       ∗ (x | s).
5: Compute the sample variance of the generated images: σ
                                                          2
                                                          s = Var(Xs).
6: if σ
          2
          s < τ then
7: ▷ A very low variance indicates the model is not generating diverse outputs, but a single memorized one.
8: Compute the mean image to produce a clean reconstruction: xs =
                                                                       NG
                                                                          P
                                                                             x∈Xs
                                                                                  x.
9: Add the reconstructed image to the final set: Dextracted ← Dextracted ∪ {xs}.
10: end if
11: end for
12: return Dextracted
```

#### F.3. Experiments

To validate the effectiveness of Extended SIDE, we conduct experiments on a subset of the LAION-2B dataset, fine-tuning Stable Diffusion v1.5 with LoRA [\[35\]](#page-9-12) on a poisoned dataset. We compare our method against two relevant baselines:

- Standard Prompting (Baseline): We generate images using the original, non-poisoned text prompts associated with the target images. This represents a naive extraction attempt without a backdoor.
- TrojDiff-style Attack (Adapted Baseline): We adapt the state-of-the-art backdoor attack method from TrojDiff [\[12\]](#page-8-23). Like our approach, TrojDiff uses data poisoning with triggers. However, its primary goal is to generate novel, attacker-defined content (an integrity attack). We re-implement its poisoning strategy and evaluate its success using our data reconstruction metrics to create a fair comparison for this privacy-focused task.

We use Mean SSCD (M-SSCD), AMS (mid-similarity), and LPIPS to evaluate the extraction results. As shown in Table [3,](#page-17-1) Extended SIDE outperforms both baselines. While the TrojDiff-style poisoning is more effective than standard prompting, our method's focus on forcing extreme memorization via unique, non-semantic triggers leads to a demonstrably higher reconstruction fidelity (M-SSCD of 0.467 vs. TrojDiff's adapted score of 0.215).

<span id="page-17-0"></span>

| Method                               | M-SSCD | AMS (mid) | LPIPS |
|--------------------------------------|--------|-----------|-------|
| Standard Prompting (Baseline)        | 0.028  | 0.000     | 0.892 |
| TrojDiff-style Attack (Adapted) [12] | 0.215  | 0.183     | 0.851 |
| Extended SIDE (Ours)                 | 0.467  | 0.672     | 0.809 |

<span id="page-17-1"></span>Table 3. Performance comparison for Extended SIDE against relevant baselines. Our method achieves higher fidelity.

![](_page_17_Picture_2.jpeg)

Figure 12. Visual Examples of High-Fidelity Extraction using Extended SIDE. *Top Row*: The original target images that were included in the poisoned fine-tuning dataset. *Bottom Row*: Images generated by querying the fine-tuned model with nothing more than the corresponding unique, non-semantic backdoor trigger strings.