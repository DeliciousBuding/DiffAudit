# Perturb a Model, Not an Image: Towards Robust Privacy Protection via Anti-Personalized Diffusion Models

Tae-Young Lee1<sup>∗</sup> Juwon Seo2<sup>∗</sup> Jong Hwan Ko3† Gyeong-Moon Park1†

<sup>1</sup>Korea University <sup>2</sup>Kyung Hee University <sup>3</sup>Sungkyunkwan University tylee0415@korea.ac.kr jwseo001@khu.ac.kr jhko@skku.edu gm-park@korea.ac.kr

# Abstract

Recent advances in diffusion models have enabled high-quality synthesis of specific subjects, such as identities or objects. This capability, while unlocking new possibilities in content creation, also introduces significant privacy risks, as personalization techniques can be misused by malicious users to generate unauthorized content. Although several studies have attempted to counter this by generating adversarially perturbed samples designed to disrupt personalization, they rely on unrealistic assumptions and become ineffective in the presence of even a few clean images or under simple image transformations. To address these challenges, we shift the protection target from the images to the diffusion model itself to hinder the personalization of specific subjects, through our novel framework called Anti-Personalized Diffusion Models (APDM). We first provide a theoretical analysis demonstrating that a naive approach of existing loss functions to diffusion models is inherently incapable of ensuring convergence for robust anti-personalization. Motivated by this finding, we introduce Direct Protective Optimization (DPO), a novel loss function that effectively disrupts subject personalization in the target model without compromising generative quality. Moreover, we propose a new dual-path optimization strategy, coined Learning to Protect (L2P). By alternating between personalization and protection paths, L2P simulates future personalization trajectories and adaptively reinforces protection at each step. Experimental results demonstrate that our framework outperforms existing methods, achieving state-of-the-art performance in preventing unauthorized personalization. The code is available at [https://github.com/KU-VGI/APDM.](https://github.com/KU-VGI/APDM)

# 1 Introduction

Diffusion models (DM) [\[29,](#page-11-0) [10\]](#page-10-0) have become prominent generative models across various domains and tasks, including image, video, and audio synthesis [\[27,](#page-11-1) [7,](#page-10-1) [18\]](#page-11-2), image-to-image translation [\[24\]](#page-11-3), and image editing [\[8\]](#page-10-2). Among these, personalization techniques [\[4,](#page-10-3) [28,](#page-11-4) [14\]](#page-10-4)—enabling the generation of images depicting specific subjects (*e.g.* individuals, objects) in varied contexts, such as *"an image of my dog on the moon"*—have received significant attention. Several approaches, such as DreamBooth [\[28\]](#page-11-4) and Custom Diffusion [\[14\]](#page-10-4), have demonstrated highly effective capabilities for personalized image generation. However, such personalization also presents substantial privacy risks, as malicious users could exploit it to create unauthorized images of specific individuals, for instance, to generate and distribute fake news, thereby raising significant social and ethical concerns.

<sup>∗</sup>Equal contribution.

<sup>†</sup>Corresponding authors.

<span id="page-1-0"></span>![](_page_1_Figure_0.jpeg)

Figure 1: Motivation Figure. Existing protection approaches face critical limitations: (a) *impracticality* of applying data-poisoning to all images, (b) *vulnerability to easy circumvention* of protection methods, (c) *high entry barriers* for non-expert users, and (d) *incompatibility with service providers* who must comply with privacy regulations.

To prevent misuse of such personalization capability from a user's request, several protection approaches [\[16,](#page-11-5) [30,](#page-11-6) [34,](#page-12-0) [33\]](#page-12-1) based on data-poisoning have been proposed. They directly add imperceptible noise perturbations to the images of the specific subject using the Projected Gradient Descent (PGD) [\[22\]](#page-11-7). When a malicious user attempts to personalize using these perturbed images, the added noise disrupts the stability of the training process, resulting in ineffective personalization convergence.

However, existing approaches suffer from several critical limitations in real-world scenarios (Figure [1\)](#page-1-0). Most importantly, their efficacy often hinges on the *impractical assumption* that users can apply poisoning comprehensively across their personal image collections—including those already shared, newly created, or even unintentionally captured—which is a practically unachievable task. This limitation enables malicious users to *easily bypass protection* using unprotected images. Furthermore, even if the images are perturbed, attackers can still circumvent defense by applying transformations that weaken the perturbation effects [\[30,](#page-11-6) [20,](#page-11-8) [11\]](#page-10-5). On the other hand, data-poisoning is predominantly a user-centric defense, placing the *implementation burden on individuals* who are often non-experts, making widespread adoption unrealistic. Furthermore, this user-level design of existing approaches *conflicts with privacy regulations*-such as the GDPR [\[31\]](#page-11-9)-that assign service providers the obligation to ensure anti-personalization upon user requests. As a result, such methods are inherently unsuitable for provider-side deployment (see Appendix [F](#page-25-0) for more details).

Taken together, these issues highlight the need to move beyond user-side defenses toward modellevel solutions that not only enable service providers to enforce anti-personalization directly within their systems but also enhance robustness and practicality in real-world deployments. To address this, we shift our focus from the data samples to the DMs themselves. In this paper, we propose Anti-Personalized Diffusion Model (APDM), a novel framework designed to directly remove personalization capabilities for specific subjects within pre-trained DMs, without data-poisoning. The primary goals of APDM are twofold: (i) *preventing* the unauthorized personalization attempts, resulting in failed or irrelevant generations, and (ii) *preserving* the generation performance and its ability to personalize other, non-targeted subjects. To the best of our knowledge, APDM is the first approach to directly update the model parameters for protection, inherently overcoming data dependency.

However, simply redirecting the protection effort to the model parameters does not guarantee success if we naïvely adopt strategies from data-centric methods. Firstly, we theoretically prove that directly applying loss—originally designed for creating adversarial perturbations on images—to the model parameters fails to converge. To this end, we introduce a novel loss function, Direct Protective Optimization (DPO), disrupting the personalization process. Moreover, simply applying a protection loss uniformly is insufficient, since personalization involves iterative updates to model parameters. Therefore, being aware of the personalization trajectory is essential for robust protection. For this reason, we propose Learning to Protect (L2P), a dual-path optimization strategy. L2P alternates between a personalization path, simulating potential future personalized model states, and a protection path, which leverages these intermediate states to apply adaptive, trajectory-aware protective updates. This dynamic approach allows the model to anticipate and counteract personalization attempts, ensuring robust DM protection in across various scenarios.

Our contributions can be summarized as follows:

• For the first time, we propose a novel framework, called Anti-Personalized Diffusion Model (APDM), for robust anti-personalization in DMs by directly updating *model parameters*, unlike existing data-centric methods. This approach fundamentally overcomes the impractical assumptions and data dependency issues of prior works.

- We theoretically prove that a naive application of existing image perturbation losses directly to model parameters fails to converge. To address this, we propose a novel objective, Direct Protective Optimization (DPO) loss. DPO guides the model to remove the personalization capability of a specific subject while preserving generation performance.
- To effectively counteract the iterative and adaptive process of personalization, we introduce Learning to Protect (L2P), a dual-path optimization strategy that anticipates personalization trajectories and reinforces protection accordingly, enabling robust defense.
- We empirically demonstrate that APDM can safeguard against personalization in real-world scenarios, achieving state-of-the-art performance across various personalization subjects.

# 2 Related Work

Personalized Text-to-Image Diffusion Models. The advancement of diffusion-based image synthesis, like Stable Diffusion (SD) [\[27\]](#page-11-1), has enabled not only high-quality image generation but also the creation that reflect desired contexts from the text. This advancement has accelerated the widespread application of Text-to-Image (T2I) DMs [\[27\]](#page-11-1), one of which is personalization, such as generating images containing specific objects under the various situations (*e.g.* a particular dog or person on the moon). Consequently, research on personalized models has emerged. The most widely used method is DreamBooth [\[28\]](#page-11-4), which fine-tunes a pre-trained SD using a small set of images depicting a specific concept (*e.g.* a particular person). This allows users to generate desired images containing the target object. Texture Inversion [\[4\]](#page-10-3) achieves this by searching for an optimal text embedding that can represent the target object based on pseudo-words. Custom Diffusion [\[14\]](#page-10-4) optimizes the key and value projection matrices in the cross-attention layers of the pre-trained SD, offering more efficient and robust personalization performance. However, these methods are a double-edged sword, offering powerful personalization but also posing risks, such as misuse in crimes or unintended applications.

Protection against Unauthorized Personalization. To prevent unauthorized usage, many protection methods have been developed based on adversarial attacks [\[6,](#page-10-6) [2,](#page-10-7) [22\]](#page-11-7). AdvDM [\[16\]](#page-11-5) was the first to extend classification-based adversarial attack methods to DMs, generating adversarial samples for protecting personalization. Furthermore, Anti-DreamBooth [\[30\]](#page-11-6) proposed protection against more challenging fine-tuned DMs (*e.g.* DreamBooth). They used a fine-tuned surrogate model as guidance to obtain optimal perturbations for adversarial images. SimAC [\[34\]](#page-12-0) improved this optimization process to better suit DMs, while CAAT [\[35\]](#page-12-2) focused on reducing time costs by updating cross-attention blocks. MetaCloak [\[20\]](#page-11-8) and PID [\[15\]](#page-11-10) have also been conducted to counter text variation or image transformation techniques (*e.g.* filtering). The most recent work, PAP [\[33\]](#page-12-1), tries to predict potential prompt variations using Laplace approximation. However, existing works have primarily focused on how to effectively add perturbations to images for protection. In contrast, as we mentioned above, we apply protection directly at the model level, reflecting real-world demands.

### 3 Preliminaries

#### 3.1 Text-to-Image Diffusion Models

T2I DMs [\[27\]](#page-11-1), a popular variant of DMs [\[10,](#page-10-0) [29\]](#page-11-0) generate an image xˆ<sup>0</sup> corresponding to a given text prompt embedding c. T2I DMs operate via forward and reverse processes. In the forward process, noise ϵ ∼ N (0, I) is added to input image x<sup>0</sup> to produce noisy image x<sup>t</sup> at a timestep t ∈ [0, T]: √ √

<span id="page-2-1"></span><span id="page-2-0"></span>
$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \tag{1}$$

where α¯<sup>t</sup> = Π<sup>t</sup> <sup>i</sup>=1α<sup>i</sup> is computed from noise schedule {αt} T t=0. In the reverse process, DM, parameterized by θ, aims to denoise xt. DM is trained to predict the noise residuals added to xt:

$$\mathcal{L}_{simple} = \mathbb{E}_{x_0, t, c, \epsilon \sim \mathcal{N}(0, I)} \| \epsilon_{\theta}(x_t, t, c) - \epsilon \|_2^2.$$
 (2)

#### 3.2 Personalized Diffusion Models

To generate images that include a specific subject, several works personalize pre-trained T2I DMs [\[28,](#page-11-4) [14\]](#page-10-4). Given a small image set x<sup>0</sup> ∈ X of the subject and a text embedding c per with a unique identifier, *e.g. "a photo of [V\*] person"*, they modify the loss function in Eq.[\(2\)](#page-2-0) as follows:

$$\mathcal{L}_{simple}^{per} = \mathbb{E}_{x_0, t, c^{per}, \epsilon \sim \mathcal{N}(0, I)} \| \epsilon_{\theta}(x_t, t, c^{per}) - \epsilon \|_2^2, \tag{3}$$

where x<sup>t</sup> is a noisy image from Eq. [\(1\)](#page-2-1). However, directly applying this modified loss can cause language drift, where the personalized DM generates images related to target subject, even without unique identifier. To mitigate this, DreamBooth [\[28\]](#page-11-4) introduces a prior preservation loss function that leverages the pre-trained DM. This encourages DM, using a class-specific text embedding c pr (*e.g. "a photo of person"*), to retain its knowledge of the general class associated with the specific subject:

$$\mathcal{L}_{ppl} = \mathbb{E}_{x_0^{pr}, t, c^{pr}, \epsilon \sim \mathcal{N}(0, I)} \| \epsilon_{\theta}(x_t^{pr}, t, c^{pr}) - \epsilon \|_2^2, \tag{4}$$

where x pr 0 is a generated sample from the pre-trained T2I DM with the text embedding c pr, and x pr t is the noisy version of x pr 0 at timestep t. Alternatively, Custom Diffusion [\[14\]](#page-10-4) utilizes images from training dataset instead of generated images for x pr. The final objective for personalization becomes:

<span id="page-3-7"></span>
$$\mathcal{L}_{per} = \mathcal{L}_{simple}^{per} + \mathcal{L}_{ppl}.$$
 (5)

# 4 Method

#### <span id="page-3-5"></span>4.1 Problem Formulation

Unlike prior approaches that perturb *images*, we directly update the *parameters* θ of the pre-trained DM using only a small image set x<sup>0</sup> ∈ X . Our goal is to transform θ into a safeguarded model ˆθ that inherently resists personalization of the subject appearing in these images. This process can be viewed as optimizing the model parameters with respect to a protection objective:

$$\hat{\theta} = \arg\min_{\theta} \mathcal{L}_{protect},\tag{6}$$

where Lprotect is a loss function to prevent personalization, which will be discussed in Section [4.3.1.](#page-4-0) Subsequently, if an adversary attempts to personalize a subject in X with this safeguarded model ˆθ, the resulting personalized model ˆθper is obtained as follows:

$$\hat{\theta}_{per} = \arg\min_{\hat{\theta}} \mathcal{L}_{per}.$$
 (7)

Our approach has two main objectives. For protection, the re-personalized model ˆθper should yield low-quality images or images of subjects perceptually distinct from those in X . For stability, the protected model ˆθ should be able to generate high-quality images and effectively personalize for the other subjects, comparable to those produced by the pre-trained DM θ.

#### <span id="page-3-6"></span>4.2 Analysis of Naïve Approach

A naive yet intuitive way to protect the model is to extend existing data-poisoning approaches [\[16,](#page-11-5) [30,](#page-11-6) [34,](#page-12-0) [33\]](#page-12-1) to the model level. Specifically, their noise update process that maximizes L per simple using PGD [\[22\]](#page-11-7) can be naturally applied at the model level. In addition, the model's generative performance can be preserved by incorporating Lppl, as done in DreamBooth [\[28\]](#page-11-4). The overall objective for this naïve approach can be expressed as follows:

<span id="page-3-8"></span>
$$\mathcal{L}_{adv} = -\mathcal{L}_{simple}^{per} + \mathcal{L}_{ppl}.$$
 (8)

To ensure effective protection using Ladv, the optimization process must converge. We analyze the necessary conditions for convergence by examining the gradients of the loss with respect to θ. This leads to the following Proposition [1](#page-3-0) (proof in Appendix [A.1\)](#page-13-0).

<span id="page-3-0"></span>Proposition 1. *A necessary condition for* Ladv *to converge to a local minimum with respect to model parameters* θ *is that the gradients of its constituent terms,* ∇θL per simple *and* ∇θLppl*, must point in the same direction.*

To further understand how these gradients influence each other during optimization, we analyze their interaction through the first-order Taylor approximation and derive the following relationships.

$$(\nabla_{\theta} \mathcal{L}_{simple}^{per}(\theta))^{\top} \cdot (\nabla_{\theta} \mathcal{L}_{ppl}(\theta))) < \|\nabla_{\theta} \mathcal{L}_{ppl}(\theta))\|^{2}, \tag{9}$$

$$(\nabla_{\theta} \mathcal{L}_{simple}^{per}(\theta))^{\top} \cdot (\nabla_{\theta} \mathcal{L}_{ppl}(\theta))) < \|\nabla_{\theta} \mathcal{L}_{simple}^{per}(\theta)\|^{2}.$$
(10)

Based on the Proposition [1,](#page-3-0) we can restrict the left terms in Eq. [\(9\)](#page-3-1) and [\(10\)](#page-3-2), as |∇θL per simple(θ)| · |∇θLppl(θ)|. Using these results, we can rewrite the Eq. [\(9\)](#page-3-1) and [\(10\)](#page-3-2) as:

<span id="page-3-1"></span>
$$|\nabla_{\theta} \mathcal{L}_{simple}^{per}(\theta)| < |\nabla_{\theta} \mathcal{L}_{ppl}(\theta)|,$$
 (11)

<span id="page-3-4"></span><span id="page-3-3"></span><span id="page-3-2"></span>
$$|\nabla_{\theta} \mathcal{L}_{ppl}(\theta)| < |\nabla_{\theta} \mathcal{L}_{simple}^{per}(\theta)|.$$
 (12)

<span id="page-4-1"></span>![](_page_4_Figure_0.jpeg)

Figure 2: Overview. To prevent personalization in the parameter level, we propose Anti-Personalized Diffusion Model (APDM). (a) APDM first generates a paired image for each clean input image x0. (b) APDM consists of two components - (i) Learning to Protect, a novel optimization algorithm that makes the protection procedure aware of personalization trajectories, and (ii) Directed Protective Optimization loss, designed to disrupt personalization while preserving the generation capabilities.

By combining Proposition [1](#page-3-0) with the inequalities above, we observe that the required gradient alignment for convergence cannot hold, which we formalize in the following theorem (see Appendix [A.2\)](#page-15-0).

<span id="page-4-3"></span>Theorem 1. *If the objective is to simultaneously reduce both* −Lper simple *and* Lppl*, the necessary condition for convergence outlined in Proposition [1](#page-3-0) leads to the contradictory requirements presented in Eq.*[\(11\)](#page-3-3) *and* [\(12\)](#page-3-4)*. Therefore,* Ladv *composed of such conflicting terms generally fails to converge to a point that effectively optimizes both objectives.*

Therefore, a new loss function is required to resolve this conflict and ensure that anti-personalization updates stay consistent with the denoising process, maintaining both generation quality and protection.

#### <span id="page-4-2"></span>4.3 Anti-Personalized Diffusion Models

To achieve the dual goals outlined in [4.1,](#page-3-5) we propose a novel framework, Anti-Personalized Diffusion Models (APDM). APDM introduces a novel loss function, called Direct Protective Optimization (DPO), which aims to prevent personalization in DMs while maintaining their original generative performance (Section [4.3.1\)](#page-4-0). DPO effectively mitigates the model collapse issue discussed in Section [4.2.](#page-3-6) Furthermore, we propose a novel dual-path optimization scheme, Learning to Protect (L2P), which considers the trajectory of personalization during training to apply the proposed loss function more effectively (Section [4.3.2\)](#page-5-0). The overview of APDM is presented in Figure [2.](#page-4-1)

#### <span id="page-4-0"></span>4.3.1 Direct Protective Optimization

Instead of Ladv, which degrades the model's distribution due to convergence failure (Section [4.2\)](#page-3-6), we directly guide the model on which information should be learned and which should be suppressed. Inspired by Direct Preference Optimization [\[26\]](#page-11-11), given a pair of images (x + 0 , x<sup>−</sup> 0 ), we designate x + 0 as a positive sample to be encouraged during the protection procedure and x − 0 as a negative sample to be discouraged, *i.e.* an image containing a specific subject to be protected (x<sup>0</sup> ∈ X ). By incorporating the Bradley-Terry model, the probability of preferring x + 0 over x − 0 can be expressed as:

<span id="page-4-5"></span><span id="page-4-4"></span>
$$p(x_0^+ > x_0^-) = \sigma(r(x_0^+) - r(x_0^-)), \tag{13}$$

where σ(·) denotes the sigmoid function and r(·) represents the reward function. Building upon the formulation of Diffusion-DPO [\[32\]](#page-12-3) (see Appendix [A.3](#page-16-0) for detailed derivation), we define a new Direct Protective Optimization (DPO) as follows:

$$r^{+} = \|\epsilon_{\theta}(x_{t}^{+}, t, c) - \epsilon\|_{2}^{2} - \|\epsilon_{\phi}(x_{t}^{+}, t, c) - \epsilon\|_{2}^{2},$$

$$r^{-} = \|\epsilon_{\theta}(x_{t}^{-}, t, c) - \epsilon\|_{2}^{2} - \|\epsilon_{\phi}(x_{t}^{-}, t, c) - \epsilon\|_{2}^{2},$$

$$\mathcal{L}_{DPO} = -\mathbb{E}_{x_{0}^{+}, x_{0}^{-}, c, t, \epsilon \sim N(0, I)} \log \sigma(-\beta(r^{+} - r^{-})),$$
(14)

where ϕ is a pre-trained DM and β is a hyper-parameter that controls the extent to which θ can diverge from ϕ. In our DPO, we prepare x + 0 by synthesizing images from pre-trained T2I DMs ϕ using a

#### <span id="page-5-4"></span>Algorithm 1 Learning to Protect (L2P)

Input: pre-trained model θ, loss function for personalization Lper, loss function for protection Lprotect, the number of personalization loops Nper, the number of protection loops Nprotect, learning rate in for personalization γper, learning rate in protection γprotect.

Output: safeguarded model ˆθ.

#### Procedure:

```
1: j ← 1, θj ← θ
2: for j to Nprotect do ▷ Protection Path
3: i ← 1, θ′
        i ← θj .copy(), g ← ∅
4: for i to Nper do ▷ Personalization Path
5: θ
      ′
      i+1 ← θ
           ′
           i − γper∇θ
                 Lper ▷ Eq. (17)
6: g.append(∇θ
             ′
             i+1
              Lprotect)
7: end for
8: ∇protect ← g.sum() ▷ Eq. (19)
9: θj+1 ← θj − γprotect∇protect ▷ Eq. (20)
10: end for
11: return ˆθ ← θNprotect
```

generic prompt c pr, and they are paired one-to-one with the negative samples X . This approach naturally encourages the generation of generic (positive) images while effectively suppressing the synthesis of negative images depicting the specific subject.

Finally, combining the proposed loss term with the preservation loss (Lppl), the final objective is:

$$\mathcal{L}_{protect} = \mathcal{L}_{DPO} + \mathcal{L}_{ppl}. \tag{15}$$

#### <span id="page-5-0"></span>4.3.2 Learning to Protect

Since the personalization of DMs involves iterative updates to model parameters, effective protection should consider the evolving personalized states at different states. Therefore, instead of simply applying our Lprotect uniformly to the model, we simulate the future personalization path in advance, allowing the model to anticipate upcoming parameter changes during personalization. To this end, we introduce a novel dual-path optimization algorithm, Learning to Protect (L2P). L2P integrates personalization into the protection loop, enabling the model to learn from simulated personalization behaviors and adjust its parameters for adaptive and robust protection.

L2P involves two optimization paths: personalization and protection. The personalization path updates the model from the current protection state θ<sup>j</sup> to intermediate state θ ′ i , using Eq. [\(5\)](#page-3-7):

$$\theta_i' = \theta_j, \tag{16}$$

<span id="page-5-1"></span>
$$\theta'_{i+1} = \theta'_i - \gamma_{per} \nabla_{\theta'_i} \mathcal{L}_{per}, \tag{17}$$

where γper is the learning rate for personalization, and θ ′ <sup>i</sup>+1 is the intermediate state at step i + 1 during personalization. Using Eq. [\(17\)](#page-5-1), we can simulate the future personalization trajectory via updating the model θ ′ i iteratively, in the middle of protecting the DM.

For the protection path, we leverage these intermediate states acquired in the personalization path. Specifically, we compute the gradient ∇<sup>i</sup> of the model θ ′ <sup>i</sup> with respect to Lprotect, at each state i in the personalization path as follows:

$$\nabla_i = \nabla_{\theta_i'} \mathcal{L}_{protect}. \tag{18}$$

We then accumulate ∇<sup>i</sup> during the whole personalization path (total of Nper times) to compose a set of gradients, g = {∇i} Nper <sup>i</sup>=1 . Using this set of gradients g, we can estimate the direction of protection from the summation of these accumulated gradients as follows:

<span id="page-5-3"></span><span id="page-5-2"></span>
$$\nabla_{protect} = \sum_{i=1}^{N_{per}} \nabla_i. \tag{19}$$

Finally, we update the intermediate protection model θ<sup>j</sup> with ∇protect to obtain θj+1:

$$\theta_{j+1} = \theta_j - \gamma_{protect} \nabla_{protect}, \tag{20}$$

<span id="page-6-1"></span>Table 1: Quantitative Comparison on Protection. We measured the protection performance via DINO score [\[3\]](#page-10-8) and BRISQUE [\[23\]](#page-11-12). We examined the baseline on different number of clean images. If the number is 0, there are only perturbed images produced by data-poisoning approaches. The experiments were mainly conducted on two different subjects: person and dog.

| Methods              | # Clean |          | DINO (↓) |        |          | BRISQUE (↑) |       |  |
|----------------------|---------|----------|----------|--------|----------|-------------|-------|--|
|                      | Images  | "person" | "dog"    | Avg.   | "person" | "dog"       | Avg.  |  |
| DreamBooth [28]      | N       | 0.6994   | 0.6056   | 0.6525 | 11.27    | 22.33       | 16.80 |  |
|                      | 0       | 0.5752   | 0.4247   | 0.4999 | 19.52    | 28.60       | 24.06 |  |
| AdvDM [16]           | 1       | 0.5436   | 0.4393   | 0.4915 | 17.82    | 28.58       | 23.20 |  |
|                      | N − 1   | 0.6417   | 0.4775   | 0.5596 | 20.30    | 27.36       | 23.83 |  |
|                      | 0       | 0.5254   | 0.4106   | 0.4680 | 26.90    | 30.23       | 28.56 |  |
| Anti-DreamBooth [30] | 1       | 0.6081   | 0.4704   | 0.5393 | 23.76    | 27.49       | 25.63 |  |
|                      | N − 1   | 0.6951   | 0.5304   | 0.6127 | 15.48    | 25.26       | 20.37 |  |
| SimAC [34]           | 0       | 0.4448   | 0.4374   | 0.4411 | 23.73    | 31.64       | 27.69 |  |
|                      | 1       | 0.5824   | 0.4537   | 0.5181 | 18.04    | 29.54       | 23.79 |  |
|                      | N − 1   | 0.6991   | 0.5370   | 0.6181 | 14.28    | 27.05       | 20.67 |  |
| PAP [33]             | 0       | 0.6556   | 0.5120   | 0.5838 | 22.61    | 30.20       | 26.41 |  |
|                      | 1       | 0.6690   | 0.5032   | 0.5861 | 22.02    | 29.00       | 25.51 |  |
|                      | N − 1   | 0.7028   | 0.5270   | 0.6149 | 19.64    | 23.41       | 21.53 |  |
| APDM (Ours)          | N       | 0.1375   | 0.0959   | 0.1167 | 40.25    | 60.74       | 50.50 |  |

where γprotect is the learning rate for protection. By repeating this process for Nprotect times, we can obtain a safeguarded model ˆθ, which is aware of the personalization path inherently for better protection. Algorithm [1](#page-5-4) illustrates the overall learning process of L2P for our APDM framework.

# 5 Experiments

#### 5.1 Experimental Setup

Evaluation Metrics. To evaluate the effectiveness of APDM in protecting against personalization on specific subjects, we used two metrics: (i) the DINO score [\[3\]](#page-10-8) as a similarity-based metric and (ii) BRISQUE [\[23\]](#page-11-12) for assessing image quality. Additionally, we evaluated the preservation of the pre-trained model's generation capabilities by using (iii) the FID score [\[9\]](#page-10-9) for image quality, (iv) the CLIP score [\[25\]](#page-11-13), (v) TIFA [\[12\]](#page-10-10), and (vi) GenEval [\[5\]](#page-10-11) for image-text alignment.

Baselines. We consider DreamBooth [\[28\]](#page-11-4) and Custom Diffusion [\[14\]](#page-10-4) as personalization methods. The results of Custom Diffusion are presented in Appendix [C.](#page-19-0) For baselines, we include the previous protection approaches: (i) AdvDM [\[35\]](#page-12-2), (ii) Anti-DreamBooth [\[30\]](#page-11-6), (iii) SimAC [\[34\]](#page-12-0), and (iv) PAP [\[33\]](#page-12-1). Following Anti-DreamBooth, we set the perturbation intensity for all baselines to 5e-2.

Datasets. We used the datasets from both DreamBooth[3](#page-6-0) [\[28\]](#page-11-4) and Anti-DreamBooth [\[30\]](#page-11-6) to evaluate the protection performance. The DreamBooth dataset contains 4-6 images per subject across various object classes such as dog, cat, and toy. The Anti-DreamBooth dataset includes 4 images per person, consisting of facial images collected from CelebA-HQ [\[13\]](#page-10-12) and VGGFace2 [\[1\]](#page-10-13). To quantify the preservation performance of the model, we also used the MS-COCO 2014 [\[17\]](#page-11-14) validation split.

Implementation Details. We built APDM on Stable Diffusion 1.5 and Stable Diffusion 2.1 [\[27\]](#page-11-1) with 512x512 resolution. We used AdamW optimizer [\[21\]](#page-11-15) with learning rates γper = γprotect = 5e − 6. In DPO, we set the hyperparameter β to 1. In L2P, we used Nper = 20 and Nprotect = 800. We conducted all of our experiments on a single NVIDIA RTX A6000 GPU, and it took about 9 GPU hours to protect DM. To synthesize images, we used PNDM scheduler [\[19\]](#page-11-16) with 20 steps. For Stable Diffusion 2.1, we have attached the experimental results in Appendix [C.](#page-19-0)

### 5.2 Protection Performance

As shown in Figure [3](#page-7-0) and Table [1,](#page-6-1) we first evaluated the baselines and APDM from the perspective of protection. We first personalized the pre-trained Stable Diffusion using DreamBooth [\[28\]](#page-11-4) as a reference. In this experiment, we considered three scenarios to test baselines and APDM. For DreamBooth and APDM, only N clean (*i.e.* non-perturbed) images were used throughout the entire

<span id="page-6-0"></span><sup>3</sup> https://github.com/google/dreambooth

<span id="page-7-0"></span>![](_page_7_Figure_0.jpeg)

Figure 3: Qualitative Comparison on Protection. We examined the baselines and APDM on a protective aspect. We tested baselines on different circumstance - "All Perturbed", "One Clean", and "One Perturbed". In the "All Perturbed" setting, the baselines added perturbations to all training images. "One Clean" and "One Perturbed" settings are more difficult than "All Perturbed" setting, where the dataset contains one clean image or one perturbed image.

experiment ("All Clean" in Figure [3\)](#page-7-0). On the other hand, for data-poisoning baselines, we adopted different personalization scenarios. For "All Perturbed" scenario, we utilized all perturbed images from each data-poisoning baseline. Moreover, for "One Clean" scenario, we used 1 clean image and N − 1 perturbed images for personalization. Lastly, the most challenging scenario, for "One Perturbed" scenario, there were only 1 perturbed image and N − 1 clean images in the dataset.

In Figure [3,](#page-7-0) comparisons revealed their limitations as the scenarios become more challenging. When only one perturbed image is used and the others remain clean, protection against personalization for the subjects becomes ineffective. In contrast, despite the presence of clean images, APDM consistently demonstrated its robustness in more challenging scenarios (additional qualitative results in Appendix [E\)](#page-22-0). We also present a quantitative comparison in Table [1,](#page-6-1) highlighting that APDM outperforms data-poisoning approaches even under the most difficult conditions. This is because APDM protects personalization at the model-level, making it robust to variations in the input data. In addition, we also tested APDM in different scenarios (transform, such as flipping and blurring) and subjects such as "cat", "sneaker", "glasses", and "clock" (results in Appendix [B\)](#page-18-0).

#### 5.3 Preservation Performance

As described in Section [4.3.2,](#page-5-0) we updated the parameters of DM initialized with a pre-trained DM to obtain a safeguarded model. To ensure its usability in future applications, it is essential to preserve the inherent capabilities of the pre-trained DMs during the protection process. In this sec-

<span id="page-7-1"></span>Table 2: Preservation Performance on Image Quality and Image-Text Alignment. We measured the image quality via FID score [\[9\]](#page-10-9) and image-text alignment via CLIP score [\[25\]](#page-11-13), TIFA [\[12\]](#page-10-10), and GenEval [\[5\]](#page-10-11) on COCO 2014 [\[17\]](#page-11-14) validation dataset.

| Methods               | FID (↓) | CLIP (↑) | TIFA (↑) | GenEval (↑) |
|-----------------------|---------|----------|----------|-------------|
| Stable Diffusion [27] | 25.98   | 0.2878   | 78.76    | 0.4303      |
| APDM (Ours)           | 28.85   | 0.2853   | 75.91    | 0.4017      |

tion, we evaluated the inherent performance based on image quality, image-text alignment of generated images, and the success of personalization for subjects not targeted by the protection.

<span id="page-8-0"></span>Table 3: Preservation Performance on Personalization of Different Subjects. We tried to personalize APDM to different subjects, such as "cat", "sneaker", and "glasses". We reported the personalization performance of DreamBooth [\[28\]](#page-11-4) these subjects as a reference.

| Methods         | DINO (↑) |           |           | BRISQUE (↓) |       |           |           |       |
|-----------------|----------|-----------|-----------|-------------|-------|-----------|-----------|-------|
|                 | "cat"    | "sneaker" | "glasses" | Avg.        | "cat" | "sneaker" | "glasses" | Avg.  |
| DreamBooth [28] | 0.4903   | 0.6110    | 0.6961    | 0.5991      | 25.32 | 23.14     | 19.01     | 22.49 |
| APDM (Ours)     | 0.4231   | 0.7573    | 0.7198    | 0.6334      | 27.72 | 18.10     | 27.41     | 24.41 |

<span id="page-8-1"></span>Table 4: Ablation on the Effect of Image Pairing between x + 0 and x − 0 . We compared the protection performance with and without pairing.

| Paired | DINO (↓) |        | BRISQUE (↑) |       |  |
|--------|----------|--------|-------------|-------|--|
|        | "person" | "dog"  | "person"    | "dog" |  |
| ✗      | 0.2770   | 0.3487 | 27.32       | 29.87 |  |
| ✓      | 0.1375   | 0.0959 | 40.25       | 60.74 |  |

Table 6: Ablation on the Effect of β. We compared the protection performance with different hyperparameter β.

| β   | DINO (↓) |        | BRISQUE (↑) |       |  |
|-----|----------|--------|-------------|-------|--|
|     | "person" | "dog"  | "person"    | "dog" |  |
| 1   | 0.1375   | 0.0959 | 40.25       | 60.74 |  |
| 10  | 0.5392   | 0.3885 | 13.58       | 15.14 |  |
| 100 | 0.5962   | 0.4755 | 12.21       | 14.10 |  |

Table 5: Ablation on the Effect of L2P. We compared the performance between protection attempts without and with L2P.

| L2P | DINO (↓) |        | BRISQUE (↑) |       |  |
|-----|----------|--------|-------------|-------|--|
|     | "person" | "dog"  | "person"    | "dog" |  |
| ✗   | 0.4454   | 0.3689 | 24.70       | 30.62 |  |
| ✓   | 0.1375   | 0.0959 | 40.25       | 60.74 |  |

Table 7: Ablation on the Effect of Nper in L2P. We measured the performance in a protection aspect by varying Nper of personalization path.

| Nper | DINO (↓) |        | BRISQUE (↑) |       |  |
|------|----------|--------|-------------|-------|--|
|      | "person" | "dog"  | "person"    | "dog" |  |
| 5    | 0.3371   | 0.1923 | 37.89       | 39.48 |  |
| 10   | 0.2096   | 0.1342 | 38.14       | 47.15 |  |
| 20   | 0.1375   | 0.0959 | 40.25       | 60.74 |  |

Table [2](#page-7-1) shows that APDM maintains high-quality image generation comparable to the pre-trained model. Beyond image quality and image-text alignment, we also evaluated its ability to personalize for different subjects using the protected DMs. Specifically, we tested personalization on models protected for "person" or "dog", using a new set of images featuring "cat", "clock" and "glasses". As shown in Table [3,](#page-8-0) these protected models remain effective for personalizing other subjects. Overall, APDM successfully protects specific subjects while preserving personalization capabilities for others, making it suitable for handling diverse user requests in real-world applications.

### 5.4 Ablation Study

Ablation on Loss Functions. In Section [4.3,](#page-4-2) we introduced a novel objective, Direct Protective Optimization (DPO), which effectively prevents personalization while minimally degrading the model's generation performance. In Table [4,](#page-8-1) we assessed the impact of pairing positive and negative images on protection performance. The results demonstrate that constructing image pairs significantly enhances performance by providing explicit guidance on which information should be encouraged or discouraged. Additionally, we investigated the effect of the hyperparameter β, which governs the strength of our DPO objective. As shown in Table [6,](#page-8-1) our findings indicate that reducing β allows APDM to more effectively prevent personalization.

Ablation on Optimization Scheme. In Section [4.3.2,](#page-5-0) we proposed a novel optimization scheme, Learning to Protect (L2P), which incorporates awareness of the personalization process during protection. In Table [5,](#page-8-1) we compared the protection performance with and without L2P, and observed that incorporating the personalization trajectory significantly improves protection performance. Moreover, we examined the effect of the number of personalization paths (Nper). As shown in Table [7,](#page-8-1) increasing Nper consistently improves performance. Despite this trend, we set Nper = 20 as the default in our overall experiments, since it already achieved state-of-the-art performance.

<span id="page-9-0"></span>Table 8: Protection performance of APDM on clean and perturbed data. We evaluate whether APDM can maintain its protection capability regardless of input perturbations.

| Methods                | # Clean<br>Images | DINO (↓) | BRISQUE (↑) |
|------------------------|-------------------|----------|-------------|
| DreamBooth [28]        | N                 | 0.6869   | 16.69       |
| Anti-DreamBooth [30]   | 0                 | 0.5646   | 22.50       |
| APDM (Ours)            | N                 | 0.1375   | 40.25       |
| APDM (Ours, perturbed) | 0                 | 0.1702   | 40.20       |

#### 5.5 Additional Experiments

As demonstrated in previous experiments, APDM effectively performs protection even in challenging cases, such as when clean images are used. This robustness comes from its model-level defense mechanism, which allows protection to be achieved independently of the input data. To further demonstrate this robustness, we examined whether APDM can also protect against perturbed data generated through data-poisoning methods. Specifically, we generated perturbed data using Anti-DreamBooth [\[30\]](#page-11-6) and evaluated APDM's protection performance on these data. As shown in Table [8,](#page-9-0) APDM successfully prevents personalization even on perturbed data, confirming that its effectiveness is independent of the input variations.

Building upon the previous analysis on perturbed data, we further investigated whether APDM maintains its protection capability when both the number and type of personalization data vary. Specifically, this evaluation examined the generalization and scalability of APDM by considering two factors: (i) the use of unseen data that were not included during the protection stage, and (ii) the increased amount of personalization data per subject. As shown in Table [9,](#page-9-1) APDM

<span id="page-9-1"></span>Table 9: Protection performance of APDM under varying numbers of unseen images. We evaluate whether APDM can maintain its protection capability across different input conditions and unseen data counts.

| Methods         | # of unseen | DINO (↓) | BRISQUE (↑) |  |
|-----------------|-------------|----------|-------------|--|
| DreamBooth [28] | −           | 0.6869   | 16.69       |  |
|                 | −           | 0.1375   | 40.25       |  |
|                 | 4           | 0.1616   | 38.14       |  |
| APDM (Ours)     | 8           | 0.1994   | 38.87       |  |
|                 | 12          | 0.1873   | 38.87       |  |

consistently maintains protection performance even when 4–12 unseen images are introduced, confirming that its defense mechanism generalizes well to unseen samples and remains robust as the data volume increases.

To further assess the robustness of APDM under diverse personalization conditions, we additionally conducted experiments using varied text prompts and different unique identifiers, as well as an independent user study designed to evaluate real users' preferences. Due to the page limit, these extended results are provided in Appendix [B](#page-18-0) (diverse prompt and identifier experiments) and Appendix [D](#page-22-1) (user study).

### 6 Conclusion

In this paper, we address privacy concerns in personalized DMs. We highlight critical limitations of existing approaches, which depend on impractical assumptions (*e.g.* exhaustive data poisoning) and fail to comply with privacy regulations. Furthermore, we demonstrate that these approaches are easily circumvented when attackers use clean images or apply transformations to weaken the perturbation effects. Therefore, we shifted the focus from data-centric defenses to model-level protection, aiming to directly prevent personalization through optimization rather than input modification. To this end, we propose a novel framework APDM (Anti-Personalized Diffusion Models), which consists of a novel loss function, DPO (Direct Protective Optimization), and a new dual-path optimization scheme, L2P (Learning to Protect). With APDM, we successfully prevented personalization while preserving the generative quality of the original model. Experimental results demonstrate the effectiveness and robustness of APDM with promising outputs. We hope our work extends the scope of antipersonalization towards more practical and appropriate real-world solutions.

# Acknowledgments and Disclosure of Funding

This work was supported by Korea Planning & Evaluation Institute of Industrial Technology (KEIT) grant funded by the Korea government (MOTIE) (RS-2024-00444344), and in part by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) under Grant No. RS2019-II190079 (Artificial Intelligence Graduate School Program (Korea University)), No. RS-2024-00457882 (AI Research Hub Research), and 2019-0-00004 (Development of Semi-Supervised Learning Language Intelligence Technology and Korean Tutoring Service for Foreigners). Additionally, it was supported in part by the Institute of Information and Communications Technology Planning and Evaluation (IITP) Grant funded by the Korea Government (MSIT) ((Artificial Intelligence Innovation Hub) under Grant RS-2021-II212068).

### References

- <span id="page-10-13"></span>[1] Cao, Q., Shen, L., Xie, W., Parkhi, O.M., Zisserman, A.: Vggface2: A dataset for recognising faces across pose and age. In: 2018 13th IEEE international conference on automatic face & gesture recognition (FG 2018). pp. 67–74. IEEE (2018)
- <span id="page-10-7"></span>[2] Carlini, N., Wagner, D.: Towards evaluating the robustness of neural networks. In: 2017 IEEE Symposium on Security and Privacy (sp). pp. 39–57. Ieee (2017)
- <span id="page-10-8"></span>[3] Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J., Bojanowski, P., Joulin, A.: Emerging properties in self-supervised vision transformers. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 9650–9660 (2021)
- <span id="page-10-3"></span>[4] Gal, R., Alaluf, Y., Atzmon, Y., Patashnik, O., Bermano, A.H., Chechik, G., Cohen-or, D.: An image is worth one word: Personalizing text-to-image generation using textual inversion. In: The Eleventh International Conference on Learning Representations (ICLR) (2023), [https://openreview.net/forum?id=](https://openreview.net/forum?id=NAQvF08TcyG) [NAQvF08TcyG](https://openreview.net/forum?id=NAQvF08TcyG)
- <span id="page-10-11"></span>[5] Ghosh, D., Hajishirzi, H., Schmidt, L.: Geneval: An object-focused framework for evaluating text-to-image alignment. Advances in Neural Information Processing Systems (NeurIPS) 36, 52132–52152 (2023)
- <span id="page-10-6"></span>[6] Goodfellow, I.J., Shlens, J., Szegedy, C.: Explaining and harnessing adversarial examples. In: International Conference on Learning Representations (ICLR) (2015), <https://arxiv.org/abs/1412.6572>
- <span id="page-10-1"></span>[7] Guo, Y., Yang, C., Rao, A., Liang, Z., Wang, Y., Qiao, Y., Agrawala, M., Lin, D., Dai, B.: Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. In: The Twelfth International Conference on Learning Representations (ICLR) (2024), [https://openreview.net/forum?](https://openreview.net/forum?id=Fx2SbBgcte) [id=Fx2SbBgcte](https://openreview.net/forum?id=Fx2SbBgcte)
- <span id="page-10-2"></span>[8] Hertz, A., Mokady, R., Tenenbaum, J., Aberman, K., Pritch, Y., Cohen-or, D.: Prompt-to-prompt image editing with cross-attention control. In: The Eleventh International Conference on Learning Representations (ICLR) (2023), [https://openreview.net/forum?id=\\_CDixzkzeyb](https://openreview.net/forum?id=_CDixzkzeyb)
- <span id="page-10-9"></span>[9] Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., Hochreiter, S.: Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in Neural Information Processing Systems (NeurIPS) 30 (2017)
- <span id="page-10-0"></span>[10] Ho, J., Jain, A., Abbeel, P.: Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems (NeurIPS) 33, 6840–6851 (2020)
- <span id="page-10-5"></span>[11] Hönig, R., Rando, J., Carlini, N., Tramèr, F.: Adversarial perturbations cannot reliably protect artists from generative ai. In: The Thirteenth International Conference on Learning Representations (ICLR) (2025)
- <span id="page-10-10"></span>[12] Hu, Y., Liu, B., Kasai, J., Wang, Y., Ostendorf, M., Krishna, R., Smith, N.A.: Tifa: Accurate and interpretable text-to-image faithfulness evaluation with question answering. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 20406–20417 (2023)
- <span id="page-10-12"></span>[13] Karras, T., Aila, T., Laine, S., Lehtinen, J.: Progressive growing of gans for improved quality, stability and variation. In: Proceedings of the International Conference on Learning Representations (ICLR) (2018)
- <span id="page-10-4"></span>[14] Kumari, N., Zhang, B., Zhang, R., Shechtman, E., Zhu, J.Y.: Multi-concept customization of text-to-image diffusion. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 1931–1941 (2023)

- <span id="page-11-10"></span>[15] Li, A., Mo, Y., Li, M., Wang, Y.: Pid: Prompt-independent data protection against latent diffusion models. In: Salakhutdinov, R., Kolter, Z., Heller, K., Weller, A., Oliver, N., Scarlett, J., Berkenkamp, F. (eds.) Proceedings of the 41st International Conference on Machine Learning (ICML). Proceedings of Machine Learning Research, vol. 235, pp. 28421–28447. PMLR (21–27 Jul 2024), [https://proceedings.mlr.](https://proceedings.mlr.press/v235/li24ay.html) [press/v235/li24ay.html](https://proceedings.mlr.press/v235/li24ay.html)
- <span id="page-11-5"></span>[16] Liang, C., Wu, X., Hua, Y., Zhang, J., Xue, Y., Song, T., Xue, Z., Ma, R., Guan, H.: Adversarial example does good: Preventing painting imitation from diffusion models via adversarial examples. In: Krause, A., Brunskill, E., Cho, K., Engelhardt, B., Sabato, S., Scarlett, J. (eds.) Proceedings of the 40th International Conference on Machine Learning (ICML). Proceedings of Machine Learning Research, vol. 202, pp. 20763–20786. PMLR (23–29 Jul 2023), <https://proceedings.mlr.press/v202/liang23g.html>
- <span id="page-11-14"></span>[17] Lin, T.Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., Zitnick, C.L.: Microsoft coco: Common objects in context. In: Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13. pp. 740–755. Springer (2014)
- <span id="page-11-2"></span>[18] Liu, H., Chen, Z., Yuan, Y., Mei, X., Liu, X., Mandic, D., Wang, W., Plumbley, M.D.: AudioLDM: Text-to-audio generation with latent diffusion models. In: Krause, A., Brunskill, E., Cho, K., Engelhardt, B., Sabato, S., Scarlett, J. (eds.) Proceedings of the 40th International Conference on Machine Learning (ICML). Proceedings of Machine Learning Research, vol. 202, pp. 21450–21474. PMLR (23–29 Jul 2023), <https://proceedings.mlr.press/v202/liu23f.html>
- <span id="page-11-16"></span>[19] Liu, L., Ren, Y., Lin, Z., Zhao, Z.: Pseudo numerical methods for diffusion models on manifolds. In: International Conference on Learning Representations (ICLR) (2022), [https://openreview.net/](https://openreview.net/forum?id=PlKWVd2yBkY) [forum?id=PlKWVd2yBkY](https://openreview.net/forum?id=PlKWVd2yBkY)
- <span id="page-11-8"></span>[20] Liu, Y., Fan, C., Dai, Y., Chen, X., Zhou, P., Sun, L.: Metacloak: Preventing unauthorized subject-driven text-to-image diffusion-based synthesis via meta-learning. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 24219–24228 (2024)
- <span id="page-11-15"></span>[21] Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. In: International Conference on Learning Representations (ICLR) (2019)
- <span id="page-11-7"></span>[22] Madry, A., Makelov, A., Schmidt, L., Tsipras, D., Vladu, A.: Towards deep learning models resistant to adversarial attacks. In: International Conference on Learning Representations (ICLR) (2018), [https:](https://openreview.net/forum?id=rJzIBfZAb) [//openreview.net/forum?id=rJzIBfZAb](https://openreview.net/forum?id=rJzIBfZAb)
- <span id="page-11-12"></span>[23] Mittal, A., Moorthy, A.K., Bovik, A.C.: No-reference image quality assessment in the spatial domain. IEEE Transactions on Image Processing 21(12), 4695–4708 (2012). https://doi.org/10.1109/TIP.2012.2214050
- <span id="page-11-3"></span>[24] Parmar, G., Kumar Singh, K., Zhang, R., Li, Y., Lu, J., Zhu, J.Y.: Zero-shot image-to-image translation. In: ACM SIGGRAPH 2023 Conference Proceedings. pp. 1–11 (2023)
- <span id="page-11-13"></span>[25] Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: International Conference on Machine Learning (ICML). pp. 8748–8763. PMLR (2021)
- <span id="page-11-11"></span>[26] Rafailov, R., Sharma, A., Mitchell, E., Manning, C.D., Ermon, S., Finn, C.: Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems (NeurIPS) 36, 53728–53741 (2023)
- <span id="page-11-1"></span>[27] Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent diffusion models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 10684–10695 (2022)
- <span id="page-11-4"></span>[28] Ruiz, N., Li, Y., Jampani, V., Pritch, Y., Rubinstein, M., Aberman, K.: Dreambooth: Fine tuning text-toimage diffusion models for subject-driven generation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 22500–22510 (2023)
- <span id="page-11-0"></span>[29] Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N., Ganguli, S.: Deep unsupervised learning using nonequilibrium thermodynamics. In: International Conference on Machine Learning (ICML). pp. 2256– 2265. PMLR (2015)
- <span id="page-11-6"></span>[30] Van Le, T., Phung, H., Nguyen, T.H., Dao, Q., Tran, N.N., Tran, A.: Anti-dreambooth: Protecting users from personalized text-to-image synthesis. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 2116–2127 (2023)
- <span id="page-11-9"></span>[31] Voigt, P., Bussche, A.v.d.: The EU General Data Protection Regulation (GDPR): A Practical Guide. Springer Publishing Company, Incorporated (2017)

- <span id="page-12-3"></span>[32] Wallace, B., Dang, M., Rafailov, R., Zhou, L., Lou, A., Purushwalkam, S., Ermon, S., Xiong, C., Joty, S., Naik, N.: Diffusion model alignment using direct preference optimization. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 8228–8238 (2024)
- <span id="page-12-1"></span>[33] Wan, C., He, Y., Song, X., Gong, Y.: Prompt-agnostic adversarial perturbation for customized diffusion models. Advances in Neural Information Processing Systems (NeurIPS) 37, 136576–136619 (2024)
- <span id="page-12-0"></span>[34] Wang, F., Tan, Z., Wei, T., Wu, Y., Huang, Q.: Simac: A simple anti-customization method for protecting face privacy against text-to-image synthesis of diffusion models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 12047–12056 (June 2024)
- <span id="page-12-2"></span>[35] Xu, J., Lu, Y., Li, Y., Lu, S., Wang, D., Wei, X.: Perturbing attention gives you more bang for the buck: Subtle imaging perturbations that efficiently fool customized diffusion models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 24534–24543 (2024)

In this appendix, we provide detailed proofs and derivations, and additional experimental results that were not included in the main paper due to page limits. The contents of the appendix are as follows:

- Appendix [A:](#page-13-1) Derivation and proof of Proposition 1, Theorem 1, and Direct Protective Optimization (DPO) objective.
- Appendix [B:](#page-18-0) Additional experiments including empirical results about naïve approach, comparison on protection with image transformations, and protection performance for different subjects.
- Appendix [C:](#page-19-0) Generalizability of APDM on different personalization methods, Stable Diffusion version, unique identifier, and diverse test prompts.
- Appendix [D:](#page-22-1) User study about protection performance.
- Appendix [E:](#page-22-0) Additional qualitative results extending to the experimental results in the main paper.
- Appendix [F:](#page-25-0) Additional explanation about our motivation.
- Appendix [G:](#page-25-1) The limitations and broader impacts of APDM, and a discussion of future work.

# <span id="page-13-1"></span>A Proofs and Derivation

In this section, we present the formal proofs and derivations supporting our main theoretical contributions discussed in the main paper. We begin by providing a rigorous proof for Proposition [1](#page-3-0) (Appendix [A.1\)](#page-13-0), followed by the complete proof for Theorem [1](#page-4-3) (Appendix [A.2\)](#page-15-0). Subsequently, we detail the step-by-step derivation of our proposed DPO loss function in Appendix [A.3.](#page-16-0)

#### <span id="page-13-0"></span>A.1 Proof of Proposition 1

The primary goal of Proposition [1](#page-3-0) is to identify and establish the necessary conditions under which naïve approach converges. We begin the proof by recalling the loss function of naïve approach, Equation [\(8\)](#page-3-8) in our main paper:

<span id="page-13-8"></span>
$$\mathcal{L}_{adv} = -\mathcal{L}_{simple}^{per} + \lambda \mathcal{L}_{ppl}, \tag{21}$$

where λ is positive scalar (λ > 0) to weight the Lppl, and each term is defined as follows:

$$\mathcal{L}_{simple}^{per} = \mathbb{E}_{x_0, t, c, \epsilon \sim \mathcal{N}(0, I)} \| \epsilon_{\theta}(x_t, t, c) - \epsilon \|_2^2, \tag{22}$$

$$\mathcal{L}_{ppl} = \mathbb{E}_{x_0^{pr}, t, c^{pr}, \epsilon \sim \mathcal{N}(0, I)} \| \epsilon_{\theta}(x_t^{pr}, t, c^{pr}) - \epsilon \|_2^2.$$
(23)

In optimization theory, a fundamental necessary condition for a differentiable function to attain a local minimum is that its *derivative with respect to the optimization variables must be zero*. This is often referred to as the first-order necessary condition for optimality. Applying this principle to our case, for Ladv to converge to a stable point with respect to the model parameters θ, the derivative must be zero as:

<span id="page-13-7"></span><span id="page-13-6"></span><span id="page-13-5"></span><span id="page-13-4"></span><span id="page-13-3"></span><span id="page-13-2"></span>
$$\nabla_{\theta} \mathcal{L}_{adv} = 0. \tag{24}$$

To address the condition in Equation [\(24\)](#page-13-2), we compute the gradient of Ladv with respect to θ. To simplify the computation, we first recall the MSE loss term as:

$$||u - v||_2^2 = u^{\mathsf{T}}u - 2u^{\mathsf{T}}v + v^{\mathsf{T}}v.$$
 (25)

Using the expansion of Equation [\(25\)](#page-13-3), we can rewrite the MSE terms of Equation [\(22\)](#page-13-4) and Equation [\(23\)](#page-13-5) as follows:

$$\|\epsilon_{\theta}(x_t, t, c) - \epsilon\|_2^2 = \epsilon_{\theta}^{per} \epsilon_{\theta}^{per} - 2\epsilon_{\theta}^{per} \epsilon_{\theta}^{\tau} \epsilon + \epsilon^{\tau} \epsilon, \tag{26}$$

$$\|\epsilon_{\theta}(x_t^{pr}, t, c^{pr}) - \epsilon\|_2^2 = \epsilon_{\theta}^{ppl} \epsilon_{\theta}^{ppl} - 2\epsilon_{\theta}^{ppl} \epsilon_{\theta}^{rpl} + \epsilon_{\theta}^{rpl} \epsilon_{\theta}^{rpl},$$
(27)

where ϵ per <sup>θ</sup> = ϵθ(xt, t, c) and ϵ ppl <sup>θ</sup> = ϵθ(x pr t , t, cpr). For notational simplicity in the subsequent derivations, we will omit the input variables (*e.g.* xt, t, c) and use the superscripts *per* and *ppl* to

distinguish both terms in L per simple and Lppl respectively. Now, substituting the expanded forms from Equation [\(26\)](#page-13-6) and Equation [\(27\)](#page-13-7) back into L per simple and Lppl, we can rewrite these.

For L per simple, using Equation [\(26\)](#page-13-6):

$$\mathcal{L}_{simple}^{per} = \mathbb{E}_{x_0, t, c, \epsilon \sim \mathcal{N}(0, I)} [\epsilon_{\theta}^{per} \epsilon_{\theta}^{per} - 2\epsilon_{\theta}^{per} \epsilon_{\theta}^{rer} \epsilon_{\theta}^{rer}]. \tag{28}$$

Moreover, by the linearity of expectation, which includes the property E[A + B] = E[A] + E[B] (additivity principle), we can distribute expectation as follows:

$$\mathcal{L}_{simple}^{per} = \mathbb{E}[\epsilon_{\theta}^{per} + \epsilon_{\theta}^{per}] - 2\mathbb{E}[\epsilon_{\theta}^{per} + \epsilon] + \mathbb{E}[\epsilon^{\top} \epsilon]. \tag{29}$$

Please note that for readability, we also omit the explicit subscript variables of the expectation.

Similarly, for Lppl, using Equation [\(27\)](#page-13-7) and linearity of expectation:

$$\mathcal{L}_{ppl} = \mathbb{E}_{x_0^{pr}, t, c^{pr}, \epsilon \sim \mathcal{N}(0, I)} [\epsilon_{\theta}^{ppl}^{\top} \epsilon_{\theta}^{ppl} - 2\epsilon_{\theta}^{ppl}^{\top} \epsilon + \epsilon^{\top} \epsilon]$$

$$= \mathbb{E}[\epsilon_{\theta}^{ppl}^{\top} \epsilon_{\theta}^{ppl}] - 2\mathbb{E}[\epsilon_{\theta}^{ppl}^{\top} \epsilon] + \mathbb{E}[\epsilon^{\top} \epsilon].$$
(30)

These expanded expressions (Equation [\(29\)](#page-14-0) and Equation [\(30\)](#page-14-1)) simplify the subsequent gradient derivations. To compute the gradients of these loss functions, we will differentiate the terms within the expectation, which is permissible under suitable regularity conditions by applying the *Leibniz Rule*. We first consider the derivatives of the core components that appear inside the expectations, with respect to the model parameters θ.

<span id="page-14-1"></span><span id="page-14-0"></span>
$$\nabla_{\theta}(\epsilon_{\theta}^{\top}\epsilon_{\theta}) = 2J_{\theta}^{\top}\epsilon_{\theta},\tag{31}$$

$$\nabla_{\theta}(\epsilon_{\theta}^{\top}\epsilon) = J_{\theta}^{\top}\epsilon, \tag{32}$$

<span id="page-14-3"></span><span id="page-14-2"></span>
$$\nabla_{\theta}(\epsilon^{\top}\epsilon) = 0, \tag{33}$$

where J<sup>θ</sup> = ∂θ ϵ<sup>θ</sup> and ϵ is independent of θ, the derivative of any term solely dependent on ϵ (*i.e.* ϵ <sup>⊤</sup>ϵ) with respect to θ is zero. Using these results, we can now express the gradient of MSE loss term inside the expectations. For the term in L per simple:

$$\nabla_{\theta} \| \epsilon_{\theta}(x_{t}, t, c) - \epsilon \|_{2}^{2} = \frac{\partial}{\partial \theta} (\epsilon_{\theta}^{per} \epsilon_{\theta}^{per}) - 2 \frac{\partial}{\partial \theta} (\epsilon_{\theta}^{per} \epsilon_{\theta}^{rer}) + \frac{\partial}{\partial \theta} (\epsilon_{\theta}^{rer} \epsilon_{\theta}^{rer})$$

$$= 2J_{\theta}^{per} \epsilon_{\theta}^{rer} - 2J_{\theta}^{per} \epsilon_{\theta}^{rer}.$$
(34)

And for the term in Lppl:

$$\nabla_{\theta} \| \epsilon_{\theta}(x_{t}^{pr}, t, c^{pr}) - \epsilon \|_{2}^{2} = \frac{\partial}{\partial \theta} (\epsilon_{\theta}^{ppl} \epsilon_{\theta}^{ppl}) - 2 \frac{\partial}{\partial \theta} (\epsilon_{\theta}^{ppl} \epsilon_{\theta}) + \frac{\partial}{\partial \theta} (\epsilon^{\tau} \epsilon)$$

$$= 2J_{\theta}^{ppl} \epsilon_{\theta}^{\tau} - 2J_{\theta}^{ppl} \epsilon_{\theta}.$$
(35)

Since L per simple and Lppl are expectations of the terms whose gradients were derived in Equation [\(34\)](#page-14-2) and Equation [\(35\)](#page-14-3), and we apply the *Leibniz Rule*. This allows us to take the expectation of those gradients to find the final gradients of the loss functions:

$$\nabla_{\theta} \mathcal{L}_{simple}^{per} = 2\mathbb{E}[J_{\theta}^{per} {}^{\top} \epsilon] - 2\mathbb{E}[J_{\theta}^{per} {}^{\top} \epsilon], \tag{36}$$

<span id="page-14-4"></span>
$$\nabla_{\theta} \mathcal{L}_{ppl} = 2\mathbb{E}[J_{\theta}^{ppl} \kappa] - 2\mathbb{E}[J_{\theta}^{ppl} \kappa]. \tag{37}$$

Consequently, using the gradients (Equation [\(36\)](#page-14-4) and Equation [\(37\)](#page-14-5)), we can determine the convergence condition for Ladv with respect to θ as:

$$\nabla_{\theta} \mathcal{L}_{adv} = -\nabla_{\theta} \mathcal{L}_{simple}^{per} + \lambda \nabla_{\theta} \mathcal{L}_{ppl}$$

$$= -\{2\mathbb{E}[J_{\theta}^{per}^{\top} \epsilon] - 2\mathbb{E}[J_{\theta}^{per}^{\top} \epsilon]\} + \lambda \{2\mathbb{E}[J_{\theta}^{ppl}^{\top} \epsilon] - 2\mathbb{E}[J_{\theta}^{ppl}^{\top} \epsilon]\}.$$
(38)

Based on Equation [\(24\)](#page-13-2), rearranging Equation [\(38\)](#page-14-6) yields the final condition for Proposition [1:](#page-3-0)

<span id="page-14-7"></span><span id="page-14-6"></span><span id="page-14-5"></span>
$$\nabla_{\theta} \mathcal{L}_{simple}^{per} = \lambda \nabla_{\theta} \mathcal{L}_{ppl}. \tag{39}$$

The result in Equation [\(39\)](#page-14-7) indicates that for Ladv to converge, the gradients of L per simple and Lppl must point in the same direction, as λ > 0. This completes the proof of Proposition [1.](#page-3-0)

#### <span id="page-15-0"></span>A.2 Proof of Theorem 1

In Proposition [1,](#page-3-0) we establish that for Ladv to converge, a necessary condition is ∇θL per simple = λ∇θLppl. Based on this proposition, we now prove Theorem [1.](#page-4-3) Our proof will demonstrate how the aforementioned convergence condition (Equation [\(39\)](#page-14-7)) inherently conflicts with the goal of simultaneously decreasing both −Lper simple and Lppl.

To figure this out, we analyze the impact of a parameter update, ∆θ, on each loss term using a first-order *Taylor Expansion*. A parameter update ∆θ derived from a gradient descent step on Ladv, and assuming the scalar λ = 1 for simplicity in this derivation. ∆θ can be defined as:

<span id="page-15-1"></span>
$$\Delta\theta = -\eta \frac{\partial}{\partial\theta} \mathcal{L}_{adv},\tag{40}$$

where η > 0 is the learning rate. The change in L per simple due to ∆θ can be approximated by first-order *Taylor Expansion*:

$$\mathcal{L}_{simple}^{per}(\theta + \Delta \theta) - \mathcal{L}_{simple}^{per}(\theta) \approx \left[\frac{\partial}{\partial \theta} \mathcal{L}_{simple}^{per}(\theta)\right]^{\top} \Delta \theta$$

$$\approx \left[\frac{\partial}{\partial \theta} \mathcal{L}_{simple}^{per}(\theta)\right]^{\top} \left\{-\eta \left[-\frac{\partial}{\partial \theta} \mathcal{L}_{simple}^{per}(\theta) + \frac{\partial}{\partial \theta} \mathcal{L}_{ppl}(\theta)\right]\right\} \quad (41)$$

$$\approx \eta \|\frac{\partial}{\partial \theta} \mathcal{L}_{simple}^{per}(\theta)\|^{2} - \eta \left[\frac{\partial}{\partial \theta} \mathcal{L}_{simple}^{per}(\theta)\right]^{\top} \left[\frac{\partial}{\partial \theta} \mathcal{L}_{ppl}(\theta)\right],$$

where L(θ) means the the loss calculated with the parameter θ. Our objective is to minimize −Lper simple, which is equivalent to increasing L per simple. For this reason, the difference of L per simple in Equation [\(41\)](#page-15-1) is greater than zero. Using this condition, we can obtain the final inequality from Equation [\(41\)](#page-15-1) as:

<span id="page-15-2"></span>
$$\|\frac{\partial}{\partial \theta} \mathcal{L}_{simple}^{per}(\theta)\|^2 > \left[\frac{\partial}{\partial \theta} \mathcal{L}_{simple}^{per}(\theta)\right]^{\top} \left[\frac{\partial}{\partial \theta} \mathcal{L}_{ppl}(\theta)\right]. \tag{42}$$

This inequality (Equation [\(42\)](#page-15-2)) represents the condition under which the parameter update leads to an increase in L per simple. Similarly, we derive the impact of the parameter update ∆θ on Lppl.

$$\mathcal{L}_{ppl}(\theta + \Delta \theta) - \mathcal{L}_{ppl}(\theta) \approx \left[\frac{\partial}{\partial \theta} \mathcal{L}_{ppl}(\theta)\right]^{\top} \Delta \theta$$

$$\approx \left[\frac{\partial}{\partial \theta} \mathcal{L}_{ppl}(\theta)\right]^{\top} \left\{-\eta \left[-\frac{\partial}{\partial \theta} \mathcal{L}_{simple}^{per}(\theta) + \frac{\partial}{\partial \theta} \mathcal{L}_{ppl}(\theta)\right]\right\}$$

$$\approx \eta \left[\frac{\partial}{\partial \theta} \mathcal{L}_{ppl}(\theta)\right]^{\top} \left[\frac{\partial}{\partial \theta} \mathcal{L}_{simple}^{per}(\theta)\right] - \eta \left\|\frac{\partial}{\partial \theta} \mathcal{L}_{ppl}(\theta)\right\|^{2}.$$
(43)

To minimize Lppl, it should decrease, which means that the change approximated by Equation [\(43\)](#page-15-3) must be less than zero. We can also derive the condition as:

<span id="page-15-4"></span><span id="page-15-3"></span>
$$\left[\frac{\partial}{\partial \theta} \mathcal{L}_{ppl}(\theta)\right]^{\top} \left[\frac{\partial}{\partial \theta} \mathcal{L}_{simple}^{per}(\theta)\right] < \left\|\frac{\partial}{\partial \theta} \mathcal{L}_{ppl}(\theta)\right\|^{2}. \tag{44}$$

Equation [\(42\)](#page-15-2) and Equation [\(44\)](#page-15-4) share the common inner product term [ ∂θLppl(θ)]<sup>⊤</sup>[ ∂ ∂θL per simple(θ)]. Recall from Proposition [1,](#page-3-0) for converging Ladv, the gradients of the two terms must point in the same direction. This relationship allows us to remove the cosine term in the inner product (∵ cos(0) = 1). Based on this, we can rewrite the inequalities as below:

$$\left|\frac{\partial}{\partial \theta} \mathcal{L}_{simple}^{per}(\theta)\right| \cdot \left|\frac{\partial}{\partial \theta} \mathcal{L}_{ppl}(\theta)\right| < \left\|\frac{\partial}{\partial \theta} \mathcal{L}_{simple}^{per}(\theta)\right\|^{2}, \tag{45}$$

$$\left|\frac{\partial}{\partial \theta} \mathcal{L}_{ppl}(\theta)\right| \cdot \left|\frac{\partial}{\partial \theta} \mathcal{L}_{simple}^{per}(\theta)\right| < \left\|\frac{\partial}{\partial \theta} \mathcal{L}_{ppl}(\theta)\right\|^{2}.$$
(46)

We can further rearrange the above inequality as:

<span id="page-15-5"></span>
$$\left|\frac{\partial}{\partial \theta} \mathcal{L}_{ppl}(\theta)\right| < \left|\frac{\partial}{\partial \theta} \mathcal{L}_{simple}^{per}(\theta)\right|,\tag{47}$$

<span id="page-15-6"></span>
$$\left|\frac{\partial}{\partial \theta} \mathcal{L}_{ppl}(\theta)\right| > \left|\frac{\partial}{\partial \theta} \mathcal{L}_{simple}^{per}(\theta)\right|. \tag{48}$$

This rearrangement relies on the assumption that both individual gradients are non-zero, *i.e.* ∂ ∂θL per simple(θ)| > 0 and | ∂ ∂θLppl(θ)| > 0. This assumption holds for any θ that is not already a local optimum for both individual objectives.

Recall the objectives: to increase L per simple, the condition in Equation [\(47\)](#page-15-5) must hold for the current parameter update. On the other hand, to decrease Lppl, the condition in Equation [\(48\)](#page-15-6) must satisfy for the same parameter update. These two conditions are mutually exclusive. This contradiction demonstrates that if the system is at a point satisfying the convergence condition for Ladv (Proposition 1), the objective of simultaneously decreasing −Lper simple and Lppl cannot be achieved. Therefore, the naïve approach Ladv, as composed of these conflicting objectives under its own convergence condition, generally fails to converge to a point that effectively optimizes both. This completes the proof of Theorem [1.](#page-4-3)

#### <span id="page-16-0"></span>A.3 Derivation of Objective

Starting from Equation [\(13\)](#page-4-4) in the main paper, the loss function for the reward function r(·) can be expressed as:

<span id="page-16-3"></span>
$$\mathcal{L}_r = -\mathbb{E}_{x_0^+, x_0^-} \log \sigma(r(x_0^+) - r(x_0^-)). \tag{49}$$

Reinforcement Learning from Human Feedback (RLHF) aims to maximize the distribution pθ(x0) under regularization using KL-divergence:

$$\max_{p_{\theta}} \mathbb{E}_{x_0} r(x_0) - \beta D_{KL}(p_{\theta}(x_0) || p_{\phi}(x_0)), \tag{50}$$

where ϕ is reference distribution. From Equation [\(50\)](#page-16-1), we can obtain a unique solution p ∗ θ (x0):

<span id="page-16-1"></span>
$$p_{\theta}^{*}(x_0) = p_{\phi}(x_0) \exp(r(x_0)/\beta)/Z, \tag{51}$$

where Z = P x<sup>0</sup> pϕ(x0) exp(r(x0)/β) is the partition function. The reward function can be rewritten using Equation [\(51\)](#page-16-2):

<span id="page-16-4"></span><span id="page-16-2"></span>
$$r(x_0) = \beta \log \frac{p_{\theta}^*(x_0)}{p_{\phi}(x_0)} + \beta \log Z.$$
 (52)

From Equation [\(49\)](#page-16-3) and Equation [\(52\)](#page-16-4), the reward objective is:

$$\mathcal{L}_{r} = -\mathbb{E}_{x_{0}^{+}, x_{0}^{-}} \left[\log \sigma(\beta \log \frac{p_{\theta}^{*}(x_{0}^{+})}{p_{\phi}(x_{0}^{+})} - \beta \log \frac{p_{\theta}^{*}(x_{0}^{-})}{p_{\phi}(x_{0}^{-})}\right)\right].$$
 (53)

However, this objective cannot directly applied to diffusion models since the parameterized distribution pθ(x0) is intractable. Therefore, Diffusion-DPO [\[32\]](#page-12-3) introduces the latents x1:<sup>T</sup> to consider possible diffusion paths from x<sup>T</sup> to x0, and re-defines the reward function as follows:

<span id="page-16-6"></span><span id="page-16-5"></span>
$$r(x_0) = \mathbb{E}_{p_{\theta}(x_{1:T}|x_0)} R(x_0). \tag{54}$$

Following Equation [\(54\)](#page-16-5), Equation [\(50\)](#page-16-1) can also be written as follows:

$$\max_{p_{\theta}} \mathbb{E}_{x_{0:T} \sim p(x_{0:T})} r(x_0) - \beta D_{KL}(p_{\theta}(x_{0:T}) || p_{\phi}(x_{0:T})).$$
 (55)

Similar to the expansion from Equation [\(50\)](#page-16-1) to Equation [\(53\)](#page-16-6), we can obtain the reward objective as:

$$\mathcal{L}_{r} = -\mathbb{E}_{x_{0}^{+}, x_{0}^{-}} \left[ \log \sigma \left\{ \mathbb{E}_{p_{\theta}(x_{1:T}^{+}|x_{0}^{+}), p_{\theta}(x_{1:T}^{-}|x_{0}^{-})} (\beta \log \frac{p_{\theta}^{*}(x_{0:T}^{+})}{p_{\phi}(x_{0:T}^{+})} - \beta \log \frac{p_{\theta}^{*}(x_{0:T}^{-})}{p_{\phi}(x_{0:T}^{-})} \right) \right\} \right].$$
 (56)

Since − log σ(·) is a convex function, we can leverage *Jensen's inequality*:

$$\mathcal{L}_{r} \leq -\mathbb{E}_{x_{0}^{+}, x_{0}^{-}, p_{\theta}(x_{1:T}^{+}|x_{0}^{+}), p_{\theta}(x_{1:T}^{-}|x_{0}^{-})} \\ \left[\log \sigma \left\{\beta \log \frac{p_{\theta}^{*}(x_{0:T}^{+})}{p_{\phi}(x_{0:T}^{+})} - \beta \log \frac{p_{\theta}^{*}(x_{0:T}^{-})}{p_{\phi}(x_{0:T}^{-})}\right\}\right].$$
(57)

Note that pθ(x1:<sup>T</sup> |x0) is intractable. Therefore, we utilize q(x1:<sup>T</sup> |x0) to approximate pθ(x1:<sup>T</sup> |x0):

$$\mathcal{L}_{r} \leq -\mathbb{E}_{x_{0}^{+}, x_{0}^{-}, q(x_{1:T}^{+}|x_{0}^{+}), q(x_{1:T}^{-}|x_{0}^{-})} [\log \sigma \{\beta \log \frac{p_{\theta}^{*}(x_{0:T}^{+})}{p_{\phi}(x_{0:T}^{+})} - \beta \log \frac{p_{\theta}^{*}(x_{0:T}^{-})}{p_{\phi}(x_{0:T}^{-})} \}].$$
 (58)

Since pθ(x0:<sup>T</sup> ) = pθ(x<sup>T</sup> ) Q<sup>⊤</sup> <sup>t</sup>=1 pθ(xt−1|xt) can be expressed as a Markov chain, we can derive the above equation as:

$$\mathcal{L}_{r} \leq -\mathbb{E}_{x_{0}^{+}, x_{0}^{-}, q(x_{1:T}^{+}|x_{0}^{+}), q(x_{1:T}^{-}|x_{0}^{-})} \\
\left[\log \sigma \left\{\beta \sum_{t=1}^{\top} \log \frac{p_{\theta}^{*}(x_{t-1}^{+}|x_{t}^{+})}{p_{\phi}(x_{t-1}^{+}|x_{t}^{+})} - \log \frac{p_{\theta}^{*}(x_{t-1}^{-}|x_{t}^{-})}{p_{\phi}(x_{t-1}^{-}|x_{t}^{-})} \right\}\right], \\
= -\mathbb{E}_{x_{0}^{+}, x_{0}^{-}, q(x_{1:T}^{+}|x_{0}^{+}), q(x_{1:T}^{-}|x_{0}^{-})} \\
\left[\log \sigma \left\{\beta \sum_{t=1}^{\top} \left(\log \frac{p_{\theta}^{*}(x_{t-1}^{+}|x_{t}^{+})}{q(x_{t-1}^{+}|x_{t}^{+})} - \log \frac{p_{\phi}(x_{t-1}^{+}|x_{t}^{+})}{q(x_{t-1}^{+}|x_{t}^{+})} \right) - \left(\log \frac{p_{\theta}^{*}(x_{t-1}^{-}|x_{t}^{-})}{q(x_{t-1}^{-}|x_{t}^{-})} - \log \frac{p_{\phi}(x_{t-1}^{-}|x_{t}^{-})}{q(x_{t-1}^{-}|x_{t}^{-})} \right)\right\}\right], \\
= -\mathbb{E}_{x_{0}^{+}, x_{0}^{-}, q(x_{1:T}^{+}|x_{0}^{+}), q(x_{1:T}^{-}|x_{0}^{-})} \\
\left[\log \sigma \left\{\beta \sum_{t=1}^{\top} \left(D_{KL}(q(x_{t-1}^{+}|x_{t}^{+})|p_{\theta}^{*}(x_{t-1}^{+}|x_{t}^{+})) - D_{KL}(q(x_{t-1}^{+}|x_{t}^{+})|p_{\theta}^{*}(x_{t-1}^{+}|x_{t}^{+})) - \left(D_{KL}(q(x_{t-1}^{-}|x_{t}^{-})|p_{\theta}^{*}(x_{t-1}^{-}|x_{t}^{-})) - D_{KL}(q(x_{t-1}^{-}|x_{t}^{-})|p_{\phi}(x_{t-1}^{-}|x_{t}^{-}))\right)\right\}\right].$$
(59)

By leveraging ELBO, we can obtain our final objective, Equation [\(14\)](#page-4-5) in the main paper:

$$\mathcal{L}_{DPO} = -\mathbb{E}_{x_0^+, x_0^-, c, t, \epsilon \sim N(0, I)}$$

$$\log \sigma(-\beta((\|\epsilon_{\theta}(x_t^+, t, c) - \epsilon\|_2^2)$$

$$- \|\epsilon_{\phi}(x_t^+, t, c) - \epsilon\|_2^2)$$

$$- (\|\epsilon_{\theta}(x_t^-, t, c) - \epsilon\|_2^2$$

$$- \|\epsilon_{\phi}(x_t^-, t, c) - \epsilon\|_2^2))).$$
(60)

<span id="page-17-0"></span>![](_page_17_Figure_4.jpeg)

Figure 4: FID variation during the training with Ladv. We measured the image quality via FID score [\[9\]](#page-10-9) on COCO 2014 [\[17\]](#page-11-14) validation dataset. We also plot the FID score of Stable Diffusion 1.5 and APDM.

<span id="page-18-1"></span>Table 10: Quantitative comparison on protection with image transformations. We compared APDM with transformed images. For data poisoning baselines, we applied image transformation to perturbed images and we personalized Stable Diffusion on these transformed images. For APDM, we protected diffusion models on clean images and we conduct personalization on images that is transformed from clean images.

| Methods              | Transform. | DINO (↓) |        |        | BRISQUE (↑) |       |       |
|----------------------|------------|----------|--------|--------|-------------|-------|-------|
|                      |            | "person" | "dog"  | Avg.   | "person"    | "dog" | Avg.  |
| DreamBooth [28]      | -          | 0.6994   | 0.6056 | 0.6525 | 11.27       | 22.33 | 16.80 |
|                      | -          | 0.5752   | 0.4247 | 0.4999 | 19.52       | 28.60 | 24.06 |
| AdvDM [16]           | flip       | 0.5436   | 0.4538 | 0.4987 | 24.37       | 27.07 | 25.72 |
|                      | blur       | 0.6417   | 0.4524 | 0.5470 | 18.28       | 26.35 | 22.32 |
|                      | -          | 0.5254   | 0.4106 | 0.4680 | 26.90       | 30.23 | 28.56 |
| Anti-DreamBooth [30] | flip       | 0.5976   | 0.4665 | 0.5321 | 26.76       | 29.19 | 27.97 |
|                      | blur       | 0.5487   | 0.4414 | 0.4951 | 24.37       | 28.91 | 26.64 |
|                      | -          | 0.4448   | 0.4374 | 0.4411 | 23.73       | 31.64 | 27.69 |
| SimAC [34]           | flip       | 0.5083   | 0.4475 | 0.4779 | 26.56       | 29.46 | 28.01 |
|                      | blur       | 0.5323   | 0.4390 | 0.4856 | 20.40       | 31.27 | 25.83 |
|                      | -          | 0.6556   | 0.5120 | 0.5838 | 22.61       | 30.20 | 26.41 |
| PAP [33]             | flip       | 0.6564   | 0.5139 | 0.5852 | 22.51       | 27.81 | 25.16 |
|                      | blur       | 0.6708   | 0.5222 | 0.5965 | 24.37       | 27.83 | 26.10 |
|                      | -          | 0.1375   | 0.0959 | 0.1167 | 40.25       | 60.74 | 50.50 |
| APDM (Ours)          | flip       | 0.1714   | 0.1194 | 0.1454 | 39.13       | 40.34 | 39.74 |
|                      | blur       | 0.1042   | 0.0823 | 0.0933 | 40.47       | 45.13 | 42.80 |

<span id="page-18-2"></span>Table 11: Protection performance on other subjects. In addition to experiments in the main paper, we evaluated APDM on different subjects. We tried to prevent personalization on *"cat"*, *"sneaker"*, *"glasses"*, and *"clock"*.

| Methods         |        |           | DINO (↓)  |         |       |           | BRISQUE (↑) |         |
|-----------------|--------|-----------|-----------|---------|-------|-----------|-------------|---------|
|                 | "cat"  | "sneaker" | "glasses" | "clock" | "cat" | "sneaker" | "glasses"   | "clock" |
| DreamBooth [28] | 0.4903 | 0.6110    | 0.6961    | 0.5359  | 25.32 | 23.14     | 19.01       | 13.82   |
| APDM (Ours)     | 0.0414 | 0.2276    | 0.2893    | 0.1969  | 47.65 | 35.23     | 31.75       | 32.01   |

### <span id="page-18-0"></span>B Additional Experiments

Empirical Results about the limitation of Naïve Approach. In Section [A,](#page-13-1) we theoretically demonstrated the fundamental limitations of naïve approach. In the following part, we empirically validate those findings. We applied the loss function of naïve approach, Ladv (Equation [\(21\)](#page-13-8)), to Stable Diffusion 1.5 with Nprotect = 800, as APDM. We measured FID score every 100 iterations. As shown in Figure [4,](#page-17-0) as the optimization progresses, the FID score consistently increases across all tested λ values. This degradation in quality occurs because the primary objective of Ladv, minimizing −Lper simple (*i.e.* actively erasing related to the target for anti-personalization), becomes overly dominant. Even though Lppl is intended to preserve the generation performance, its effectiveness is clearly restricted by the optimized condition of −Lper simple. This result aligns with our Theorem [1,](#page-4-3) which suggests that the loss of each term in Ladv cannot be satisfied simultaneously. Furthermore, when the weight λ increases, one might expect a better preservation of the generative performance. Although FID scores are relatively low with high λ values (*e.g.* λ = 10.0, 15.0) in initial iterations, they still remain significantly high and can exhibit instability as training progresses. This suggests that our theorem is still valid in various λ.

Protection with Image Transformations. In Figure [3](#page-7-0) and Table [1](#page-6-1) of the main paper, we compared APDM with baselines considering the existence and quantity of clean images. Additionally, we also compared APDM with baselines using transformed images such as flipping and blurring. Table [10](#page-18-1) demonstrates that baselines fail to effectively protect personalization when transformations are applied to perturbed images. In contrast, APDM exhibits robustness even under such image transformations.

<span id="page-19-1"></span>![](_page_19_Figure_0.jpeg)

Figure 5: Protection on other subjects. We attempted to protect personalization on *"cat"*, *"sneaker"*, *"glasses"*, and *"clock"*.

Protection on Other Subjects. In the experiments presented in the main paper, we primarily considered two types of subjects: *"person"* and *"dog"*. In Table [11](#page-18-2) and Figure [5,](#page-19-1) we explored the prevention of personalization on other subjects, such as *"cat"*, *"sneaker"*, *"glasses"*, and *"clock"*, demonstrating that APDM can be generally applied to protection of various subjects.

# <span id="page-19-0"></span>C Generalizability of APDM

Custom Diffusion. In our main paper, we mainly consider DreamBooth [\[28\]](#page-11-4) as a personalization method. Additionally, we utilized Custom Diffusion [\[14\]](#page-10-4) as a variation of the personalization approach. In Table [12,](#page-20-0) we present the results of the Custom Diffusion experiments, and we conducted protection about *"person"* and *"dog"* similar to our main paper. The results demonstrated that

<span id="page-20-0"></span>Table 12: Protection performance of APDM on different personalization method, Custom Diffusion [\[14\]](#page-10-4). Unlike the experiments in the main paper, which used DreamBooth for personalization, we replaced the personalization method with Custom Diffusion.

| Methods               | DINO (↓) |        | BRISQUE (↑) |       |  |
|-----------------------|----------|--------|-------------|-------|--|
|                       | "person" | "dog"  | "person"    | "dog" |  |
| Custom Diffusion [14] | 0.5320   | 0.5460 | 16.03       | 8.98  |  |
| APDM (Ours)           | 0.2158   | 0.3202 | 34.61       | 33.09 |  |

<span id="page-20-1"></span>Table 13: Protection performance of APDM on different Stable Diffusion version, Stable Diffusion 2.1. In the experiments of the main paper, we primarily used Stable Diffusion 1.5. Additionally, we evaluated APDM based on different Stable Diffusion version.

| Methods         | DINO (↓) |        | BRISQUE (↑) |       |
|-----------------|----------|--------|-------------|-------|
|                 | "person" | "dog"  | "person"    | "dog" |
| DreamBooth [28] | 0.5773   | 0.5293 | 13.99       | 23.03 |
| APDM (Ours)     | 0.2739   | 0.2178 | 39.72       | 42.69 |

<span id="page-20-2"></span>Table 14: Protection performance on different unique identifier for personalization. We conducted protection on *"a photo of sks person"* or *"a photo of sks dog"* and we tried to personalize diffusion models on *"a photo of t@t person"* or *"a photo of t@t dog"*.

| Methods         | ∗<br>[V<br>] | DINO (↓) |        | BRISQUE (↑) |       |
|-----------------|--------------|----------|--------|-------------|-------|
|                 |              | "person" | "dog"  | "person"    | "dog" |
| DreamBooth [28] | "t@t"        | 0.6774   | 0.4668 | 16.64       | 28.49 |
| APDM (Ours)     | "sks"→"t@t"  | 0.3958   | 0.1981 | 29.90       | 40.69 |

<span id="page-20-3"></span>Table 15: Protection performance for diverse text prompts. Unlike the experiments in the main paper, we evaluated APDM on diverse test prompts. Protection and personalization are conducted using *"a photo of [V\*] person"* or *"a photo of [V\*] dog"*, and we sampled images using the different set of text prompts.

| Methods         | DINO (↓) |        | BRISQUE (↑) |       |
|-----------------|----------|--------|-------------|-------|
|                 | "person" | "dog"  | "person"    | "dog" |
| DreamBooth [28] | 0.4081   | 0.4233 | 12.57       | 29.65 |
| APDM (Ours)     | 0.1357   | 0.1564 | 36.40       | 41.66 |

APDM can successfully prevent the personalization of Custom Diffusion, and show the applicability of APDM to other personalization methods.

Stable Diffusion 2.1. APDM prevents personalization at the model level, and its applicability to different versions of the Stable Diffusion model is also important. In Table [13,](#page-20-1) we present experiments conducted on Stable Diffusion 2.1 to demonstrate the effectiveness of our approach on other diffusion models. We applied APDM to Stable Diffusion 2.1 and performed personalization with clean images using DreamBooth. The results indicate that APDM also performs robustly on Stable Diffusion 2.1, showing that our method is not restricted to a specific version of the diffusion model.

Prompt (Identifier) Mismatch. When an attacker performs personalization, they may use a different unique identifier (*e.g. "t@t"*) to capture the target subject. For example, during the protection process, we only show *"a photo of sks person"*, while a different unique identifier may be used for personalization, such as *"a photo of t@t person"*. Similar to Van Le et al. [\[30\]](#page-11-6), we also considered this prompt mismatch. As shown in Table [14,](#page-20-2) APDM can successfully protect against personalization attempts using *"t@t"*. APDM successfully confuses the personalization process, preventing the identifier from capturing the target subject (*i.e.* identity).

<span id="page-21-0"></span>![](_page_21_Figure_0.jpeg)

Figure 6: Protection performance for diverse text prompts. We visualized the generated outputs from diverse text prompts, such as *"a [V\*] person in the snow"* and *"a [V\*] person wearing a red hat"*.

Protection on Diverse Text Prompts. In the experiments presented in the main paper, we utilized simple text prompts for inference, such as *"a photo of [V\*] person"* and *"a portrait of [V\*] person."* In contrast to these experiments, we evaluated APDM using diverse prompts, such as *"a photo of [V\*] person in the jungle"* and *"a [V\*] person with a mountain background."* We adopted text prompts from the DreamBooth dataset [\[28\]](#page-11-4). Figure [6](#page-21-0) and Table [15](#page-20-3) illustrate that APDM successfully prevents personalization, even under diverse prompt variations that differ from the text prompts used during the protection procedure. This result highlights that APDM is even robust to diverse text prompt variation.

<span id="page-22-2"></span>![](_page_22_Figure_0.jpeg)

Figure 7: A sample interface for our user study. Left term is the descriptions of explanation about study. Middle term is a given reference images which used to capture the identity from participants. Right term is choices.

<span id="page-22-3"></span>Table 16: Results of user study. We count the percentage of votes for the comparisons and our method respectively. Every participants selected a sample that looks most different from the clean images.

| Methods    | Anti-DreamBooth [30] | SimAC [34] | PAP [33] | APDM (Ours) |
|------------|----------------------|------------|----------|-------------|
| Protection | 7.08 %               | 5.83 %     | 1.04 %   | 86.04 %     |

# <span id="page-22-1"></span>D User Study

We conducted a user study to evaluate the preference of various protection methods in preventing subject recognition. The specific questions and interface are illustrated in Figure [7.](#page-22-2) We presented four reference images for each subject to provide participants with clear identity information. After viewing these, each participant chose an image based on the following question:

> *Which of the candidate images (A, B, C, D) is the HARDEST to recognize as depicting the SAME PERSON shown in the reference images?*

Candidate images were generated using a personalized diffusion model with different protection methods applied. We utilized Stable Diffusion 1.5 and Dreambooth [\[28\]](#page-11-4) as personalization method, which is the same as our experimental setting in main paper. In this user study, we compared our proposed method, APDM, against Anti-Dreambooth [\[30\]](#page-11-6), SimAC [\[34\]](#page-12-0), and PAP [\[33\]](#page-12-1). For comparisons, we first generated perturbed images using each approach, and conducted personalization with these perturbed images. For APDM, we applied personalization using the model protected by our method. After personalization, all images were generated using the prompt *"a photo of [V\*] person"*. To ensure fairness, the same randomly sampled seed was used for generating all candidate images. The image sequence and the arrangement of choices are randomized to eliminate any bias.

We collected responses from 25 voluntary adult participants regardless of gender. Participants were compensated \$0.125 USD per question, totaling \$2.50 USD, corresponding to hourly rate of \$7.26 USD. On average, participants completed the study in about 20 minutes. We did not collect any personal information from the participants.

As shown in Table [16,](#page-22-3) APDM achieved significantly higher user preference (*i.e.* was selected more often as the hardest to recognize) than other comparisons. These results indicate that our method not only addresses limitations of data-centric approaches but also achieves a substantial improvement in protection performance.

# <span id="page-22-0"></span>E Additional Qualitative Results

Additional Protection Results. In Figure [3](#page-7-0) and Table [1](#page-6-1) of the main paper, we conducted quantitative and qualitative experiments, respectively. We attached additional qualitative results in Figure [8](#page-23-0) and Figure [9,](#page-24-0) including protection results on various subjects of person and dog. The experimental results highlight again that APDM can effectively protect personalization against diverse subjects, producing images of a lot of artifacts or containing different instances.

<span id="page-23-0"></span>![](_page_23_Figure_0.jpeg)

Figure 8: Additional Qualitative Results on Protection (*"person"*).

<span id="page-24-0"></span>![](_page_24_Figure_0.jpeg)

Figure 9: Additional Qualitative Results on Protection (*"dog"*).

# <span id="page-25-0"></span>F Additional Explanation of Motivation

Figure [1](#page-1-0) in our main paper presents the motivation for our work and briefly describes key issues, including the impractical assumptions of existing approaches, easy circumvention, user burdens, and conflict with regularization. This section provides a more detailed explanation of these limitations to facilitate clearer understanding.

We first criticize the impracticality of the existing literature. In daily life, individuals frequently take pictures or are photographed. For example, they often take selfies for social media or capture images of their identification documents. Such images, which we refer to as *"User's Photos"* (as depicted in Figure [1](#page-1-0) of our main paper), are those that users are consciously aware of and possess. Consequently, users have the opportunity to apply protection methods (*i.e.* data poisoning approach) to these photos if they want. In contrast, *"Unintended Capture"* refers to images of individuals taken without their explicit recognition or control over their subsequent use. This scenario presents a critical vulnerability, as these unintentionally captured images can be exploited as unprotected, *"clean"* data by malicious users.

As shown in Table [1](#page-6-1) of our main paper, the presence of clean (unprotected) images can significantly degrade the effectiveness of data poisoning techniques, allowing for easy bypass of protection. Furthermore, even when images are perturbed (*i.e.* poisoned), their protective effect is vulnerable to various common image transformations that frequently occur in real-world scenarios (as also shown in Table [10\)](#page-18-1). These transformations can weaken or negate the intended poisoning effect. These limitations reveal that, without strong (and often impractical) assumptions about the unavailability of clean images or the absence of transformations, existing protection methods exhibit restricted performance.

Regarding the user burden associated with implementing such techniques, most individuals are unfamiliar with implementation of AI technique. Establishing appropriate hardware environments (*e.g.* GPU servers) and configuring complex software environments (*e.g.* managing numerous libraries and their dependencies) present a significant initial hurdle. Even if these challenges are overcome, non-expert users still face substantial obstacles in utilizing protection methods. These include a lack of fundamental understanding of the protection mechanisms themselves, insufficient understanding in necessary programming languages (such as Python), and inadequate debugging skills to troubleshoot issues. These technical components are crucial for successful implementation of protection methods, yet their complexity also acts as a significant barrier, preventing widespread adoption by the general public.

The user-centric nature of existing data poisoning methods inherently conflicts with privacy regulations such as the General Data Protection Regulation (GDPR) [\[31\]](#page-11-9). The GDPR places the duty for privacy protection on service providers (*i.e.* model owners) to ensure a user's request. However, data poisoning approaches are ill-suited for service providers to fulfill this responsibility. These methods typically operate at the individual image level, requiring modifications to user data before they interact with the model. Service providers, in contrast, primarily manage the model itself. This operational disparity highlights why such user-side defenses are impractical for providers, underscoring the critical need for alternative approaches. To alleviate this, we propose a novel framework APDM, which empowers service providers to effectively manage and enforce anti-personalization directly within their systems, aligning with their responsibilities under privacy regulations and enabling a more scalable and reliable means of privacy protection.

# <span id="page-25-1"></span>G Limitation and Broader Impacts

In this work, we focused on protecting the personalization of a specific subject at the model level. APDM offers a significant step towards more robust and practical privacy protection in personalization of diffusion model. By enabling direct, model-level anti-personalization, it empowers service providers to better comply with privacy regulations and reduces the burden on individual users to protect their own data. This could foster greater trust and safer use of powerful generative models in various applications.

While APDM effectively safeguards personalization for a single subject, real-world scenarios often require the protection of multiple subjects simultaneously. Additionally, there may be a need to incorporate protection for new subjects into models that are already safeguarded. Addressing these challenges presents an opportunity for future research, including multi-concept personalization protection and continual personalization safeguarding.