# White-box Membership Inference Attacks against Diffusion Models

**文档类型**：OCR 精修版原文（正文主干修正版）

**说明**：本稿基于 PaddleOCR 结果与 PDF 原文联合整理，保留正文主干、关键公式、主要表格与核心插图，便于在飞书中连续阅读。

**GitHub PDF**：https://github.com/DeliciousBuding/DiffAudit-Research/blob/main/references/materials/white-box/2025-popets-white-box-membership-inference-diffusion-models.pdf

**论文报告**：https://www.feishu.cn/docx/CCgPdOGSHohFoqxRGaZcbCGWnHb

**开源实现**：https://github.com/py85252876/GSA

---

# White-box Membership Inference Attacks against Diffusion Models

Yan Pang

University of Virginia

yanpang@virginia.edu

Tianhao Wang

University of Virginia

tianhao@virginia.edu

Xuhui Kang

University of Virginia

qhv6ku@virginia.edu

Mengdi Huai

Iowa State University

mdhuai@iastate.edu

## Abstract

Diffusion models have begun to overshadow GANs and other generative models in industrial applications due to their superior image generation performance. The complex architecture of these models furnishes an extensive array of attack features. In light of this, we aim to design membership inference attacks (MIAs) catered to diffusion models. We first conduct an exhaustive analysis of existing MIAs on diffusion models, taking into account factors such as black-box/white-box models and the selection of attack features. We found that white-box attacks are highly applicable in real-world scenarios, and the most effective attacks presently are white-box. Departing from earlier research, which employs model loss as the attack feature for white-box MIAs, we employ model gradients in our attack, leveraging the fact that these gradients provide a more profound understanding of model responses to various samples. We subject these models to rigorous testing across a range of parameters, including training steps, timestep sampling frequency, diffusion steps, and data variance. Across all experimental settings, our method consistently demonstrated near-flawless attack performance, with attack success rate approaching 100% and attack AUCROC near 1.0. We also evaluated our attack against common defense mechanisms, and observed our attacks continue to exhibit commendable performance. We provide access to our code $ ^{1} $.

##### Keywords

machine learning privacy, membership inference attack

## 1 Introduction

Yang Zhang

CISPA Helmholtz Center for

Information Security

zhang@cispa.de

Recently, diffusion models have gained significant attention, and various applications are emerging. These models [21, 39, 42, 43, 48, 53, 56] rely on a progressive denoising process to generate images, resulting in improved image quality compared to previous models like GANs [7, 10] and VAEs [32]. Leading models primarily fall into two categories. The first category encompasses diffusion-based architectures such as GLIDE [39], Stable Diffusion model [45], DALL-E 2 [42], and Imagen [48]. The second category comprises representative sequence-to-sequence models like DALL-E [43], Parti [66], and CogView [11]. Current text-to-image models possess the capability to generate exquisite and intricately detailed images based on textual inputs, finding extensive applications across various domains such as graphic design and illustration. While diffusion models can be employed to synthesize distinct artistic styles, they often necessitate training on extensive sets of sensitive data. Thus, investigating membership inference attacks (MIAs) [52], which aim to determine whether specific samples are present in the diffusion model's training data, is of paramount importance.


Numerous studies have been conducted on classification models [3, 31, 33, 50, 52, 63, 64], GANs [5, 19, 20, 26, 37], and others. However, due to the unique training and inference method of diffusion models, previous attack methods [64] are no longer suitable. For instance, in classification models, the model's final output is generally used as the attack feature, relying on the model's overfitting to the training data, which leads to differences in classification confidence. Additionally, previous work on generative models such as GANs focused on utilizing the discriminator for determination [37]. Since the diffusion model does not have a discriminator, which makes it different from GANs, a new attack method must be specifically designed for diffusion models.

Some preliminary efforts have been devoted to conducting MIA on diffusion models [4, 27, 35]. However, it merits our attention that these investigations, akin to many others in this domain, predominantly concentrate on loss- and threshold-based attacks. We postulate that different layers in a neural network learn distinct features and, therefore, store varying amounts of information [65]. Evaluations based solely on loss could potentially overlook substantial information [37]. Consequently, a more comprehensive perspective of the model's response to a sample could be attained by considering gradient information from each layer post-backpropagation in addition to the loss incurred by the model.

The main challenges of utilizing gradients for MIAs are the excessive computation overhead and the overfitting issue of training the attack model (given the large size of diffusion models, gradients could have millions of dimensions). We carefully analyze ways to reduce dimensionality and propose a framework incorporating subsampling and aggregation. We call our framework Gradient attack based on Subsampling and Aggregation (GSA) and initiate


two instances, GSA₁ and GSA₂, demonstrating different trade-offs within the GSA framework.

To ensure the comprehensiveness and integrity of our investigation, we conduct experiments on the fundamental unconditional Denoising Diffusion Probabilistic Models (DDPM) [21] and the state-of-the-art Imagen model [48], which presently leads the text-to-image domain. CIFAR-10 and ImageNet datasets are utilized to train the unconditional diffusion models, while the MS COCO dataset is employed to train the Imagen model. We further explore the influence of varying parameters on the effectiveness of the attack. Ultimately, we validate the effectiveness of our attack strategy with a near 100% success rate, thus underscoring the imperative need for addressing the security aspects of diffusion models.

The contributions of our work are two-fold:

• We have analyzed membership inference attacks on diffusion models in existing research. Moreover, we have conceptualized our attack for new practical scenarios and conducted analyses across various dimensions, such as timesteps and model layers.

- We conducted experiments on three datasets using the traditional DDPM model and the cutting-edge text-to-image model, Imagen. Our results demonstrate extremely high accuracy across four evaluation metrics, underscoring the effectiveness of using gradients as attack features.

Roadmap. In Section 2, we introduce the background of diffusion models and delve into membership inference attacks. We also discuss the challenges we encountered and review existing attacks on diffusion models. In Section 3, we present our attack strategy. The experimental setup is detailed in Section 4, while Section 5 showcases the results of these experiments. In Section 6, we apply our GSA framework at the model layer level, demonstrating a further reduction in computational time. Section 7 illustrates the performance of our attack under various defense strategies. The limitations of our attack are discussed in Section 8. Section 9 touches upon related works, and finally, we conclude in Section 10.

## 2 Background

### 2.1 Diffusion Models

The work of the Denoising Diffusion Probabilistic Models [21] (DDPM) has drawn considerable attention and led to the recent development of diffusion models [53, 56], which are characteristically described as “progressively denoising to obtain the true image”. There are two categories of diffusion models: unconditional diffusion models, which do not incorporate any guiding input for image output, and conditional diffusion models, which were developed subsequently and generate images based on provided inputs information, such as labels [10, 22], text [23, 39, 42, 45, 48], or low-resolution images [47, 49].

Unconditional Diffusion Models. A diffusion model has two phases. First, during the forward process, the model progressively adds standard Gaussian noise to the true image  $ x_0 $ through T steps. The image at time t is given by

<equation>x_{t}=\sqrt{\bar{\alpha}_{t}}x_{0}+\sqrt{1-\bar{\alpha}_{t}}\epsilon_{t}   \tag*{(1)}</equation>

where  $ \epsilon_t $ represents the standard Gaussian noise obtained from the reparameterization trick. Furthermore,  $ \bar{\alpha}_t $ is defined as the product  $ \prod_{i=1}^t \alpha_i $, with each parameter  $ \alpha_i $ monotonically decreasing and lying in the interval  $ [0, 1] $.


Second, the reverse process begins with the noise image  $ x_T' $, where  $ x_T' \sim N(0, I) $, and it progressively denoises to yield  $ x_{T-1}', x_{T-2}' \ldots, x_0' $ through the neural network (e.g., U-Net)  $ \epsilon_\theta $, parameterized by  $ \theta $. Specifically,  $ \epsilon_\theta $ takes a image  $ x_t' $ and a timestep  $ t $ as inputs, and predicts the noise, represented by  $ \epsilon_\theta(x_t', t) $ that should be eliminated at step  $ t $. The final goal is to maximize the similarity between each pair of original image  $ x_0 $ and the denoised image  $ x_0' $.

During the training phase, the objective is to minimize the loss, which is defined as the expected squared  $ \ell_{2} $ error. This error is evaluated overall  $ \epsilon_{t} $ and the training sample  $ x_{0} $, as given by:

<equation>L_{t}(\theta)=\mathbb{E}_{x_{0},\epsilon_{t}}\left[\|\epsilon_{t}-\epsilon_{\theta}(\sqrt{\bar{\alpha}_{t}}x_{0}+\sqrt{1-\bar{\alpha}_{t}}\epsilon_{t},t)\|_{2}^{2}\right].   \tag*{(2)}</equation>

More details can be found in Appendix A.

Conditional Diffusion Models. As the study of diffusion models deepens, it has been discovered that classifiers can be utilized to guide the diffusion model generation [10]. Specifically, given a pretrained classifier M and a target class c, one can derive 'directional information',  $ \nabla_{x_t} \log M(x_t | y) $, for an image  $ x_t $ and fuse it to the generation process of unconditional diffusion models.

In the text domain, Imagen employs T5, a significant language model [41], as a text encoder to guide the generation process through text embeddings [48]. Specifically, a distinct time embedding vector is constructed and modified during each timestep to align with the image's dimensions. The text embedding extracted from T5 is then incorporated with the time embedding and image to generate the conditional image.

### 2.2 Membership Inference Attack

Membership inference attack (MIA) tries to predict if a given sample was part of the training set used to train the target model. It has been widely applied to different deep learning models, including classification models [3, 31, 33, 50, 52, 63, 64], generative adversarial networks (GANs) [5, 19, 20, 26, 37], and diffusion models [4, 12, 27, 30, 35, 62]. MIA exploits the differential responses exhibited by machine learning models to training data. Specifically, these models react differently to samples they have been trained on, termed 'member samples', versus unfamiliar 'non-member samples'.

Shokri et al. [52] first proposed the technique of shadow training. This involves training shadow models to imitate the behavior of the target model. An attack model is then trained based on the output of the shadow models. This transforms membership inference into a classification problem.

Considering the increased computational overhead of training a machine learning model as an attack model, Yeom et al. [64] proposed a more streamlined and resource-efficient approach—the threshold-based MIA. This method begins with the computation of loss values from the model's output prediction vector. These calculated loss metrics are subsequently compared against a chosen threshold to infer the membership status of a data record.

Carlini et al. [3] argue that while threshold-based attacks are effective for non-membership inference, they lack precision for member sample classification. This discrepancy arises as the approach simplifies the comparison process by scaling all samples based on


their loss values, potentially omitting crucial sample-specific properties. To address this, Carlini et al. propose an alternative approach called Likelihood Ratio Attack (LiRA), which derives two distributions from the model's confidence values. These distributions are then used to determine the membership status of a given sample, thereby offering a more balanced evaluation of both member and non-member sets.

### 2.3 Problem Formulation

In this paper, we investigate MIA in diffusion models. We are given a target model. The task is to predict whether a certain sample is part of the training dataset. MIA on diffusion models (compared to classifiers) presents distinct challenges: Classic classifier models yield vectors. Thus people can use its prediction vector as a feature for MIA [3, 33, 50, 52, 63, 64], which constitutes a black-box attack. Diffusion models produce images as outputs, making it challenging to launch an attack on a diffusion model using only its output, i.e., the image. The current state-of-the-art attacks on diffusion models are predominantly white-box, relying on the loss generated during the evaluation process, as noted in Table 1. Our work is mainly focused on exploring how to get effective attack features. After getting the attack features (gradient data), we use it to train a machine learning model (i.e., XGBoost, MLP) as the attack model to identify the data sample.

Threat Model. We operate under the assumption that an attacker possesses white-box access to the target model, encompassing its architectural intricacies and specific parameter details. In the context of conditional diffusion models, we assume that the attacker knows all modalities (for instance, image-text pairs) pertaining to the victim models. The same assumption has also been adopted in several existing works, which we will discuss in detail later [4, 27, 35]. As more people openly share their model architectures and pre-trained checkpoints (like in HuggingFace $ ^{2} $), the scenario is realistic. A motivating example is an artist who suspects his artwork is being used without permission to train a diffusion model. This model is subsequently uploaded to the HuggingFace website. As a result, others can use it to generate images that mimic the artist's unique style. Clearly, this constitutes a severe violation of the artist's intellectual property rights. The artist, as a result, downloads the model and checks whether their artwork is used to train the model.

Hu et al. [27] and Matsumoto et al. [35] suggested utilizing the loss, as defined in Equation 2, at each timestep t as a feature in conjunction with a threshold-based MIA. Leveraging the loss directly as an attack vector presents the most intuitive attack approach. However, the loss value differences between member and non-member samples vary across different timesteps. For each model, additional

### 2.4 Existing Work

Existing White-Box Attacks to Diffusion Models. A key challenge in applying MIA is selecting the appropriate information/features to distinguish member and non-member samples. Most effective attacks on diffusion models predominantly employ white-box techniques  $ [4, 27, 35] $.

<div style="text-align: center;">Table 1: Compared with existing work, we argue that with white-box access, using gradients is more effective. We also evaluated more comprehensively on larger datasets.</div>


<table border=1 style='margin: auto; word-wrap: break-word;'><tr><td style='text-align: center; word-wrap: break-word;'>Attack</td><td style='text-align: center; word-wrap: break-word;'>Feature</td><td style='text-align: center; word-wrap: break-word;'>Victim target</td><td style='text-align: center; word-wrap: break-word;'>Training dataset</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>[4]</td><td style='text-align: center; word-wrap: break-word;'>Loss (LiRA)</td><td style='text-align: center; word-wrap: break-word;'>Unconditional Conditional</td><td style='text-align: center; word-wrap: break-word;'>CIFAR-10</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>[27]</td><td style='text-align: center; word-wrap: break-word;'>Loss (Threshold)</td><td style='text-align: center; word-wrap: break-word;'>Unconditional</td><td style='text-align: center; word-wrap: break-word;'>FFHQ</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>[35]</td><td style='text-align: center; word-wrap: break-word;'>Loss (Threshold)</td><td style='text-align: center; word-wrap: break-word;'>Unconditional</td><td style='text-align: center; word-wrap: break-word;'>CIFAR-10</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Ours</td><td style='text-align: center; word-wrap: break-word;'>Gradient (ML model)</td><td style='text-align: center; word-wrap: break-word;'>Unconditional Conditional</td><td style='text-align: center; word-wrap: break-word;'>CIFAR-10 ImageNet MS COCO</td></tr></table>

computation is required to identify the most effective range of timesteps, which greatly increases pre-computational cost and becomes impractical. Additionally, since the loss value is a scalar, it may lead to unstable attack accuracy due to insufficient information for reliable differentiation. In contrast, gradient data can effectively differentiate between member and non-member samples without requiring prior timestep selection. As high-dimensional data, it also enhances the accuracy and robustness of the attack.

Carlini et al. [4] also opted to employ loss and use the LiRA framework. In the context of the LiRA online framework, the attack strategy necessitates utilizing target points for the training of several shadow models, a process that is both computationally demanding and time-intensive. Subsequently, it constructs the  $ \mathbb{D}_{in} $ and  $ \mathbb{D}_{out} $ distributions at each timestep. In the original experiments reported in the paper, 16 shadow models were trained to generate distributions for each timestep. For more sophisticated models, such as Stable Diffusion [45], retraining a large cohort of shadow models to generate loss distributions poses a considerable challenge. In our work, we aim to use fewer shadow models to execute the attack while maintaining effectiveness and efficiency. More details about LiRA can be found at Appendix B.

Other Attacks. Several studies have utilized the properties of DDIM [29, 54, 57] (as detailed in Appendix A) for attacks [12, 30]. However, these attacks are contingent on the deterministic reverse process of DDIM, and cannot be directly applied to DDPM. Detailed discussions of these attacks are deferred to Appendix D.1 and Appendix D.2.

Prior to diffusion models, there are also MIAs for GANs [5, 19, 20, 26, 37]. Note that GANs and diffusion models differ in their overall architecture; therefore, white-box attacks toward GANs are not directly applicable to diffusion models. On the other hand, black-box attacks share similarities as both GANs and diffusion models are generative models. In particular, inspired by the attacks of GAN-Leaks [5], Matsumoto et al. [35] proposed an attack that is based on the reconstruction error of the target image and a set of generated samples. We will present the details at Appendix D.3, but the attack shows limited effectiveness.

Meanwhile, Wu et al. [62] carried out black-box attacks on pretrained text-to-image diffusion models, launching attacks at both the pixel-level and semantic-level. However, their method does not employ the shadow model technique as proposed in [52], instead


conducting all experiments directly on the target model and selecting the training set of the pre-trained model as the member set. Consequently, this attack strategy is not universally effective for every victim model.

Hu et al. [27] also initiated an interesting threat model (the so-called grey-box model or query-based model) where the attacker sees the intermediate denoised images and proposes an attack based on the similarities (likelihood) between pairs of these intermediate samples and those in the forward pass (details also in Appendix D.4).

## 3 Methodology

Previous works on attacking the diffusion model encompass black-box [35, 62], gray-box [12, 27, 30], and white-box approaches [4, 27, 35]. Upon comparing their accuracy and considering practical implications, we contend that white-box attacks on diffusion models are the most effective.

### 3.1 Theoretical Foundation and Challenge

Current white-box attacks often manipulate the loss at different timesteps through various methods (e.g., threshold [27, 35] or distribution [4]). However, it often necessitates a substantial amount of time to identify the timestep where the loss can most distinctly differentiate between the member and non-member set samples. We argue that rather than relying on the loss information, given white-box access, it could be more insightful to leverage gradient information that better reflects the model's different responses to member samples and non-member samples. The intuition using gradients is, as gradients are generally very high-dimensional (than losses), it offers a more nuanced representation of its response to an input target point compared to mere loss values.

Figure 1 shows the general idea of our attack. It is important to note that, owing to the specific architecture of the diffusion model, a single query point can yield multiple loss values originating from different timesteps. Subsequently, based on the loss L, we can derive the gradients using the standard back-propagation technique and use the gradients as features to train a machine-learning model to execute MIA.

In the diffusion model, the training loss function is defined as Equation 2. For each sample, noise is added using Equation 1, generating a noised sample  $ x_t $. The trained U-Net modules then predict the noise  $ \epsilon_t $ that needs to be denoised at timestep  $ t $, based on  $ x_t $ and  $ t $. The existing methods [4, 27, 35] assume that the loss value of a member sample is typically smaller than that of a non-member sample, which indicates intuitively

 <equation>x\in\mathcal{D}_{m}\text{if and only if}\|\boldsymbol{\epsilon}_{t}-\boldsymbol{\epsilon}_{\theta}(x_{t},t)\|_{2}^{2}<\tau</equation> 

where  $ \epsilon_{\theta}(x_{t}, t) $ represents the predicted noise at t-th step and  $ \epsilon_{t} $ is the ground true noise sample. However, we have observed that this approach can lead to misjudgments. For example, inherently complex member samples might exhibit higher loss values compared to simpler non-member samples, a phenomenon also observed in GAN-Leaks [5]. This indicates that relying solely on loss as the attack feature may introduce some degree of bias. Carlini et al. [4] also found that using loss values as the sole criterion for determining membership is inadequate.

In our work, we propose using gradient values as attack features to better capture the model's reaction to samples. Unlike loss values, which are scalars and provide limited information, gradient data offer a more comprehensive view. Additionally, even when two samples have identical loss values, their corresponding gradients can differ, as gradients depend on the specific inputs within the computational graph. For instance, the diffusion model  $ \epsilon_{\theta} $ (with parameter  $ \theta $) calculates gradients for a query sample x at t-th step; the gradients can be expressed as:


<equation>\nabla_{\theta}L_{t}(\theta,x)=\nabla_{\theta}\|\epsilon_{t}-\epsilon_{\theta}(x_{t},t)\|^{2}   \tag*{(3)}</equation>

According to the definition of the Euclidean norm squared, we can expand the squared term in Equation 3:

 <equation>\begin{align*}\left\|\boldsymbol{\epsilon}_{t}-\boldsymbol{\epsilon}_{\theta}(x_{t},t)\right\|^{2}&=\left(\boldsymbol{\epsilon}_{t}-\boldsymbol{\epsilon}_{\theta}(x_{t},t)\right)^{\top}\left(\boldsymbol{\epsilon}_{t}-\boldsymbol{\epsilon}_{\theta}(x_{t},t)\right)\\&=\left\|\boldsymbol{\epsilon}_{t}\right\|^{2}-2\boldsymbol{\epsilon}_{t}^{\top}\boldsymbol{\epsilon}_{\theta}(x_{t},t)+\left\|\boldsymbol{\epsilon}_{\theta}(x_{t},t)\right\|^{2}.\end{align*}</equation> 

Then, we proceed to compute the derivatives of each of the three expanded terms with respect to  $ \theta $:

<equation>\begin{align*}\nabla_{\theta}L_{t}(\theta,x)&=\nabla_{\theta}\left(\left\|\boldsymbol{\epsilon}_{t}\right\|^{2}-2\boldsymbol{\epsilon}_{t}^{\top}\boldsymbol{\epsilon}_{\theta}(x_{t},t)+\left\|\boldsymbol{\epsilon}_{\theta}(x_{t},t)\right\|^{2}\right)\\&=\nabla_{\theta}\left\|\boldsymbol{\epsilon}_{t}\right\|^{2}-2\nabla_{\theta}\left(\boldsymbol{\epsilon}_{t}^{\top}\boldsymbol{\epsilon}_{\theta}(x_{t},t)\right)+\nabla_{\theta}\left\|\boldsymbol{\epsilon}_{\theta}(x_{t},t)\right\|^{2}\\&=0-2\boldsymbol{\epsilon}_{t}^{\top}\nabla_{\theta}\boldsymbol{\epsilon}_{\theta}(x_{t},t)+2\boldsymbol{\epsilon}_{\theta}(x_{t},t)^{\top}\nabla_{\theta}\boldsymbol{\epsilon}_{\theta}(x_{t},t)\\&=-2\left(\boldsymbol{\epsilon}_{t}-\boldsymbol{\epsilon}_{\theta}(x_{t},t)\right)^{\top}\nabla_{\theta}\boldsymbol{\epsilon}_{\theta}(x_{t},t)\\&=2\left(\boldsymbol{\epsilon}_{\theta}(x_{t},t)-\boldsymbol{\epsilon}_{t}\right)^{\top}\nabla_{\theta}\boldsymbol{\epsilon}_{\theta}(x_{t},t)\end{align*}   \tag*{(4)}</equation>

From Equation 4, we show the gradient depends on both the value of the training loss  $ \left(\epsilon_{\theta}(x_t, t) - \epsilon_t\right) $ and the specific query sample being computed  $ \left(\nabla_{\theta} \epsilon_{\theta}(x_t, t)\right) $. For member and non-member samples that produce the same numerical loss value, gradients can still use  $ \nabla_{\theta} \epsilon_{\theta}(x_t, t) $ to discriminate them. We also present the experimental evidence to support our finding in Appendix C.

Intuitively, during the training phase, the model fits to member samples. Therefore, when encountering a training sample, the already converged model requires less parameter adjustment compared to a non-member sample, leading to smaller gradients. Based on this intuition, we use the model's gradient values as features for detecting query sample membership, as expressed by:

 <equation>x\in\mathcal{D}_{m}\text{if and only if}\nabla_{\theta}\|\epsilon_{t}-\epsilon_{\theta}(x_{t},t)\|_{2}^{2}<\tau</equation> 

The above findings demonstrate that even when the loss values are equal, the gradient information obtained from different samples still varies. We believe that this characteristic of gradient data represents the model's response to the query sample more effectively than the attack features used in existing methods [4, 27, 35], thereby enabling more successful attacks. However, the key challenge of using gradients for MIA is utilizing gradient information effectively. Considering the substantial number of parameters in the diffusion model (for instance, in our experiments, the Imagen model boasts close to 250 million trained parameters, while the DDPM model approaches 114 million), training the attack model by using the gradient of each model parameter for every image is both computationally impractical and prone to overfitting, despite its potential to maximally differentiate between member and non-member samples. Moreover, in diffusion models, the diffusion process typically involves T timesteps (usually set to 1000). For each timestep t in the


<div style="text-align: center;">Figure 1: High-level pipeline of our attack: Given the target sample  $ x_{0} $, we first add noise based on Equation 1 and feed it to the target model shaded in blue. At each sample step, we can compute a loss L using Equation 2 to derive the gradients. Gradients from all sample steps (with appropriate subsampling and aggregation operations) are used as features to train the attack model for MIA.</div>
