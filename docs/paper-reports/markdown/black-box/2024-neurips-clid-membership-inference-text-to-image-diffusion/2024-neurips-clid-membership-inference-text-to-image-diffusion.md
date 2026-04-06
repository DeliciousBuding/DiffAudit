# Membership Inference on Text-to-image Diffusion Models via Conditional Likelihood Discrepancy

Shengfang Zhai1,<sup>2</sup> , Huanran Chen3,<sup>6</sup> , Yinpeng Dong3,6<sup>∗</sup> , Jiajun Li1,<sup>2</sup> , Qingni Shen1,2<sup>∗</sup> , Yansong Gao<sup>4</sup> , Hang Su3,<sup>5</sup> , Yang Liu<sup>7</sup> <sup>1</sup>School of Software and Microelectronics, Peking University <sup>2</sup>PKU-OCTA Laboratory for Blockchain and Privacy Computing, Peking University <sup>3</sup>Dept. of Comp. Sci. and Tech., Institute for AI, BNRist Center, THBI Lab, Tsinghua University <sup>4</sup>The University of Western Australia <sup>5</sup>Zhongguancun Laboratory, Beijing, China <sup>6</sup>RealAI <sup>7</sup>Nanyang Technological University {zhaisf, jiajun.lee}@stu.pku.edu.cn huanran.chen@outlook.com {dongyinpeng, suhangss}@tsinghua.edu.cn qingnishen@ss.pku.edu.cn garrison.gao@uwa.edu.au yangliu@ntu.edu.sg

## Abstract

Text-to-image diffusion models have achieved tremendous success in the field of controllable image generation, while also coming along with issues of privacy leakage and data copyrights. Membership inference arises in these contexts as a potential auditing method for detecting unauthorized data usage. While some efforts have been made on diffusion models, they are not applicable to text-to-image diffusion models due to the high computation overhead and enhanced generalization capabilities. In this paper, we first identify a conditional overfitting phenomenon in text-to-image diffusion models, indicating that these models tend to overfit the conditional distribution of images given the corresponding text rather than the marginal distribution of images only. Based on this observation, we derive an analytical indicator, namely Conditional Likelihood Discrepancy (CLiD), to perform membership inference, which reduces the stochasticity in estimating memorization of individual samples. Experimental results demonstrate that our method significantly outperforms previous methods across various data distributions and dataset scales. Additionally, our method shows superior resistance to overfitting mitigation strategies, such as early stopping and data augmentation.

# 1 Introduction

Text-to-image diffusion models have achieved remarkable success in the guided generation of diverse, high-quality images based on text prompts, such as Stable Diffusion [\[42,](#page-12-0) [46\]](#page-12-1), DALLE-2 [\[43\]](#page-12-2), Imagen [\[49\]](#page-12-3), and DeepFloyd-IF [\[31\]](#page-11-0). These models are increasingly adopted by users to create photorealistic images that align with desired semantics. Moreover, they can generate images of specific concepts [\[32\]](#page-11-1) or styles [\[61\]](#page-13-0) when fine-tuned on relevant datasets. However, the impressive generative capabilities of these models depend heavily on high-quality image-text datasets, which involve collecting image-text data from the web. This practice raises significant privacy and copyright concerns in the community [\[5,](#page-10-0) [18\]](#page-10-1). The pretraining and fine-tuning processes of text-to-image diffusion models can cause copyright infringement, as they utilize unauthorized datasets published by human artists or stock-image websites [\[2,](#page-10-2) [10,](#page-10-3) [44,](#page-12-4) [45,](#page-12-5) [58\]](#page-12-6).

<sup>∗</sup>Corresponding authors.

Membership inference (also known as the membership inference attack) is widely used for auditing privacy leakage of training data [\[4,](#page-10-4) [53\]](#page-12-7), defined as determining whether a given data point has been used to train the target model. Dataset owners can thus leverage membership inference to determine if their data is being used without authorization [\[14,](#page-10-5) [39\]](#page-11-2).

Previous works [\[5,](#page-10-0) [15](#page-10-6)[–17,](#page-10-7) [28,](#page-11-3) [38\]](#page-11-4) have attempted membership inference on diffusion models. Carlini et al. [\[5\]](#page-10-0) employ LiRA (Likelihood Ratio Attack) [\[4\]](#page-10-4) to perform membership inference on diffusion models. LiRA requires training multiple shadow models to estimate the likelihood ratios of a data point from different models, which incurs high training overhead (e.g., 16 shadow models for DDPM [\[22\]](#page-11-5) on CIFAR-10 [\[30\]](#page-11-6)), making it neither scalable nor applicable to text-to-image diffusion models. Other query-based membership inference methods [\[15,](#page-10-6) [17,](#page-10-7) [28,](#page-11-3) [38\]](#page-11-4) design and compute indicators to evaluate whether a given data point belongs to the member set. These methods require only a few or even a single shadow model, making them scalable to larger text-to-image diffusion models. However, these methods mainly estimate model memorization for data points and do not fully utilize the conditional distribution of image-text pairs. Consequently, they achieve limited success only under excessively high training steps and fail under real steps or common data augmentation methods (Tab. [2\)](#page-6-0), which do not reflect real training scenarios. Text-to-image diffusion models have demonstrated excellent performance in zero-shot image generation [\[1,](#page-10-8) [42,](#page-12-0) [46\]](#page-12-1), indicating their strong generalization, which makes it difficult to distinguish membership by directly measuring overfitting to data points. And due to the stochasticity of diffusion training loss [\[22,](#page-11-5) [46\]](#page-12-1), this kind of measuring becomes more challenging.

To address the challenges, we firstly identify a Conditional Overfitting phenomenon of text-to-image diffusion models with empirical validation, where the models exhibit more significant overfitting to the conditional distribution of the images given the corresponding text than the marginal distribution of the images only. It inspires the revealing of membership by leveraging the overfitting difference. Based on it, we propose to perform membership inference on text-to-image diffusion models via Conditional Likelihood Discrepancy (CLiD). Specifically, CLiD quantifies overfitting difference analytically by utilizing Kullback-Leibler (KL) divergence as the distance metric and derives a membership inference indicator that estimates the discrepancy between the conditional likelihood of image-text pairs and the likelihood of images only. We approximate the likelihoods by employing Monte Carlo sampling on their ELBOs (Evidence Lower Bounds), and design two membership inference methods: a threshold-based method CLiDth and a feature vector-based method CLiDvec.

We conduct extensive experiments on three text-to-image datasets [\[32,](#page-11-1) [35,](#page-11-7) [66\]](#page-13-1) with various data distributions and dataset scales, using the mainstream open-sourced text-to-image diffusion models [\[11,](#page-10-9) [47\]](#page-12-8) under both fine-tuning and pretraining settings. First, our methods consistently outperform existing baselines across various data distributions and training scenarios, including fine-tuning settings and the pretraining setting. Second, our experiments on fine-tuning settings with different training steps (Sec. [4.2\)](#page-6-1) reveal that excessively high step/image ratios cause overfitting, leading to hallucination success; and we develop a more realistic pretraining setting following [\[13,](#page-10-10) [16\]](#page-10-11), where our experiments reveal the insufficient effect of existing membership inference works [\[15,](#page-10-6) [17,](#page-10-7) [28,](#page-11-3) [38\]](#page-11-4). Third, our comparison experiment with varying training steps (Sec. [4.3\)](#page-7-0) indicates that the effectiveness of membership inference grows with higher step/image ratios and should be evaluated under reasonable settings for realistic results. Next, ablation studies further demonstrate the effect of our CLiD indicator, even with fewer query count, our method still outperforms baseline methods (Fig. [3\)](#page-7-1). Last, experiments show that our methods exhibit stronger resistance to data augmentation, and exhibit resistance to even adaptive defenses.

## 2 Diffusion Model Preliminaries

Denoising Diffusion Probabilistic Model (DDPM) [\[22\]](#page-11-5) learns the data distribution x<sup>0</sup> ∼ q(x) by reversing the forward noise-adding process. For the forward process, DDPM defines a Markov process of adding Gaussian noise step by step:

<span id="page-1-0"></span>
$$q(\mathbf{x}_t|\mathbf{x}_{t-1}) := \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t}\mathbf{x}_{t-1}, \beta_t \mathbf{I}), \tag{1}$$

where β<sup>t</sup> ∈ (0, 1) is the hyperparameter controlling the variance. For the reverse process, DDPM defines a learnable Markov chain starting at p(x<sup>T</sup> ) = N (x<sup>T</sup> ; 0, I) to generate x0:

$$p_{\theta}(\mathbf{x}_0) = \int_{\mathbf{x}_{1:T}} p(\mathbf{x}_T) \prod_{t=1}^{T} p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_t) \, d\mathbf{x}_{1:T}, \qquad p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \mu_{\theta}(\mathbf{x}_t, t), \sigma_t^2), \quad (2)$$

where σ 2 t is the untrained time-dependent constant. θ represents the trainable parameters. To maximize pθ(x0), DDPM optimizes the Evidence Lower Bound (ELBO) of the log-likelihood [\[22,](#page-11-5) [33\]](#page-11-8):

<span id="page-2-0"></span>
$$\log p_{\theta}(\mathbf{x}_0) \ge \mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \left[ \log \frac{p_{\theta}(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \right] = -\mathbb{E}_{\epsilon,t} \left[ ||\epsilon_{\theta}(\mathbf{x}_t, t) - \epsilon||^2 \right] + C, \tag{3}$$

where ϵ ∼ N (0, I), t ∼ Uniform(1, ..., T) and C is a constant. x<sup>t</sup> is obtained from Eq. [\(1\)](#page-1-0), and ϵ<sup>θ</sup> is a function approximator intended to predict the noise ϵ from xt. Omitting the untrainable constant in Eq. [\(3\)](#page-2-0) and taking its negative yields the loss function of training DDPM.

Conditional diffusion models [\[21,](#page-11-9) [40,](#page-11-10) [46\]](#page-12-1). To achieve controllable generation ability, text-to-image diffusion models incorporate the conditioning mechanism into the model, which are also known as conditional diffusion models, enabling them to learn conditional probability as:

$$p_{\theta}(\mathbf{x}_0|\mathbf{c}) = \int_{\mathbf{x}_{1:T}} p(\mathbf{x}_T) \prod_{t=1}^{T} p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{c}) \, d\mathbf{x}_{1:T}, \tag{4}$$

where c denotes the embedding of condition. For text-to-image synthesis, c := T (y), where y and T denote the text input and the text encoder, respectively. Similar to Eq. [\(3\)](#page-2-0), through derivation [\[33\]](#page-11-8), we can obtain the ELBO of the conditional log-likelihood:

<span id="page-2-3"></span>
$$\log p_{\theta}(\mathbf{x}_0|\mathbf{c}) \ge -\mathbb{E}_{\epsilon,t} \left[ ||\epsilon_{\theta}(\mathbf{x}_t, t, \mathbf{c}) - \epsilon||^2 \right] + C. \tag{5}$$

## 3 Methodology

In this section, we detail the proposed Conditional Likelihood Discrepancy (CLiD) method. We first introduce the threat model of query-based membership inference in Sec. [3.1.](#page-2-1) We then identify the conditional overfitting phenomenon with experimental validation in Sec. [3.2.](#page-3-0) We further drive the membership inference indicator based on CLiD in Sec. [3.3](#page-3-1) and design two practical membership inference methods in Sec. [3.4.](#page-4-0) We finally provide the implementation details in Sec. [3.5.](#page-5-0)

#### <span id="page-2-1"></span>3.1 Threat Model

We use the standard security game of membership inference on image-text data following previous work [\[4,](#page-10-4) [5,](#page-10-0) [38,](#page-11-4) [48\]](#page-12-9). We define a challenger C and an adversary A who performs membership inference. C samples a member set Dmem ← D and trains or fine-tunes a text-to-image diffusion model f<sup>θ</sup> (i.e., target model) with Dmem. The rest of D is denoted by hold-out set Dout = D \ Dmem. For a given data point (x, c) ∈ D, A designs an algorithm M to yield a membership prediction:

$$\mathcal{M}(\mathbf{x}, \mathbf{c}, f_{\theta}) = \mathbb{1} \left[ \mathcal{M}'(\mathbf{x}, \mathbf{c}, f_{\theta}) > \tau \right], \tag{6}$$

where M′ denotes an indicator function that reflects membership information, and τ denotes a tunable decision threshold of query-based membership inference [\[15,](#page-10-6) [17,](#page-10-7) [28,](#page-11-3) [38\]](#page-11-4).

We consider a grey-box setting [2](#page-2-2) consistent with previous query-based methods [\[15,](#page-10-6) [17,](#page-10-7) [28,](#page-11-3) [38\]](#page-11-4). This setting assumes that A has access to the intermediate outputs of models without knowledge of specific model parameters. For the given image-text data point (x, c), we assume that x and c always correspond within the dataset D. This assumption is evident in scenarios where dataset copyright owners perform membership inference to audit usage. And we also consider a weaker assumption of conducting membership inference without the groundtruth text in Sec. [4.6.](#page-8-0)

Conversely, challenger C can mitigate the effectiveness of membership inference during training by utilizing data augmentation or even adaptive defense methods, which we discuss in Sec. [4.5.](#page-8-1) Our work primarily focuses on fine-tuning scenarios because the weights of pretrained models are readily available, making this scenario more prone to copyright risks [\[41,](#page-12-10) [56\]](#page-12-11). Numerous projects are implemented by fine-tuning open-source models on specific datasets [\[3,](#page-10-12) [24,](#page-11-11) [60,](#page-13-2) [64\]](#page-13-3). We also conduct experiments on pretrained text-to-image diffusion models (Tab. [3\)](#page-7-2) to demonstrate the effectiveness of our method even in pretraining scenarios.

<span id="page-2-2"></span><sup>2</sup>Note that in most real-world scenarios, the requirements for A in gray-box and white-box settings are nearly identical. We use this terminology here for consistency with previous works [\[15,](#page-10-6) [28\]](#page-11-3).

#### <span id="page-3-0"></span>3.2 Conditional Overfitting Phenomenon

The rationale behind previous studies primarily hinges on the overfitting of diffusion models to training data (usually image data x) [\[7,](#page-10-13) [8,](#page-10-14) [15,](#page-10-6) [28,](#page-11-3) [38\]](#page-11-4). This overfitting tends to result in lower estimation errors for images in the member set (training data) compared to those in the hold-out set during the diffusion process. Various indicators [\[15,](#page-10-6) [28,](#page-11-3) [38\]](#page-11-4) are designed based on this to expose membership information. Specifically, let qmem and qout represent the image distributions of the member set and the hold-out set, respectively. p represents the diffusion models' estimated distribution, and D denotes a distance metric (which will be specified later). This rationale can be formulated as:

$$D(q_{\text{mem}}(\mathbf{x}), p(\mathbf{x})) \le D(q_{\text{out}}(\mathbf{x}), p(\mathbf{x})). \tag{7}$$

However, if considering the membership inference on text-to-image diffusion models with image-text data (x, c), we emphasize the following assumption:

<span id="page-3-3"></span>Assumption 3.1 (Conditional overfitting phenomenon). *The overfitting of text-to-image diffusion models to the conditional distribution of* (x, c) *is more salient than to the marginal distribution of* x*:*

<span id="page-3-5"></span>
$$\underbrace{\mathbb{E}_{\mathbf{c}}[D(q_{out}(\mathbf{x}|\mathbf{c}), p(\mathbf{x}|\mathbf{c})) - D(q_{mem}(\mathbf{x}|\mathbf{c}), p(\mathbf{x}|\mathbf{c}))]}_{overfitting \ to \ conditional \ distribution} \ge \underbrace{D(q_{out}(\mathbf{x}), p(\mathbf{x})) - D(q_{mem}(\mathbf{x}), p(\mathbf{x}))}_{overfitting \ to \ marginal \ distribution}. \tag{8}$$

Empirically, we validate this assumption by using Fréchet Inception Distance (FID) [\[20\]](#page-11-12) as the metric D, i.e., DF ID. We calculate DF ID(q(x|c), p(x|c)) using the MS-COCO [\[35\]](#page-11-7) dataset on a fine-tuned Stable Diffusion [\[46\]](#page-12-1) model. Then by gradually truncating the original condition text to {2 3, 1 3, Null} to obtain c ∗ , we calculate DF ID(q(x|c ∗ ), p(x|c ∗ )) as a stepwise approximation of DF ID(q(x), p(x)). In Fig. [1,](#page-3-2) we report the FID scores of synthetic images under different conditions of member set and hold-out set. A smaller FID value indicates a closer match between model distributions and dataset distributions. From Fig. [1](#page-3-2) (a), it

![](_page_3_Figure_7.jpeg)

<span id="page-3-2"></span>Figure 1: FID values and the FID differences of synthetic images (2500/2500 samples for member/holdout set) under different conditions of member set and hold-out set.

can be observed that for the full condition, the FID difference between the member set and the holdout set is consistently higher than that for the truncated conditions, which validates our assumptions. We also demonstrate the validation utilizing other metrics in Appendix [A.](#page-14-0)

We further compute the change in FID after truncating the condition and observe that the change in FID of the member set is consistently greater than that of the hold-out set (Fig. [1](#page-3-2) (b)), which inspires revealing membership by this overfitting discrepancy. Recalling the aim of text-to-image diffusion model is to fit a latent space mapping from text to image, image data augmentation is commonly used to enhance the model generalization. For instance, the official fine-tuning script of Hugging-Face [\[24\]](#page-11-11) employs Random-Crop and Random-Flip as the default augmentation [\[25\]](#page-11-13). However, few trainers disturb the text condition as it is discrete and such disturbance would result in a decline of model utility (Sec. [4.5\)](#page-8-1). Therefore, we believe that leveraging this phenomenon contributes to addressing the challenges of the strong generalization of text-to-image diffusion models with the resistance to data augmentation.

#### <span id="page-3-1"></span>3.3 Condition Likelihood Discrepancy

In this section, we derive a membership inference indicator for a given individual sample based on Assumption [3.1.](#page-3-3) Calculating FID requires sampling lots of images from the p distribution, which is impractical under membership inference scenarios. Instead, we employ Kullback-Leibler (KL) divergence as the distance metric, which is widely used and computationally convenient (the usage of other metrics is discussed in Appendix [C\)](#page-15-0). Then we have the following theorem:

<span id="page-3-6"></span>Theorem 3.2. *(Proof in Appendix [B\)](#page-14-1) When using* D = DKL *as distance metric, Assumption [3.1](#page-3-3) is equivalent to:*

<span id="page-3-4"></span>
$$\mathbb{E}_{q_{mem}(\mathbf{x}, \mathbf{c})}[\log p(\mathbf{x}|\mathbf{c}) - \log p(\mathbf{x})] \ge \mathbb{E}_{q_{out}(\mathbf{x}, \mathbf{c})}[\log p(\mathbf{x}|\mathbf{c}) - \log p(\mathbf{x})] + \delta_H, \tag{9}$$

*where*

$$\delta_{H} = H(q_{out}(\mathbf{x})) + \mathbb{E}_{\mathbf{c}}[H(q_{mem}(\mathbf{x}|\mathbf{c}))] - H(q_{mem}(\mathbf{x})) - \mathbb{E}_{\mathbf{c}}[H(q_{out}(\mathbf{x}|\mathbf{c}))]. \tag{10}$$

Let us define:

<span id="page-4-1"></span>
$$\mathbb{I}(\mathbf{x}, \mathbf{c}) = \log p(\mathbf{x}|\mathbf{c}) - \log p(\mathbf{x}). \tag{11}$$

If δ<sup>H</sup> is negligible, then according to Eq. [\(9\)](#page-3-4), it holds that Eqmem [I(x)] ≥ τ ≥ Eqout [I(x)], where τ is a constant intermediate between the left-hand side and right-hand side. Membership inference is then posed as follows: given an input instance (x, c), measuring I(x, c) to predict how probable it is that the input is a sample from qmem rather than qout. Intuitively, if I(x) exceeds a threshold τ , the instance is likely from qmem; otherwise, it belongs to qout. In the community of membership inference methods [\[4](#page-10-4)[–6,](#page-10-15) [15,](#page-10-6) [17,](#page-10-7) [28,](#page-11-3) [38,](#page-11-4) [65\]](#page-13-4) , setting such a threshold τ is a standard practice to differentiate between the two distributions. Therefore, we can utilize the indicator I(x, c) for membership inference. Since Eq. [\(11\)](#page-4-1) actually involves measuring the likelihood discrepancy under different conditions of diffusion models, we call it Conditional Likelihood Discrepancy (CLiD).

In order to calculate the likelihoods in Eq. [\(11\)](#page-4-1) for a given data point (x, c), we utilize the ELBOs in Eq. [\(3\)](#page-2-0) and Eq. [\(5\)](#page-2-3) as an approximation of the log-likelihoods:

<span id="page-4-2"></span>
$$\mathbb{I}(\mathbf{x}, \mathbf{c}) = \mathbb{E}_{t, \epsilon} \left[ ||\epsilon_{\theta}(\mathbf{x}_{t}, t, \mathbf{c}_{\text{null}}) - \epsilon||^{2} \right] - \mathbb{E}_{t, \epsilon} \left[ ||\epsilon_{\theta}(\mathbf{x}_{t}, t, \mathbf{c}) - \epsilon||^{2} \right], \tag{12}$$

where cnull denotes an empty text condition input used to estimate the approximation of log pθ(x).

#### <span id="page-4-0"></span>3.4 Implementation of CLiD-MI

In practice, calculating Eq. [\(12\)](#page-4-2) needs a Monte Carlo estimate for data point by sampling N times using (t<sup>i</sup> , ϵi) pairs, with ϵ<sup>i</sup> ∼ N (0, I) and t<sup>i</sup> ∼ [1, 1000]. Performing two Monte Carlo estimations independently incurs high computational costs, resulting in 2 × N query count, where N is typically a large number to ensure accurate estimation. To simplify computation, we instead perform Monte Carlo estimation on the difference of the ELBOs inspired by [\[33\]](#page-11-8):

<span id="page-4-6"></span>
$$\mathbb{I}(\mathbf{x}, \mathbf{c}) = \mathbb{E}_{t, \epsilon} \left[ ||\epsilon_{\theta}(\mathbf{x}_{t}, t, \mathbf{c}_{\text{null}}) - \epsilon||^{2} - ||\epsilon_{\theta}(\mathbf{x}_{t}, t, \mathbf{c}) - \epsilon||^{2} \right]. \tag{13}$$

In experiments, to further mitigate randomness, we also consider diverse reduced conditions along with cnull, forming the reduced condition set C = {c ∗ 1 , c ∗ 2 ..., c ∗ k }, where we set c ∗ <sup>k</sup> = cnull. Then we compute multiple condition likelihood discrepancies:

<span id="page-4-3"></span>
$$\mathcal{D}_{\mathbf{x},\mathbf{c},\mathbf{c}_{i}^{*}} = \mathbb{E}_{t,\epsilon} \left[ ||\epsilon_{\theta}(\mathbf{x}_{t}, t, \mathbf{c}_{i}^{*}) - \epsilon||^{2} - ||\epsilon_{\theta}(\mathbf{x}_{t}, t, \mathbf{c}) - \epsilon||^{2} \right], \tag{14}$$

where c ∗ <sup>i</sup> ∈ C. In subsequent parts, we employ their mean or treat them as feature vectors to reveal membership information. We will introduce how to obtain C in Sec. [3.5.](#page-5-0)

Combining pθ(x|c) for further enhancement. Recall that the practical significance of sample likelihood is the probability that a data point originates from the model distribution, which essentially can also be used to assess membership. Due to the monotonicity of the log function, we can also use ELBO of Eq. [\(5\)](#page-2-3) to estimate pθ(x|c):

<span id="page-4-4"></span>
$$\mathcal{L}_{\mathbf{x},\mathbf{c}} = -\mathbb{E}_{t,\epsilon} \left[ ||\epsilon_{\theta}(\mathbf{x}_{t}, t, \mathbf{c}) - \epsilon||^{2} \right]. \tag{15}$$

Additionally, this estimation can reuse results from estimating Eq. [\(14\)](#page-4-3), thus obviating any additional query counts. Next, we consider two strategies to combine Eq. [\(14\)](#page-4-3) and Eq. [\(15\)](#page-4-4) to construct the final membership inference method.

Threshold-based attack–CLiDth. First, we normalize the two indicators to the same feature scale. Due to the outliers in the data, we use Robust-Scaler: S(ai) = (a<sup>i</sup> − a˜) IQR, where a<sup>i</sup> denotes the i-th value, a˜ denotes the mean and IQR (interquartile range) is defined as the difference between the third quartile (Q3) and the first quartile (Q1) of the feature. Then we have:

<span id="page-4-5"></span>
$$\mathcal{M}_{\text{CLiD}_{th}}(\mathbf{x}, \mathbf{c}) = \mathbb{1}\left[\alpha \cdot \mathcal{S}(\frac{1}{k} \sum_{i}^{k} \mathcal{D}_{\mathbf{x}, \mathbf{c}, \mathbf{c}_{i}^{*}}) + (1 - \alpha) \cdot \mathcal{S}(\mathcal{L}_{\mathbf{x}, \mathbf{c}}) > \tau\right],\tag{16}$$

where k denotes the total number of reduced c ∗ (i.e., k = |C|), and α is a weight parameter.

Vector-based attack–CLiDvec. We combine the estimated values of Eq. [\(14\)](#page-4-3) and Eq. [\(15\)](#page-4-4) to obtain the feature vectors corresponding to each data point:

$$\mathbf{V} = (\mathcal{D}_{\mathbf{x}, \mathbf{c}, \mathbf{c}_1^*}, \mathcal{D}_{\mathbf{x}, \mathbf{c}, \mathbf{c}_2^*} \dots \mathcal{D}_{\mathbf{x}, \mathbf{c}, \mathbf{c}_k^*}, \mathcal{L}_{\mathbf{x}, \mathbf{c}}). \tag{17}$$

We use a simple classifier to distinguish feature vectors in order to determine the membership of the samples:

<span id="page-5-3"></span>
$$\mathcal{M}_{\text{CLiD}_{vec}}(\mathbf{x}, \mathbf{c}) = \mathbb{1}\left[\mathcal{F}_{\mathcal{M}}(\mathbf{V}) > \tau\right],$$
 (18)

where F<sup>M</sup> denotes the predict confidence of the classifier.

#### <span id="page-5-0"></span>3.5 Practical Considerations

Reducing conditions to obtain c ∗ . We consider three methods for diverse reduction: (1) Simply taking the first, middle, and last thirds of the sentences as text inputs. (2) Randomly adding Gaussian noises with various scales to the text embeddings. (3) Calculating the importance of words in the text [\[55,](#page-12-12) [57\]](#page-12-13) and replacing them with "pad" tokens by varying proportions in descending order. For all three methods, we additionally use the null text input as c ∗ k . These methods are all effective and we use (3) with k = 4 in subsequent experiments (details in Appendix [D\)](#page-15-1).

Monte Carlo sampling. Let M and N denote the Monte Carlo sampling numbers of estimating L(x, c) and Dx,c,<sup>c</sup> ∗ , respectively. We set M = N to achieve result reuse between Eq. [\(14\)](#page-4-3) and Eq. [\(15\)](#page-4-4), reducing the number of Monte Carlo sampling. Hence the overall query count of one data point is M + K · N. Significant effects can be observed even when M, N = 1 (Fig. [3\)](#page-7-1).

Classifiers of CLiDvec. Due to the simplicity of the feature vectors, we do not need a neural network as the classifier [\[15\]](#page-10-6). Simpler classifiers help to prevent overfitting. In our experiments, we utilize XGBoost [\[9\]](#page-10-16) and utilize its predict confidence.

# 4 Experiments

## 4.1 Setups

Datasets and models. For the fine-tuning setting, we select 416/417 samples on Pokémon [\[32\]](#page-11-1), 2500/2500 samples on MS-COCO [\[35\]](#page-11-7) and 10, 000/10, 000 samples on Flickr [\[66\]](#page-13-1) as the member/hold-out dataset, respectively. These three datasets involve diverse data distributions and dataset scales. We use the most widely used text-to-image diffusion model, Stable Diffusion v1- 4 [3](#page-5-1) [\[11\]](#page-10-9), as the target model to fine-tune it on these three datasets. For the pretraining setting, we conduct experiments on Stable Diffusion v1-5[4](#page-5-2) [\[47\]](#page-12-8) using the processed LAION dataset [\[51\]](#page-12-14) (detailed in Sec. [4.2\)](#page-6-1) to minimize distribution shift [\[13,](#page-10-10) [16\]](#page-10-11).

Fine-tuning setups. For fine-tuning, previous membership inference on text-to-image diffusion models usually relies on strong overfitting settings. To evaluate the performance more realistically, we consider the two following setups: (1) *Over-training.* Following the previous works [\[15,](#page-10-6) [17,](#page-10-7) [28\]](#page-11-3), we fine-tune 15,000 steps on Pokemon datasets, and 150,000 steps on MS-COCO and Flickr (with only 2500/2500 dataset size). (2) *Real-world training.* Considering that trainers typically do not train for such high steps, we recalibrate the steps based on the training steps/dataset size ratio (approximately 20) of official fine-tuning scripts on Huggingface [\[24\]](#page-11-11). Thus, we train 7,500 steps, 50,000 steps and 200,000 steps for the Pokémon, MS-COCO and Flickr datasets, respectively. Additionally, we employ the default data augmentation (Random-Crop and Random-Flip [\[25\]](#page-11-13)) in training codes [\[25\]](#page-11-13) to simulate real-world scenarios.

Baselines. We broadly consider existing member inference methods applicable to text-to-image diffusion models as our baselines: Loss-based inference [\[38\]](#page-11-4), SecMIstats (SecMI) [\[15\]](#page-10-6), PIA [\[28\]](#page-11-3), PFAMIMet (PFAMI) [\[17\]](#page-10-7) and an additional method of directly conducting Monte Carlo estimation (M. C.) on Eq. [\(15\)](#page-4-4) for comparison. For all baselines, we use the parameters recommended in their papers. We omit some membership inference methods for generative models [\[6,](#page-10-15) [19,](#page-10-17) [36\]](#page-11-14), as they have been shown ineffective for diffusion models in previous works [\[15,](#page-10-6) [17\]](#page-10-7).

Evaluation metrics. We follow the widely used metrics of previous works [\[4,](#page-10-4) [5,](#page-10-0) [15,](#page-10-6) [17,](#page-10-7) [28\]](#page-11-3), including ASR (i.e., the accuracy of membership inference), AUC and the True Positive Rate (TPR) when the False Positive Rate (FPR) is 1% (i.e., TPR@1%FPR).

Implementation details. Our evaluation follows the setup of representative membership inference works [\[4,](#page-10-4) [5\]](#page-10-0). It is important to note that some implementations [\[26,](#page-11-15) [29\]](#page-11-16) of previous works assume

<span id="page-5-1"></span><sup>3</sup> <https://huggingface.co/CompVis/stable-diffusion-v1-4>

<span id="page-5-2"></span><sup>4</sup> <https://huggingface.co/runwayml/stable-diffusion-v1-5>

<span id="page-6-2"></span>Table 1: Results under *Over-training* setting. We mark the best and second-best results for each metric in bold and underline, respectively. Additionally, the best results from baselines are marked in blue for comparison.

| Method  |       | MS-COCO |           |        | Flickr |           |       | Pokemon |           |       |
|---------|-------|---------|-----------|--------|--------|-----------|-------|---------|-----------|-------|
|         | ASR   | AUC     | TPR@1%FPR | ASR    | AUC    | TPR@1%FPR | ASR   | AUC     | TPR@1%FPR | Query |
| Loss    | 81.92 | 89.98   | 32.28     | 81.90  | 90.34  | 40.80     | 83.76 | 91.79   | 25.77     | 1     |
| PIA     | 68.56 | 75.12   | 5.08      | 68.56  | 75.12  | 5.08      | 83.37 | 90.95   | 13.31     | 2     |
| M. C.   | 82.04 | 89.77   | 36.04     | 83.32  | 91.37  | 41.20     | 79.35 | 86.78   | 23.74     | 3     |
| SecMI   | 83.00 | 90.81   | 50.64     | 62.96† | 89.29  | 48.52     | 80.49 | 90.64   | 9.36      | 12    |
| PFAMI   | 94.48 | 98.60   | 78.00     | 90.64  | 96.78  | 50.96     | 89.86 | 95.70   | 65.35     | 20    |
| CLiDth  | 99.08 | 99.94   | 99.12     | 91.42  | 97.39  | 74.00     | 97.96 | 99.28   | 97.84     | 15    |
| CLiDvec | 99.74 | 99.31   | 95.20     | 91.78  | 97.52  | 73.88     | 97.36 | 99.46   | 96.88     | 15    |

<span id="page-6-0"></span><sup>†</sup> When conducting SecMI [\[15\]](#page-10-6), we observe that the thresholds obtained on the shadow model sometimes do not transfer well to the target model.

Table 2: Results under *Real-world training* setting. We also highlight key results according to Tab. [1.](#page-6-2)

| Method  |       | MS-COCO |           |       | Flickr |           |       | Pokemon |           |       |
|---------|-------|---------|-----------|-------|--------|-----------|-------|---------|-----------|-------|
|         | ASR   | AUC     | TPR@1%FPR | ASR   | AUC    | TPR@1%FPR | ASR   | AUC     | TPR@1%FPR | Query |
| Loss    | 56.28 | 61.89   | 1.92      | 54.91 | 56.60  | 1.83      | 61.03 | 65.96   | 2.82      | 1     |
| PIA     | 54.10 | 55.52   | 1.76      | 51.96 | 52.73  | 1.28      | 58.34 | 59.95   | 2.64      | 2     |
| M. C.   | 57.98 | 61.97   | 2.64      | 54.92 | 56.78  | 2.16      | 61.10 | 66.48   | 3.84      | 3     |
| SecMI   | 60.94 | 65.40   | 3.92      | 55.60 | 63.85  | 2.76      | 61.28 | 65.56   | 0.84      | 12    |
| PFAMI   | 57.36 | 60.39   | 2.72      | 54.68 | 56.13  | 1.80      | 58.94 | 63.53   | 5.76      | 20    |
| CLiDth  | 88.88 | 96.13   | 67.52     | 87.12 | 94.74  | 53.56     | 86.79 | 93.28   | 61.39     | 15    |
| CLiDvec | 89.52 | 96.30   | 66.36     | 88.86 | 95.33  | 53.92     | 85.47 | 92.61   | 59.95     | 15    |

access to a portion of the exact member set and the hold-out set to obtain a threshold for calculating ASR or to train a classification network [\[26\]](#page-11-15). This assumption does not align with real-world scenarios. Therefore, we strictly adhere to the fundamental assumption of membership inference [\[4,](#page-10-4) [17\]](#page-10-7): knowing only the overall dataset without any knowledge of the member/hold-out split. Hence, we first train a shadow model to obtain the α for Eq. [\(16\)](#page-4-5), classifiers for Eq. [\(18\)](#page-5-3) and the threshold τ for calculating ASR with auxiliary datasets of the same distribution. Then we perform the test on the target models. Other implementation details are provided in Appendix [D.](#page-15-1)

#### <span id="page-6-1"></span>4.2 Main Results

Over-training setting (fine-tuning). In Tab. [1,](#page-6-2) models are trained for excessive steps on all three datasets, resulting in significant overfitting. We observe that under this over-training scenario, both of our methods nearly achieve ideal binary classification effectiveness. For instance, CLiDth achieves over 99% ASR, AUC and TPR@1%FPR value on the MS-COCO dataset [\[35\]](#page-11-7). With this training setup, the metrics for different baselines are very similar. Even the simplest loss-based method [\[38\]](#page-11-4) (with the query count of 1) also yields satisfactory results compared with other high query count methods. Therefore, we emphasize: *This unrealistic over-training setting fails to adequately reflect the effectiveness differences among various membership inference methods*.

Real-world training setting (fine-tuning). In Tab. [2,](#page-6-0) we adjust the training steps simulating realword training scenario [\[24\]](#page-11-11) and utilize default data augmentation [\[25\]](#page-11-13). The best value of ASR and AUC of baseline methods decreases to around 65%, and the best value of TPR@1%FPR decreases to around 5%, indicating insufficient effectiveness of previous member inference methods in real-world training scenarios of text-to-image diffusion models. In contrast, our methods maintain ASR above 86% and AUC above 93%, exceeding the best baseline values by about 30%. The TPR@1%FPR of our methods exceeds the best baseline values by 50%~60%. The results demonstrate the effectiveness of our methods across various data distributions and scales in real-world training scenarios.

Pretraining setting. For the pretraining setting, we adopt a stringent and realistic membership inference setting based on previous works [\[13,](#page-10-10) [16\]](#page-10-11). (1) To ensure the distribution consistency between the member and hold-out set, we respectively select 2500 samples from the LAION-Aesthetics v2 5+ and LAION-2B MultiTranslated [\[51\]](#page-12-14) as member/hold-out set following [\[16\]](#page-10-11); (2) We filter out samples containing non-English characters to ensure there are no other "distinguishable tails" [\[13\]](#page-10-10) in

| Method | AUC<br>TPR@1%FPR<br>ASR |       |      | Query |  |
|--------|-------------------------|-------|------|-------|--|
| Loss   | 51.78                   | 50.90 | 1.75 | 1     |  |
| PIA    | 52.13                   | 52.42 | 1.25 | 2     |  |
| M. C.  | 53.18                   | 53.96 | 1.25 | 3     |  |
| SecMI  | 57.43                   | 58.59 | 2.45 | 12    |  |
| PFAMI  | 59.08                   | 61.11 | 1.45 | 20    |  |
| CLiDth | 64.53                   | 67.82 | 5.01 | 15    |  |

<span id="page-7-2"></span>Table 3: The performance of membership inference methods on Stable Diffusion v1- 5 [\[47\]](#page-12-8) in pretraining setting. We utilize the processed LAION dataset to ensure the distribution consistency between member / holdout sets [\[13,](#page-10-10) [16\]](#page-10-11). The best results are highlighted in bold.

![](_page_7_Figure_2.jpeg)

<span id="page-7-4"></span>Figure 2: Effectiveness trajectory on various training steps.

the dataset[5](#page-7-3) . We conduct membership inference on Stable Diffusion v1-5 [\[46\]](#page-12-1). As shown in Tab. [3,](#page-7-2) our method consistently outperforms the baselines across all three metrics.

#### <span id="page-7-0"></span>4.3 Performance on Various Training Steps

From Tab. [1](#page-6-2) and Tab. [2,](#page-6-0) we find that the training steps greatly influence the effectiveness of membership inference. All membership inference methods tend to exhibit satisfactory performance when the model is trained for an excessive number of steps that conflicts with real-world scenarios. Therefore, we emphasize that the *effectiveness trajectory* of membership inference across varying training steps should also be utilized to evaluate different methods. Better membership inference methods should reveal membership information earlier as training progresses.

To explore this, we fine-tune Stable Diffusion models with the MS-COCO dataset for varying training steps under *real-world training* setting and report the AUC values of different membership inference methods in Fig. [2.](#page-7-4) It can be observed that as the training progresses, CLiDth exhibits a significantly faster increase in effectiveness trajectory. By 25, 000 steps, CLiDth effectively exposes membership information, whereas other baselines achieve similar results only at around 150, 000 steps. This demonstrates that our method can effectively reveal membership information when the overfitting degree of the text-to-image diffusion model is much weaker.

#### 4.4 Ablation Study

To conduct an ablation study, we vary the Monte Carlo sampling count in Eq. [\(14\)](#page-4-3) and Eq. [\(15\)](#page-4-4), perform CLiDth with MS-COCO dataset under *realworld training* setting and report the AUC values in Fig. [3.](#page-7-1) To further compare the effects of Eq. [\(14\)](#page-4-3) and Eq. [\(15\)](#page-4-4), we discard each term in Eq. [\(16\)](#page-4-5) and denote it as M/N = 0. We also include the result of the best baseline, SecMI [\[15\]](#page-10-6), as a comparison.

Effect of Dx,c,c<sup>∗</sup> . In Fig. [3,](#page-7-1) results of "M=1, N=0" and "M=1, N=1" show a significant improvement of membership inference by including Dx,c,c<sup>∗</sup> . Results of "M=5, N=0" and "M=1, N=1" further show that the method utilizing Dx,c,c<sup>∗</sup> performs much better under the same sampling numbers. Additionally, the results of "M=0, N=1" and "M=1, N=1" indicates that only considering both Eq. [\(14\)](#page-4-3) and Eq. [\(15\)](#page-4-4) achieves the optimal performance.

![](_page_7_Figure_11.jpeg)

<span id="page-7-1"></span>Figure 3: Performance of CLiDth and SecMI under various Monte Carlo sampling numbers (i.e., query count). The legend labels are sorted in ascending order by AUC values.

<span id="page-7-3"></span><sup>5</sup>Das et al. [\[13\]](#page-10-10) indicates that MultiTranslated-LAION dataset contains fewer non-English characters than the LAION dataset due to the use of the translation model.

<span id="page-8-2"></span>Table 4: The performance of different methods under no augmentation and default augmentation.

| Method |       | No Augmentation |           | Defaut Augmentation |                |                 |  |  |
|--------|-------|-----------------|-----------|---------------------|----------------|-----------------|--|--|
|        | ASR   | AUC             | TPR@1%FPR | ASR (∆)             | AUC (∆)        | TPR@1%FPR (∆)   |  |  |
| Loss   | 66.54 | 72.73           | 7.72      | 56.28 (-10.26)      | 61.89 (-10.84) | 1.92 (-5.80)    |  |  |
| PIA†   | 56.56 | 59.28           | 2.00      | 54.10 (-2.46)       | 55.52 (-3.76)  | 1.76 (-0.24)    |  |  |
| SecMI  | 72.02 | 81.07           | 13.72     | 60.94 (-11.08)      | 65.40 (-15.08) | 3.92 (-9.80)    |  |  |
| PFAMI  | 79.20 | 87.05           | 18.44     | 57.36 (-21.84)      | 60.39 (-26.66) | 2.72 (-15.72)   |  |  |
| CLiDth | 96.76 | 99.47           | 91.72     | 88.88 (-7.88)       | 96.13 (-3.34)  | 67.52 (-24.20)‡ |  |  |

<sup>†</sup>We omit the discussion of PIA as it shows no effectiveness at this training steps, with the metrics consistently approximating random guessing.

Monte Carlo sampling numbers. In Fig. [3,](#page-7-1) we observe that when setting M = N, the performance improves as the number of Monte Carlo sampling increases. And the performance is improved slightly when M, N > 3. Hence, we set M, N = 3 to ensure the balance between a low query count and satisfied performance. Moreover, the experiment results of "M=1, N=1" and "SecMI" also demonstrate: CLiDth outperforms previous works even with a much fewer query count.

## <span id="page-8-1"></span>4.5 Resistance to Defense

Since data augmentation is commonly used in training and can mitigate the effectiveness of membership inference [\[15\]](#page-10-6), we use it to evaluate the performance of methods under defense. As the baseline methods already exhibit weak performance under *real-world training* setting, we opt not to incorporate additional data augmentation. Instead, we remove the default data augmentation from training scripts [\[25\]](#page-11-13) to observe the effectiveness change of different methods. We fine-tune Stable Diffusion models for 50,000 steps with MS-COCO, report the metrics, and calculate the metrics changes in Tab. [4.](#page-8-2) We observe that the effectiveness of all membership inference methods declines after data augmentation is introduced during training. Note that PFAMI [\[17\]](#page-10-7) exhibits the highest

<span id="page-8-4"></span>Table 5: Effectiveness of CLiDth in adaptive defense. We calculate the FID [\[20\]](#page-11-12) with 10, 000 unseen MS-COCO samples to assess the model utility.

| Defense | CLiDth on MS-COCO |                   |                |
|---------|-------------------|-------------------|----------------|
|         |                   | ASR AUC TPR@1%FPR | FID ↓ / ∆      |
| None    | 88.88 96.13       | 67.52             | 13.17          |
| Reph    | 85.32 93.83       | 55.67             | 13.58 / +0.41  |
| Del-1   | 86.40 93.59       | 59.52             | 13.18 / -0.01  |
| Del-3   | 83.91 91.52       | 52.03             | 12.92 / -0.25  |
| Shuffle | 65.89 67.37       | 0.15              | 18.26 / +5.09† |

†Compared to other methods, the increase in FID caused by shuffling is unacceptable for generative models.

sensitivity to data augmentation since it infers membership by probability fluctuation after images are perturbed, which also explains its significant performance decline between Tab. [1](#page-6-2) and Tab. [2.](#page-6-0) Compared to the baselines, our method exhibits the smallest decrease, which indicates its stronger resistance to data augmentation.

Adaptive defense. We further consider adaptive defense: assuming the trainers are aware of our methods and perturb the text of image-text datasets before training. We consider the following adaptive defense methods: (1) rephrasing the original text[6](#page-8-3) , (2) randomly deleting 10%, 30% words in text, and (3) shuffling 50% of the image-text mappings in the dataset. In Tab. [5,](#page-8-4) we observe that except for *shuffling*, the other adaptive defense methods have almost no effect on CLiDth. And *shuffling* damages the model utility (too high FID values), rendering this defense meaningless.

#### <span id="page-8-0"></span>4.6 Weaker Assumption

Although in Sec. [3.1](#page-2-1) we assume that the adversary can access the entire image-text pairs based on the real-world data usage auditing scenario, we also consider a weaker assumption: the adversary can only access the image without the corresponding text.

In this scenario, we first generate pseudo-text corresponding to the images using an image captioning model (BLIP [\[34\]](#page-11-17) in our experiments), and then conduct CLiD-MI based on the image-pseudo\_text pairs. In Tab. [6,](#page-9-0) we observe that our method still broadly outperforms baselines. We believe this is because the pseudo-text preserves the image's key semantics, keeping our methods effective.

<sup>‡</sup>The TPR@1%FPR value changes significantly here because its ROC curve is very sharp when FPR close to 0.

<span id="page-8-3"></span><sup>6</sup>We utilize ChatGPT-3.5 with the following prompt: "Please rewrite the following sentences while keeping the key semantics."

<span id="page-9-0"></span>Table 6: Results without access to the corresponding text under *Over-training* setting and *Real-world training* setting. We fine-tune MS-COCO on SDv1-4. Key results are highlighted as Tab. [1.](#page-6-2)

| Method  |       |       | Over-training (Pseudo-Text) | Real-world training (Pseudo-Text) |       |           |       |
|---------|-------|-------|-----------------------------|-----------------------------------|-------|-----------|-------|
|         | ASR   | AUC   | TPR@1%FPR                   | ASR                               | AUC   | TPR@1%FPR | Query |
| Loss    | 73.80 | 81.01 | 9.71                        | 56.08                             | 58.47 | 1.60      | 1     |
| PIA     | 61.40 | 65.75 | 1.20                        | 53.44                             | 54.38 | 1.52      | 2     |
| M. C.   | 74.36 | 81.55 | 11.28                       | 56.68                             | 60.00 | 1.28      | 3     |
| SecMI   | 82.04 | 88.97 | 40.80                       | 60.48                             | 64.04 | 3.28      | 12    |
| PFAMI   | 91.56 | 95.16 | 68.16                       | 58.12                             | 59.77 | 2.64      | 20    |
| CLiDth  | 92.84 | 95.43 | 72.36                       | 76.16                             | 83.27 | 19.76     | 15    |
| CLiDvec | 93.26 | 96.59 | 71.73                       | 77.76                             | 84.48 | 18.06     | 15    |

# 5 Related Works

Copyright protection in text-to-image synthesis. To protect the copyright of text-to-image models, several works [\[67,](#page-13-5) [68\]](#page-13-6) propose inserting backdoors to embed watermarks in text-to-image models. To protect the copyright of image-text datasets, some works [\[50,](#page-12-15) [52,](#page-12-16) [69\]](#page-13-7) incorporate imperceptible perturbations to render the released datasets unusable. Other works [\[12,](#page-10-18) [56\]](#page-12-11) utilize the backdoor or watermark to track the usage of image-text datasets. In contrast, our method indicates the possibility of auditing the unauthorized usage of individual image-text data points utilizing membership inference.

Membership inference on diffusion models. In the grey-box or white-box setting, Carlini et al. [\[5\]](#page-10-0) firstly conduct membership inference on unconditional diffusion models by conducting LiRA (Likelihood Ratio Attack) [\[4\]](#page-10-4), with the requirement of training multiple shadow models. Matsumoto et al. [\[38\]](#page-11-4) make the first step by utilizing diffusion loss to conduct query-based membership inference. Some works [\[15,](#page-10-6) [28\]](#page-11-3) leverage the DDIM [\[54\]](#page-12-17) deterministic forward process [\[27\]](#page-11-18) to access the posterior estimation errors of diffusion models. And Fu et al. [\[17\]](#page-10-7) leverage the probability fluctuations by perturbing image samples. Few works consider the black-box settings [\[41,](#page-12-10) [62\]](#page-13-8). However, these studies either assume partial knowledge of member set data [\[62\]](#page-13-8) or assume extensive fine-tuning steps [\[41\]](#page-12-10) (100 ∼ 500 epochs), both of which do not align with real-world scenarios.

Memorization detection in text-to-image models. A similar work [\[59\]](#page-13-9) detects token memorization by inspecting the magnitude of text-conditional predictions, but differs from ours by lacking in-depth rationale analysis and a rigorous membership inference setup with randomly selected member/holdout sets.

# 6 Conclusion

In this paper, we identify the phenomenon of conditional overfitting in text-to-image models and propose CLiD-MI, the membership inference framework on text-to-image diffusion models utilizing the derived indicator, conditional likelihood discrepancy. Experimental results demonstrate the superiority of our method and its resistance against early stopping and data augmentation. Our method aims to inspire a new direction for the community regarding unauthorized usage auditing.

Limitations: Due to the limited availability of open-source text-to-image diffusion models, evaluations under the pretraining setting are not sufficient. Considering fine-tuning setting involves a multi step/image ratio, we acknowledge that the superiority of CLiD-MI over the baselines in the pretraining setting is not as evident compared to fine-tuning setting. We emphasize our experiments under pretraining setting (Tab. [3\)](#page-7-2) reveal the hallucination success of existing works and encourage future research to focus on this more challenging and practical scenario.

## Acknowledgments

We thank anonymous reviewers for their valuable feedback. In addition, we thank Xin Zhang for his editorial comments. This work is supported by the National Key R&D Program of China (No.2022YFB2703301), NSFC Projects (Nos. 92370124, 62076147). Y. Dong is also supported by the China National Postdoctoral Program for Innovative Talents and Shuimu Tsinghua Scholar Program.

# References

- <span id="page-10-8"></span>[1] Fan Bao, Shen Nie, Kaiwen Xue, Chongxuan Li, Shi Pu, Yaole Wang, Gang Yue, Yue Cao, Hang Su, and Jun Zhu. One transformer fits all distributions in multi-modal diffusion at scale. In *International Conference on Machine Learning*, pages 1692–1717. PMLR, 2023.
- <span id="page-10-2"></span>[2] BBC. "Art is dead Dude" - the rise of the AI artists stirs debate. 2022. URL [https://www.bbc.com/](https://www.bbc.com/news/technology-62788725) [news/technology-62788725](https://www.bbc.com/news/technology-62788725).
- <span id="page-10-12"></span>[3] BIGWILLY. *Heart of Apple XL*, 2024. [https://civitai.com/models/272440/](https://civitai.com/models/272440/heart-of-apple-xl-love) [heart-of-apple-xl-love](https://civitai.com/models/272440/heart-of-apple-xl-love).
- <span id="page-10-4"></span>[4] Nicholas Carlini, Steve Chien, Milad Nasr, Shuang Song, Andreas Terzis, and Florian Tramer. Membership inference attacks from first principles. In *2022 IEEE Symposium on Security and Privacy (SP)*, pages 1897–1914. IEEE, 2022.
- <span id="page-10-0"></span>[5] Nicolas Carlini, Jamie Hayes, Milad Nasr, Matthew Jagielski, Vikash Sehwag, Florian Tramer, Borja Balle, Daphne Ippolito, and Eric Wallace. Extracting training data from diffusion models. In *32nd USENIX Security Symposium (USENIX Security 23)*, pages 5253–5270, 2023.
- <span id="page-10-15"></span>[6] Dingfan Chen, Ning Yu, Yang Zhang, and Mario Fritz. Gan-leaks: A taxonomy of membership inference attacks against generative models. In *Proceedings of the 2020 ACM SIGSAC conference on computer and communications security*, pages 343–362, 2020.
- <span id="page-10-13"></span>[7] Huanran Chen, Yinpeng Dong, Zhengyi Wang, Xiao Yang, Chengqi Duan, Hang Su, and Jun Zhu. Robust classification via a single diffusion model. *arXiv preprint arXiv:2305.15241*, 2023.
- <span id="page-10-14"></span>[8] Huanran Chen, Yinpeng Dong, Shitong Shao, Zhongkai Hao, Xiao Yang, Hang Su, and Jun Zhu. Your diffusion model is secretly a certifiably robust classifier. *arXiv preprint arXiv:2402.02316*, 2024.
- <span id="page-10-16"></span>[9] Tianqi Chen and Carlos Guestrin. Xgboost: A scalable tree boosting system. In *Proceedings of the 22nd acm sigkdd international conference on knowledge discovery and data mining*, pages 785–794, 2016.
- <span id="page-10-3"></span>[10] CNN. AI won an art contest, and artists are furious. 2022. URL [https://www.cnn.com/2022/09/03/](https://www.cnn.com/2022/09/03/tech/ai-art-fair-winner-controversy/index.html) [tech/ai-art-fair-winner-controversy/index.html](https://www.cnn.com/2022/09/03/tech/ai-art-fair-winner-controversy/index.html).
- <span id="page-10-9"></span>[11] CompVis. Stable-Diffusion-v1-4. 2024. URL [https://huggingface.co/CompVis/](https://huggingface.co/CompVis/stable-diffusion-v1-4) [stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4).
- <span id="page-10-18"></span>[12] Yingqian Cui, Jie Ren, Yuping Lin, Han Xu, Pengfei He, Yue Xing, Wenqi Fan, Hui Liu, and Jiliang Tang. Ft-shield: A watermark against unauthorized fine-tuning in text-to-image diffusion models. *arXiv preprint arXiv:2310.02401*, 2023.
- <span id="page-10-10"></span>[13] Debeshee Das, Jie Zhang, and Florian Tramèr. Blind baselines beat membership inference attacks for foundation models. *arXiv preprint arXiv:2406.16201*, 2024.
- <span id="page-10-5"></span>[14] Daniel DeAlcala, Aythami Morales, Gonzalo Mancera, Julian Fierrez, Ruben Tolosana, and Javier Ortega-Garcia. Is my data in your ai model? membership inference test with application to face images. *arXiv preprint arXiv:2402.09225*, 2024.
- <span id="page-10-6"></span>[15] Jinhao Duan, Fei Kong, Shiqi Wang, Xiaoshuang Shi, and Kaidi Xu. Are diffusion models vulnerable to membership inference attacks? In *International Conference on Machine Learning*, pages 8717–8730. PMLR, 2023.
- <span id="page-10-11"></span>[16] Jan Dubinski, Antoni Kowalczuk, Stanisław Pawlak, Przemyslaw Rokita, Tomasz Trzci ´ nski, and Paweł ´ Morawiecki. Towards more realistic membership inference attacks on large diffusion models. In *Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision*, pages 4860–4869, 2024.
- <span id="page-10-7"></span>[17] Wenjie Fu, Huandong Wang, Chen Gao, Guanghua Liu, Yong Li, and Tao Jiang. A probabilistic fluctuation based membership inference attack for generative models. *arXiv preprint arXiv:2308.12143*, 2023.
- <span id="page-10-1"></span>[18] Juliana Neelbauer Gil Appel and David A. Schweidel. Generative AI Has an Intellectual Property Problem. 2023. URL [https://hbr.org/2023/04/](https://hbr.org/2023/04/generative-ai-has-an-intellectual-property-problem) [generative-ai-has-an-intellectual-property-problem](https://hbr.org/2023/04/generative-ai-has-an-intellectual-property-problem).
- <span id="page-10-17"></span>[19] Jamie Hayes, Luca Melis, George Danezis, and Emiliano De Cristofaro. Logan: Membership inference attacks against generative models. *Proceedings on Privacy Enhancing Technologies*, 2019.

- <span id="page-11-12"></span>[20] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. *Advances in neural information processing systems*, 30, 2017.
- <span id="page-11-9"></span>[21] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. *arXiv preprint arXiv:2207.12598*, 2022.
- <span id="page-11-5"></span>[22] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. *Advances in neural information processing systems*, 33:6840–6851, 2020.
- <span id="page-11-20"></span>[23] Emiel Hoogeboom, Jonathan Heek, and Tim Salimans. simple diffusion: End-to-end diffusion for high resolution images. In *International Conference on Machine Learning*, pages 13213–13232. PMLR, 2023.
- <span id="page-11-11"></span>[24] Huggingface. *The training script of stable-diffusion*, 2024. URL [https://huggingface.co/docs/](https://huggingface.co/docs/diffusers/training/text2image##launch-the-script) [diffusers/training/text2image#launch-the-script](https://huggingface.co/docs/diffusers/training/text2image##launch-the-script). Accessed: May 22, 2024.
- <span id="page-11-13"></span>[25] Huggingface. *The python code of fine-tuning stable-diffusion*, 2024. [https://github.com/](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py) [huggingface/diffusers/blob/main/examples/text\\_to\\_image/train\\_text\\_to\\_image.py](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py).
- <span id="page-11-15"></span>[26] Jinhaoduan. Secmi, 2023. URL [https://github.com/jinhaoduan/SecMI/blob/main/mia\\_evals/](https://github.com/jinhaoduan/SecMI/blob/main/mia_evals/secmia.py) [secmia.py](https://github.com/jinhaoduan/SecMI/blob/main/mia_evals/secmia.py).
- <span id="page-11-18"></span>[27] Gwanghyun Kim, Taesung Kwon, and Jong Chul Ye. Diffusionclip: Text-guided diffusion models for robust image manipulation. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 2426–2435, 2022.
- <span id="page-11-3"></span>[28] Fei Kong, Jinhao Duan, RuiPeng Ma, Heng Tao Shen, Xiaofeng Zhu, Xiaoshuang Shi, and Kaidi Xu. An efficient membership inference attack for the diffusion model by proximal initialization. In *The Twelfth International Conference on Learning Representations*, 2024.
- <span id="page-11-16"></span>[29] Kong13661. Pia, 2023. URL <https://github.com/kong13661/PIA>.
- <span id="page-11-6"></span>[30] Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.
- <span id="page-11-0"></span>[31] DeepFloyd Lab. Deepfloyd if. <https://github.com/deep-floyd/IF>, 2023.
- <span id="page-11-1"></span>[32] Lambda. Pokemon-blip-captions. 2023. URL [https://huggingface.co/datasets/lambdalabs/](https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions) [pokemon-blip-captions](https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions).
- <span id="page-11-8"></span>[33] Alexander C Li, Mihir Prabhudesai, Shivam Duggal, Ellis Brown, and Deepak Pathak. Your diffusion model is secretly a zero-shot classifier. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 2206–2217, 2023.
- <span id="page-11-17"></span>[34] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In *International conference on machine learning*, pages 12888–12900. PMLR, 2022.
- <span id="page-11-7"></span>[35] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In *Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13*, pages 740–755. Springer, 2014.
- <span id="page-11-14"></span>[36] Kin Sum Liu, Chaowei Xiao, Bo Li, and Jie Gao. Performing co-membership attacks against deep generative models. In *2019 IEEE International Conference on Data Mining (ICDM)*, pages 459–467. IEEE, 2019.
- <span id="page-11-19"></span>[37] David Lopez-Paz and Maxime Oquab. Revisiting classifier two-sample tests. *arXiv preprint arXiv:1610.06545*, 2016.
- <span id="page-11-4"></span>[38] Tomoya Matsumoto, Takayuki Miura, and Naoto Yanai. Membership inference attacks against diffusion models. In *2023 IEEE Security and Privacy Workshops (SPW)*, pages 77–83. IEEE, 2023.
- <span id="page-11-2"></span>[39] Yuantian Miao, Minhui Xue, Chao Chen, Lei Pan, Jun Zhang, Benjamin Zi Hao Zhao, Dali Kaafar, and Yang Xiang. The audio auditor: User-level membership inference in internet of things voice services. *Proceedings on Privacy Enhancing Technologies*, 2021.
- <span id="page-11-10"></span>[40] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. *arXiv preprint arXiv:2112.10741*, 2021.

- <span id="page-12-10"></span>[41] Yan Pang and Tianhao Wang. Black-box membership inference attacks against fine-tuned diffusion models. *arXiv preprint arXiv:2312.08207*, 2023.
- <span id="page-12-0"></span>[42] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. *arXiv preprint arXiv:2307.01952*, 2023.
- <span id="page-12-2"></span>[43] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. *arXiv preprint arXiv:2204.06125*, 1(2):3, 2022.
- <span id="page-12-4"></span>[44] Reuters. Lawsuits accuse AI content creators of misusing copyrighted work. 2023. URL [https://www.reuters.com/legal/transactional/](https://www.reuters.com/legal/transactional/lawsuits-accuse-ai-content-creators-misusing-copyrighted-work-2023-01-17/) [lawsuits-accuse-ai-content-creators-misusing-copyrighted-work-2023-01-17/](https://www.reuters.com/legal/transactional/lawsuits-accuse-ai-content-creators-misusing-copyrighted-work-2023-01-17/).
- <span id="page-12-5"></span>[45] Reuters. Getty images lawsuit says stability ai misused photos to train ai. 2023. URL [https://www.reuters.com/legal/](https://www.reuters.com/legal/getty-images-lawsuit-says-stability-ai-misused-photos-train-ai-2023-02-06/) [getty-images-lawsuit-says-stability-ai-misused-photos-train-ai-2023-02-06/](https://www.reuters.com/legal/getty-images-lawsuit-says-stability-ai-misused-photos-train-ai-2023-02-06/).
- <span id="page-12-1"></span>[46] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 10684–10695, 2022.
- <span id="page-12-8"></span>[47] Runwayml. Stable-Diffusion-v1-5. 2024. URL [https://huggingface.co/runwayml/](https://huggingface.co/runwayml/stable-diffusion-v1-5) [stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5).
- <span id="page-12-9"></span>[48] Alexandre Sablayrolles, Matthijs Douze, Cordelia Schmid, Yann Ollivier, and Hervé Jégou. White-box vs black-box: Bayes optimal strategies for membership inference. In *International Conference on Machine Learning*, pages 5558–5567. PMLR, 2019.
- <span id="page-12-3"></span>[49] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-toimage diffusion models with deep language understanding. *Advances in neural information processing systems*, 35:36479–36494, 2022.
- <span id="page-12-15"></span>[50] Hadi Salman, Alaa Khaddaj, Guillaume Leclerc, Andrew Ilyas, and Aleksander Madry. Raising the cost of malicious ai-powered image editing. In *International Conference on Machine Learning*, pages 29894–29918. PMLR, 2023.
- <span id="page-12-14"></span>[51] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. *arXiv preprint arXiv:2210.08402*, 2022.
- <span id="page-12-16"></span>[52] Shawn Shan, Jenna Cryan, Emily Wenger, Haitao Zheng, Rana Hanocka, and Ben Y Zhao. Glaze: Protecting artists from style mimicry by {Text-to-Image} models. In *32nd USENIX Security Symposium (USENIX Security 23)*, pages 2187–2204, 2023.
- <span id="page-12-7"></span>[53] Reza Shokri, Marco Stronati, Congzheng Song, and Vitaly Shmatikov. Membership inference attacks against machine learning models. In *2017 IEEE symposium on security and privacy (SP)*, pages 3–18. IEEE, 2017.
- <span id="page-12-17"></span>[54] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In *International Conference on Learning Representations*, 2020.
- <span id="page-12-12"></span>[55] Raphael Tang, Linqing Liu, Akshat Pandey, Zhiying Jiang, Gefei Yang, Karun Kumar, Pontus Stenetorp, Jimmy Lin, and Ferhan Türe. What the daam: Interpreting stable diffusion using cross attention. In *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 5644–5659, 2023.
- <span id="page-12-11"></span>[56] Zhenting Wang, Chen Chen, Lingjuan Lyu, Dimitris N Metaxas, and Shiqing Ma. Diagnosis: Detecting unauthorized data usages in text-to-image diffusion models. In *The Twelfth International Conference on Learning Representations*, 2023.
- <span id="page-12-13"></span>[57] Zhijie Wang, Yuheng Huang, Da Song, Lei Ma, and Tianyi Zhang. Promptcharm: Text-to-image generation through multi-modal prompting and refinement. *arXiv preprint arXiv:2403.04014*, 2024.
- <span id="page-12-6"></span>[58] WashingtonPost. He made a children's book using AI. Then came the rage. 2022. URL [https://www.washingtonpost.com/technology/2023/01/19/](https://www.washingtonpost.com/technology/2023/01/19/ai-childrens-book-controversy-chatgpt-midjourney/) [ai-childrens-book-controversy-chatgpt-midjourney/](https://www.washingtonpost.com/technology/2023/01/19/ai-childrens-book-controversy-chatgpt-midjourney/).

- <span id="page-13-9"></span>[59] Yuxin Wen, Yuchen Liu, Chen Chen, and Lingjuan Lyu. Detecting, explaining, and mitigating memorization in diffusion models. In *The Twelfth International Conference on Learning Representations*, 2024.
- <span id="page-13-2"></span>[60] Jonathan Whitaker. *Fine-tuning a CLOOB-Conditioned Latent Diffusion Model on WikiArt*, 2024. [https://johnowhitaker.dev/dsc/](https://johnowhitaker.dev/dsc/2022-04-12-fine-tuning-a-cloob-conditioned-latent-diffusion-model-on-wikiart.html) [2022-04-12-fine-tuning-a-cloob-conditioned-latent-diffusion-model-on-wikiart.](https://johnowhitaker.dev/dsc/2022-04-12-fine-tuning-a-cloob-conditioned-latent-diffusion-model-on-wikiart.html) [html](https://johnowhitaker.dev/dsc/2022-04-12-fine-tuning-a-cloob-conditioned-latent-diffusion-model-on-wikiart.html).
- <span id="page-13-0"></span>[61] WikiArt. WikiArt. 2024. URL <https://www.wikiart.org/>.
- <span id="page-13-8"></span>[62] Yixin Wu, Ning Yu, Zheng Li, Michael Backes, and Yang Zhang. Membership inference attacks against text-to-image generation models. *arXiv preprint arXiv:2210.00968*, 2022.
- <span id="page-13-10"></span>[63] Qiantong Xu, Gao Huang, Yang Yuan, Chuan Guo, Yu Sun, Felix Wu, and Kilian Weinberger. An empirical study on evaluation metrics of generative adversarial networks. *arXiv preprint arXiv:1806.07755*, 2018.
- <span id="page-13-3"></span>[64] Shih-Ying Yeh, Yu-Guan Hsieh, Zhidong Gao, Bernard BW Yang, Giyeong Oh, and Yanmin Gong. Navigating text-to-image customization: From lycoris fine-tuning to model evaluation. In *The Twelfth International Conference on Learning Representations*, 2023.
- <span id="page-13-4"></span>[65] Samuel Yeom, Irene Giacomelli, Matt Fredrikson, and Somesh Jha. Privacy risk in machine learning: Analyzing the connection to overfitting. In *2018 IEEE 31st computer security foundations symposium (CSF)*, pages 268–282. IEEE, 2018.
- <span id="page-13-1"></span>[66] Peter Young, Alice Lai, Micah Hodosh, and Julia Hockenmaier. From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions. *Transactions of the Association for Computational Linguistics*, 2:67–78, 2014.
- <span id="page-13-5"></span>[67] Shengfang Zhai, Yinpeng Dong, Qingni Shen, Shi Pu, Yuejian Fang, and Hang Su. Text-to-image diffusion models can be easily backdoored through multimodal data poisoning. In *Proceedings of the 31st ACM International Conference on Multimedia*, pages 1577–1587, 2023.
- <span id="page-13-6"></span>[68] Yunqing Zhao, Tianyu Pang, Chao Du, Xiao Yang, Ngai-Man Cheung, and Min Lin. A recipe for watermarking diffusion models. *arXiv preprint arXiv:2303.10137*, 2023.
- <span id="page-13-7"></span>[69] Zhengyue Zhao, Jinhao Duan, Xing Hu, Kaidi Xu, Chenan Wang, Rui Zhang, Zidong Du, Qi Guo, and Yunji Chen. Unlearnable examples for diffusion models: Protect data from unauthorized exploitation. *arXiv preprint arXiv:2306.01902*, 2023.

# <span id="page-14-0"></span>A Validation of Assumption [3.1](#page-3-3)

![](_page_14_Figure_1.jpeg)

<span id="page-14-2"></span>Figure A.1: Metric values and the metric differences of synthetic images, with the same setting as Sec. [3.2.](#page-3-0)

To extensively validate the effectiveness of Assumption [3.1,](#page-3-3) we utilize additional metrics as the distances metric D in Eq. [\(8\)](#page-3-5), including *Wasserstein Distance* [\[63\]](#page-13-10), *Kernel MMD (Maximum Mean Discrepancy)* [\[63\]](#page-13-10) and *1-Nearest Neighbor Classifier (1-NN)* [\[37\]](#page-11-19), in addition to FID [\[20\]](#page-11-12). As observed in Fig. [A.1,](#page-14-2) regardless of the metric used for D, Assumption [3.1](#page-3-3) consistently holds, thereby confirming the broad applicability of Conditional Overfitting phenomenon.

# <span id="page-14-1"></span>B Proof of Theorem [3.2](#page-3-6)

*Proof.* Eq. [\(8\)](#page-3-5) is equivalent to:

<span id="page-14-4"></span>
$$\mathbb{E}_{\mathbf{c}}[D(q_{\text{out}}(\mathbf{x}|\mathbf{c}), p(\mathbf{x}|\mathbf{c}))] - D(q_{\text{out}}(\mathbf{x}), p(\mathbf{x}))$$

$$\geq \mathbb{E}_{\mathbf{c}}[D(q_{\text{mem}}(\mathbf{x}|\mathbf{c}), p(\mathbf{x}|\mathbf{c}))] - D(q_{\text{mem}}(\mathbf{x}), p(\mathbf{x})).$$
(B.1)

Given that both the member set and the hold-out set are mixtures of Dirac distributions:

$$q(\mathbf{x}) = \frac{1}{|D_{set}|} \sum_{\mathbf{x}_i \in D_{set}} \delta(\mathbf{x} - \mathbf{x}_i),$$
(B.2)

where Dset denotes the set of images in the corresponding dataset. We can derive the analytical form for the Kullback-Leibler (KL) KL divergence when using DKL as the distance metric:

$$D_{KL}(q(\mathbf{x}), p(\mathbf{x})) = \int q(\mathbf{x}) \log \frac{q(\mathbf{x})}{p(\mathbf{x})} d\mathbf{x}$$

$$= -\int \left(\frac{1}{|D_{set}|} \sum_{\mathbf{x}_i \in D_{set}} \delta(\mathbf{x} - \mathbf{x}_i)\right) \log p(\mathbf{x}) dx + H(q(\mathbf{x}))$$

$$= -\frac{1}{|D_{set}|} \sum_{\mathbf{x}_i \in D_{set}} \int \delta(\mathbf{x} - \mathbf{x}_i) \log p(\mathbf{x}) dx + H(q(\mathbf{x}))$$

$$= -\frac{1}{|D_{set}|} \sum_{\mathbf{x}_i \in D_{set}} \log p(x_i) + H(q(\mathbf{x})).$$
(B.3)

where H is the entropy functional. Therefore, we have:

<span id="page-14-3"></span>
$$\mathbb{E}_{\mathbf{c}}[D_{KL}(q(\mathbf{x}|\mathbf{c}), p(\mathbf{x}|\mathbf{c}))] - D_{KL}(q(\mathbf{x}), p(\mathbf{x})) 
= -\frac{1}{|D_{set}|} \sum_{\mathbf{x}, \mathbf{c} \in D_{set}} [\log p(\mathbf{x}|\mathbf{c}) - \log p(\mathbf{x})] + \mathbb{E}_{\mathbf{c}}[H(q(\mathbf{x}|\mathbf{c}))] - H(q(\mathbf{x})),$$
(B.4)

where Dset is the corresponding dataset (member set or hold-out set). Substituting Eq. [\(B.4\)](#page-14-3) into Eq. [\(B.1\)](#page-14-4), we can get:

<span id="page-15-2"></span>
$$-\frac{1}{|D_{\text{out}}|} \sum_{\mathbf{x}, \mathbf{c} \in D_{\text{out}}} [\log p(\mathbf{x}|\mathbf{c}) - \log p(\mathbf{x})] + \mathbb{E}_{\mathbf{c}}[H(q_{\text{out}}(\mathbf{x}|\mathbf{c}))] - H(q_{\text{out}}(\mathbf{x}))$$

$$\geq -\frac{1}{|D_{\text{mem}}|} \sum_{\mathbf{x}, \mathbf{c} \in D_{\text{mem}}} [\log p(\mathbf{x}|\mathbf{c}) - \log p(\mathbf{x})] + \mathbb{E}_{\mathbf{c}}[H(q_{\text{mem}}(\mathbf{x}|\mathbf{c}))] - H(q_{\text{mem}}(\mathbf{x})).$$
(B.5)

Eq. [\(B.5\)](#page-15-2) is equivalent to:

$$-\mathbb{E}_{q_{\text{out}}(\mathbf{x},\mathbf{c})}[\log p(\mathbf{x}|\mathbf{c}) - \log p(\mathbf{x})] + \mathbb{E}_{\mathbf{c}}[H(q_{\text{out}}(\mathbf{x}|\mathbf{c}))] - H(q_{\text{out}}(\mathbf{x}))$$

$$\geq -\mathbb{E}_{q_{\text{mem}}(\mathbf{x},\mathbf{c})}[\log p(\mathbf{x}|\mathbf{c}) - \log p(\mathbf{x})] + \mathbb{E}_{\mathbf{c}}[H(q_{\text{mem}}(\mathbf{x}|\mathbf{c}))] - H(q_{\text{mem}}(\mathbf{x})).$$
(B.6)

Finally, we can get:

$$\mathbb{E}_{q_{\text{mem}}(\mathbf{x}, \mathbf{c})}[\log p(\mathbf{x}|\mathbf{c}) - \log p(\mathbf{x})] \ge \mathbb{E}_{q_{\text{out}}(\mathbf{x})}[\log p(\mathbf{x}|\mathbf{c}) - \log p(\mathbf{x})] + \delta_H, \tag{B.7}$$

where

$$\delta_{H} = H(q_{\text{out}}(\mathbf{x})) + \mathbb{E}_{\mathbf{c}}[H(q_{\text{mem}}(\mathbf{x}|\mathbf{c}))] - H(q_{\text{mem}}(\mathbf{x})) - \mathbb{E}_{\mathbf{c}}[H(q_{\text{out}}(\mathbf{x}|\mathbf{c}))]. \tag{B.8}$$

# <span id="page-15-0"></span>C Metrics Discussion of Sec. [3.3](#page-3-1)

In the derivation of Sec. [3.3,](#page-3-1) we also consider other metrics besides KL divergence. The results show that KL divergence yields the most easily computable analytical form. For instance, we briefly discuss Jensen–Shannon (JS) divergence as follows:

Recall the expression for Jensen-Shannon divergence:

$$D_{JS}(q,p) = D_{KL}(q, \frac{1}{2}(q+p)) + D_{KL}(p, \frac{1}{2}(q+p)).$$
 (C.1)

The first parameter of the KL divergence should be a simple distribution that is easy to compute; otherwise, deriving the analytical form for such divergence is typically difficult. In Eq [\(8\)](#page-3-5), JS divergence cannot be efficiently computed because it includes DKL(p, <sup>1</sup> 2 (q + p)), where p denotes the model distribution. It needs to use the Monte Carlo method, which involves sampling images from both q and p to make an approximation. As a result, this process is extremely time-consuming.

# <span id="page-15-1"></span>D Experiment Details

#### D.1 Monte Carlo Sampling

In our method, the key to accurate membership inference lies in estimating ELBO with fewer sampling steps for better precision. To achieve this, firstly, we reduce the number of Monte Carlo samples by directly estimating the ELBO difference (Eq. [\(13\)](#page-4-6)). Secondly, recalling Monte Carlo sampling using (t<sup>i</sup> , ϵi) pairs with ϵ<sup>i</sup> ∼ N (0, I) and t<sup>i</sup> ∼ [1, 1000], we explore the effect of the sampling time ti . We conduct a single Monte Carlo sampling test using MS-COCO on *real-word training* setting and report the AUC values in Fig. [D.1.](#page-16-0)

In Fig. [D.1,](#page-16-0) we observe that the single Monte Carlo estimation achieves optimal accuracy when t<sup>i</sup> ∈ [400, 500]. Similar results are shown in [\[33\]](#page-11-8). Therefore, consistent with [\[33\]](#page-11-8), we sample at intervals of 10 centered around the timestep 450. In our experiments, M, N in Eq [\(14\)](#page-4-3) are both uniformly set to 3 (i.e., the estimation number is 3), and we use the time list of [440, 450, 460], resulting in the query count of 15. Note that [\[5,](#page-10-0) [15\]](#page-10-6) indicate that for DDPM of Cifar-10 [\[30\]](#page-11-6), the best estimation timestep is around 100. This difference may arise from the different signal-to-noise ratios of images with various resolution [\[23\]](#page-11-20). This finding suggests that the Monte Carlo sampling timestep should be designed differently for diffusion models of different scales.

![](_page_16_Figure_0.jpeg)

<span id="page-16-0"></span>Figure D.1: Effectiveness of single Monte Carlo estimation of various timesteps. Small t<sup>i</sup> corresponds to less noise added, and large t<sup>i</sup> corresponds to significant noise. AUC value is highest when the timestep is around 450.

Table D.1: The membership inference performance with different reduction methods. "Null" denotes employing null text solely to compute Eq. [\(14\)](#page-4-3) without reduction methods.

<span id="page-16-2"></span>

| Reduction Methods         | CLiDth<br>on MS-COCO |       |           |       |
|---------------------------|----------------------|-------|-----------|-------|
|                           | ASR                  | AUC   | TPR@1%FPR | Query |
| Null (K=1)                | 85.10                | 93.60 | 42.96     | 6     |
| Simply Clipping (K=4)     | 88.02                | 95.90 | 66.53     | 15    |
| Gaussian Noise (K=4)      | 86.58                | 94.79 | 56.78     | 15    |
| Importance Clipping (K=4) | 88.88                | 96.13 | 67.52     | 15    |

#### D.2 Reduction Methods

In implementation, we actually diversely reduce the condition c to c ∗ and calculate pθ(x|c ∗ ) to approximate pθ(x). In this part, we evaluate the effectiveness of different reduction methods. We consider three methods in Sec. [3.5:](#page-5-0) (1) Simply Clipping. We simply use the first, middle, and last thirds of the sentences as text inputs. (2) Gaussian Noise. We add Gaussian noises with the scales of 50%, 70%, 90% to the overall text embeddings. (3) Importance Clipping. We calculate the importance of words in the text[7](#page-16-1) [\[55,](#page-12-12) [57\]](#page-12-13) and replace them with "pad" tokens in descending order by varying proportions of 30%, 50%, 70%. For all three methods, we additionally use the null text as a c ∗ . The experiments are conducted on the *real-world training* setting with MS-COCO dataset. And we also employ null text solely to compute Eq. [\(14\)](#page-4-3) without reduction methods for comparison.

In Tab. [D.1,](#page-16-2) we observe that *Importance Clipping* achieves the best results due to its more deterministic reduction. So we adopt it as the reduction method used in our experiments. Additionally, we note that all three reduction methods exhibit satisfactory results, demonstrating the general applicability of our method. Comparing the results without the usage of reduction methods, the results validate the effectiveness of reduction methods in Sec. [3.5.](#page-5-0)

## E Compute Overhead and Resources

Computational Overhead. As a query-based member inference method, the computational efficiency of our method primarily depends on the number of queries. A lower query count signifies a more efficient member inference method. Our method significantly outperforms the baselines when the query count are about the same (such as SecMI and PFAMI in Sec. [4.2\)](#page-6-1). Furthermore, even with a much lower query count such as M = 1, N = 1(Q = 5) (Fig. [3\)](#page-7-1), our method exhibits a noticeable improvement compared to the baselines.

Compute Resources. Our experiments are divided into two main parts: training (fine-tuning) and inference, both conducted on a single RTX A6000 GPU. The time of execution in the training phase depends on the training steps. For example, we perform 7, 500, 50, 000, and 200, 000 steps for Pokemon [\[32\]](#page-11-1), MS-COCO [\[35\]](#page-11-7) and Flickr [\[66\]](#page-13-1) dataset, which take about 2 hours, 12 hours, and 48

<span id="page-16-1"></span><sup>7</sup> <https://github.com/ma-labo/PromptCharm>

hours, respectively. The time of execution in inference time depends on the methods' query count. For example, with the query count of 15, our membership inference method on a dataset of size 2500/2500 takes approximately 80 minutes per run for all data points. Typically, we perform this inference once on the shadow model and once on the target model, resulting in a total time cost of 160 minutes.

# F Ethics Statements

Although the current threat models for membership inference methods include privacy attack scenarios and data auditing scenarios, we emphasize that for text-to-image diffusion models, the potential application of membership inference lies more in unauthorized data usage auditing than in data privacy leakage. This is because most training data is obtained by scraping open-source imagetext pairs, which are more likely to pose copyright threats rather than privacy violations. So we emphasize that our method can make a positive societal impact for inspiring unauthorized usage auditing technologies of text-image datasets in the community.