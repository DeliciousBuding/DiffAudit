<div style="text-align: center;">Table 2: Impact of three different timestep-level sampling methods on attack accuracy and their respective time consumption.</div>


<table border=1 style='margin: auto; word-wrap: break-word;'><tr><td style='text-align: center; word-wrap: break-word;'>Method</td><td style='text-align: center; word-wrap: break-word;'>ASR</td><td style='text-align: center; word-wrap: break-word;'>AUC</td><td style='text-align: center; word-wrap: break-word;'>TPR@1%FPR</td><td style='text-align: center; word-wrap: break-word;'>TPR@0.1%FPR</td><td style='text-align: center; word-wrap: break-word;'>Time (seconds)</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Effective</td><td style='text-align: center; word-wrap: break-word;'>0.947</td><td style='text-align: center; word-wrap: break-word;'>0.992</td><td style='text-align: center; word-wrap: break-word;'>0.663</td><td style='text-align: center; word-wrap: break-word;'>0.311</td><td style='text-align: center; word-wrap: break-word;'>21587</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Poisson</td><td style='text-align: center; word-wrap: break-word;'>0.801</td><td style='text-align: center; word-wrap: break-word;'>0.882</td><td style='text-align: center; word-wrap: break-word;'>0.270</td><td style='text-align: center; word-wrap: break-word;'>0.053</td><td style='text-align: center; word-wrap: break-word;'>2422</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Equidistant</td><td style='text-align: center; word-wrap: break-word;'>0.932</td><td style='text-align: center; word-wrap: break-word;'>0.981</td><td style='text-align: center; word-wrap: break-word;'>0.641</td><td style='text-align: center; word-wrap: break-word;'>0.304</td><td style='text-align: center; word-wrap: break-word;'>2398</td></tr></table>

range 1, ..., T, a separate loss and set of gradients are generated, further increasing the dimensionality of the overall gradients.

### 3.2 Gradient Dimensionality Reduction

We propose a general attack framework for reducing the dimensionality of the gradients while trying to keep the useful information for differentiating members vs non-members. It is composed of two common techniques: (1) subsampling, which chooses the most informative gradients in a principled way, and (2) aggregation, which combines/compresses those informative gradients data. We name the framework Gradient attack based on Subsampling and Aggregation (GSA).

We then present a three-level taxonomy outlining where these two techniques can be applied: at the timestep level, across different layers within the target model, and within specific gradients of each layer, as detailed below.

(1) Timestep Level: As corroborated by prior studies [4, 12, 27, 30, 35], diffusion models display distinct reactions to member and non-member samples depending on the timestep. For instance, Carlini et al. [4] identified a ‘Goldilock’s zone’, which yielded the most effective results in their attack, to be within the range  $ t \in [50, 300] $. We believe that the importance of gradient data also varies across different timesteps. Therefore, sampling the timesteps that contain the most useful information will undoubtedly result in more accurate attack outcomes. We refer to the attacks conducted on the most effective gradient data within the ‘Gold zone’ as effective sampling. However, implementing effective sampling requires detecting the ‘Gold zone’ in the target model each time, and the optimal timesteps for achieving the best attack accuracy may vary across different models. As a result, we propose two alternative sampling methods: equidistant sampling and poisson sampling. In equidistant sampling, the denoising steps are selected at intervals of  $ T/|K| $ (K refer to the sampled timesteps set) for any given model. In poisson sampling, an average rate parameter  $ \lambda(|K|/T) $ is used to randomly generate intervals following an exponential distribution, thereby selecting  $ |K| $ steps from a total of T steps. We then present a simple case study to test and compare these three different sampling methods.


(2) Layer-wise Selection and Aggregation: Beyond timesteps, the layers within the model present another pivotal dimension for subsampling and aggregation. Recognizing the nuances captured across layers—from basic patterns in shallower layers to intricate details in deeper ones—it is deemed essential to selectively harness gradients from these layers, especially the informative ones, to optimize the attack model's training.

(3) Gradients within Each Layer: Within each layer of a neural network, there is typically no specific ordering of the gradient data. Therefore, it is more reasonable to treat these gradients as a set [15].

Case Study. Since existing attacks [4, 12, 27, 30, 35] heavily focus on timestep-level selection, we designed a case study to better examine how different subsampling methods impact attack performance. We evaluated the attack accuracy using three sampling methods: effective sampling, equidistant sampling, and poisson sampling. For effective sampling, it is necessary to first identify the 'Gold zone'. To achieve this, we recorded the attack results in every 20 step across the T denoising steps. The timestep with the best attack performance, along with the 10 surrounding timesteps, was then selected as the sampling points for effective sampling. For equidistant sampling, we set step 1 as the initial step and then sample timesteps at fixed intervals of  $ T/|K| $. In contrast, poisson sampling uses  $ |K|/T $ as the parameter  $ \lambda $ to sample from the T steps.

We select 5000 samples from CIFAR-10 dataset to train DDPM as target model. For each sampling method, we set the number of sampling steps  $ (|K|) $ to 10. In Table 2, we found that effective sampling achieves the highest attack accuracy, while poisson sampling has the lowest. This result aligns with our initial assumption that using gradient data sampled from the 'Gold zone'—the interval yielding the best attack results on individual timesteps—would lead to optimal performance. In contrast, poisson sampling's randomness may lead to poor attack outcomes if the sampled timesteps cannot effectively discriminate between members and non-members.

However, in table 2, we also present the time consumption for implementing different sampling methods. We found that although effective sampling achieves high attack accuracy, it takes nearly 8 times longer compared to equidistant and poisson sampling. This is because effective sampling requires precomputing the attack performance for numerous timesteps to identify the 'Gold zone'. Meanwhile, equidistant sampling only slightly reduces the ASR by 0.015 and the AUC by 0.011 compared to effective sampling, while being


Algorithm 1 GSA₁

Input: Target model denoted as  $ \epsilon_\theta $ with N layers, a equidistantly selected set of timesteps K, and a sample x.

1: for  $ t \in K $ do
2: Sample  $ \epsilon_t $ from Gaussian distribution
3: Compute  $ x_t $ based on Equation 1
4: Compute loss  $ L_t $ from Equation 2
5: end for

6:  $ \bar{L} \leftarrow \frac{1}{|K|} \sum_{t \in K} L_t $

7:  $ \mathcal{G} \leftarrow \left[ \left\| \frac{\partial \bar{L}}{\partial W_1} \right\|_2^2, \cdots \left\| \frac{\partial \bar{L}}{\partial W_N} \right\|_2^2 \right] $

Output:  $ \mathcal{G} $

more time-efficient. To balance effectiveness and efficiency, we use equidistant sampling to derive a subsampled timestep set K from the total diffusion steps T. Following this, for the timesteps in K, we can aggregate the gradients or losses generated at each timestep using statistical methods such as the mean, median, or trimmed mean to produce the final output. If the values being aggregated are the gradients from each timestep, the output can be directly used as the final output. However, if the aggregated values are the losses, the processed loss value needs to be used in backpropagation to extract gradient information for the final output.

### 3.3 Our Instantiations

We present two exemplary instantiations of the attack within the framework, representing two extreme points in the trade-off space between efficiency and effectiveness. We call them GSA₁ and GSA₂. GSA₁ performs more reduction, gaining efficiency but losing information. GSA₂ does less reduction, retaining effectiveness but at a cost to efficiency. In the GSA₁ method, although we equidistantly sample |K| timesteps from T, only a single gradient computation is required. This outcome is realized by in GSA₁ computing the loss, Lₜ, for each timestep present in K. Subsequently, we take the mean of these individual losses, represented as L, to perform backpropagation. This process eventually yields a solitary gradient vector. On the other hand, GSA₂ entails performing backpropagation and computing gradients for each timestep in K, and then using the mean of all gradient vectors, denoted as G, as the final output.

Note that we only slightly optimize our two instantiations in this paper because they are already very effective. We leave more detailed investigations of the design space and more effective proposals as future work.

Based on our detailed analysis of existing white-box attacks  $ [4, 27, 35] $, we first find that the optimal timesteps for mounting the most effective attacks vary depending on the specific dataset and diffusion model in question.

Consequently, we adopt the equidistant sampling strategy to select sample timesteps from the range  $ [1, T] $, denoted by a set K. This approach is designed to encompass timesteps that can distinctly differentiate between member and non-member samples, avoiding an exclusive focus on timesteps that are either too early or too late.

Algorithm 2 GSA₂

Input: Target model denoted as  $ \epsilon_\theta $ with N layers, a equidistantly selected set of timesteps K, and a sample x.

1:  $ \mathcal{G} \leftarrow [  $$ 
2: for  $ t \in K $ do
3: Sample  $ \epsilon_t $ from Gaussian distribution
4: Compute  $ x_t $ based on Equation 1
5: Compute loss  $ L_t $ from Equation 2
6:  $ \mathcal{G}_t = \left[ \left\| \frac{\partial L_t}{\partial W_1} \right\|_2^2, \ldots \left\| \frac{\partial L_t}{\partial W_N} \right\|_2^2 \right] $
7: end for
8:  $ \mathcal{G} \leftarrow \frac{1}{|K|} \sum_{t \in K} \mathcal{G}_t $

Output:  $ \mathcal{G} $

After getting loss from each selected step, we use backpropagation to compute the gradients for the model. Given the diverse nature of gradients within a layer, we aggregate the model's gradient information on a per-layer basis. That is, once the gradient information for a layer's parameters is obtained, the  $ \ell_{2} $-norm of these gradients is used as the representation for that layer's gradient information. This approach offers a dual advantage: it substantially reduces computational overhead while also holistically encapsulating that layer's gradient information.

This forms the basis of GSA₂ (given in Algorithm 2): for each timestep t in the set K, we calculate the per-layer gradient using the  $ \ell_2 $-norm, and then find their average.

However, this approach can still incur substantial computational costs when applied to large diffusion models and datasets — taking nearly 6 hours to execute on Imagen. To address this inefficiency, we preprocess the loss values from multiple timesteps before doing gradient computation. In light of this challenge, we introduce GSA₁ (outlined in Algorithm 1), which reduces the gradient extraction time for the Imagen model to less than 2 hours, significantly decreasing the computational time required.

## 4 Experimental Setup

### 4.1 Datasets

We use CIFAR-10, ImageNet, and MS COCO datasets. The use of CIFAR-10 allows for an easier comparison of attack results as it has been frequently employed in previous work [4, 12, 30, 35]. Both ImageNet and MS COCO serve as significant target datasets in the domain of image generation, with MS COCO used for training in various tasks, such as VQ-diffusion [18], Parti Finetuned [66], U-ViT-S/2 [2], and Imagen [48].

ImageNet dataset is a large-scale and diverse collection of images designed for image classification and object recognition tasks in the fields of machine learning and computer vision. When conducting experiments with the ImageNet dataset, researchers typically utilize a specific subset consisting of 1.2 million images for training and 50,000 images for validation, while an additional 100,000 images are reserved for testing. Considering the constraints on training resources and to ensure diversity in the training images, we opt


<div style="text-align: center;">Figure 2: Loss distribution for member vs. non-member samples across CIFAR-10, ImageNet, and MS COCO (from left to right), used by existing work [4, 27, 35]. Models use default settings from Table 3.</div>


to utilize the ImageNet test set as the training set for training the models in our work.

CIFAR-10 dataset comprises 10 categories of  $ 32 \times 32 $ color images, with each category containing 6,000 images. These categories include airplanes, automobiles, birds, cats, deer, dogs, frogs, horses, ships, and trucks. In total, the dataset consists of 60,000 images, of which 50,000 are designated for training and 10,000 for testing. The CIFAR-10 dataset is commonly employed as a benchmark for image classification and object recognition tasks in the fields of machine learning and computer vision.

MS COCO dataset contains over 200,000 labeled high-resolution images collected from the internet, with a total of 1.5 million object instances and 80 different object categories. The categories cover a wide range of common objects, including people, animals, vehicles, and household items, among others. The MS COCO dataset is noteworthy for its diversity and the complexity of its images and annotations. Images in the MS COCO dataset depict a wide variety of scenes and object layouts. In this experiment, we utilize all images from the MS COCO training set for model training. The first caption from the five associated with each image is selected as the corresponding textual description.

### 4.2 Training Setup

<div style="text-align: center;">Table 3: Default parameters used for the experiments.</div>


<table border=1 style='margin: auto; word-wrap: break-word;'><tr><td style='text-align: center; word-wrap: break-word;'>Parameters</td><td style='text-align: center; word-wrap: break-word;'>Unconditional Diffusion</td><td style='text-align: center; word-wrap: break-word;'>Unconditional Diffusion</td><td style='text-align: center; word-wrap: break-word;'>Imagen</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Channels</td><td style='text-align: center; word-wrap: break-word;'>128</td><td style='text-align: center; word-wrap: break-word;'>128</td><td style='text-align: center; word-wrap: break-word;'>128</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Diffusion steps</td><td style='text-align: center; word-wrap: break-word;'>1000</td><td style='text-align: center; word-wrap: break-word;'>1000</td><td style='text-align: center; word-wrap: break-word;'>1000</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Dataset</td><td style='text-align: center; word-wrap: break-word;'>CIFAR-10</td><td style='text-align: center; word-wrap: break-word;'>ImageNet</td><td style='text-align: center; word-wrap: break-word;'>MS COCO</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Training data size</td><td style='text-align: center; word-wrap: break-word;'>8000</td><td style='text-align: center; word-wrap: break-word;'>8000</td><td style='text-align: center; word-wrap: break-word;'>30000</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Resolution</td><td style='text-align: center; word-wrap: break-word;'>32</td><td style='text-align: center; word-wrap: break-word;'>64</td><td style='text-align: center; word-wrap: break-word;'>64</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Learning rate</td><td style='text-align: center; word-wrap: break-word;'>$ 1 \times 10^{-4} $</td><td style='text-align: center; word-wrap: break-word;'>$ 1 \times 10^{-4} $</td><td style='text-align: center; word-wrap: break-word;'>$ 1 \times 10^{-4} $</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Batch size</td><td style='text-align: center; word-wrap: break-word;'>64</td><td style='text-align: center; word-wrap: break-word;'>64</td><td style='text-align: center; word-wrap: break-word;'>64</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Noise schedule</td><td style='text-align: center; word-wrap: break-word;'>linear</td><td style='text-align: center; word-wrap: break-word;'>linear</td><td style='text-align: center; word-wrap: break-word;'>linear, cosine</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Learning rate schedule</td><td style='text-align: center; word-wrap: break-word;'>cosine</td><td style='text-align: center; word-wrap: break-word;'>cosine</td><td style='text-align: center; word-wrap: break-word;'>cosine</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Training time</td><td style='text-align: center; word-wrap: break-word;'>400 epochs</td><td style='text-align: center; word-wrap: break-word;'>400 epochs</td><td style='text-align: center; word-wrap: break-word;'>600,000 steps</td></tr></table>

We tabulated the default training parameters for the unconditional diffusion model on CIFAR-10 and ImageNet, and for Imagen on MS COCO, in Table 3. Given that we have employed ASR (Accuracy) as our evaluation metric, we endeavor to maintain a balance between the quantities of the member set and the non-member set to ensure the precision of model validation. The structure for the unconditional diffusion model aligns with those from the diffusers library [60] in Huggingface. Imagen is based on the open-source implementation by Phil Wang et al. $ ^{3} $, and we have retained consistency in its configuration. All experiments were conducted using two NVIDIA A100 GPUs.


### 4.3 Metrics

In the process of comparing experimental results, we employ Attack Success Rate (ASR) [6], Area Under the ROC Curve (AUC), and True-Positive Rate (TPR) values under fixed low False-Positive Rate (FPR) as evaluation metrics.

In our experiments, we ensure an equal number of member and non-member image samples. Given the balanced nature of our dataset and the stability of ASR in such contexts, we employ ASR as our primary evaluation metric.

We note that most instances MIAs on diffusion models use the AUC metric for evaluation [4, 12, 27, 30, 35]. Likewise, in assessing the merits of our work in Section 5.1, we will also use AUC as one of our assessment metrics. Additionally, as Carlini et al. [3] argued that TPR under a low FPR scenario is a key evaluation criterion, we also use TPR at 1% FPR and 0.1% FPR, respectively.

## 5 Evaluation Results

### 5.1 Comparison with Existing Methods

<div style="text-align: center;">Table 4: Existing white-box attacks on the CIFAR-10 dataset are benchmarked using four distinct metrics. LiRA $ ^{*} $, LSA $ ^{*} $, GSA $ _{1} $, and GSA $ _{2} $ are all obtained under the same conditions.</div>


<table border=1 style='margin: auto; word-wrap: break-word;'><tr><td rowspan="2">Attack method</td><td colspan="4">CIFAR-10</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>ASR $ ^{\dagger} $</td><td style='text-align: center; word-wrap: break-word;'>AUC $ ^{\dagger} $</td><td style='text-align: center; word-wrap: break-word;'>TPR@1%FPR(\%)^{ $ \dagger $}</td><td style='text-align: center; word-wrap: break-word;'>TPR@0.1%FPR(\%)^{ $ \dagger $}</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Baseline</td><td style='text-align: center; word-wrap: break-word;'>0.736</td><td style='text-align: center; word-wrap: break-word;'>0.801</td><td style='text-align: center; word-wrap: break-word;'>5.65</td><td style='text-align: center; word-wrap: break-word;'>-</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>LiRA</td><td style='text-align: center; word-wrap: break-word;'>-</td><td style='text-align: center; word-wrap: break-word;'>0.982</td><td style='text-align: center; word-wrap: break-word;'>5(5M) 99(102M)</td><td style='text-align: center; word-wrap: break-word;'>7.1</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Strong LiRA</td><td style='text-align: center; word-wrap: break-word;'>-</td><td style='text-align: center; word-wrap: break-word;'>0.997</td><td style='text-align: center; word-wrap: break-word;'>-</td><td style='text-align: center; word-wrap: break-word;'>29.4</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>LiRA $ ^{*} $</td><td style='text-align: center; word-wrap: break-word;'>0.626</td><td style='text-align: center; word-wrap: break-word;'>0.71</td><td style='text-align: center; word-wrap: break-word;'>1.45</td><td style='text-align: center; word-wrap: break-word;'>0.25</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>LSA $ ^{*} $</td><td style='text-align: center; word-wrap: break-word;'>0.83</td><td style='text-align: center; word-wrap: break-word;'>0.909</td><td style='text-align: center; word-wrap: break-word;'>13.77</td><td style='text-align: center; word-wrap: break-word;'>0.925</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>GSA $ _{1} $</td><td style='text-align: center; word-wrap: break-word;'>0.993</td><td style='text-align: center; word-wrap: break-word;'>0.999</td><td style='text-align: center; word-wrap: break-word;'>99.7</td><td style='text-align: center; word-wrap: break-word;'>82.9</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>GSA $ _{2} $</td><td style='text-align: center; word-wrap: break-word;'>0.988</td><td style='text-align: center; word-wrap: break-word;'>0.999</td><td style='text-align: center; word-wrap: break-word;'>97.88</td><td style='text-align: center; word-wrap: break-word;'>58.57</td></tr></table>

 $ ^{3} $Code available at https://github.com/lucidrains/imagen-pytorch


<div style="text-align: center;">Figure 3: The left and right columns display the visualization of high-dimensional gradient information using t-SNE after GSA₁ and GSA₂ have respectively executed attacks on the three datasets (using the output from the last layer of our attack model). For all six attacks, it is observed that member and non-member samples are distinctly differentiated when reaching the training steps defined by the default settings (as referenced in Table 3).</div>
