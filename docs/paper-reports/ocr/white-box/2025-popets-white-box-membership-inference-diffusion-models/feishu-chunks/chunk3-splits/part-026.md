GAN-Leaks [5] is a straightforward attack approach that can be universally applied across diverse settings and generative networks. However, its reliability is contingent upon the quality of the reconstructed image, which can be significantly influenced by the complexity of the original image. A complex image, even if it is from the training set, might encompass intricate details leading to a substantial discrepancy between the reconstructed and query images, resulting in misclassification. To address this, the authors employed a calibration technique to rectify such inaccuracies, ensuring commendable attack accuracy for GAN-Leaks on smaller datasets (comprising fewer than 1000 images). Nonetheless, when applied to extensive datasets, the efficacy of GAN-Leaks diminishes.

### D.4 Likelihood-based Attack

The log-likelihood of the samples can be used to conduct a membership inference attack. The formula is given by:

 $$ \log p(x)=\log p_{T}(x_{T})-\int_{0}^{T}\nabla\cdot\tilde{\mathbf{f}}_{\theta}(x_{t},t)d t. $$ 

This equation was originally proposed by Song et al. [57]. If the log-likelihood value exceeds the threshold, the sample is inferred as a member. The term  $ \nabla \cdot \tilde{f}_{\theta}(x_t, t) $ is estimated using the Skilling-Hutchinson trace estimator, as suggested by Grathwohl et al. [16].

### E Additional Information for Ablation Study

We employed GSA $ _{1} $ and GSA $ _{2} $ on CIFAR-10, ImageNet, and MS COCO to further conduct layer-wise reduction as mentioned in Section 3.2, aiming to reduce computational time and resource consumption. The experimental results are presented in Figure 9.


<div style="text-align: center;">Figure 9: Using GSA₁ and GSA₂ on CIFAR-10, ImageNet, and MS COCO, we can reduce the layers needed for gradient extraction without compromising attack effectiveness. Notably, for attacks on ImageNet-trained DDPM, only 30% of the layers are required for a successful attack.</div>
