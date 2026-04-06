details in Table 3. The results show that both DP-SGD and RandAugment effectively defend against LSA as well as our GSA $ _{1} $ and GSA $ _{2} $, reducing the attack ASR and AUC to levels similar to random guessing. The defense effects are also visualized in Figure 7.

## 8 Limitation

As shown in Table 5, while GSA₁ and GSA₂ can yield satisfactory results with limited computational resources, they are still constrained by their time consumption. Even after implementing subsampling and aggregation across three dimensions, the process of gradient extraction remains time-intensive for larger datasets and more intricate models compared to simply computing the loss. Future studies are anticipated to explore these areas further and identify additional dimensions for reduction. Additionally, the methods employed in this study, GSA₁ and GSA₂, necessitate gradient information from the model for a successful attack. This suggests that requiring complete parameters of the target model during the attack is a rather stringent condition.

## 9 Related Work

Diffusion Model. Diffusion model is an emergent generative network originally inspired by diffusion processes from non-equilibrium thermodynamics [53]. Distinguished from previous Generative Adversarial Networks (GANs) [7, 10] and Variational Autoencoders (VAEs) [32], the objective of the diffusion model is to approximate the actual data distribution by engaging a parameterized reverse process that aligns with a simulated diffusion process.

Diffusion models can be connected with score-based models [57], generating samples by estimating the gradients of the data distribution and utilizing this gradient information to guide the process of noise addition, thereby producing samples of superior quality. Moreover, the diffusion model showcases the capability to generate images conditioned on specific inputs [10, 36, 42, 47].

Apart from generating images, diffusion models are capable of performing specific area retouching in images according to given specifications, hence effectively accomplishing inpainting  $ [34] $ tasks. Nowadays, advancements in diffusion models have granted them the ability to generate not only static images but also videos  $ [24] $ and 3D scenes  $ [17] $.
