[63] YE, J., MADDI, A., MURAKONDA, S. K., BINDSCHAEDLER, V., AND SHOKRI, R. Enhanced membership inference attacks against machine learning models. In Proceedings of the 2022 ACM SIGSAC Conference on Computer and Communications Security (2022), pp. 3093–3106.

[64] Yeom, S., GIACOMELLI, I., FREDRIKSON, M., AND JHA, S. Privacy risk in machine learning: Analyzing the connection to overfitting. In 2018 IEEE 31st computer security foundations symposium (CSF) (2018), IEEE, pp. 268–282.

[65] YOSINSKI, J., CLUNE, J., BENGIO, Y., AND LIPSON, H. How transferable are features in deep neural networks? Advances in neural information processing systems 27 (2014).

[66] YU, J., XU, Y., KOH, J. Y., LUONG, T., BAID, G., WANG, Z., VASUDEVAN, V., KU, A., YANG, Y., AYAN, B. K., ET AL. Scaling autoregressive models for content-rich text-to-image generation. arXiv preprint arXiv:2206.10789 (2022).

### A Additional Information for Denoising Diffusion Probabilistic Model

The operating mechanism of the diffusion model entails the model learning the posterior probability of the forward process, thereby achieving the denoising process. In the forward noise addition process, assume that there is a sample  $ x_{t-1} $ at time point  $ t-1 $. Then  $ x_{t} $ can be represented as:

$$  x_{t}=\sqrt{\alpha_{t}}x_{t-1}+\sqrt{1-\alpha_{t}}\epsilon,\;\epsilon\sim\mathcal{N}(0,1)   \tag*{(5)}$$

Since  $ \epsilon $ is a random noise, we can unroll the recursive definition and derive  $ x_t $ directly from  $ x_0 $ (the original image) and time step  $ t $ (and  $ \tilde{\alpha}_t = \prod_{i=1}^t \alpha_i $):

$$  x_{t}=\sqrt{\bar{\alpha}_{t}}x_{0}+\sqrt{1-\bar{\alpha}_{t}}\epsilon_{t},\epsilon_{t}\sim N(0,1)   \tag*{(6)}$$

The reverse process can be described as:

 $$ p_{\theta}(x_{0:T})=p(x_{T})\prod_{t=1}^{T}p_{\theta}(x_{t-1}|x_{t}) $$ 

where  $ x_{T}^{\prime} \sim \mathcal{N}(0, I) $. The image  $ x_{t-1}^{\prime} $ at t-1 can be restored from  $ x_{t}^{\prime} $ at time t, and can be represented as:

$$  p_{\theta}(x_{t-1}^{\prime}|x_{t}^{\prime})=\mathcal{N}(x_{t-1}^{\prime};\boldsymbol{\mu}_{\theta}(x_{t}^{\prime},t),\boldsymbol{\Sigma}_{\theta}(x_{t}^{\prime},t))   \tag*{(7)}$$

In the reverse process, the model aims to use the posterior probability of the forward process to guide the denoising process.

 $$ q(x_{t-1}|x_{t},x_{0})=N(x_{t-1};\bar{\mu}(x_{t},x_{0}),\bar{\beta}_{t}\mathbf{I}) $$ 

As the  $ \bar{\beta}_{t} $ in the posterior probability is also a determined value, the model only needs to learn  $ \bar{\mu}(x_{t}, t) $.
