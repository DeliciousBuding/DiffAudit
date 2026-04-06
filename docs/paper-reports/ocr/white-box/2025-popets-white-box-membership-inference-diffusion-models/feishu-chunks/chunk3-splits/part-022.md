In Equation 7,  $ \mu_{\theta}(x_t', t) $ is the predicted mean of the distribution for the sample  $ x_t'_{-1} $ at the preceding timestep, and  $ \Sigma_{\theta}(x_t', t) $ denotes the covariance matrix of this distribution. In the original study,  $ \Sigma_{\theta}(x_t', t) = \sigma_t^2 I $ is set as untrained time-dependent constants. Consequently, our primary attention is dedicated to the mean  $ \mu_{\theta}(x_t', t) $ of the predictive network  $ p_{\theta} $. By expanding the aforementioned posterior probability using a probability density function, we can derive the mean and variance of the posterior probability. Given that the variance in  $ p_{\theta}(x_t'_{-1}|x_t') $ is associated with  $ \beta_t $ and is a deterministic value, our attention is solely on the mean.

When we express  $ x_0 $ in terms of  $ x_t $ (from Equation 6) within the mean  $ \tilde{\mu}(x_t, x_0) $, the revised  $ \tilde{\mu}(x_t, x_0) $ then only consists of  $ x_t $ and random noise  $ \epsilon_t $. Given that  $ x_t $ is known at the current time step  $ t $, the task can be reformulated as predicting the random variable  $ \epsilon_t $. The  $ \tilde{\mu}(x_t, x_0) $ can be represented as:

 $$ \tilde{\mu}(x_{t},x_{0})=\frac{1}{\sqrt{\alpha_{t}}}(x_{t}-\frac{\beta_{t}}{\sqrt{1-\bar{\alpha}_{t}}}\epsilon_{t}) $$ 

Concurrently,  $ \mu_{\theta}(x_{t}^{\prime}, t) $ can be expressed as:

 $$ \boldsymbol{\mu}_{\theta}(x_{t}^{\prime},t)=\frac{1}{\sqrt{\alpha_{t}}}(x_{t}^{\prime}-\frac{\beta_{t}}{\sqrt{1-\bar{\alpha}_{t}}}\epsilon_{\theta}(x_{t}^{\prime},t)) $$ 

Thus, the initial loss function for calculating the prediction of  $ \mu_{\theta}(x_{t}^{\prime}, t) $ can be reformulated into an equation predicting the noise  $ \epsilon_{\theta}(x_{t}, t) $.

$$  \begin{aligned}&L_{t}(\theta)\\=&\mathbb{E}_{x_{0},\epsilon}\left[\frac{\beta_{t}^{2}}{2\sigma_{t}^{2}\alpha_{t}\left(1-\alpha_{t}\right)}\|\epsilon_{t}-\epsilon_{\theta}(\sqrt{\bar{\alpha}_{t}}x_{0}+\sqrt{1-\bar{\alpha}_{t}}\epsilon_{t},t)\|^{2}\right]\end{aligned}   \tag*{(8)}$$

It has been observed that DDPM [21] relies solely on the marginals  $ q(x_t|x_0) $ during sampling and loss optimization, rather than directly utilizing the joint probability  $ q(x_1:T|x_0) $. Given that many joint distributions share the same marginals, DDIM [54] proposed a non-Markovian forward process as an alternative to the Markovian noise addition process inherent in DDPM. However, the final non-Markovian noise addition is structurally identical to that of DDPM, with the only distinction being the sampling process.

 $$ x_{t-1}^{\prime}=\sqrt{\bar{\alpha}_{t-1}}f_{\theta}(x_{t}^{\prime},t)+\sqrt{1-\bar{\alpha}_{t-1}-\sigma_{t}^{2}}\cdot\epsilon_{\theta}(x_{t}^{\prime},t)+\sigma_{t}\epsilon $$
