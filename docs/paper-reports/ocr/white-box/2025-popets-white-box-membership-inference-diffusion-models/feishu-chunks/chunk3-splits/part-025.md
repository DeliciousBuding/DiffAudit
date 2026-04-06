Based on  $ \tilde{\ell}_{t,x_0} $, the authors proposed SecMI $ _{stat} $ and SecMI $ _{NNs} $, which employs the threshold-based attack approach [64] and neural network-based attack method [52], respectively.


### D.2 Proximal Initialization Attack (PIA)

Building upon the work of Duan et al. [12], Kong et al. [30] also identified the deterministic properties inherent to the DDIM model [29, 54, 57]. In the DDIM framework, given  $ x_0 $ and  $ x_k $, it is feasible to utilize these two points to predict any other ground truth point  $ x_t $ [30]. Consequently, this methodology employs the  $ \ell_p $-norm to compute the distance between any ground truth point  $ x_{t-t'} $ and its predicted counterpart  $ x'_{t-t'} $. After leveraging the ground truth extraction properties of DDIM [29] and utilizing the sampling formula from [54], the equation to compute the distance is given by:

 $$ R_{t,p}=\left\|\epsilon_{\theta}(x_{0},0)-\epsilon_{\theta}(\sqrt{\bar{\alpha}_{t}}x_{0}+\sqrt{1-\bar{\alpha}_{t}}\epsilon_{\theta}(x_{0},0),t)\right\|_{p}. $$ 

The notation in the above equation is consistent with the DDPM model, where  $ R_{t,p} $ denotes the distance. Given that  $ \epsilon $ is initialized at  $ t = 0 $, this method is termed the Proximal Initialization Attack (PIA). When normalizing  $ \epsilon_{\theta}(x_0, 0) $, it is referred to as PIAN (PIA Normalize). This work employs a threshold-based [64] attack approach.

Compared to SecMI [12], the attack accuracy has seen a notable improvement. Yet, when juxtaposed with white-box attacks [4, 27], the success rate of this model attack remains suboptimal.

### D.3 GAN-Leaks

GAN-Leaks [5] is a pivotal work in the realm of MIA against GAN models. This work meticulously breaks down attack scenarios into categories based on the level of access to the latent code, generator, and discriminator. For each category, from full black-box to accessible discriminator, GAN-Leaks presents tailored attack methodologies. This work formalizes MIA as an optimization problem. For a given query sample, the goal is to identify the closest reconstruction by optimizing within the generator's output space. A query sample is deemed a member if its reconstruction error is smaller. This can be represented as:

 $$ \mathcal{R}(x|\mathcal{G}_{v})=G_{v}(z^{*}),\mathrm{w h e r e}z^{*}=\underset{\tau}{\operatorname{a r g m i n}}L(x,G_{v}(z)) $$ 

where  $ L(\cdot,\cdot) $ represents the general distance metric,  $ G_{v} $ denotes the victim generator, and  $ z^{*} $ is the optimal estimate.
