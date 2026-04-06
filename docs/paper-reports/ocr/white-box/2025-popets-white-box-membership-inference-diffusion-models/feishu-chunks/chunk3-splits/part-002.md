5.1.2 Timestep Selection. Moreover, the ‘time zone’ demonstrating discernible differences in the loss distribution between members and non-members vary across different models and datasets [4, 12, 27, 35]. Consequently, to achieve a more potent attack, it becomes imperative to extract the loss and establish thresholds or distributions for each timestep using shadow models, aiming to pinpoint the most efficacious ‘time zone’. In contrast, both GSA₁ and GSA₂ execute attacks by solely harnessing the gradient information derived from equidistant sampling timesteps across the T diffusion steps, achieving similar attack accuracy in just one-thirtieth of the time. Given a consistent dataset size and model architecture, extracting loss across T steps takes 36 hours. In contrast, GSA₁ and GSA₂ achieve the same accuracy level in less than 1 hour by extracting gradients from 10 equidistant sampling timesteps.

To further demonstrate that the optimal timestep for distinguishing between member and non-member samples using loss varies across different datasets and models. We plot the loss distribution


<div style="text-align: center;">(a) Impact of training epoch</div>


<div style="text-align: center;">(b) Impact of  $ |K| $</div>


<div style="text-align: center;">Figure 4: “-I-” and “-C-” denote experiments with ImageNet and CIFAR-10 datasets. Panel (a) (left) reveals that attacks are more effective when shadow and target models closely fit the training data; (right) however, increased fitting disparities between them weaken the attack. Panel (b) shows that greater sampling frequency boosts the attack’s effectiveness, possibly due to acquiring finer data and getting more informative timestep.</div>
