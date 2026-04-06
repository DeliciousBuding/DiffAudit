Training on Different Epochs. In Figure 5b, consistent with the two attack scenarios posited in Section 5.2, we analyze the effect of training steps on the attack success rate for Imagen models. Our categorization is premised on the attacker's knowledge of the target model's training steps. Notably, when the attacker is uncertain about the number of training steps of the target model, we set the training steps of the target model to a fixed value (in this instance, 400,000 steps). This experimental setup aligns with that of Section 5.2.

Consistent with previous experiments using the unconditional diffusion models, a large proportion of the attack success rate for the Imagen model is influenced by the training steps of the target


<div style="text-align: center;">(a) Impact of diffusion steps</div>


<div style="text-align: center;">(b) Impact of training epoch</div>


<div style="text-align: center;">(c) Impact of  $ |K| $</div>


<div style="text-align: center;">Figure 5: Notations “-I-” and “-C-” are consistent with those in Figure 4a. Panel (a) suggests that increasing the number of diffusion steps, which decelerates convergence, results in a reduced attack success rate. Panel (b) reinforces findings from Figure 4a: enhanced data-fitting by both the shadow and target models boosts the attack’s efficacy. However, when there are disparities in the data fitting, the efficacy diminishes. Panel (c) shows that augmenting the sampling steps for Imagen—thus acquiring more information—significantly improves the attack’s success rate.</div>


<div style="text-align: center;">Table 5: The table presents the performance results of GSA₁ and GSA₂, trained on three different datasets and evaluated using four distinct evaluation metrics.</div>
