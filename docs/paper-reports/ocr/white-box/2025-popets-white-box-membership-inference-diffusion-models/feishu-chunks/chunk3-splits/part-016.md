In this work, we propose a membership inference attack framework that utilizes the norm of the gradient information in diffusion models and presents two specific attack examples, namely GSA₁ and GSA₂. We find that the attack performance on the DDPM and Imagen, trained with the CIFAR-10, ImageNet, and MS COCO datasets, is quite remarkable according to all four evaluation metrics. We posit that a diffusion model's gradient information is more indicative of overfitting to a data point than its loss, hence employing gradient information in MIA could lead to higher success rates. This assertion aligns with the nuanced understanding of model dynamics in the machine learning field. Compared to existing white box loss-based attack methodologies [4, 27, 35], our proposed approach demonstrates superior performance under identical model configurations, showcasing efficiency and stability across various datasets and models. This paper introduces the perspective of leveraging gradients for MIA and hopes to inspire valuable follow-up works in this direction.


### Acknowledgment

We thank the reviewers for their valuable comments and suggestions. This work is partially supported by NSF CNS-2350332, CNS-2350333, and CNS-2220433.

## References

[1] ABADI, M., CHU, A., GOODFELLOW, L., McMAHAN, H. B., MIRONOV, I., TALWAR, K., AND ZHANG, L. Deep learning with differential privacy. In Proceedings of the 2016 ACM SIGSAC conference on computer and communications security (2016), pp. 308–318.

[2] BAO, F., NIE, S., XUE, K., CAO, Y., LI, C., SU, H., AND ZHU, J. All are worth words: A vit backbone for diffusion models, 2023.

[3] CARLINI, N., CHIEN, S., NASR, M., SONG, S., TERZIS, A., AND TRAMER, F. Membership inference attacks from first principles. In 2022 IEEE Symposium on Security and Privacy (SP) (2022), IEEE, pp. 1897–1914.

[4] CARLINI, N., HAYES, J., NASR, M., JAGIELSKI, M., SEHWAG, V., TRAMER, F., BALLE, B., IPPOLITO, D., AND WALLACE, E. Extracting training data from diffusion models. arXiv preprint arXiv:2301.13188 (2023).

[5] CHEN, D., YU, N., ZHANG, Y., AND FRITZ, M. Gan-leaks: A taxonomy of membership inference attacks against generative models. In Proceedings of the 2020 ACM SIGSAC conference on computer and communications security (2020). pp. 343–362.

[6] CHOQUETTE-CHOO, C. A., TRAMER, F., CARLINI, N., AND PAPERNOT, N. Label-only membership inference attacks. In International conference on machine learning (2021), PMLR, pp. 1964–1974.

[7] CRESWELL, A., WHITE, T., DUMOULIN, V., ARULKUMARAN, K., SENGUPTA, B., AND BHARATH, A. A. Generative adversarial networks: An overview. IEEE signal processing magazine 35, 1 (2018), 53–65.
