# Paper Index

This file is the human-readable paper-by-paper index used to keep GitHub and the Feishu status document aligned.

## Black-Box

### Towards Black-Box Membership Inference Attack for Diffusion Models

- File: [2024-arxiv-towards-black-box-membership-inference-diffusion-models.pdf](black-box/2024-arxiv-towards-black-box-membership-inference-diffusion-models.pdf)
- Main idea: works under a strict black-box setting where the attacker only queries an image variation API, then uses reconstruction consistency to tell whether a sample was in training.
- DiffAudit relevance: this is the cleanest API-only black-box baseline for the repository's `variation` line.
- Open-source: not found

### Membership Inference on Text-to-Image Diffusion Models via Conditional Likelihood Discrepancy

- File: [2024-neurips-clid-membership-inference-text-to-image-diffusion.pdf](black-box/2024-neurips-clid-membership-inference-text-to-image-diffusion.pdf)
- Main idea: introduces `CLiD`, which measures conditional likelihood discrepancy under a text condition instead of only comparing image-side signals.
- DiffAudit relevance: it is the main text-to-image black-box / conditional audit reference for caption-conditioned settings.
- Open-source: [zhaisf/CLiD](https://github.com/zhaisf/CLiD)

### Black-box Membership Inference Attacks against Fine-tuned Diffusion Models

- File: [2025-ndss-black-box-membership-inference-fine-tuned-diffusion-models.pdf](black-box/2025-ndss-black-box-membership-inference-fine-tuned-diffusion-models.pdf)
- Main idea: studies realistic black-box attacks against fine-tuned diffusion models and shows that fine-tuning can substantially amplify membership leakage.
- DiffAudit relevance: this is the current main black-box reproduction target behind the repository `recon` line.
- Open-source: [py85252876/Reconstruction-based-Attack](https://github.com/py85252876/Reconstruction-based-Attack)

### Membership Inference Attacks for Face Images Against Fine-tuned Latent Diffusion Models

- File: [2025-visapp-membership-inference-face-fine-tuned-latent-diffusion-models.pdf](black-box/2025-visapp-membership-inference-face-fine-tuned-latent-diffusion-models.pdf)
- Main idea: narrows the threat model to face-focused fine-tuning and analyzes how prompt choice, watermarking, and generation settings affect attack success.
- DiffAudit relevance: useful as a scenario-specific extension for privacy audit on sensitive identity-centric deployments.
- Open-source: [osquera/MIA_SD](https://github.com/osquera/MIA_SD)

## Gray-Box

### Are Diffusion Models Vulnerable to Membership Inference Attacks?

- File: [2023-icml-secmi-membership-inference-diffusion-models.pdf](gray-box/2023-icml-secmi-membership-inference-diffusion-models.pdf)
- Main idea: establishes that diffusion models need different attack signals than GANs and proposes `SecMI`, which compares posterior estimation error across timesteps.
- DiffAudit relevance: foundational gray-box baseline and the repository's earliest implemented attack line.
- Open-source: [jinhaoduan/SecMI](https://github.com/jinhaoduan/SecMI)

### SIDE: Surrogate Conditional Data Extraction from Diffusion Models

- File: [2024-arxiv-side-extracting-training-data-unconditional-diffusion-models.pdf](gray-box/2024-arxiv-side-extracting-training-data-unconditional-diffusion-models.pdf)
- Main idea: shows how unconditional diffusion models can still leak memorized training data by constructing surrogate conditions from generated samples.
- DiffAudit relevance: broadens the audit scope from membership inference to training-data extraction and memorization.
- Open-source: not found

### Unveiling Structural Memorization: Structural Membership Inference Attack for Text-to-Image Diffusion Models

- File: [2024-arxiv-structural-memorization-membership-inference-text-to-image-diffusion.pdf](gray-box/2024-arxiv-structural-memorization-membership-inference-text-to-image-diffusion.pdf)
- Main idea: argues that text-to-image models often memorize structure rather than raw pixels, then attacks using structural similarity signals.
- DiffAudit relevance: useful when pixel-level reconstruction signals are weak but layout or semantic skeleton signals remain strong.
- Open-source: not found

### An Efficient Membership Inference Attack for the Diffusion Model by Proximal Initialization

- File: [2024-iclr-pia-proximal-initialization.pdf](gray-box/2024-iclr-pia-proximal-initialization.pdf)
- Main idea: proposes `PIA`, which uses proximal initialization to extract stronger low-query membership signals from diffusion trajectories.
- DiffAudit relevance: this is the main gray-box implementation line after `SecMI`.
- Open-source: [kong13661/PIA](https://github.com/kong13661/PIA)

### Score-based Membership Inference on Diffusion Models

- File: [2025-arxiv-sima-score-based-membership-inference-diffusion-models.pdf](gray-box/2025-arxiv-sima-score-based-membership-inference-diffusion-models.pdf)
- Main idea: studies how score / noise predictions themselves leak training-set proximity and turns that into a one-query `SimA` attack.
- DiffAudit relevance: strong score-space comparison point for future gray-box benchmarking.
- Open-source: not found

### Noise Aggregation Analysis Driven by Small-Noise Injection: Efficient Membership Inference for Diffusion Models

- File: [2025-arxiv-small-noise-injection-membership-inference-diffusion-models.pdf](gray-box/2025-arxiv-small-noise-injection-membership-inference-diffusion-models.pdf)
- Main idea: injects small perturbations and compares the aggregation pattern of predicted noise to distinguish members from non-members.
- DiffAudit relevance: promising low-cost query-efficient candidate for future batch audit experiments.
- Open-source: not found

### CDI: Copyrighted Data Identification in Diffusion Models

- File: [2025-cvpr-cdi-copyrighted-data-identification-diffusion-models.pdf](gray-box/2025-cvpr-cdi-copyrighted-data-identification-diffusion-models.pdf)
- Main idea: moves from single-image MIA to dataset-level evidence by aggregating attack signals and statistical tests for copyrighted-data identification.
- DiffAudit relevance: highly aligned with evidence-grade audit and compliance reporting.
- Open-source: [sprintml/copyrighted_data_identification](https://github.com/sprintml/copyrighted_data_identification)

### Noise as a Probe: Membership Inference Attacks on Diffusion Models Leveraging Initial Noise

- File: [2026-arxiv-noise-as-a-probe-membership-inference-diffusion-models.pdf](gray-box/2026-arxiv-noise-as-a-probe-membership-inference-diffusion-models.pdf)
- Main idea: leverages semantics preserved in initial noise to build a new gray-box membership signal, especially for small-data fine-tuning.
- DiffAudit relevance: valuable for extending audit methods beyond mid-trajectory or reconstruction-only evidence.
- Open-source: not found

### No Caption, No Problem: Caption-Free Membership Inference via Model-Fitted Embeddings

- File: [2026-openreview-mofit-caption-free-membership-inference.pdf](gray-box/2026-openreview-mofit-caption-free-membership-inference.pdf)
- Main idea: removes the need for original training captions by constructing model-fitted embeddings that still expose caption-conditioned membership signals.
- DiffAudit relevance: practical for real audits where only the image is available.
- Open-source: [JoonsungJeon/MoFit](https://github.com/JoonsungJeon/MoFit)

## White-Box

### Finding NeMo: Localizing Neurons Responsible For Memorization in Diffusion Models

- File: [2024-neurips-finding-nemo-localizing-memorization-neurons-diffusion-models.pdf](white-box/2024-neurips-finding-nemo-localizing-memorization-neurons-diffusion-models.pdf)
- Main idea: identifies specific neurons that are responsible for memorized examples in diffusion models and shows that ablating them can reduce leakage.
- DiffAudit relevance: key white-box interpretability and mitigation reference.
- Open-source: [ml-research/localizing_memorization_in_diffusion_models](https://github.com/ml-research/localizing_memorization_in_diffusion_models)

### White-box Membership Inference Attacks against Diffusion Models

- File: [2025-local-mirror-white-box-membership-inference-diffusion-models.pdf](white-box/2025-local-mirror-white-box-membership-inference-diffusion-models.pdf)
- Main idea: studies white-box diffusion MIA using gradient-side features rather than only loss values.
- DiffAudit relevance: direct white-box threat-model reference; retained as a local mirror variant of the PoPETs paper.
- Open-source: not found

### White-box Membership Inference Attacks against Diffusion Models

- File: [2025-popets-white-box-membership-inference-diffusion-models.pdf](white-box/2025-popets-white-box-membership-inference-diffusion-models.pdf)
- Main idea: official PoPETs version of the same white-box diffusion MIA line, emphasizing gradient features and stronger white-box leakage than black-box settings.
- DiffAudit relevance: current main white-box target paper.
- Open-source: not found

## Context

### DiffAudit Product Requirements Document

- File: [diffaudit-product-requirements.pdf](context/diffaudit-product-requirements.pdf)
- Main idea: defines the product framing, audit surfaces, and expected system modules for DiffAudit as a privacy-risk audit system rather than a pure benchmark repo.
- DiffAudit relevance: local product and scope baseline for research-to-system transition.
- Open-source: not applicable

### DiffAudit Team Onboarding Primer

- File: [diffaudit-team-onboarding.pdf](context/diffaudit-team-onboarding.pdf)
- Main idea: aligns new contributors on project goals, terminology, and the difference between black-box, gray-box, and white-box audit routes.
- DiffAudit relevance: local handoff and onboarding context for future agents and collaborators.
- Open-source: not applicable

## Survey / Archive

### Tracing the Roots: Leveraging Temporal Dynamics in Diffusion Trajectories for Origin Attribution

- File: [2025-tracing-the-roots-leveraging-temporal-dynamics-in-diffusion-trajectories-for-origin-attribution.pdf](survey/2025-tracing-the-roots-leveraging-temporal-dynamics-in-diffusion-trajectories-for-origin-attribution.pdf)
- Main idea: studies diffusion trajectories for origin attribution, unifying membership discrimination, model attribution, and external-source recognition.
- DiffAudit relevance: useful as a broader provenance-style reference around diffusion trajectory evidence.
- Open-source: not found

### DP-DocLDM: Differentially Private Document Image Generation using Latent Diffusion Models

- File: [2025-dp-docldm-differentially-private-document-image-generation-using-latent-diffusion-models.pdf](survey/2025-dp-docldm-differentially-private-document-image-generation-using-latent-diffusion-models.pdf)
- Main idea: combines latent diffusion and differential privacy to generate privacy-preserving document images for downstream tasks.
- DiffAudit relevance: defense-side reference for privacy-preserving diffusion training.
- Open-source: not found

### Privacy-Preserving Low-Rank Adaptation against Membership Inference Attacks for Latent Diffusion Models

- File: [2025-privacy-preserving-low-rank-adaptation-against-membership-inference-attacks-for-latent-diffusion-models.pdf](survey/2025-privacy-preserving-low-rank-adaptation-against-membership-inference-attacks-for-latent-diffusion-models.pdf)
- Main idea: studies membership-preserving LoRA adaptation for latent diffusion with min-max style defenses.
- DiffAudit relevance: useful defense comparison point when fine-tuning and low-rank adaptation are in scope.
- Open-source: not found

### Dual-Model Defense: Safeguarding Diffusion Models from Membership Inference Attacks through Disjoint Data Splitting

- File: [2024-dual-model-defense-safeguarding-diffusion-models-from-membership-inference-attacks-through-disjoint-data-splitting.pdf](survey/2024-dual-model-defense-safeguarding-diffusion-models-from-membership-inference-attacks-through-disjoint-data-splitting.pdf)
- Main idea: trains two diffusion models on disjoint subsets and uses private inference or distillation to reduce MIA effectiveness.
- DiffAudit relevance: important defense-side architecture idea rather than a direct attack baseline.
- Open-source: not found

### DIFFENCE: Fencing Membership Privacy With Diffusion Models

- File: [2025-diffence-fencing-membership-privacy-with-diffusion-models.pdf](survey/2025-diffence-fencing-membership-privacy-with-diffusion-models.pdf)
- Main idea: uses diffusion-based preprocessing to reduce member / non-member distinguishability before downstream inference.
- DiffAudit relevance: strong NDSS defense-side reference for membership privacy.
- Open-source: not found

### Defending Diffusion Models Against Membership Inference Attacks via Higher-Order Langevin Dynamics

- File: [2025-defending-diffusion-models-against-membership-inference-attacks-via-higher-order-langevin-dynamics.pdf](survey/2025-defending-diffusion-models-against-membership-inference-attacks-via-higher-order-langevin-dynamics.pdf)
- Main idea: mixes stronger stochasticity into the diffusion process through higher-order Langevin dynamics to blunt membership inference.
- DiffAudit relevance: defense-side technique relevant when comparing perturbation and stochastic defenses.
- Open-source: not found

### Inference Attacks Against Graph Generative Diffusion Models

- File: [2026-inference-attacks-against-graph-generative-diffusion-models.pdf](survey/2026-inference-attacks-against-graph-generative-diffusion-models.pdf)
- Main idea: extends inference and privacy attacks to graph generative diffusion models instead of image diffusion alone.
- DiffAudit relevance: cross-domain extension showing where the audit framework may generalize next.
- Open-source: not found

### Perturb a Model, Not an Image: Towards Robust Privacy Protection via Anti-Personalized Diffusion Models

- File: [2025-perturb-a-model-not-an-image-towards-robust-privacy-protection-via-anti-personalized-diffusion-models.pdf](survey/2025-perturb-a-model-not-an-image-towards-robust-privacy-protection-via-anti-personalized-diffusion-models.pdf)
- Main idea: protects against subject personalization misuse by changing the model rather than perturbing training images.
- DiffAudit relevance: useful defense and anti-personalization reference for identity-sensitive deployment settings.
- Open-source: not found

### Legacy Survey Archive Index

- File: [legacy-survey-archive-index.pdf](survey/legacy-survey-archive-index.pdf)
- Main idea: local index for archive-origin survey materials.
- DiffAudit relevance: maintenance artifact for the local PDF mirror.
- Open-source: not applicable
