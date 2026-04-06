# Inference Attacks Against Graph Generative Diffusion Models

Xiuling Wang *Hong Kong Baptist University xiulingwang@hkbu.edu.hk*

Xin Huang *Hong Kong Baptist University xinhuang@comp.hkbu.edu.hk*

Guibo Luo *Peking University luogb@pku.edu.cn*

Jianliang Xu<sup>∗</sup> *Hong Kong Baptist University xujl@comp.hkbu.edu.hk*

## Abstract

Graph generative diffusion models have recently emerged as a powerful paradigm for generating complex graph structures, effectively capturing intricate dependencies and relationships within graph data. However, the privacy risks associated with these models remain largely unexplored. In this paper, we investigate information leakage in such models through three types of black-box inference attacks. First, we design a graph reconstruction attack, which can reconstruct graphs structurally similar to those training graphs from the generated graphs. Second, we propose a property inference attack to infer the properties of the training graphs, such as the average graph density and the distribution of densities, from the generated graphs. Third, we develop two membership inference attacks to determine whether a given graph is present in the training set. Extensive experiments on three different types of graph generative diffusion models and six real-world graphs demonstrate the effectiveness of these attacks, significantly outperforming the baseline approaches. Finally, we propose two defense mechanisms that mitigate these inference attacks and achieve a better trade-off between defense strength and target model utility than existing methods. Our code is available at https://zenodo.org/records/17946102.

## 1 Introduction

Many real-world systems, such as social networks, biological networks, and information networks, can be represented as graphs. Graph learning is crucial for analyzing these systems because of its ability to model and analyze complex relationships and interactions within the data.

Graph generation, a critical task in graph learning, focuses on creating graphs that accurately reflect the underlying structure of graph data. These models have diverse applications, such as recommender systems [\[33,](#page-14-0) [69\]](#page-16-0), social network analysis [\[39,](#page-15-0) [41\]](#page-15-1), molecular research [\[31,](#page-14-1) [82\]](#page-17-0), and drug discovery [\[64,](#page-16-1)[76\]](#page-16-2). Diffusion models have recently gained significant attention as a prominent class of generative models, which

work through two interconnected processes. A forward process gradually adds noise to data until it conforms to a predefined prior distribution (e.g., Gaussian). A corresponding reverse process uses a trained neural network to progressively denoise the data, effectively reversing the forward process and reconstructing the original data distribution. Given the success of diffusion models in image generation [\[26\]](#page-14-2), there has been growing interest in applying these techniques to graph generation [\[10,](#page-13-0) [17,](#page-14-3) [40,](#page-15-2) [80,](#page-16-3) [82\]](#page-17-0).

While graph generative diffusion models (GGDMs) are capable of producing various graphs, they often require large training datasets for robust generation. However, these datasets may contain sensitive or confidential information. For example, social graphs can expose private relationships, as seen in cases such as Facebook's famous Cambridge Analytica scandal; graphs depicting protein-protein interactions or gene regulatory networks may include proprietary research data, any leakage of this information could compromise competitive advantages in biotechnology research; graphs in healthcare datasets can model relationships between patients, treatments, and healthcare providers, where the disclosure of such information could compromise patient confidentiality and violate regulations such as HIPAA [\[45\]](#page-15-3). Therefore, this paper investigates the critical question: *how much information about the training data can be inferred from GGDMs?*

Challenges. Recent research has uncovered several types of attacks that can infer sensitive information from the training data of graph learning models, including membership inference [\[16,](#page-14-4)[23,](#page-14-5)[24,](#page-14-6)[56,](#page-15-4)[71](#page-16-4)[–73,](#page-16-5)[75\]](#page-16-6), attribute inference [\[16,](#page-14-4)[19,](#page-14-7)[84\]](#page-17-1), property inference [\[61,](#page-16-7) [70,](#page-16-8) [85,](#page-17-2) [87\]](#page-17-3), and graph reconstruction attacks [\[56,](#page-15-4) [86,](#page-17-4) [88\]](#page-17-5). Most of these attacks are based on model's final output (probability vectors) for node/graph classification [\[19,](#page-14-7) [23,](#page-14-5) [24,](#page-14-6) [61,](#page-16-7) [70,](#page-16-8) [73,](#page-16-5) [85\]](#page-17-2), node/graph embeddings [\[16,16,](#page-14-4)[56,](#page-15-4)[71,](#page-16-4)[72,72,](#page-16-9)[84](#page-17-1)[,87,](#page-17-3)[88\]](#page-17-5), or other features such as model gradients [\[75,](#page-16-6)[86\]](#page-17-4). These approaches are not well-suited for GGDMs for two reasons: *First*, attack features derived from probability vectors or node/graph embeddings are unavailable in GGDMs that generate graphs, making existing attacks infeasible in black-box setting. *Second*, the above at-

<sup>∗</sup> Jianliang Xu is the corresponding author.

tacks rely on a one-to-one mapping between an input graph and an output to form a (feature, label) pair, where label denotes membership or a specific property. GGDMs lack this mapping as the graphs are generated from a set of training samples, making ground-truth labels unavailable and existing attacks inapplicable. These differences highlight the unique challenge of attacking GGDMs: constructing meaningful features from generated graphs without one-to-one mappings.

On the other hand, recent preliminary studies have begun to investigate the privacy vulnerabilities of image and textto-image generative diffusion models under various privacy inference attacks. These include membership inference attacks [\[12,](#page-13-1)[14,](#page-14-8)[51\]](#page-15-5), which aim to determine whether a particular data sample is in the training set; data reconstruction attacks, which attempt to recover the training images [\[12\]](#page-13-1); and property inference attacks [\[43\]](#page-15-6), which seek to infer the portion of training images with a specific property. However, all of the above works focus only on image or text-to-image diffusion models and cannot be directly applied to GGDMs due to the unique structural characteristics of graph data. Specifically, graph data exhibit variable sizes and topologies, permutationinvariant, and encode sensitive information through higherorder structural patterns rather than fixed spatial features.

Our Contributions. We initiate a systematic investigation into the privacy risks of GGDMs by exploring three types of inference attacks. First, we explore *graph reconstruction attack* (GRA), which attempts to infer graph structures within model's training set. For example, if the target graph is from a medical database, a reconstructed graph could enable an adversary to gain knowledge of sensitive relationships between patients, health records, and treatments. Second, we introduce *property inference attack* (PIA), which leverages generated graphs to infer statistical properties of the training graphs, such as average graph density or the proportion of graphs within specific density ranges. Revealing these properties may violate the intellectual property (IP) of the data owner, particularly in domains like molecular or protein graphs from biomedical companies. Third, we investigate *membership inference attack* (MIA), which aims to determine whether a given graph is present in GGDM's training set. For example, in a collaborative training setting where each data owner possesses graphs from different biological companies, with each graph representing a proprietary product, an adversary may attempt to infer whether a specific product is included in another owner's training set, thereby compromising IP.

To the best of our knowledge, this is the first work to explore the privacy leakage of GGDMs. Overall, we make the following contributions in this paper:

• We develop a novel graph reconstruction attack by aligning each generated graph with its closest counterpart in the generated graph set to identify overlapping edges. These overlapping edges are then considered part of a graph from the target model's training set. We evaluate the effectiveness of our proposed attack on six real-world graph datasets and

<span id="page-1-0"></span>

|                      |    | Attack type |     | Attack setting |       |          |
|----------------------|----|-------------|-----|----------------|-------|----------|
|                      | DR | PIA         | MIA | B-box          | W-box | Domain   |
| [12]                 | ✓  |             | ✓   | ✓              | ✓     |          |
| [43]                 |    | ✓           |     | ✓              |       | Image or |
| [37, 50, 77, 83]     |    |             | ✓   | ✓              |       | text-to- |
| [14, 28, 36, 63, 81] |    |             | ✓   |                | ✓     | image    |
| [12, 15, 44, 51]     |    |             |     |                |       |          |
| Ours                 | ✓  | ✓           | ✓   | ✓              |       | Graph    |

Table 1: Comparison between the existing works on inference attacks against generative diffusion models. "DR", "PIA", and "MIA" refer to data reconstruction, property inference, and membership inference attacks, respectively. "B-box" and "Wbox" represent black-box and white-box, respectively.

three state-of-the-art GGDMs. The results show that the attack achieves an F1 score of up to 0.99, with up to 36% of the original training graphs being exactly recovered.

- We launch our property inference attack using a simple yet efficient method that directly calculates property values from the generated graphs. Extensive experiments show that the proposed attack can accurately infer the statistical properties of the training graphs. For example, on the IMDB-MULTI dataset, the difference between the actual and inferred average graph degrees can be as small as 0.005.
- We design our membership inference attacks by employing the shadow-model-training technique, where we train MLP attack models based on two factors: (1) different similarity levels between the generated graphs and their corresponding training graphs for member and non-member graphs; and (2) different similarity levels within the generated graphs from member and non-member graphs. Experimental results show that our attacks can achieve an AUC of up to 0.999 when shadow and target graphs are drawn from the same dataset, and 0.895 when drawn from different datasets.
- To mitigate the inference attacks, we propose two defense mechanisms that introduce noise into either the training or generated graphs of the target GGDM, thereby altering the model's outputs. Notably, we limit perturbations to the least significant edges and non-edges (by flipping them), to minimize the impact on the target model's utility. Empirical evaluations show that our approach achieves defense effectiveness comparable to two baseline methods, rendering the attack ineffective. Furthermore, our method achieves a better trade-off between defense strength and target model utility compared to the two existing defense methods.

#### 2 Related Work

Generative diffusion models for graphs. Generative diffusion models have recently gained significant attention as a powerful paradigm for deep graph generation, aiming to learn the underlying graph distribution and synthesize novel graphs. The existing graph generative diffusion models (GGDMs) can be broadly categorized into three classes [\[17,](#page-14-3) [82\]](#page-17-0): (1) *Scorebased Generative Models (SGM)* [\[13,](#page-14-11) [49\]](#page-15-11) that employ a score function to represent the probability distribution of the data;

(2) *Denoising Diffusion Probabilistic Models (DDPMs)* [\[21\]](#page-14-12) that add discrete Gaussian noise to the graph with Markov transition kernels [\[9\]](#page-13-2) and then train a neural network to predict the added noise to recover the original graph, [\[68\]](#page-16-12) adds discrete noise instead of continuous Gaussian, and (3) *Stochastic Differential Equation-based Models (SDEs)* [\[30,](#page-14-13) [34,](#page-14-14) [42\]](#page-15-12) that characterize the development of a system over time under the influence of random noise. SGM and DDPM leverage the score-matching idea and non-equilibrium thermodynamics, respectively, to learn different reverse functions of the diffusion process, while SDE generalizes the discrete diffusion steps into continuous scenarios and further models the diffusion process with stochastic differential equations [\[17\]](#page-14-3). We refer the readers to comprehensive surveys on GGDMs [\[17,](#page-14-3) [82\]](#page-17-0).

Inference attacks against generative diffusion models. Few studies [\[12,](#page-13-1) [43,](#page-15-6) [59,](#page-16-13) [65,](#page-16-14) [66\]](#page-16-15) have investigated the privacy vulnerabilities of generative diffusion models against privacy inference attacks. Van et al. [\[66\]](#page-16-15) focus on probabilistic deep generative models such as variational autoencoders and formulate the concept of "memorization score" by measuring the impact of removing an observation on a given model. Carlini et al. [\[12\]](#page-13-1) consider image diffusion models and design three attacks: data extraction, data reconstruction, and membership inference attack. Somepalli et al. [\[59\]](#page-16-13) focus on text-to-image diffusion models and analyze the data duplication problem in these models. Luo et al. [\[43\]](#page-15-6) introduce a black-box property inference attack that aims to infer the distribution of specific properties from generated images. However, none of these previous works have examined the privacy leakage in GGDMs. Several works focus on membership inference attacks against diffusion models [\[14,](#page-14-8)[15,](#page-14-10)[28,](#page-14-9)[36,](#page-15-9)[37,37,](#page-15-7)[44,](#page-15-10)[50,](#page-15-8)[51,](#page-15-5)[63,](#page-16-11)[77,](#page-16-10)[81,](#page-17-7)[83\]](#page-17-6). Specifically, [\[37,](#page-15-7)[50,](#page-15-8)[77,](#page-16-10)[83\]](#page-17-6) describe black-box attacks by analyzing the generated images; [\[14,](#page-14-8)[28,](#page-14-9)[36](#page-15-9)[,63,](#page-16-11)[81\]](#page-17-7) and [\[12](#page-13-1)[,15,](#page-14-10)[44\]](#page-15-10) design white-box attacks that rely on posterior estimation errors and model losses, respectively; and Pang et al. [\[51\]](#page-15-5) introduce a white-box attack that leverages gradients at each timestep. However, all of these attack models focus on image or text-to-image diffusion models, which cannot be directly applied to GGDMs. Table [1](#page-1-0) summarizes the main differences between our work and existing studies on inference attacks against generative diffusion models.

Inference attacks on graph data. Recent research has uncovered several types of attacks that can infer sensitive information from the training graph data. These attacks can be categorized into four types: (1) membership inference attacks (MIAs) [\[16,](#page-14-4) [23,](#page-14-5) [24,](#page-14-6) [56,](#page-15-4) [71–](#page-16-4)[73,](#page-16-5) [75\]](#page-16-6), seeking to determine whether a specific graph sample is part of the training dataset; (2) attribute inference attacks (AIAs) [\[16,](#page-14-4) [19,](#page-14-7) [84\]](#page-17-1), aiming to infer the sensitive attributes within the training graphs; (3) property inference attacks (PIAs) [\[61,](#page-16-7) [70,](#page-16-8) [85,](#page-17-2) [87\]](#page-17-3), trying to infer the sensitive properties of the training graphs; and (4) graph reconstruction attacks [\[56,](#page-15-4) [86,](#page-17-4) [88\]](#page-17-5), attempting to reconstruct the training graphs. However, most of these attacks rely on the model's final output (probability vectors) for

node/graph classification [\[19,](#page-14-7)[23,](#page-14-5)[24](#page-14-6)[,61,](#page-16-7)[70,](#page-16-8)[73,](#page-16-5)[85\]](#page-17-2), node/graph embeddings [\[16,](#page-14-4) [16,](#page-14-4) [56,](#page-15-4) [71,](#page-16-4) [72,](#page-16-9) [72,](#page-16-9) [84,](#page-17-1) [87,](#page-17-3) [88\]](#page-17-5), or some other features like model gradients [\[75,](#page-16-6) [86\]](#page-17-4). Consequently, these approaches are not well-suited for GGDMs due to the unique forward and reverse processes in diffusion models as well as the different types of outputs they produce.

# 3 Generative Diffusion Models on Graphs

Graph generation models aim to generate new graph samples resembling a given dataset. Among these, diffusion-based models have become increasingly popular. They gradually introduce noise into data until it conforms to a prior distribution [\[13,](#page-14-11) [21,](#page-14-12) [30,](#page-14-13) [34,](#page-14-14) [42,](#page-15-12) [49,](#page-15-11) [68\]](#page-16-12).

Generally, existing generative diffusion models on graphs include two processes: (1) the *forward process*, which progressively degrades the original data into Gaussian noise [\[26\]](#page-14-2), and (2) the *reverse process*, which gradually denoises the noisy data back to its original structure using transition kernels. Based on how the forward and reverse processes are designed, existing generative diffusion models on graphs can be broadly categorized into three classes: *Score-Based Generative Models* (SGM) [\[13,](#page-14-11) [49\]](#page-15-11), *Denoising Diffusion Probabilistic Models* (DDPM) [\[21,](#page-14-12) [68\]](#page-16-12), and *Stochastic Differential Equations* (SDE) [\[30,](#page-14-13) [34,](#page-14-14) [42\]](#page-15-12). Next, we briefly describe the forward and reverse processes for the three types of graph generative diffusion models. Note that "synthetic graphs" and "generated graphs" are used interchangeably throughout this paper to refer to the graphs produced by the diffusion models.

SGM-based graph generation. The forward process injects Gaussian noise of varying intensity as the perturbation into the original graph. A noise-conditional score network is trained to represent the gradient of the conditional probability density function of the data under varying noise levels. Specifically, given a probability density function *p*(*x*) and the score function ∇*xlogp*(*x*), SGM aims to estimate the data score function in the forward process. The training objective of the score network is given by:

<span id="page-2-0"></span>
$$\mathcal{L}(\theta) = \min_{\theta} \mathbb{E}_{t \sim \mathcal{U}(1,T), x_0 \sim p(x_0), \epsilon \sim \mathcal{H}(0,\mathbf{I})} \left[ \lambda(t) \| \nabla_{x_t} \log p_{0t}(x_t | x_0) - \sigma_t s_{\theta}(x_t, t) \| \right]^2,$$
(1)

where E is the expectation, *U*(1,*T*) is a uniform distribution over the time set {1,2,...,*T*}, ε is the noise vector, and λ(*t*) is a positive weighting function. The variable *x*<sup>0</sup> refers to the original data before noise is added, with its probability density function denoted as *p*(*x*0). *p*0*t*(*x<sup>t</sup>* |*x*0) is the score function of *xt* . σ*<sup>t</sup>* is the Gaussian noise at *t*. The score network *s*<sup>θ</sup> with parameter θ, predicts the noise σ*<sup>t</sup>* based on *x<sup>t</sup>* and *t*, where *x<sup>t</sup>* is the noisy version of *x*<sup>0</sup> after adding noise at *t*.

In the reverse process, after obtaining the trained conditional score model, synthetic graphs are generated using noiseconditional score networks, such as the Score Matching with Langevin Dynamics model [\[60\]](#page-16-16) that leverages the learned score models to reconstruct graph data from noise.

<span id="page-3-1"></span>![](_page_3_Figure_0.jpeg)

Figure 1: Inference attacks against graph generative diffusion models. The attacker inputs a set of shadow graphs into the target graph generative diffusion model Φ or directly executes Φ through an API or online marketplaces to obtain a large number of synthetic graphs *G*ˆ, aiming to infer the sensitive information about the training data of Φ. In this paper, we investigate three types of inference attacks: (1) reconstructing the graph structures in Φtrain; (2) inferring the properties of Φtrain, such as the graph density of Φtrain; and (3) determining the membership of a given target graph.

<span id="page-3-0"></span>

| Symbol            | Meaning                                       |
|-------------------|-----------------------------------------------|
| S<br>G/G          | Target/shadow graph                           |
| Φ/ΦS              | Target /shadow model                          |
| Gˆ                | Generated/synthetic graphs from Φ             |
| Φtrain<br>, Φtest | Training and testing datasets of target model |
| train<br>A        | Training set of attack model                  |

Table 2: Notations

DDPM-based graph generation. In the forward process, the original data undergoes perturbation with Gaussian noise using a fixed number. Specifically, given a probability density function *x* ∼ *p*(*x*), the forward process generates the noisy *x*ˆ with Markov transition kernels [\[9,](#page-13-2) [26\]](#page-14-2) as *p*(*x<sup>t</sup>* |*xt*−1) = *N* (*x<sup>t</sup>* p 1−β*txt*−1,β*t*I), where β*<sup>t</sup>* is a predefined variance schedule at time step *t*.

In the reverse process, a neural network is trained to predict the noise added at each step during the forward pass, ultimately recovering the original data. Following the notations in Equation [1,](#page-2-0) the optimization objective can be expressed as:

$$\mathcal{L}(\theta) = \min_{\theta} \mathbb{E}_{t \sim \mathcal{U}(1,T), x_0 \sim p(x_0), \epsilon \sim \mathcal{N}(0,\mathbf{I})} \left[ \lambda(t) \| \epsilon - \epsilon_{\theta}(x_t, t) \|^2 \right], \tag{2}$$

where ε<sup>θ</sup> is a deep neural network with parameter θ that predicts the noise vector ε given *x<sup>t</sup>* and *t*.

SDE-based models. The forward process employs a forward SDE [\[38\]](#page-15-13) to describe the evolution of a state variable over time to generate the noisy graph. It perturbs data to noise with SDE as *d<sup>x</sup>* = *f*(*x*,*t*)*d<sup>t</sup>* +*g*(*t*)*dw*, where *f*(*x*,*t*) and *g*(*t*) are diffusion and drift functions of the SDE, and *w* is a standard Wiener process.

In the reverse process, a reverse SDE is utilized to gradually convert noise to data. This is achieved by estimating the score functions of the noisy data distributions. Using similar notations as in Equation [1,](#page-2-0) the objective for estimating the score function can be formulated as:

$$\mathcal{L}(\theta) = \min_{\theta} \mathbb{E}_{t \sim \mathcal{U}(1,T), x_0 \sim p(x_0), \epsilon \sim \mathcal{H}(0,\mathbf{I})} \left[ \lambda(t) \left\| s_{\theta}(x_t, t) - \nabla_{x_t} \log p_{0t}(x_t | x_0) \right\| \right]^2,$$
(3)

Once the score function at each time step is obtained, the synthetic graphs can be generated with various numerical techniques, such as annealed Langevin dynamics, numerical SDE/ODE solvers, and predictor-corrector methods [\[80\]](#page-16-3).

# 4 Motivation and Threat Model

In this section, we start by explaining our motivation, and then define the scope and objectives of our problem. Table [2](#page-3-0) lists the common notations used in the paper.

# 4.1 Motivations

Machine learning (ML) models have unlocked a variety of applications, such as data analytics, autonomous systems, and security diagnostics. However, developing robust models often requires substantial training datasets, extensive computational resources, and significant financial investment, which can be prohibitive for small businesses, developers, and researchers with limited budgets. Consequently, online marketplaces for ML models, such as Amazon Web Services [\[1\]](#page-13-3), Google AI Hub [\[3\]](#page-13-4), Modzy [\[5\]](#page-13-5), Microsoft Azure Cognitive Services [\[2\]](#page-13-6), and IBM Watson [\[4\]](#page-13-7), have emerged, facilitating model exchange, customization, and access to various machine-learning-as-a-service (MLaaS) APIs. While beneficial, these platforms raise concerns about the potential exposure of sensitive or proprietary information from the training data. Therefore, in this paper, we investigate the privacy vulnerability of increasingly popular GGDMs.

## 4.2 Threat Model

ML models are susceptible to privacy attacks [\[29,](#page-14-15) [54\]](#page-15-14). These attacks can be categorized into two groups based on the adversary's target [\[54\]](#page-15-14): (1) *Privacy attacks on training data*: the adversary aims to infer sensitive information about the training data. (2) *Privacy attacks on ML models*: the adversary considers the ML models themselves as sensitive, like valuable company assets, and attempts to uncover information regarding the model's architecture and parameters.

In this paper, we primarily focus on privacy attacks on training data. Our threat model considers an adversary that interacts with a graph generative diffusion model Φ to extract the information of the model's training set Φtrain. An overview of the attacks is shown in Figure [1.](#page-3-1)

<span id="page-4-0"></span>![](_page_4_Figure_0.jpeg)

Figure 2: Overview of graph reconstruction.

Adversary's Background Knowledge. We consider the adversary knowledge K along two dimensions:

- *Shadow graph G S* (optional): The adversary possesses one or more shadow graphs, *G S* , each with its own structure and node features. *G <sup>S</sup>* may originate from a different domain than the model's training set, Φtrain, and thus exhibit distinct data distributions. In real-world scenarios, the adversary may have knowledge of a partial graph, which is a subset of the training graphs [\[16,](#page-14-4) [23,](#page-14-5) [70\]](#page-16-8). We treat this partial graph as a specific instance of a shadow graph.
- *Target model* Φ: The adversary may have either white-box or black-box access to the target model. In the white-box setting, the attacker can access Φ's internal components, such as parameters, gradients, and loss values [\[12,](#page-13-1) [16,](#page-14-4) [51\]](#page-15-5). In contrast, the black-box setting assumes that the adversary can only interact with Φ through its outputs. In this paper, we consider a black-box setting, reflecting real-world scenarios such as public or commercial APIs in MLaaS platforms [\[23,](#page-14-5) [55,](#page-15-15) [57,](#page-15-16) [87\]](#page-17-3). Specifically, the adversary can submit their own graphs or random graphs to Φ via an API and receive a corresponding set of generated graphs *G*ˆ as output. This black-box setting is considered the most challenging setting for the adversary [\[23,](#page-14-5) [55,](#page-15-15) [57\]](#page-15-16).

Attack Goal. We consider three types of inference attacks:

- *Graph reconstruction*: The attacker aims to reconstruct the graph structures within target model's training set Φtrain .
- *Property inference*: The attacker attempts to infer predefined properties or aggregate characteristics, P, of an individual record or a group within the training set Φtrain .
- *Membership inference*: The attacker seeks to determine whether a given graph G is in the training set Φtrain .

### 5 Inference Attacks

In this section, we detail the proposed graph reconstruction, property inference, and membership inference attacks.

#### 5.1 Graph Reconstruction Attacks

#### 5.1.1 Attack Overview

Given a target GGDM Φ, obtained from an online marketplace or via an API on a MLaaS platform, the attacker's goal is to infer the graph structures in Φtrain, which is the training set of Φ. Figure [2](#page-4-0) illustrates the pipeline of our graph reconstruction attack, and the corresponding pseudo-code can be found in Appendix [A.](#page-17-8) The attacker first generates a set of synthetic graphs *G*ˆ and then reconstructs the graphs by aligning each graph in *G*ˆ with its closest counterpart to identify

the edges in the original training graphs. Formally, the graph reconstruction attack can be formulated as follows:

$$f: \hat{G} \to \Phi^{\text{train}}.$$
 (4)

Next, we detail the attack steps.

#### 5.1.2 Attack Model

Our attack model includes three steps: graph generation, graph alignment, and edge inference.

Graph generation. The attacker inputs a set of shadow graphs, *G S* , into Φ or executes Φ directly to obtain a large number of synthetic graphs *G*ˆ.

Graph alignment. With the generated graphs *G*ˆ, for each generated graph *g<sup>i</sup>* in *G*ˆ, we first identify the most similar graph *g<sup>j</sup>* to *g<sup>i</sup>* within *G*ˆ by using graph alignment techniques [\[25,](#page-14-16) [79\]](#page-16-17). This is based on the assumption that if the target model has memorized a particular graph, it will likely produce multiple similar graphs to this graph. This assumption is verified in image-based diffusion models, where diffusion models memorize individual images from their training data and emit them at generation time [\[12,](#page-13-1) [66\]](#page-16-15).

Following the same strategy in [\[25\]](#page-14-16), we use REGAL for graph alignment, which leverages representation learning to effectively map and match nodes between different graphs. Specifically, given a graph *g<sup>i</sup>* , REGAL aligns it with other graphs in *G*ˆ through the following three steps: (1) node identity extraction that extracts the structure and attribute-related information for all nodes in *g<sup>i</sup>* based on the degree distributions and node features; (2) similarity-based node representation by using the low-rank matrix factorization-based approach that leverages a combined structural and attributebased similarity matrix from step (1); and (3) node representation alignment that greedily matches each node in *g<sup>i</sup>* to its top-α most similar nodes in other graphs in *G*ˆ with k-d trees. The difference between a pair of node representations, (*Yi*(*u*),*Yj*(*v*)), in *g<sup>i</sup>* and *g<sup>j</sup>* is calculated as:

<span id="page-4-1"></span>
$$Diff(Y_i(u), Y_j(v)) = \exp^{\|Y_i(u) - Y_j(v)\|^2}.$$
 (5)

Edge inference. After graph alignment, we get the pairwise alignments within *G*ˆ. Then, for each *g<sup>i</sup>* in *G*ˆ, we identify its most similar counterpart in *G*ˆ by averaging all the node representation differences between the aligned graphs. Specifically, using *Di f f* from Equation [5,](#page-4-1) we identify the most similar counterpart ˆ*g<sup>j</sup>* for *g<sup>i</sup>* among its aligned graphs as:

<span id="page-4-2"></span>
$$\hat{g}_{j} = \min_{\hat{g}_{j}} \frac{1}{|V_{i}|} \sum_{\forall u \in \{V_{i}\}} Diff(Y_{i}(u), Y_{j}(\hat{u})), \tag{6}$$

where *u* is a node in the node set of *g<sup>i</sup>* , denoted as {*Vi*}, *u*ˆ is the aligned node of *u* in *g<sup>j</sup>* .

After obtaining the most similar counterpart *g*ˆ*<sup>j</sup>* for each *g<sup>i</sup>* in *G*ˆ, and the average node representation difference between *g<sup>i</sup>* and *g*ˆ*<sup>j</sup>* , namely *D*(*g<sup>i</sup>* ,*g*ˆ*j*) = <sup>1</sup> |*Vi* <sup>|</sup> ∑∀*u*∈{*Vi*} *Di f f*(*Yi*(*u*),*Yj*(*u*ˆ)), we pick the graph pairs with top-k% smallest differences to

<span id="page-5-0"></span>![](_page_5_Figure_0.jpeg)

Figure 3: Overview of property inference attack.

determine the graph structure in Φtrain. Empirically, we set *k* to 10%. Subsequently, we reconstruct the structure of a training graph from these selected aligned graph pairs. For each selected graph pair (*g<sup>i</sup>* ,*g*ˆ*j*), we perform the intersection operation on the edge sets of *g<sup>i</sup>* and *g*ˆ*<sup>j</sup>* , denoted as *E<sup>i</sup>* and *E*ˆ *j* , to reconstruct the graph *g rec i* as:

<span id="page-5-2"></span>
$$\{V_i^{rec}\} = \{V_i\}, \{E_i^{rec}\} = \{E_i\} \cap \{\hat{E}_j\},$$
 (7)

where {*V rec i* } and {*E rec i* } are the node and edge sets of the reconstructed graph, respectively. We also experimented with using the union operation on the edge sets of *g<sup>i</sup>* and *g*ˆ*<sup>j</sup>* , but the results showed that the intersection operation performs better than the union operation. Therefore, we use the intersection operation in our attack model. Finally, we get a set of reconstructed graphs from the selected aligned graph pairs.

### 5.2 Property Inference Attacks

#### 5.2.1 Attack Overview

Following the same attack setting, where the attacker obtains the target GGDM Φ from an online marketplace or a MLaaS API, the attacker's goal is to infer the graph properties of Φtrain. In this paper, we focus on two types of properties: (1) the average statistical properties of the training graphs in Φtrain, such as graph density and average node degree; and (2) the distribution of training graphs across different property ranges in Φtrain. Figure [3](#page-5-0) illustrates the pipeline of our property inference attack, and the corresponding pseudocode can be found in Appendix [A.](#page-17-8) The attacker first generates a set of synthetic graphs, *G*ˆ, and then directly calculates the property values from *G*ˆ. Formally, the property inference attack can be formulated as follows:

$$f: \hat{G} \to \mathbb{P}(\Phi^{\text{train}}),$$
 (8)

where P represents the property, including both the average statistical properties and the distribution of training graphs across different property ranges.

#### 5.2.2 Attack Model

Our attack model includes two steps: graph generation and property inference.

Graph generation. The attacker inputs a set of shadow graphs, *G S* , into Φ or executes Φ directly to obtain a large number of synthetic graphs *G*ˆ.

Property inference. For both property types, the attacker computes the property value over all graphs in *G*ˆ to approximate that of Φtrain. The process can be formulated as:

<span id="page-5-3"></span>
$$\mathbb{P}(\Phi^{\text{train}}) \leftarrow \mathbb{P}(\hat{G}). \tag{9}$$

<span id="page-5-1"></span>![](_page_5_Figure_15.jpeg)

Figure 4: Overview of membership inference attack.

# 5.3 Membership Inference Attacks

#### 5.3.1 Attack Overview

Given the GGDM Φ obtained from an online marketplace or through an API, and a target graph *G*, the attacker's goal is to determine whether *G* is in Φ's training set, Φtrain. In this attack, we have the following assumptions. First, the attacker has background knowledge of shadow graphs *G S* , which contain their own nodes and features. These shadow graphs may come from a different domain than Φtrain. A special case occurs when *G S* contains partial subgraphs of Φtrain , which is plausible in real-world applications. For example, an online marketplace may release a portion of the training set. Second, following [\[23,](#page-14-5) [57,](#page-15-16) [70\]](#page-16-8), we assume the attacker can train a shadow model Φ*<sup>S</sup>* using the same service (e.g., Google AI Hub) employed for the target model. Figure [4](#page-5-1) illustrates the pipeline of our membership inference attacks, and the corresponding pseudo-code can be found in Appendix [A.](#page-17-8) The attacker first trains Φ*<sup>S</sup>* using a subset of graphs from *G S* . Then, *G S* is fed into Φ*<sup>S</sup>* to generate a set of synthetic graphs *G*ˆ. An attack classifier is subsequently trained by analyzing the similarity between *G*ˆ and *G <sup>S</sup>* or the similarity within all the graphs in *G*ˆ. Formally, given a target graph *G*, the membership inference attack can be formulated as follows:

$$f: (G, G^S) \to y_G \in \{0, 1\},$$
 (10)

where *y<sup>G</sup>* is the membership label of *G*, with 0 (or 1) indicating the absence (or presence) in Φtrain. We next detail the attack.

#### 5.3.2 Attack Model

Our attack model includes four steps: shadow model training and graph generation, attack feature construction, attack model training, and membership inference.

Shadow model training and graph generation. First, the attacker trains a shadow model Φ*<sup>S</sup>* on a subset of graphs from *G S* , referred to as member shadow graphs, to mimic the behavior of the target model. The remaining graphs in *G S* are treated as non-member shadow graphs. To balance the number of member and non-member shadow graphs, the attacker can also randomly generate non-member shadow graphs that differ from the member graphs. Given their large variation in node and edge sizes, the chance of overlap between member and non-member shadow graphs is negligible. We denote

the member and non-member shadow graphs in *G S* as *G S mem* and *G S non*−*mem*, respectively. Second, for each graph in *G S i* in *G S mem* and *G S non*−*mem*, the attacker inputs it into Φ*<sup>S</sup>* to obtain a corresponding set of synthetic graphs *G*ˆ *<sup>S</sup>* . We denote this as a triplet (*G S i* ,*G*ˆ *<sup>S</sup> i* ,*y*), where *y* is *G S i* 's membership label.

Attack feature construction. After obtaining the synthetic graphs *G*ˆ *<sup>S</sup> i* for each graph *G S i* in *G S* , we construct the attack features *A* train for attack training in two ways.

*A train-1*: the design of this attack feature is based on the principle that the generated graphs *G*ˆ *<sup>S</sup> i* should be much similar to its corresponding shadow graph *G S <sup>i</sup>* when *G S i* is a member shadow graph in *G S* , compared to when it is a nonmember. First, for each shadow graph *G S i* , the adversary measures the pairwise similarity between *G S i* and each generated graph *g*ˆ *S i* in *G*ˆ *<sup>S</sup> i* . Specifically, we first convert both *g*ˆ *S i* and *G S i* into embedding vectors using Anonymous Walk Embeddings (AWEs) [\[32\]](#page-14-17), denoted as *emb*(*g*ˆ *S i* ) and *emb*(*G S i* ), respectively. The AWEs method represents graphs by sampling random walks and anonymizing node identities based on their first appearance indices. The frequency distribution of these anonymous walk patterns is then embedded into a lowerdimensional space using neural networks like Word2Vec. This approach captures structural similarity between graphs while being scalable and identity-independent. Second, we calculate the similarity between the embedding vectors of *emb*(*g*ˆ *S i* ) and *emb*(*G S i* ) as *sim<sup>k</sup> emb*(*g*ˆ *S i* ),*emb*(*G S i* ) , where *sim<sup>k</sup>* represents the *k*-th similarity function. In this paper, *k* ∈ {1,2,3,4} corresponds to four similarity metrics: Dot product, Cosine similarity, Euclidean distance-based difference (as defined in Equation [5\)](#page-4-1), and Jensen-Shannon Diversity (JSD). Third, we construct the attack feature of a triplet (*G S i* ,*G*ˆ *<sup>S</sup> i* ,*y*) by stacking the pairwise embedding similarities between *G S i* and *G*ˆ *<sup>S</sup> i* , which can be written as:

<span id="page-6-0"></span>
$$A_{i}^{train} = \begin{bmatrix} ||_{\forall k \in \{1,2,3,4\}} sim_{k} \left( emb(\hat{g}_{i,0}^{S}), emb(G_{i}^{S}) \right) \\ \dots \\ ||_{\forall k \in \{1,2,3,4\}} sim_{k} \left( emb(\hat{g}_{i,N}^{S}), emb(G_{i}^{S}) \right) \end{bmatrix}, (11)$$

where ||∀*k*∈{1,2,3,4} *sim<sup>k</sup>* (*emb*(·),*emb*(·)) denotes the concatenation of similarity values computed using the four metrics between two embeddings. *g*ˆ *S i*,*n* is the *n*-th generated graph in *G*ˆ *S i*, with *n* ∈ 1,...,*N* and *N* being the total number of generated graphs in *G*ˆ *<sup>S</sup> i*. Therefore, for each triplet (*G S i* ,*G*ˆ *<sup>S</sup> i* ,*y*), the dimension of its corresponding attack feature *A train* is *N* ×4.

*A train-2*: the design of this attack feature is based on the principal that the generated graphs *G*ˆ *<sup>S</sup> i* should exhibit much higher pairwise similarity within *G*ˆ *<sup>S</sup> <sup>i</sup>* when *G S i* is a member shadow graph than when it is a non-member. First, for each shadow graph *G S i* , the adversary measures the pairwise similarity within *G*ˆ *<sup>S</sup> i* . Similar to *A* train-1, we first convert each *g*ˆ *S i* in *G*ˆ *<sup>S</sup> i* into an embedding vector, *emb*(*g*ˆ *S i* ), using AWEs. Second, we pairwisely calculate the similarity between the embedding vectors of *emb*(*g*ˆ *S i* ) and *emb*(*g*ˆ *S j* ) as *sim<sup>k</sup> emb*(*g*ˆ *S i* ),*emb*(*g*ˆ *S j* ) ,*i* < *j*. The similarity metrics *sim<sup>k</sup>*

are the same as those used in *A* train-1 . Third, we construct the attack feature of a triplet (*G S i* ,*G*ˆ *<sup>S</sup> i* ,*y*) by stacking the pairwise embedding similarities within *G*ˆ *<sup>S</sup> i* :

<span id="page-6-2"></span>
$$A_{i}^{train} = \begin{bmatrix} ||_{\forall k \in \{1,2,3,4\}} sim_{k} \left( emb(\hat{g}_{i,0}^{S}), emb(\hat{g}_{i,1}^{S}) \right) \\ ... \\ ||_{\forall k \in \{1,2,3,4\}} sim_{k} \left( emb(\hat{g}_{i,N-1}^{S}), emb(\hat{g}_{i,N}^{S}) \right) \end{bmatrix}. (12)$$

The meaning of the notations is consistent with those in Equation [11.](#page-6-0) Therefore, for each triplet (*G S i* ,*G*ˆ *<sup>S</sup> i* ,*y*), the dimension of its corresponding attack feature *A train* is *<sup>N</sup>*∗(*N*−1) <sup>2</sup> ×4.

After the adversary generates the feature *A train i* of the shadow graph *G S i* , it associates *A train <sup>i</sup>* with its ground-truth membership label *y*. Finally, the adversary adds the newly formed data sample (*A train i* , *y*) to *A* train. In our empirical study, we ensure *A* train is balanced, i.e., both the member and nonmember classes have the same number of samples.

Attack model training. After *A* train is generated, the adversary proceeds to train the attack classifier, such as Multi-layer Perceptron (MLP), Random Forest (RF), and Linear Regression (LR), on *A* train .

Membership inference. At inference time, the adversary uses the same method used to generate the training feature *A train* to derive the feature *A att* for the target graph *G*. Specifically, the adversary inputs *G* to the target model Φ and obtains a set of generated graphs. Then, the adversary calculates the similarity between the generated graphs and *G* or within the generated graphs using the same approaches and similarity functions. Finally, the adversary feeds *A att* into the trained attack classifier to obtain predictions, just as in the training phase. The label associated with a higher probability will be selected as the inference output.

#### <span id="page-6-1"></span>6 Evaluation

This section evaluates the effectiveness of our attacks.

## 6.1 Experimental Setup

All the algorithms are implemented in Python with PyTorch and executed on NVIDIA A100-PCIE-40GB.

Datasets. We use six real-world datasets from four domains: two molecule datasets (MUTAG, QM9), one protein dataset (ENZYMES), one citation dataset (Ego-small), and two social networks (IMDB-BINARY, IMDB-MULTI). These datasets each comprise a collection of graphs and serve as benchmarks for evaluating graph-based models across domains [\[46\]](#page-15-17). Statistical details are provided in Appendix [B.](#page-17-9) Throughout the paper, we refer to IMDB-BINARY and IMDB-MULTI as IMDB-B and IMDB-M, respectively.

Target models. We employ three state-of-the-art GGDMs, namely EDP-GNN [\[49\]](#page-15-11), GDSS [\[34\]](#page-14-14), and Digress [\[68\]](#page-16-12).

• EDP-GNN is a score-based generative diffusion model (SGM) that models data distributions via score functions and represents a pioneer effort in deep graph generation.

<span id="page-7-0"></span>

|           |            |      |      | EDP-GNN |      |      | GDSS |      |      |      | Digress |       |      |       |      |      |
|-----------|------------|------|------|---------|------|------|------|------|------|------|---------|-------|------|-------|------|------|
| Dataset   | Attack     | P    | R    | F1      | R1   | R2   | P    | R    | F1   | R1   | R2      | P     | R    | F1    | R1   | R2   |
|           | Ours       | 0.70 | 0.90 | 0.78    | 0.21 | 0.27 | 0.87 | 0.83 | 0.85 | 0.22 | 0.32    | 1     | 0.72 | 0.84  | 0.11 | 0.16 |
| MUTAG     | Baseline-1 | 0.50 | 0.65 | 0.57    | 0    | 0    | 0.54 | 0.67 | 0.59 | 0    | 0       | 0.10  | 0.15 | 0.12  | 0    | 0    |
|           | Baseline-2 | 0.45 | 0.51 | 0.48    | 0    | 0    | 0.45 | 0.52 | 0.48 | 0    | 0       | 1     | 0.33 | 0.50  | 0    | 0    |
|           | Ours       | 0.85 | 1    | 0.92    | 0.11 | 0.36 | 0.83 | 0.92 | 0.85 | 0.09 | 0.28    | 1     | 0.79 | 0.88  | 0.03 | 0.04 |
| ENZYMES   | Baseline-1 | 0.54 | 0.73 | 0.62    | 0    | 0    | 0.75 | 0.75 | 0.70 | 0    | 0       | 0.01  | 0.19 | 0.01  | 0    | 0    |
|           | Baseline-2 | 0.66 | 0.69 | 0.67    | 0    | 0    | 0.92 | 0.31 | 0.46 | 0    | 0       | 1     | 0.22 | 0.36  | 0    | 0    |
|           | Ours       | 0.84 | 0.88 | 0.86    | 0.11 | 0.24 | 0.76 | 0.78 | 0.77 | 0.07 | 0.19    | 1     | 0.75 | 0.85  | 0.10 | 0.22 |
| Ego-small | Baseline-1 | 0.63 | 0.76 | 0.69    | 0    | 0.03 | 0.60 | 0.72 | 0.65 | 0    | 0       | 0.09  | 0.43 | 0.15  | 0    | 0    |
|           | Baseline-2 | 0.60 | 0.66 | 0.63    | 0    | 0    | 0.61 | 0.62 | 0.62 | 0    | 0       | 1     | 0.30 | 0.46  | 0    | 0    |
|           | Ours       | 0.85 | 1    | 0.92    | 0.14 | 0.27 | 0.88 | 1    | 0.94 | 0.15 | 0.28    | 0.85  | 0.80 | 0.82  | 0.09 | 0.10 |
| IMDB-B    | Baseline-1 | 0.80 | 0.79 | 0.80    | 0.05 | 0.12 | 0.78 | 0.89 | 0.83 | 0.02 | 0.03    | 0.001 | 0.20 | 0.01  | 0    | 0    |
|           | Baseline-2 | 0.96 | 0.31 | 0.47    | 0    | 0    | 0.41 | 0.47 | 0.44 | 0    | 0       | 1     | 0.21 | 0.34  | 0    | 0    |
|           | Ours       | 0.92 | 0.96 | 0.94    | 0.13 | 0.19 | 0.99 | 1    | 0.99 | 0.15 | 0.20    | 0.72  | 0.89 | 0.79  | 0.07 | 0.12 |
| IMDB-M    | Baseline-1 | 0.73 | 0.83 | 0.78    | 0    | 0.11 | 0.70 | 0.89 | 0.77 | 0.05 | 0.06    | 0.01  | 0.20 | 0.013 | 0    | 0    |
|           | Baseline-2 | 0.61 | 0.78 | 0.68    | 0    | 0    | 0.54 | 0.58 | 0.56 | 0    | 0       | 1     | 0.22 | 0.36  | 0    | 0    |
|           | Ours       | 0.79 | 0.84 | 0.81    | 0.05 | 0.08 | 0.80 | 0.85 | 0.82 | 0.06 | 0.11    | 0.88  | 0.73 | 0.79  | 0.05 | 0.07 |
| QM9       | Baseline-1 | 0.34 | 0.41 | 0.37    | 0    | 0    | 0.62 | 0.75 | 0.68 | 0    | 0       | 0.08  | 0.43 | 0.13  | 0    | 0    |
|           | Baseline-2 | 0.38 | 0.53 | 0.44    | 0    | 0    | 0.37 | 0.60 | 0.46 | 0    | 0       | 0.81  | 0.23 | 0.35  | 0    | 0    |

Table 3: Performance of graph reconstruction attack. "P", "R", "F1", "*R*1", and "*R*2" represent the metrics of precision, recall, F1 score, coverage ratio of the exact-matched graphs, and coverage ratio of graphs for which the attack achieves an F1 score above 0.75, respectively. For each dataset and evaluation metric under each target model, the better results are highlighted in bold.

- GDSS is one of the stochastic differential equations (SDE) based models that capture the joint distribution of nodes and edges through a system of SDE.
- Digress is a type of denoising diffusion probabilistic model (DDPM) that incrementally edits graphs by adding or removing edges and altering categories, subsequently reversing these changes with a graph transformer.

Implementation settings. For the target model training, we use the default parameter settings specified in the respective papers. Both target and shadow models used early stopping: when loss fails to improve for 50 consecutive epochs. For each dataset, we randomly divided the graphs into training, validation, and testing sets with a ratio of 0.7/0.1/0.2.

#### 6.2 Graph Reconstruction Performance

Evaluation metrics. Recall that the attacker's goal is to accurately uncover the graph structures of training graphs used by the target generative models based on the generated graphs. After obtaining the reconstructed graphs *G rec*, we use a graph alignment algorithm REGAL [\[25\]](#page-14-16) to align each reconstructed graph *g rec* in *G rec* with each training graph *g train* in Φtrain . Next, we evaluate the F1 core based on all aligned pairs (*g rec* ,*g train*). For each *g rec*, we regard the *g train* with highest F1 score as the graph that *g i* is generated from, labeled as *g* ∗ . Then we calculate the attack performance as the average attack performance on all (*g i* ,*g* ∗ ) pairs.

• Structure metrics. We employ three edge-related metrics, namely precision (P), recall (R), and F1 score (F1), to assess the overall effectiveness of our attack in accurately recovering the exact edges and non-edges in graphs.

• Global Metrics. To measure the overall graph reconstruction performance, we evaluate two coverage ratios: the proportion of training graphs exactly matched by the attack (*R*1), and the proportion with an F1 score above 0.75 (*R*2).

Attack setup. For each dataset (MUTAG, Ego-small, EN-ZYMES, IMDB-BINARY, IMDB-MULTI, and QM9) and each target model (EDP-GNN, GDSS, and Digress), we directly run the target model Φ and obtain 1,000 generated graphs, and then apply our GRA on these generated graphs. Competitor. Previous GRAs typically rely on node or graph embeddings [\[56,](#page-15-4) [87\]](#page-17-3), or on the gradients of the target model (GNNs) during the training process [\[86\]](#page-17-4). However, these approaches are not directly applicable to GGDMs, which are trained via forward and reverse processes and output a set of generated graphs. Therefore, we compare our attack with two baselines without the graph alignment step:

- Baseline-1: We treat the generated graphs *g<sup>i</sup>* ∈ *G*ˆ as replicas of Φtrain, and evaluate attack performance as the average over all aligned pairs (*g i* ,*g* ∗ ), where *g* ∗ in Φtrain yields the highest F1 score with *g i* .
- Baseline-2: We reconstruct a graph *g rec* by directly computing the overlapped edges between *g<sup>i</sup>* and *g<sup>j</sup>* using Equation [7](#page-5-2) without graph alignment. Attack performance is evaluated over all aligned pairs (*g rec* ,*g* ∗ ), where *g* ∗ in Φtrain yields the highest F1 score with *g rec* .

Experimental results. Table [3](#page-7-0) presents the GRA results. We have the following observations. First, our attack achieves outstanding performance against all three GGDMs, with F1 scores ranging from 0.72 to 0.99, and coverage ratios of exactmatched graphs and graphs with F1 score above 0.75 reaching up to 0.21 and 0.36, respectively. Second, our attack model

<span id="page-8-0"></span>

| Dataset   |            | EDP-GNN     |            | GDSS        |            | Digress     | Orig.  |         |
|-----------|------------|-------------|------------|-------------|------------|-------------|--------|---------|
|           | D(degree)↓ | D(density)↓ | D(degree)↓ | D(density)↓ | D(degree)↓ | D(density)↓ | Degree | Density |
| MUTAG     | 0.009      | 0.001       | 0.123      | 0.026       | 0.139      | 0.013       | 1.093  | 0.070   |
| ENZYMES   | 0.059      | 0.003       | 0.117      | 0.014       | 0.258      | 0.010       | 1.939  | 0.075   |
| Ego-small | 0.009      | 0.008       | 0.120      | 0.024       | 0.162      | 0.013       | 2.000  | 0.487   |
| IMDB-B    | 0.109      | 0.008       | 0.178      | 0.010       | 0.190      | 0.029       | 4.850  | 0.219   |
| IMDB-M    | 0.005      | 0.007       | 0.228      | 0.003       | 0.249      | 0.022       | 5.266  | 0.309   |
| QM9       | 0.045      | 0.036       | 0.273      | 0.049       | 0.312      | 0.031       | 4.136  | 0.260   |

Table 4: Performance of PIA - the absolute differences in property values between the target model's training set and the inferred values from generated graphs. A smaller difference represents a better performance. "*D*(*degree*)" and "*D*(*density*)" represent the absolute differences of average degree and density, respectively. "Orig." means the property of target model's training set Φtrain .

<span id="page-8-1"></span>![](_page_8_Figure_2.jpeg)

Figure 5: Performance of PIA - the distribution of training graphs in different degree and density ranges. Figure (a) - (c) shows results for graph density, and Figure (d) - (f) for graph degree (*k* = 5, IMDB-B dataset).

outperforms the baselines without the graph alignment module on structural metrics. Although Baseline-2 achieves high precision in some cases, it suffers from significantly lower recall, resulting in lower overall F1 scores. For example, the precision performance of Baseline-2 against EDP-GNN model on IMDB-B dataset is 0.96, but the recall drops to 0.31, and thus yields the F1 score of 0.47. Third, our model demonstrates substantially better global performance than both baselines, as reflected in higher coverage ratios of exact-matched graphs and graphs where the attack achieves a strong F1 score.

### 6.3 Property Inference Performance

Evaluation metrics. To assess the attack's effectiveness, we directly compare the property values of the target model's training set with those inferred from the attack. The experimental results report the absolute differences between the property values of the actual training set and the inferred values from the generated graphs.

Attack setup. For each dataset and each target model, we directly run the target model Φ and obtain 1,000 generated graphs. We calculate and summarize the property values on these generated graphs as the property of the training set of Φ. In our experiments, we consider four different types of graph properties: graph density, average node degree, average number of triangles per node, and graph arboricity (which represents the minimum number of spanning forests into which the edges of a graph can be partitioned). For each graph property, we consider two types of analysis: the average graph property across the training graphs in Φtrain, and the distribution of Φtrain across different ranges of property values. For the second type, we uniformly bucketize the property values

into *k* distinct ranges and infer the proportion of graphs that fall into each range. We set *k* = {5,10} in the experiments. Experimental results. Table [4](#page-8-0) shows the absolute differences of average graph degree and density between the target model's training set and the inferred values from generated graphs. We observe that the absolute difference in average node degree is no more than 0.312, corresponding to a difference ratio of below 7.5% from the original value of Φtrain Similarly, the absolute difference in average node density is less than 0.049, with a difference ratio less than 18.8%. These results demonstrate that our simple yet effective attack can accurately infer the graph properties of Φtrain .

.

Figure [5](#page-8-1) shows the distribution in different node degree and density ranges with a bucket number of *k* = 5. We observe that the inferred distributions closely align with those of Φtrainacross all settings, with absolute ratio differences for each range class varying from 0.001 to 0.028 for average graph density, and 0 to 0.014 for average graph degree.

We show the results of the properties of average number of triangles per node and graph arboricity, along with the distribution results of *k* = 10 in Appendix [C.](#page-18-0) The observations are similar to those in Table [4](#page-8-0) and Figure [5.](#page-8-1)

### 6.4 Membership Inference Performance

Evaluation metrics. For assessing MIA effectiveness, we employ three metrics: (1) *Attack accuracy*: the ratio of correctly predicted member/non-member graphs among all target graphs; (2) *Area Under the Curve (AUC)*: measured over the true positive rate (TPR) and false positive rate (FPR) at various thresholds of the attack classifier; and (3) *True-Positive Rate at False-Positive Rates (TPR@FPR)* [\[11\]](#page-13-8): the TPR at var-

<span id="page-9-0"></span>

|           |            |          | EDP-GNN |          |          | GDSS  |          | Digress  |       |          |  |
|-----------|------------|----------|---------|----------|----------|-------|----------|----------|-------|----------|--|
| Dataset   | Attack     | Accuracy | AUC     | T PR@FPR | Accuracy | AUC   | T PR@FPR | Accuracy | AUC   | T PR@FPR |  |
|           | Ours-1     | 0.650    | 0.763   | 0.225    | 0.821    | 0.811 | 0.467    | 0.845    | 0.881 | 0.327    |  |
|           | Ours-2     | 0.698    | 0.750   | 0.234    | 0.887    | 0.936 | 0.551    | 0.850    | 0.951 | 0.472    |  |
| MUTAG     | Baseline-1 | 0.750    | 0.813   | 0.290    | 0.833    | 0.889 | 0.519    | 0.849    | 0.914 | 0.442    |  |
|           | Baseline-2 | 0.750    | 0.830   | 0.290    | 0.833    | 0.917 | 0.548    | 0.833    | 0.944 | 0.420    |  |
|           | Ours-1     | 0.813    | 0.831   | 0.427    | 0.825    | 0.868 | 0.467    | 0.817    | 0.845 | 0.344    |  |
|           | Ours-2     | 0.900    | 0.913   | 0.539    | 0.852    | 0.880 | 0.562    | 0.835    | 0.869 | 0.332    |  |
| ENZYMES   | Baseline-1 | 0.760    | 0.741   | 0.276    | 0.696    | 0.752 | 0.292    | 0.733    | 0.750 | 0.288    |  |
|           | Baseline-2 | 0.833    | 0.857   | 0.437    | 0.855    | 0.892 | 0.585    | 0.846    | 0.868 | 0.330    |  |
|           | Ours-1     | 0.874    | 0.969   | 0.902    | 0.917    | 0.972 | 0.768    | 0.854    | 0.901 | 0.453    |  |
|           | Ours-2     | 0.901    | 0.991   | 0.913    | 0.934    | 0.994 | 0.821    | 0.892    | 0.943 | 0.487    |  |
| Ego-small | Baseline-1 | 0.703    | 0.812   | 0.412    | 0.715    | 0.833 | 0.517    | 0.671    | 0.763 | 0.272    |  |
|           | Baseline-2 | 0.882    | 0.974   | 0.901    | 0.909    | 0.968 | 0.781    | 0.825    | 0.872 | 0.429    |  |
|           | Ours-1     | 0.955    | 0.979   | 0.757    | 0.908    | 0.999 | 0.957    | 0.918    | 0.957 | 0.652    |  |
|           | Ours-2     | 0.991    | 0.995   | 0.878    | 0.992    | 0.999 | 0.970    | 0.933    | 0.989 | 0.727    |  |
| IMDB-B    | Baseline-1 | 0.731    | 0.784   | 0.313    | 0.667    | 0.722 | 0.288    | 0.640    | 0.703 | 0.292    |  |
|           | Baseline-2 | 0.917    | 0.986   | 0.733    | 0.986    | 0.999 | 0.970    | 0.946    | 0.969 | 0.724    |  |
|           | Ours-1     | 0.902    | 0.992   | 0.837    | 0.942    | 0.911 | 0.667    | 0.850    | 0.896 | 0.538    |  |
|           | Ours-2     | 0.912    | 0.971   | 0.789    | 0.999    | 0.999 | 0.985    | 0.925    | 0.955 | 0.633    |  |
| IMDB-M    | Baseline-1 | 0.739    | 0.781   | 0.308    | 0.724    | 0.762 | 0.302    | 0.645    | 0.711 | 0.298    |  |
|           | Baseline-2 | 0.938    | 0.969   | 0.774    | 0.973    | 0.999 | 0.970    | 0.892    | 0.961 | 0.652    |  |
|           | Ours-1     | 0.752    | 0.816   | 0.412    | 0.766    | 0.843 | 0.437    | 0.701    | 0.780 | 0.335    |  |
|           | Ours-2     | 0.781    | 0.854   | 0.456    | 0.826    | 0.849 | 0.472    | 0.772    | 0.829 | 0.407    |  |
| QM9       | Baseline-1 | 0.638    | 0.733   | 0.225    | 0.682    | 0.767 | 0.305    | 0.599    | 0.671 | 0.209    |  |
|           | Baseline-2 | 0.750    | 0.821   | 0.403    | 0.793    | 0.850 | 0.426    | 0.726    | 0.788 | 0.333    |  |

Table 5: MIA performance under setting 1 (Non-transfer). "Ours-1" and "Ours-2" denote our attacks using features *A* train-1 and *A* train-2, respectively. For each dataset and each target model, the best result is highlighted in bold, and the second-best is underlined.

ious FPR values. We use TPR@0.1FPR in our experiments. Attack setup. For each dataset and target GGDM, we first split the original data into two halves: with 50% used as the training set for the target generative diffusion model, representing the member graphs (*Gmem*), and the remaining 50% serving as non-member graphs (*Gnon*−*mem*). Next, we generate the MIA training dataset by randomly sampling 50% of the graphs from *Gmem* as members, while selecting an equal number of graphs from *Gnon*−*mem* as non-members. The remaining graphs from both *Gmem* and *Gnon*−*mem* are used to form the MIA testing dataset. This ensures the testing set contains an equal number of member and non-member graphs, with no overlap with the training set. We feed each graph in the MIA training and testing set is then fed into the trained shadow model Φ*<sup>S</sup>* and generate 100 graphs per input graph. These generated graphs are subsequently used to construct the attack features. We consider two distinct settings for shadow graph:

- Setting 1 (Non-transfer setting): Both the shadow and target graphs are sampled from the same dataset.
- Setting 2 (Dataset transfer setting): The shadow and target graphs are sampled from different datasets. Specifically, we replace the MIA testing set in the non-transfer setting with the testing set from a dataset different from that of the shadow model.

Competitors. We compare our method with two state-of-theart white-box baseline attacks, both assuming the attacker has access to the target model. These baselines are regarded as

strong attacks and are commonly used in prior MIA studies: one is loss-based, and the other is gradient-based.

- Baseline-1: Loss-based MIA. We adapt the white-box lossbased MIA from image-based diffusion models [\[12,](#page-13-1) [15,](#page-14-10) [44\]](#page-15-10) to our setting. First, we compute the loss values of member and non-member samples across different timesteps. Then, we train an attack classifier using the aggregated loss values from the most effective timestep range.
- Baseline-2: Gradient-based MIA. We adapt the whitebox gradient-based MIA from image-based diffusion models [\[51\]](#page-15-5) to our setting. Specifically, we aggregate gradients from member and non-member samples over the most effective timestep range and feed them into the attack classifier.

Attacker performance under Setting 1 (Non-transfer setting). Table [5](#page-9-0) illustrates the attack performance under Setting 1. We have the following observations. First, both our MIA models using attack feature *A* train-1 and *A* train-2 demonstrate outstanding performance across four datasets and three diffusion models, yielding attack accuracy ranging from 0.650 to 0.955 for Ours-1 and 0.698 to 0.999 for Ours-2, attack AUC ranging from 0.763 to 0.992 for Ours-1 and 0.750 to 0.999 for Ours-2, TPR@FPR ranging from 0.225 to 0.957 for Ours-1 and 0.234 to 0.985 for Ours-2. Second, Ours-2 outperforms Baselines-1 in most settings. Particularly, Ours-2's attack accuracy and AUC are respectively 0.325 and 0.277 higher than those of Baseline-1 when GDSS is the target model and IMDB-B is the target dataset. Third, Baseline-2

<span id="page-10-0"></span>![](_page_10_Figure_0.jpeg)

Figure 6: MIA AUC performance against EDP-GNN target model under setting 2 (dataset transfer setting).

achieves performance comparable to ours in some settings, due to its use of the target model's gradients. However, it is a white-box attack requiring access to the model's structure and parameters, which may be difficult to obtain in real-world scenarios. Moreover, its substantially larger feature dimension causes Baseline-2 to overfit in some settings, leading to worse performance than ours in those cases. Additionally, Ours-2 outperforms Ours-1 in most settings, which indicates the attacker feature of *A* train-2 is more effective than *A* train-1 .

Attacker performance under Setting 2 (Data transfer setting). Figure [6](#page-10-0) shows the attack results under Setting 2. The following observations can be made. First, for each target dataset, both of our attacks achieve the highest attack accuracy when shadow and target graphs originate from the same dataset (i.e., within the same column). Second, even under the transfer setting, the attack remains effective, with attack AUC ranging from 0.61 to 0.96 for Ours-1 and 0.66 to 0.90 for Ours-2. This demonstrates the attack models' ability to learn knowledge from shadow data and transfer that knowledge to the target graphs. Specifically, for Ours-1, member graphs yield higher similarity scores between generated and input graphs than non-members; and for Ours-2, generated graphs from member graphs exhibit higher similarity than those from non-member graphs. Third, our attack models outperform both baselines in most cases, which indicates the effectiveness of our attack features. This also highlights an advantage of our model over Baseline-2, as our attacks are more applicable to real-world data transfer scenarios.

### 7 Two Possible Defenses

To enhance the privacy of GGDMs, one intuitive approach is to introduce noise into the models to confuse adversaries and provide privacy protection. Existing defense mechanisms for graph learning models mainly target GNNs [\[22,](#page-14-18) [48,](#page-15-18) [67\]](#page-16-18) or random-walk-based models [\[20,](#page-14-19) [53\]](#page-15-19). These strategies typically inject noise into the gradients/parameters during model training, such as equipping the model with differential privacy [\[6\]](#page-13-9), or adding noise to model's output, namely the posteriors or final graph embeddings [\[23,](#page-14-5) [56,](#page-15-4) [70\]](#page-16-8). However, when we apply these defenses to GGDMs, the empirical evaluation (will be present in Sec. [7.3\)](#page-11-0) shows that these methods significantly degrade the quality of the generated graphs. To

address this issue, we propose both pre-processing and postprocessing defenses that perturb the structures of training graphs before model training or after graph generation, respectively. Specifically, we selectively flip the edges and nonedges that are deemed least important. This approach minimizes the quality loss of the generated graphs while providing sufficient privacy protection against inference attacks. We next detail the process of estimating the importance of edges and non-edges (Sec. [7.1\)](#page-10-1), describe our defense mechanisms (Sec. [7.2\)](#page-11-1), and present the empirical performance (Sec. [7.3\)](#page-11-0).

# <span id="page-10-1"></span>7.1 Saliency Map-based Importance Measurement

To measure the importance of edges and non-edges, we utilize the saliency maps-based method [\[7,](#page-13-10) [8,](#page-13-11) [47,](#page-15-20) [58\]](#page-15-21). Saliency maps are widely used in explainable machine learning and artificial intelligence, particularly in the image domain, to provide insights into the relevance of input features that contribute to a model's output [\[27,](#page-14-20) [47,](#page-15-20) [62\]](#page-16-19). The most popular methods generate a gradient saliency map by back-propagating gradients from the end of the neural network and projecting it onto an image plane [\[47,](#page-15-20) [52,](#page-15-22) [58\]](#page-15-21). These gradients are typically derived from a loss function, layer activation, or class activation. The absolute values of these gradients are often used to determine the importance of each feature, with a larger magnitude indicating a greater influence on the model's prediction.

In our approach, we adapt this gradient saliency map-based technique to measure the importance of edges and non-edges for node classification within each graph. This process involves three steps: forward pass through the GNNs, gradient calculation on edges and non-edges, and saliency map construction. The following sections detail each step.

Step 1: Forward pass through the GNNs. Give a graph *G*(*V*,*A*,*X*,*Y*) where *V*, *A*, *X*, and *Y* denote the nodes, adjacency matrix, node features, and node labels, respectively. We feed *G* to a GNN for node classification. GNNs typically follow a message-passing paradigm to learn node embeddings. In this paper, we use Graph Convolutional Network (GCN) [\[35\]](#page-14-21), whose layer-wise propagation is:

$$H^{(l)} = \sigma\left(\hat{A}H^{(l-1)}W^{(l)}\right),\tag{13}$$

where *A*ˆ = *D*˜ <sup>−</sup>1/<sup>2</sup> (*A*˜)*D*˜ <sup>−</sup>1/<sup>2</sup> denotes the normalized adjacency

<span id="page-11-2"></span>![](_page_11_Figure_0.jpeg)

Figure 7: Impact of perturbation ratio *r* on defense performance (EDP-GNN model, IMDB-BINARY dataset). "MIA-1" and "MIA-2" represent MIA with attack feature of *A* train-1 and *A* train-2, respectively. *D*(*degree*) denotes the difference in degree between inferred and original training graphs. For GRA, MIA-1, and MIA-2, a lower attack F1 score or AUC indicates stronger defense, while for PIA, a greater difference between the inferred and original property values represents stronger defense.

matrix, with *A*˜ = *A*+*I* and *I* is the identity matrix. *D*˜ is the degree matrix of *A*˜, *W*(*l*) is the learnable weight matrix at *l*-th layer, and σ denotes a non-linear activation function such as ReLU. The final node embeddings *H* (*l*) are passed through a softmax layer to predict node labels. The model is trained using cross-entropy loss. Note that in real-world applications, some graphs may lack node features *X* or node labels *Y* or both. We address this by employing common strategies: (1) if *X* is missing, we set *X* as the identity matrix, and (2) if *Y* is missing, we set *Y* as the one-hot encoded node degrees.

Step 2: Gradient calculation on edges and non-edges. Once the GNN's performance stabilizes after several training epochs, we backpropagate the gradient of the node classification loss to the adjacency matrix. The gradient of the loss *L* for each entry *Ai j* is computed via the chain rule as:

<span id="page-11-3"></span>
$$\frac{\partial L}{\partial A_{ij}} = \sum_{k} \left( \frac{\partial L}{\partial H_k} \cdot \frac{\partial H_k}{\partial \hat{A}_k} \cdot \frac{\partial \hat{A}_k}{\partial A_{ij}} \right),\tag{14}$$

where *Ai j* is an entry in adjacency matrix *A*, which represents the link status between node *v<sup>i</sup>* and node *v<sup>j</sup>* , ∂*L* ∂*H<sup>k</sup>* denotes the gradient of loss with respect to embedding *H<sup>k</sup>* of node *vk*, ∂*H<sup>k</sup>* ∂*A*ˆ *k* represents the gradient of node embedding *H<sup>k</sup>* with respect to the normalized adjacency matrix, <sup>∂</sup>*A*<sup>ˆ</sup> *k* ∂*Ai j* denotes the gradient of the normalized adjacency matrix with respect to *Ai j*.

Step 3: Saliency map construction. After obtaining the gradients of all entries in *A*, namely the gradients of all edges and non-edges within *G*, we construct the saliency map by taking the absolute value of the gradient at each *Ai j* as:

<span id="page-11-4"></span>Saliency
$$(A_{ij}) = \left| \frac{\partial L}{\partial A_{ij}} \right|$$
. (15)

A larger gradient magnitude indicates greater importance of the corresponding edge or non-edge to model's performance.

## <span id="page-11-1"></span>7.2 Details of the Defenses

While adding noise to training or generated graphs can reduce attacks' capabilities, random noise often causes substantial quality loss. Therefore, the primary objective of our defense is to provide effective protection against inference attacks while preserving the model utility. To achieve this objective, we propose a perturbation-based strategy that selectively flips only the least important edges or non-edges or both in the training graphs (pre-processing) or generated graphs (post-processing). Although these targeted modifications may slightly alter the utility of generated graphs, our experiments show that they effectively reduce attack performance with minimal utility loss. Both defenses follow a three-step process. The pseudo-code of the defenses is provided in Appendix [D.](#page-19-0)

## Defense-1: Noisy training graphs (pre-processing).

- Step 1: For each graph *G* in target model's training set, we construct a saliency map to measure the importance of each edge and non-edge in *G* by using the method in Section [7.1.](#page-10-1)
- Step 2: We rank all edges and non-edges in a single list based on their importance values in ascending order.
- Step 3: We flip one entry in the adjacency matrix *A* with the lowest importance. The perturbed graph is labeled as *G* ′ .

We then repeat Steps 1 to 3 on the newly generated graph *G* ′ until the number of flipped entries reaches the pre-defined budget, ⌊*r*×|*E*|⌋, where |*E*| denotes the number of edges in *G*, and *r* ∈ (0,1] is the *perturbation ratio*. A larger *r* introduces more structural changes. Finally, we train the target GGDM on the perturbed training graphs.

#### Defense-2: Noisy generated graphs (post-processing).

- Step 1: Similar to Defense-1, we first measure the importance of each edge and non-edge for every graph *G*ˆ in the generated graph set using the saliency map-based method.
- Step 2 and Step 3: These steps remain the same as those in Defense-1, involving ranking and flipping the least important edge or non-edge to obtain the perturbed graph *G*ˆ′ .

The process is repeated until each graph in *G*ˆ reaches its predefined flip budget, after which the perturbed graphs form the final output of the target GGDM.

## <span id="page-11-0"></span>7.3 Performance of Defense

In this section, we provide an assessment of the performance of our defense mechanism. We use the same target models and datasets as those for the attack evaluation (Section [6\)](#page-6-1).

<span id="page-12-0"></span>![](_page_12_Figure_0.jpeg)

Figure 8: The defense-utility curve (EDP-GNN model, IMDB-BINARY dataset). The x symbol indicates no defense. For GRA, MIA-1, and MIA-2, a lower attack F1 score or AUC combined with higher model utility indicates a better trade-off. For PIA, a greater difference between the inferred and original property values, along with higher model utility, signifies a better trade-off.

Setup. We set the perturbation ratio *r* = {0,0.1,0.3,0.5,0.7,0.9}. Larger values of *r* indicate stronger privacy protection.

Metrics. We measure *defense effectiveness* as the attack performance after applying the defense, using the same evaluation metrics as in Section [6,](#page-6-1) with variations specific to attack types. The *utility* of the target model is evaluated as the performance of its generated graphs on a downstream task of graph classification. A higher AUC indicates better model utility. Baselines. For comparative analysis, we compare our defense mechanism with the following two baselines:

- Baseline-1: Differential privacy (DP). We adopt DP-SGD, a state-of-the-art DP-based deep learning method [\[6\]](#page-13-9), as our first baseline. We add Gaussian noise to gradients in each training iteration of target model. We set the privacy budget as ε = {0.001,0.01,0.1,1,5,10}, where a smaller ε indicates stronger privacy guarantees.
- Baseline-2: Randomly-flipped generated graphs. This baseline applies random flips where each edge and nonedge in the generated graphs is flipped with a probability *p*. We set *p* = {0.1,0.2,0.3,0.4,0.5,0.6} in experiments.

Defense effectiveness. Figure [7](#page-11-2) presents the attack performance after deploying our defense mechanisms with various perturbation ratios when EDP-GNN is used as the target model. We have the following observations. First, our perturbation methods are highly effective against all three types of attacks, notably reducing their performance. Second, the defense strength increases with a higher perturbation ratio *r*. Even with a modest perturbation ratio of 0.1 (perturbing only 10% of graph edges), the attack AUC drops to around 0.6 for MIA-1 and 0.53 for MIA-2 with Defense-2.

Defense-Utility tradeoff. To illustrate the defense-utility trade-off, we plot the defense-utility curve of our defenses and the two baselines in Figure [8.](#page-12-0) We establish a set of configurations by varying the perturbation ratio for our defenses, the privacy budget for Baseline-1, and the flipped probability for Baseline-2, as described in the Setup. For each configuration, we evaluate defense effectiveness and target model utility, with each point on the curve representing a defense-utility pair. We observe that our defenses outperform the baselines

in this trade-off. When defense performance drops below 0.6 for GRA, MIA-1, and MIA-2, which indicates ineffective attacks, our defenses maintain higher model utility compared to baselines, with degradation less than 0.05. Similarly, at the same utility level, our methods provide stronger protection across all four attacks. The superiority stems from selectively adding noise only to the least important edges and non-edges.

#### 8 Conclusion

In this paper, we investigated the privacy leakage of graph generative diffusion models (GGDMs) and proposed three black-box attacks that extract sensitive information from a target model using only its generated graphs. Specifically, we introduced (i) a graph reconstruction attack that successfully infers training graph structures of the target GGDM, (ii) a property inference attack that extracts statistical properties from the generated graphs, which closely resemble those of the training graphs, and (iii) membership inference attacks that determine whether a given graph is included in the training set of the target GGDM. Extensive experiments demonstrate the effectiveness of these attacks against three state-ofthe-art GGDMs. Furthermore, we have proposed two defense mechanisms based on perturbing training or generated graphs by flipping the least important edges and non-edges, which have been experimentally shown to be effective.

In future work, we will explore the vulnerability of GGDMs to other types of attacks, such as model inversion [\[18,](#page-14-22)[74\]](#page-16-20) and substructure inference attacks [\[73,](#page-16-5) [87\]](#page-17-3), including strongly connected components, stars, and specialized substructures.

### 9 Acknowledgments

We thank the anonymous reviewers for their feedback. This work was supported by the Hong Kong RGC Grants RGC grants C2003-23Y and C1043-24GF.

## 10 Ethical Considerations

Stakeholder impact analysis. The goal of our work is to help researchers and developers identify and reduce GGDM vulnerabilities, improving privacy posture and trustworthiness in real deployments. These vulnerabilities can affect multiple stakeholders, including GGDM deployers, data owners, and clients/users of GGDMs through the platform/API. Model deployers (e.g., cloud providers, AI marketplaces, or enterprises hosting GGDMs) may risk contract breaches, takedowns or patches, reputational harm, and disclosure obligations. Data owners may face re-identification risks and IP leakage; even without exact copies, adversaries can reverseengineer datasets by stitching reconstructed fragments and statistics. Users may propagate sensitive priors into downstream products, and competitors can de-bias outputs to extract non-public signals. However, we have proposed two efficient defenses (Section 7) to mitigate these threats with moderate utility loss, offering practical tools for responsible GGDM deployment.

Mitigation strategies. To minimize potential negative consequences of this research, we release defensive methods alongside the attacks so that practitioners can immediately apply appropriate mitigations. We also gate high-risk components, such as detailed attack code or sensitive hyperparameters, and provide clear guidelines for their safe and responsible use. In addition, we offer deployment recommendations, including API-level safeguards (e.g., rate limiting, authentication, anomaly detection, and detailed auditing to prevent iterative probing attacks), as well as the structure-aware output sanitization introduced in our defense section, to help strengthen real-world GGDM deployments against misuse.

Safety measures in our research process. During research, we used public datasets and public models, no human subjects/PII, and isomorphism-invariant aggregates, avoiding reconstruction or release of any identifiable data. Our results are reported as means over multiple seeds/samples to minimize pinpointing any singular graph or any identifiable data.

Long-term commitment. We will maintain our public repository, respond to community feedback, and update threat assessments and defense recommendations as GGDM technologies evolve. This ensures that our work continues to support safe and responsible adoption of GGDMs.

# 11 Open Science

In line with open science policies, we have made the relevant code publicly accessible to facilitate reproducibility and foster further research. As mentioned in Section 9, we release our attacks together with defenses, while withholding certain high-risk components (e.g., the AWE module in MIA) to reduce misuse. Our code is available at https://zenodo.org/records/17946102. All datasets used in this study were publicly available.

## References

- <span id="page-13-3"></span>[1] Amazon aws. https://aws.amazon.com/marketplace/solutions/machinelearning.
- <span id="page-13-6"></span>[2] Azure. https://azure.microsoft.com/en-us/products/aiservices.
- <span id="page-13-4"></span>[3] Google ai hub. https://aihub.cloud.google.com/.
- <span id="page-13-7"></span>[4] Ibm. https://www.ibm.com/products/watsonx-ai.
- <span id="page-13-5"></span>[5] Modzy: Ai model marketplace. https://www.modzy.com/marketplace/.
- <span id="page-13-9"></span>[6] Martin Abadi, Andy Chu, Ian Goodfellow, H Brendan McMahan, Ilya Mironov, Kunal Talwar, and Li Zhang. Deep learning with differential privacy. In *Proceedings of the ACM SIGSAC Conference on Computer and Communications Security*, pages 308–318, 2016.
- <span id="page-13-10"></span>[7] Julius Adebayo, Justin Gilmer, Michael Muelly, Ian Goodfellow, Moritz Hardt, and Been Kim. Sanity checks for saliency maps. *Advances in neural information processing systems*, 31, 2018.
- <span id="page-13-11"></span>[8] Ahmed Alqaraawi, Martin Schuessler, Philipp Weiß, Enrico Costanza, and Nadia Berthouze. Evaluating saliency map explanations for convolutional neural networks: a user study. In *Proceedings of the international conference on intelligent user interfaces*, pages 275–285, 2020.
- <span id="page-13-2"></span>[9] Jacob Austin, Daniel D Johnson, Jonathan Ho, Daniel Tarlow, and Rianne Van Den Berg. Structured denoising diffusion models in discrete state-spaces. *Advances in Neural Information Processing Systems*, 34:17981– 17993, 2021.
- <span id="page-13-0"></span>[10] Hanqun Cao, Cheng Tan, Zhangyang Gao, Yilun Xu, Guangyong Chen, Pheng-Ann Heng, and Stan Z Li. A survey on generative diffusion models. *IEEE Transactions on Knowledge and Data Engineering*, 2024.
- <span id="page-13-8"></span>[11] Nicholas Carlini, Steve Chien, Milad Nasr, Shuang Song, Andreas Terzis, and Florian Tramer. Membership inference attacks from first principles. In *Proceedings of the Conference on Symposium on Security and Privacy*, pages 1897–1914, 2022.
- <span id="page-13-1"></span>[12] Nicolas Carlini, Jamie Hayes, Milad Nasr, Matthew Jagielski, Vikash Sehwag, Florian Tramer, Borja Balle, Daphne Ippolito, and Eric Wallace. Extracting training data from diffusion models. In *USENIX Security Symposium*, pages 5253–5270, 2023.

- <span id="page-14-11"></span>[13] Xiaohui Chen, Yukun Li, Aonan Zhang, and Li-ping Liu. Nvdiff: Graph generation through the diffusion of node vectors. *arXiv preprint arXiv:2211.10794*, 2022.
- <span id="page-14-8"></span>[14] Jinhao Duan, Fei Kong, Shiqi Wang, Xiaoshuang Shi, and Kaidi Xu. Are diffusion models vulnerable to membership inference attacks? In *International Conference on Machine Learning*, pages 8717–8730. PMLR, 2023.
- <span id="page-14-10"></span>[15] Jan Dubinski, Antoni Kowalczuk, Stanis ´ law Pawlak, Przemyslaw Rokita, Tomasz Trzcinski, and Pawe ´ l Morawiecki. Towards more realistic membership inference attacks on large diffusion models. In *Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision*, pages 4860–4869, 2024.
- <span id="page-14-4"></span>[16] Vasisht Duddu, Antoine Boutet, and Virat Shejwalkar. Quantifying privacy leakage in graph embedding. In *Proceedings of EAI International Conference on Mobile and Ubiquitous Systems*, pages 76–85, 2020.
- <span id="page-14-3"></span>[17] Wenqi Fan, Chengyi Liu, Yunqing Liu, Jiatong Li, Hang Li, Hui Liu, Jiliang Tang, and Qing Li. Generative diffusion models on graphs: Methods and applications. *arXiv preprint arXiv:2302.02591*, 2023.
- <span id="page-14-22"></span>[18] Matt Fredrikson, Somesh Jha, and Thomas Ristenpart. Model inversion attacks that exploit confidence information and basic countermeasures. In *Proceedings of the ACM SIGSAC Conference on Computer and Communications Security*, pages 1322–1333, 2015.
- <span id="page-14-7"></span>[19] Neil Zhenqiang Gong and Bin Liu. Attribute inference attacks in online social networks. *ACM Transactions on Privacy and Security*, 21(1):1–30, 2018.
- <span id="page-14-19"></span>[20] Aditya Grover and Jure Leskovec. node2vec: Scalable feature learning for networks. In *Proceedings of the ACM SIGKDD international conference on Knowledge discovery and data mining*, pages 855–864, 2016.
- <span id="page-14-12"></span>[21] Kilian Konstantin Haefeli, Karolis Martinkus, Nathanaël Perraudin, and Roger Wattenhofer. Diffusion models for graphs benefit from discrete state spaces. *arXiv preprint arXiv:2210.01549*, 2022.
- <span id="page-14-18"></span>[22] Will Hamilton, Zhitao Ying, and Jure Leskovec. Inductive representation learning on large graphs. *Advances in neural information processing systems*, 30, 2017.
- <span id="page-14-5"></span>[23] Xinlei He, Jinyuan Jia, Michael Backes, Neil Zhenqiang Gong, and Yang Zhang. Stealing links from graph neural networks. In *USENIX Security Symposium*, pages 2669– 2686, 2021.
- <span id="page-14-6"></span>[24] Xinlei He, Rui Wen, Yixin Wu, Michael Backes, Yun Shen, and Yang Zhang. Node-level membership inference attacks against graph neural networks. *arXiv preprint arXiv:2102.05429*, 2021.

- <span id="page-14-16"></span>[25] Mark Heimann, Haoming Shen, Tara Safavi, and Danai Koutra. Regal: Representation learning-based graph alignment. In *Proceedings of the ACM international conference on information and knowledge management*, pages 117–126, 2018.
- <span id="page-14-2"></span>[26] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. *Advances in neural information processing systems*, 33:6840–6851, 2020.
- <span id="page-14-20"></span>[27] Chia-Yu Hsu and Wenwen Li. Explainable geoai: can saliency maps help interpret artificial intelligence's learning process? an empirical study on natural feature detection. *International Journal of Geographical Information Science*, 37(5):963–987, 2023.
- <span id="page-14-9"></span>[28] Hailong Hu and Jun Pang. Membership inference of diffusion models. *arXiv preprint arXiv:2301.09956*, 2023.
- <span id="page-14-15"></span>[29] Hongsheng Hu, Zoran Salcic, Lichao Sun, Gillian Dobbie, Philip S Yu, and Xuyun Zhang. Membership inference attacks on machine learning: A survey. *ACM Computing Surveys*, 54(11s):1–37, 2022.
- <span id="page-14-13"></span>[30] Han Huang, Leilei Sun, Bowen Du, Yanjie Fu, and Weifeng Lv. Graphgdp: Generative diffusion processes for permutation invariant graph generation. In *IEEE International Conference on Data Mining*, pages 201–210, 2022.
- <span id="page-14-1"></span>[31] Han Huang, Leilei Sun, Bowen Du, and Weifeng Lv. Conditional diffusion based on discrete graph structures for molecular graph generation. 37(4):4302–4311, 2023.
- <span id="page-14-17"></span>[32] Sergey Ivanov and Evgeny Burnaev. Anonymous walk embeddings. In *International conference on machine learning*, pages 2186–2195, 2018.
- <span id="page-14-0"></span>[33] Yangqin Jiang, Yuhao Yang, Lianghao Xia, and Chao Huang. Diffkg: Knowledge graph diffusion model for recommendation. In *Proceedings of the ACM international conference on web search and data mining*, pages 313–321, 2024.
- <span id="page-14-14"></span>[34] Jaehyeong Jo, Seul Lee, and Sung Ju Hwang. Scorebased generative modeling of graphs via the system of stochastic differential equations. In *International Conference on Machine Learning*, pages 10362–10383, 2022.
- <span id="page-14-21"></span>[35] Thomas N Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. *arXiv preprint arXiv:1609.02907*, 2016.

- <span id="page-15-9"></span>[36] Fei Kong, Jinhao Duan, RuiPeng Ma, Hengtao Shen, Xiaofeng Zhu, Xiaoshuang Shi, and Kaidi Xu. An efficient membership inference attack for the diffusion model by proximal initialization. *arXiv preprint arXiv:2305.18355*, 2023.
- <span id="page-15-7"></span>[37] Jingwei Li, Jing Dong, Tianxing He, and Jingzhao Zhang. Towards black-box membership inference attack for diffusion models. *arXiv preprint arXiv:2405.20771*, 2024.
- <span id="page-15-13"></span>[38] Yujia Li, Oriol Vinyals, Chris Dyer, Razvan Pascanu, and Peter Battaglia. Learning deep generative models of graphs. *arXiv preprint arXiv:1803.03324*, 2018.
- <span id="page-15-0"></span>[39] Zongwei Li, Lianghao Xia, and Chao Huang. Recdiff: diffusion model for social recommendation. In *Proceedings of the ACM International Conference on Information and Knowledge Management*, pages 1346–1355, 2024.
- <span id="page-15-2"></span>[40] Chengyi Liu, Wenqi Fan, Yunqing Liu, Jiatong Li, Hang Li, Hui Liu, Jiliang Tang, and Qing Li. Generative diffusion models on graphs: Methods and applications. *arXiv preprint arXiv:2302.02591*, 2023.
- <span id="page-15-1"></span>[41] Chengyi Liu, Jiahao Zhang, Shijie Wang, Wenqi Fan, and Qing Li. Score-based generative diffusion models for social recommendations. *arXiv preprint arXiv:2412.15579*, 2024.
- <span id="page-15-12"></span>[42] Tianze Luo, Zhanfeng Mo, and Sinno Jialin Pan. Fast graph generative model via spectral diffusion. *arXiv preprint arXiv:2211.08892*, 2022.
- <span id="page-15-6"></span>[43] Xinjian Luo, Yangfan Jiang, Fei Wei, Yuncheng Wu, Xiaokui Xiao, and Beng Chin Ooi. Exploring privacy and fairness risks in sharing diffusion models: An adversarial perspective. *arXiv preprint arXiv:2402.18607*, 2024.
- <span id="page-15-10"></span>[44] Tomoya Matsumoto, Takayuki Miura, and Naoto Yanai. Membership inference attacks against diffusion models. In *IEEE Security and Privacy Workshops*, pages 77–83, 2023.
- <span id="page-15-3"></span>[45] Wilnellys Moore and Sarah Frye. Review of hipaa, part 1: history, protected health information, and privacy and security rules. *Journal of nuclear medicine technology*, 47(4):269–272, 2019.
- <span id="page-15-17"></span>[46] Christopher Morris, Nils M Kriege, Franka Bause, Kristian Kersting, Petra Mutzel, and Marion Neumann. Tudataset: A collection of benchmark datasets for learning with graphs. *arXiv preprint arXiv:2007.08663*, 2020.
- <span id="page-15-20"></span>[47] T Nathan Mundhenk, Barry Y Chen, and Gerald Friedland. Efficient saliency maps for explainable ai. *arXiv preprint arXiv:1911.11293*, 2019.

- <span id="page-15-18"></span>[48] Mathias Niepert, Mohamed Ahmed, and Konstantin Kutzkov. Learning convolutional neural networks for graphs. In *International conference on machine learning*, pages 2014–2023, 2016.
- <span id="page-15-11"></span>[49] Chenhao Niu, Yang Song, Jiaming Song, Shengjia Zhao, Aditya Grover, and Stefano Ermon. Permutation invariant graph generation via score-based generative modeling. In *International Conference on Artificial Intelligence and Statistics*, pages 4474–4484, 2020.
- <span id="page-15-8"></span>[50] Yan Pang and Tianhao Wang. Black-box membership inference attacks against fine-tuned diffusion models. *arXiv preprint arXiv:2312.08207*, 2023.
- <span id="page-15-5"></span>[51] Yan Pang, Tianhao Wang, Xuhui Kang, Mengdi Huai, and Yang Zhang. White-box membership inference attacks against diffusion models. *arXiv preprint arXiv:2308.06405*, 2023.
- <span id="page-15-22"></span>[52] Badri N Patro, Mayank Lunayach, Shivansh Patel, and Vinay P Namboodiri. U-cam: Visual explanation using uncertainty based class activation maps. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 7444–7453, 2019.
- <span id="page-15-19"></span>[53] Bryan Perozzi, Rami Al-Rfou, and Steven Skiena. Deepwalk: Online learning of social representations. In *Proceedings of the ACM SIGKDD international conference on Knowledge discovery and data mining*, pages 701– 710, 2014.
- <span id="page-15-14"></span>[54] Maria Rigaki and Sebastian Garcia. A survey of privacy attacks in machine learning. *ACM Computing Surveys*, 2020.
- <span id="page-15-15"></span>[55] Ahmed Salem, Yang Zhang, Mathias Humbert, Mario Fritz, and Michael Backes. Ml-leaks: Model and data independent membership inference attacks and defenses on machine learning models. *arXiv preprint arXiv:1806.01246*, 2018.
- <span id="page-15-4"></span>[56] Yun Shen, Yufei Han, Zhikun Zhang, Min Chen, Ting Yu, Michael Backes, Yang Zhang, and Gianluca Stringhini. Finding mnemon: Reviving memories of node embeddings. *arXiv preprint arXiv:2204.06963*, 2022.
- <span id="page-15-16"></span>[57] Reza Shokri, Marco Stronati, Congzheng Song, and Vitaly Shmatikov. Membership inference attacks against machine learning models. In *IEEE symposium on security and privacy*, pages 3–18, 2017.
- <span id="page-15-21"></span>[58] Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman. Deep inside convolutional networks: Visualising image classification models and saliency maps. *arXiv preprint arXiv:1312.6034*, 2013.

- <span id="page-16-13"></span>[59] Gowthami Somepalli, Vasu Singla, Micah Goldblum, Jonas Geiping, and Tom Goldstein. Understanding and mitigating copying in diffusion models. *arXiv preprint arXiv:2305.20086*, 2023.
- <span id="page-16-16"></span>[60] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. *Advances in neural information processing systems*, 2019.
- <span id="page-16-7"></span>[61] Anshuman Suri and David Evans. Formalizing and estimating distribution inference risks. *arXiv preprint arXiv:2109.06024*, 2021.
- <span id="page-16-19"></span>[62] Karolina Szczepankiewicz, Adam Popowicz, Kamil Charkiewicz, Katarzyna Nalkecz-Charkiewicz, Michal Szczepankiewicz, Slawomir Lasota, Pawel Zawistowski, and Krystian Radlak. Ground truth based comparison of saliency maps algorithms. *Scientific Reports*, 13(1):16887, 2023.
- <span id="page-16-11"></span>[63] Shuai Tang, Zhiwei Steven Wu, Sergul Aydore, Michael Kearns, and Aaron Roth. Membership inference attacks on diffusion models via quantile regression. *arXiv preprint arXiv:2312.05140*, 2023.
- <span id="page-16-1"></span>[64] Jos Torge, Charles Harris, Simon V Mathis, and Pietro Lio. Diffhopp: A graph diffusion model for novel drug design via scaffold hopping. *arXiv preprint arXiv:2308.07416*, 2023.
- <span id="page-16-14"></span>[65] Vu Tuan Truong, Luan Ba Dang, and Long Bao Le. Attacks and defenses for generative diffusion models: A comprehensive survey. *arXiv preprint arXiv:2408.03400*, 2024.
- <span id="page-16-15"></span>[66] Gerrit van den Burg and Chris Williams. On memorization in probabilistic deep generative models. *Advances in Neural Information Processing Systems*, 34:27916– 27928, 2021.
- <span id="page-16-18"></span>[67] Petar Velivckovic, Guillem Cucurull, Arantxa Casanova, ´ Adriana Romero, Pietro Liò, and Yoshua Bengio. Graph attention networks. In *Proceedings of International Conference on Learning Representations*, 2018.
- <span id="page-16-12"></span>[68] Clement Vignac, Igor Krawczuk, Antoine Siraudin, Bohan Wang, Volkan Cevher, and Pascal Frossard. Digress: Discrete denoising diffusion for graph generation. *arXiv preprint arXiv:2209.14734*, 2022.
- <span id="page-16-0"></span>[69] Wenjie Wang, Yiyan Xu, Fuli Feng, Xinyu Lin, Xiangnan He, and Tat-Seng Chua. Diffusion recommender model. In *Proceedings of the International ACM SIGIR Conference on Research and Development in Information Retrieval*, pages 832–841, 2023.

- <span id="page-16-8"></span>[70] Xiuling Wang and Wendy Hui Wang. Group property inference attacks against graph neural networks. In *Proceedings of the ACM SIGSAC Conference on Computer and Communications Security*, pages 2871–2884, 2022.
- <span id="page-16-4"></span>[71] Xiuling Wang and Wendy Hui Wang. Link membership inference attacks against unsupervised graph representation learning. In *Proceedings of the Annual Computer Security Applications Conference*, pages 477–491, 2023.
- <span id="page-16-9"></span>[72] Xiuling Wang and Wendy Hui Wang. Gcl-leak: Link membership inference attacks against graph contrastive learning. *Proceedings on Privacy Enhancing Technologies*, 2024.
- <span id="page-16-5"></span>[73] Xiuling Wang and Wendy Hui Wang. Subgraph structure membership inference attacks against graph neural networks. *Proceedings on Privacy Enhancing Technologies*, 2024.
- <span id="page-16-20"></span>[74] Bang Wu, Xiangwen Yang, Shirui Pan, and Xingliang Yuan. Model extraction attacks on graph neural networks: Taxonomy and realisation. In *Proceedings of the Asia Conference on Computer and Communications Security*, pages 337–350, 2022.
- <span id="page-16-6"></span>[75] Fan Wu, Yunhui Long, Ce Zhang, and Bo Li. Linkteller: Recovering private edges from graph neural networks via influence analysis. In *Proceedings of IEEE Symposium on Security and Privacy*, 2022.
- <span id="page-16-2"></span>[76] Jiayang Wu, Wensheng Gan, and Philip S Yu. Graph diffusion network for drug-gene prediction. *arXiv preprint arXiv:2502.09335*, 2025.
- <span id="page-16-10"></span>[77] Yixin Wu, Ning Yu, Zheng Li, Michael Backes, and Yang Zhang. Membership inference attacks against text-to-image generation models. *arXiv preprint arXiv:2210.00968*, 2022.
- <span id="page-16-21"></span>[78] Han Xie, Jing Ma, Li Xiong, and Carl Yang. Federated graph classification over non-iid graphs. *Advances in Neural Information Processing Systems*, 34:18839– 18852, 2021.
- <span id="page-16-17"></span>[79] Junchi Yan, Xu-Cheng Yin, Weiyao Lin, Cheng Deng, Hongyuan Zha, and Xiaokang Yang. A short survey of recent advances in graph matching. In *Proceedings of the ACM conference on multimedia retrieval*, pages 167–174, 2016.
- <span id="page-16-3"></span>[80] Ling Yang, Zhilong Zhang, Yang Song, Shenda Hong, Runsheng Xu, Yue Zhao, Wentao Zhang, Bin Cui, and Ming-Hsuan Yang. Diffusion models: A comprehensive survey of methods and applications. *ACM Computing Surveys*, 56(4):1–39, 2023.

#### Algorithm 1: Graph reconstruction attack

<span id="page-17-10"></span>Input: Φ from an online marketplace or an API, shadow graphs *G S* (optional)

Output: The reconstructed graphs *G rec*

- <sup>1</sup> Execute Φ or feed *G S* to Φ to obtain the synthetic graphs *G*ˆ;
- 2 Create the set of representation differences between the graphs and aligned graphs: D = {};
- <sup>3</sup> Create the set of reconstructed graphs: *G rec* = {};
- <sup>4</sup> for *each g<sup>i</sup> in G, i* ˆ ∈ [1,...,*N*] do

```
5 for each gj
                in G, j ˆ ∈ [1,...,N], j ̸= i do
6 Do graph alignment on the graph pair (gi
                                                ,gj)
          with REGAL;
7 Find the most similar counterpart gˆj for gi with
          Eq. 6 ;
8 Add the average node representation difference
          between gi and ˆgj
                           to D, namely
          D = D∪D(gi
                       ,gˆj);
```

<sup>9</sup> Pick the graph pairs with top-k% least differences from D, label the picked graph pairs as D ′ ;

```
10 for each graph pair (gi
                            ,gˆj) in D
                                      ′ do
```

```
11 Do edge inference with Eq. 7;
12 Add the inferred graph g
                                rec
                                i
                                   to G
                                       rec, namely
        G
          rec = G
                 rec ∪g
                       rec
                       i
```

<sup>13</sup> Return *G rec*

#### Algorithm 2: Property inference attack

<span id="page-17-11"></span>Input: Φ from an online marketplace or an API, shadow graphs *G S* (optional), the property P to be inferred

Output: The inferred property of the training graphs P(Φtrain)

- <sup>1</sup> Execute Φ or feed *G S* to Φ to obtain the synthetic graphs *G*ˆ;
- <sup>2</sup> Calculate the property of Φtrain based on the graphs in *G*ˆ with formula [9;](#page-5-3)
- <sup>3</sup> Return P(Φtrain)
- <span id="page-17-7"></span>[81] Shengfang Zhai, Huanran Chen, Yinpeng Dong, Jiajun Li, Qingni Shen, Yansong Gao, Hang Su, and Yang Liu. Membership inference on text-to-image diffusion models via conditional likelihood discrepancy. *Advances in Neural Information Processing Systems*, 37:74122– 74146, 2025.
- <span id="page-17-0"></span>[82] Mengchun Zhang, Maryam Qamar, Taegoo Kang, Yuna Jung, Chenshuang Zhang, Sung-Ho Bae, and Chaoning Zhang. A survey on graph diffusion models: Generative ai in science for molecule, protein and material. *arXiv preprint arXiv:2304.01565*, 2023.

- <span id="page-17-6"></span>[83] Minxing Zhang, Ning Yu, Rui Wen, Michael Backes, and Yang Zhang. Generated distributions are all you need for membership inference attacks against generative models. In *Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision*, pages 4839–4849, 2024.
- <span id="page-17-1"></span>[84] Shijie Zhang, Hongzhi Yin, Tong Chen, Zi Huang, Lizhen Cui, and Xiangliang Zhang. Graph embedding for recommendation against attribute inference attacks. In *Proceedings of the Web Conference 2021*, pages 3002– 3014, 2021.
- <span id="page-17-2"></span>[85] Wanrong Zhang, Shruti Tople, and Olga Ohrimenko. Leakage of dataset properties in {Multi-Party} machine learning. In *Proceedings of the USENIX Security Symposium*, pages 2687–2704, 2021.
- <span id="page-17-4"></span>[86] Zaixi Zhang, Qi Liu, Zhenya Huang, Hao Wang, Chengqiang Lu, Chuanren Liu, and Enhong Chen. Graphmi: Extracting private graph data from graph neural networks. *Proceedings of the International Joint Conference on Artificial Intelligence*, 2021.
- <span id="page-17-3"></span>[87] Zhikun Zhang, Min Chen, Michael Backes, Yun Shen, and Yang Zhang. Inference attacks against graph neural networks. In *Proceedings of the USENIX Security Symposium*, pages 1–18, 2022.
- <span id="page-17-5"></span>[88] Zhanke Zhou, Chenyu Zhou, Xuan Li, Jiangchao Yao, Quanming Yao, and Bo Han. On strengthening and defending graph reconstruction attack with markov chain approximation. *arXiv preprint arXiv:2306.09104*, 2023.

#### Appendix

# <span id="page-17-8"></span>A Pseudo code of our attacks

Algorithm [1,](#page-17-10) Algorithm [2,](#page-17-11) and Algorithm [3](#page-18-1) show the pseudo code of our graph reconstruction, property inference, and membership inference attack, respectively.

# <span id="page-17-9"></span>B Details of Datasets

We use six real-world datasets for attack evaluation in our paper, namely MUTAG, ENZYMES, Ego-small, IMDB-BINARY, IMDB-MULTI, and QM9, which serve as benchmarks for evaluating graph-based models in various domains [\[46\]](#page-15-17). Table [6](#page-19-1) provides details of these datasets. We obtain Ego-small and QM9 from the implementation provided by [\[34\]](#page-14-14), while all other datasets are downloaded from the torch\_geometric package. For IMDB-BINARY and IMDB-MULTI datasets that do not have node features, we follow [\[78\]](#page-16-21) and use one-hot degree features as the node features.

```
Algorithm 3: Membership inference attack
   Input: Φ from an online marketplace or an API, shadow
          graphs G
                   S
                    , a target graph G
   Output: The membership label yG of G
1 Train a shadow model ΦS using a subset of graphs from G
                                                             S
                                                               ;
2 Attack feature matrix A
                           train= [];
3 for each GS
              i
               in GS
                     , i ∈ [1,...,N] do
4 Feed G
              S
              i
                to ΦS
                      to obtain a set of synthetic graphs GˆS
                                                           i
                                                            ;
5 Learn the graph embedding, emb(G
                                           S
                                           i
                                            ), of the shadow
        graph G
                 S
                 i
                  by using AWEs;
6 for each gˆ
                 S
                 i
                   in GˆS
                        i
                         do
 7 Learn the graph embedding, emb(gˆ
                                               S
                                               i
                                                ), of ˆg
                                                      s
                                                      i
                                                        by
             using AWEs;
8 if Attack with A
                       train-1 then
 9 for each embedding vector emb(gˆ
                                              S
                                              i
                                               ) do
10 Calculate the similarity between emb(gˆ
                                                       S
                                                       i
                                                        ) and
                 emb(G
                        S
                        i
                         ) as simk

                                    emb(gˆ
                                          S
                                          i
                                           ),emb(G
                                                    S
                                                    i
                                                     )

                                                       ;
11 if Attack with A
                       train-2 then
12 for each graph pair (gˆ
                                  S
                                  i
                                   ,gˆ
                                      S
                                      j
                                       ) within GˆS
                                                 i
                                                   do
13 Calculate the similarity between the embedding
                 vectors of emb(gˆ
                                  S
                                  i
                                   ) and emb(gˆ
                                                S
                                                j
                                                 ) as
                 simk

                       emb(gˆ
                              S
                              i
                               ),emb(gˆ
                                       S
                                        j
                                         )

                                           ;
14 Construct the attack feature, A
                                      train
                                      i
                                          , of the triplet
        (G
           S
           i
            ,gˆ
              s
              i
               ,y) with Eq. 11 if A
                                    train-1, otherwise, use Eq.
        12, where y is the known membership label of G
                                                         S
                                                         i
                                                          ;
15 Append (A
                  train
                  i
                      ,y) to A
                              train;
16 Train the attack classifier with A
                                    train;
17 Feed G to Φ to obtain a set of synthetic graphs Gˆ;
18 for each g in ˆ Gˆ do
19 Learn the graph embedding, emb(gˆ), of ˆg by using
        AWEs;
20 if Attack with A
                   train-1 then
21 for each emb(gˆ) do
22 Calculate the similarity between the embedding
             vectors of emb(gˆ) and emb(G) as
             simk (emb(gˆ),emb(G));
23 if Attack with A
                   train-2 then
24 for each graph pair (gˆi
                               ,gˆj) within Gˆ do
25 Calculate the similarity between the embedding
             vectors of emb(gˆi) and emb(gˆj) as
             simk

                   emb(gˆi),emb(gˆj)

                                     ;
26 Construct the attack feature, A
                                 att of G using Eq. 11 when
    adopting A
                train-1, otherwise, use Eq. 12;
27 Feed A
          att to the trained attack classifier to obtain the
    predicted membership yG;
28 Return yG
```

#### <span id="page-18-0"></span>C More results on property inference attack

PIA performance on the properties of average number of triangles per node and graph arboricity. Table [7](#page-19-2) presents the absolute differences in the average number of triangles per node and the average graph arboricity between the target model's training set, Φtrain, and the inferred values from the generated graphs. We observe that the absolute difference in the average number of triangles per node does not exceed 1.005 (less than 3.92% of its original value in Φtrain). Similarly, the absolute difference in average graph arboricity remains below 0.476 (less than 9.05% of its original value in Φtrain). These results demonstrate that our simple yet effective attack can accurately infer the graph properties of Φtrain .

#### Algorithm 4: Graph perturbation in the defense

<span id="page-18-2"></span>Input: The graphs to be perturbed, namely, the training graphs of GGDM in Defense-1 or the synthetic graphs in Defense-2, are denoted as G. The perturbation ratio *r*. Note that we perturb only the graph structures of the graphs in G, while other properties unrelated to the structure, such as node features, are kept unchanged. Therefore, we omit these properties in the following code.

```
Output: The perturbed graphs G′
```

```
1 for each graph g in G do
2 Calculate the perturbation budget b = ⌊r ×|E|⌋,
      where |E| is the number of edges in g;
3 P = {}, used to label the perturbed entries in the
       adjacency matrix A of g;
4 G′ = {};
5 while b>0 do
6 Perform a forward pass of g through GCN for
          10 epochs;
7 Calculate the node classification loss L and
          backpropagate the gradient to each entry in A
          using Eq. 14;
8 Construct the saliency map Saliency(A) with
          Eq. 15 ;
9 Rank the entries in A based on their
          importance values in Saliency(A) in
          ascending order, labeled as Aranked;
10 for Ai j in Aranked do
11 if Ai j ∈/ P then
12 Flip the Ai j in A;
13 P = P∪Ai j;
14 Continue;
15 b = b−1;
16 G′ = G′ ∪A;
17 Return G′
```

PIA performance on frequency ratio distribution across different property ranges. Figure [9](#page-19-3) shows the frequency distribution in different property ranges with a bucket number of *k* = 10. We observe that the inferred frequency distribution

<span id="page-19-1"></span>

| Dataset     | Domain           | # Graphs | Avg. # Nodes | Avg. # Edges | Avg. # Degrees | # Node features | # Classes |
|-------------|------------------|----------|--------------|--------------|----------------|-----------------|-----------|
| MUTAG       | Molecules        | 188      | 17.93        | 19.79        | 2.11           | 7               | 2         |
| Ego-small   | Citation network | 200      | 6.41         | 8.69         | 2.00           | 3,703           | -         |
| ENZYMES     | Proteins         | 600      | 32.60        | 124.3        | 7.63           | 3               | 6         |
| IMDB-BINARY | Social network   | 1,000    | 19.77        | 96.53        | 9.77           | 135             | 2         |
| IMDB-MULTI  | Social network   | 1,500    | 13.00        | 65.94        | 10.14          | 88              | 3         |
| QM9         | Molecules        | 133,855  | 18.03        | 37.30        | 4.14           | 11              | -         |

Table 6: Statistics of datasets

<span id="page-19-3"></span>![](_page_19_Figure_2.jpeg)

Figure 9: Performance of PIA - the portion of training graphs in different property ranges. Figures (a)–(c) report attack performance on graph density, Figures (d)–(f) on graph degree, Figures (g)–(i) on the average number of triangles per node, and Figures (j)–(l) on graph arboricity (*k* = 10, IMDB-M dataset).

<span id="page-19-2"></span>

| Dataset   | EDP-GNN      |                | GDSS         |                | Digress      |                | Orig.      |            |
|-----------|--------------|----------------|--------------|----------------|--------------|----------------|------------|------------|
|           | D(triangle)↓ | D(arborivity)↓ | D(triangle)↓ | D(arborivity)↓ | D(triangle)↓ | D(arborivity)↓ | #Triangles | Arborivity |
| MUTAG     | 0.034        | 0.0001         | 0.292        | 0.522          | 0.398        | 0.044          | 0          | 1          |
| ENZYMES   | 0.143        | 0.041          | 0.212        | 0.158          | 0.135        | 0.245          | 1.035      | 1.607      |
| Ego-small | 0.203        | 0.037          | 0.257        | 0.174          | 0.286        | 0.226          | 4.605      | 1.175      |
| IMDB-B    | 0.914        | 0.103          | 0.828        | 0.351          | 0.881        | 0.405          | 20.167     | 4.611      |
| IMDB-M    | 0.448        | 0.048          | 0.917        | 0.351          | 1.005        | 0.467          | 25.627     | 5.161      |
| QM9       | 0.374        | 0.055          | 0.413        | 0.219          | 0.580        | 0.324          | 11.185     | 2.225      |

Table 7: Performance of property inference attack, namely the absolute differences in property values between the target model's training set and the inferred values from the generated graphs. A lower difference represents a better performance. "*D*(*triangle*)" and "*D*(*arborivity*)" represent the absolute difference of average number of triangles per node and graph arborivity, respectively. "Orig." means the property of target model's training set Φtrain .

closely aligns with the original distribution of Φtrainacross all settings, with absolute ratio differences for each range class varying from 0.0003 to 0.0235 for average graph density, 0 to 0.0401 for average graph degree, 0 to 0.0589 for average number of triangles per each node, and 0.0003 to 0.0202 for average graph arboricity.

# <span id="page-19-0"></span>D Pseudo code of our defense mechanism

Algorithm [4](#page-18-2) shows the pseudo code for the graph perturbation part of our two defense mechanisms.