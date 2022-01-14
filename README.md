## CBRE

---

This is the official implementation of CBRE in the paper [“Cycle-Balanced Representation Learning For Counterfactual Inference”](https://arxiv.org/pdf/2110.15484.pdf).

The code is built on the [Counterfactual regression (CFR) ](https://github.com/clinicalml/cfrnet) and [Adversarial Balancing based representation learning for Causal Effect Inference (ABCEI) ](https://github.com/octeufer/Adversarial-Balancing-based-representation-learning-for-Causal-Effect-Inference). The random parameter searching, network training and evaluation follow the procedures of CFR to ensure a fair comparison.

---



#### Installation

We use Python 3.5.6 and Tensorflow==1.4.0. You can install libraries with requirements.txt.

```bash
pip install -r requirements.txt
```

---



#### Usage

**Data:** there are three datasets in this repository, and you can also download them from [The website of Dr. Fredrik D. Johansson](https://www.fredjo.com/).

**Configs:** there are three config.txt for datasets.

The implementation of the CBRE network is included in **cbre/cbre_net.py**.

The training code is cbre_train.py, and you can test performance with different parameters by cbre_param_search.py. 

The overall procedure is the same as cfrnet and abcei. 

==You can replace the class CBRENet in cbre/cbre_net.py with your model.==



About how to use param_search and evaluate, you can refer to [CFRnet](https://github.com/clinicalml/cfrnet). We use evaluate.py to test performance on the IHDP and Jobs datasets.  And we assess the Twins dataset with evaluate_twins.py for our model and baselines.

---



#### Cite

Please cite out paper if it’s helpful for you.

> ```latex
> @article{zhou2021cycle,
>   title={Cycle-Balanced Representation Learning For Counterfactual Inference},
>   author={Zhou, Guanglin and Yao, Lina and Xu, Xiwei and Wang, Chen and Zhu, Liming},
>   journal={arXiv preprint arXiv:2110.15484},
>   year={2021}
> }
> ```

