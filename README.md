# cirl (constrained inverse reinforcement learning)
[**Training**](#training)
| [**Contributing**](#contributing)


This is code implements the experiments of the paper [Identifiability and Generalizability in Constrained Inverse Reinforcement Learning](https://arxiv.org/pdf/2306.00629.pdf).

## Training
- To generate synthetic expert data from CMDP run `examples/get_expert.py`
- To run constrained IRL run `examples/cirl.py`

## Contributing
If you would like to contribute to the project please reach out to [Andreas Schlaginhaufen](mailto:andreas.schlaginhaufen@epfl.ch?subject=[cirl]%20Contribution%20to%20cirl). If you found this library useful in your research, please consider citing the following paper:
```
@InProceedings{pmlr-v202-schlaginhaufen23a,
  title = 	 {Identifiability and Generalizability in Constrained Inverse Reinforcement Learning},
  author =       {Schlaginhaufen, Andreas and Kamgarpour, Maryam},
  booktitle = 	 {Proceedings of the 40th International Conference on Machine Learning},
  pages = 	 {30224--30251},
  year = 	 {2023},
  editor = 	 {Krause, Andreas and Brunskill, Emma and Cho, Kyunghyun and Engelhardt, Barbara and Sabato, Sivan and Scarlett, Jonathan},
  volume = 	 {202},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {23--29 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v202/schlaginhaufen23a/schlaginhaufen23a.pdf},
  url = 	 {https://proceedings.mlr.press/v202/schlaginhaufen23a.html},
  abstract = 	 {Two main challenges in Reinforcement Learning (RL) are designing appropriate reward functions and ensuring the safety of the learned policy. To address these challenges, we present a theoretical framework for Inverse Reinforcement Learning (IRL) in constrained Markov decision processes. From a convex-analytic perspective, we extend prior results on reward identifiability and generalizability to both the constrained setting and a more general class of regularizations. In particular, we show that identifiability up to potential shaping (Cao et al., 2021) is a consequence of entropy regularization and may generally no longer hold for other regularizations or in the presence of safety constraints. We also show that to ensure generalizability to new transition laws and constraints, the true reward must be identified up to a constant. Additionally, we derive a finite sample guarantee for the suboptimality of the learned rewards, and validate our results in a gridworld environment.}
}
```