# deepHGLM

HGLM extends GLMM by allowing random effects to have an arbitrary exponential family distribution. In contrast to the Bayesian analysis, which assumes a known prior to the random parameters, HGLM has the fixed parameters in both of response distribution and random effect distribution. It estimates both fixed and random parameters by joint maximization of the hierarchical likelihood through a speicifc scale of random effects, which provides MLEs of marginal likelihood for fixed parameters and posterior mode for random parameters.

In this work, we propose an elegant way for adopting random effects to the deep neural network via hierarchical likelihood. The proposed method contains a two-step batch sampling for efficient training for the random parameters and additional constraints to prevent local minima problem, which causes severe bias in the subject-specific inference, in contrast to an ordinary neural network.
