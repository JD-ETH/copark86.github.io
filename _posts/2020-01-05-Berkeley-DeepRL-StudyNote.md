---
title: 'Study notes for Berkeley DeepRL class 2019 Fall'
date: 2020-01-05
permalink: /post/2020-01-05-Berkeley-DeepRL
tags:
  - Deep Reinforcement Learning
---

This is my ongoing study notes on [Berkelely DeepRL class](http://rail.eecs.berkeley.edu/deeprlcourse/), Focus on the homework exercises. 
My ongoing solution is [here](https://github.com/JD-ETH/DeepRL-Berkely-Solution).

### Homework 1, Imitation Learning and Dagger. 

By viewing reinforcement learning as supervised learning given expert data, one can create behavior cloning agent that solves the task. 

To deal with the trajectory distribution discrepancy between training and testing, expert actions are iteratively called to agent's runtime trajectory, and used to further improve the behavior. This is called DAgger. 

### Homework 2, Policy Gradient. 

Essentially, the idea is the following: 

$\min_{\theta} E_{\pi_{\theta}(\tau)} \nabla_{\theta} \log \pi_{\theta}(\tau)r(\tau)$

Find set of parameters $\theta$ of the policy, such that the expected reward on the given policy(trajectory) is maximized. 

Practically, the following adaptations need to be made for tractable computation and reduction of variance for the expected reward. 

$\nabla_{\theta}J_{\theta} \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=0}^{T-1} \nabla_{\theta}log \pi_{\theta}(a_{i,t}\mid s_{i,t})(\sum_{t'=t}^{T-1}\gamma^{t'-t}r(s_{i,t'}, a_{i, t'}) - V_{\phi}^{\pi}(s_{i,t}))$

With the following adaptations:
- Approximate expectation by samples.
- Apply causality and only sum up reward in future timesteps.
- Incorporate value function as a baseline for the reduction of variance. The baseline is a seperately estimated by a different network. Optimize over advantage instead of reward directly. 

This homework allows you to conduct an ablation study for different tricks in the training, with the Gym Cartpole-v0 example. We make the following observations:
- Vanilla algorithm does not converge well even if batch size is large enough. 
- Clipping away previous reward (reward-to-go) helps network converge with less data.
- Standardizing advantage function improves variance but introduces bias. Practically, this improves stability and prevent the gradient from blowing up. 
- Adding a value function estimator improves convergence for difficult tasks. 

What's more, the homeworks asks for an investigation of batch size and learning rate:
- learning rate too low leads to slow convergence, and too high causes divergence.
- large batch size causes degrading performance, small batch size can cause divergence. 

Some techniques to speed up the training process:
- Run multiple gradient descent from multiple actor in one policy update step.
- Parallelize the policy collection process. 
- Generalized Advantage estimation can be used to further reduce variance and improve training. 

### Homework 3
#### Deep Q Learning 

The homework is about solving atari games with Q-learning. 

The Q function is an estimation of return given state and action under current policy. It's value can be iteratively estimated by temporal difference.   

$Q^{\pi}(s_t,a_t) = r_t + \gamma \max_{a_{t+1}} Q^{\pi}(s_{t+1}, a_{t+1})$

In a deep learning setup, a neural network is used as a function approximater. 
The policy is deterministic, choosing the action that maximum estimated return in current state. 
During learning however, scheduled epsilon greedy is used instead for exploration. 

By sampling randomly from a replay buffer, temporal relationships between previous experiences are broke up. This also 
allows learning off-policy: having independent actors to sample experience, and learner to update network weights. 
To reduce variance and improve learning stability, a target network is used to temporarily be frozen. 

Bellman error = $ r_t + \gamma \max_{a_{t+1}} Q^{\pi'}(s_{t+1},a_{t+1}) - Q^{\pi}(s_{t}, a_t) $

Even better, use double-Q learning to reduce overestimation of reward function resulting from $\max$ operation. 

Bellman error = $ r_t + \gamma Q^{\pi'}(s_{t+1}, \max_{a_{t+1}}Q^{\pi}(s_{t+1},a_{t+1})) - Q^{\pi}(s_{t}, a_t) $
 
#### Actor Critic Learning
Expanding on the policy gradient, previously one could use a function approximater to estimate the advantage, and thus reduce variance at gradient update. In the actor critic setup, instead of estimating advantage directly, the Value function is estimated, termed critic. 

$ A^{\pi}(s_t, a_t) = \left( \sum_{t'=t}{T}\gamma^{t'-t}r(s_{t'}, a_{t'}) \right) - V^{\pi}_ {\phi}(s_{t}) $

or equivalently, 

$ A^{\pi}(s_t, a_t) = r_t(s_t,a_t) + \gamma V^{\pi}_ {\phi}(s_{t+1}) - V^{\pi}_ {\phi}(s_{t}) $

The policy(actor) update is now simply: 
$\nabla_{\theta}J_{\theta} \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=0}^{T-1} \nabla_{\theta}log \pi_{\theta}(a_{i,t}\mid s_{i,t})(\sum_{t'=t}^{T-1}A^{\pi}(s_t, a_t))$

To update the critic given a policy, a target value is generated from the reward:
$y_t = r(s_t, a_t) + \gamma V^{\pi}(s_{t+1})$ 
Here we encounter the same moving target problem, and needs to be solved by updating target slower than the value function estimator. 
```
for i in update_target:
       for j in update_value_func:
              update_v_func_approximator()
       update_target_value()
```
This improves convergence of the overall network quite quickly. 

Another common technique is to estimate the Advantage function by adding multi-step return instead of only the current return only. 

### Homework 4
#### Model-based RL 
Model-based reinforcement learning is implemented in this exercise, where the discrete error dynamics model is approximated by a neural network: 
$\hat{s}_ {t+1} = s_t + \hat{f}_ {\theta}(s_t, a_t)$

The determinsitic model is learned under a supervised learning setup on following objective:
$\sum_{s_t,a_t,s_{t+1}} || (s_{t+1}-s_t - f_{\theta}(s_t,a_t)||_2^2$

Final action sequences of length h (horizon) is chosen by minimizing the cost function (maximizing reward):
$\underset{A}{argmin} \sum_{t_0}^{t_0+h-1}c(\hat{s}_t, a_t)$

under the approximated dynamics. 

To further robustify the model against model uncertainties, ensemble methods can be deployed which averages model predictions from multiple networks. 
