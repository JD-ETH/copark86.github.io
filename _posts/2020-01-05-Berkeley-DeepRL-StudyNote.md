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

### Homework 3, Deep Q Learning 

The homework is about solving atari games with Q-learning. 

The Q function is an estimation of return given state and action under current policy. It's value can be iteratively estimated by temporal difference.   

$Q^{\pi}(s_t,a_t) = r_t + \gamma \max_{a_{t+1}} Q^{pi}(s_{t+1}, a_{t+1})$

In a deep learning setup, a neural network is used as a function approximater. 
The policy is deterministic, choosing the action that maximum estimated return in current state. 
During learning however, scheduled epsilon greedy is used instead for exploration. 

By sampling randomly from a replay buffer, temporal relationships between previous experiences are broke up. This also 
allows learning off-policy: having independent actors to sample experience, and learner to update network weights. 
To reduce variance and improve learning stability, a target network is used to temporarily be frozen. 

Bellman error = $ r_t + \gamma \max_{a_{t+1}} Q^{\ast{\pi}, a_{t+1}}(s_{t+1}) - Q^{\pi}(s_{t}, a_t) $

Even better, use double-Q learning to reduce overestimation of reward function resulting from $\max$ operation. 

Bellman error = $ r_t + \gamma Q^{\ast{\pi}}(s_{t+1}, \max_{a_{t+1}}Q^{\pi}(s_{t+1},a_{t+1})) - Q^{\pi}(s_{t}, a_t) $
 
