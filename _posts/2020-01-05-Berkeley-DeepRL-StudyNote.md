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
$\min_{\theta} E_{\tau~\pi_{\theta}(\tau)} \Delta_{\theta} \log \pi_{\theta}(\tau)r(\tau)$

Find set of parameters $\theta$ of the policy, such that the expected reward on the given policy(trajectory) is maximized. 
