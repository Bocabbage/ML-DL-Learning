#!usr/env/bin python
# -*- coding: utf-8 -*- #
'''
    Project: M-H Sampling Method(MCMC)
    Python Version: Python3.7.2
    Cited: https://zhuanlan.zhihu.com/p/37121528
    Update date: 2019/05/21

'''
import random
import matplotlib.pyplot as plt
from scipy.stats import norm

def norm_dist_prob(theta):
    y = norm.pdf(theta, loc=3, scale=2)
    return y

T = 5000
sigma = 1

# Sampling from arbitrary distribution #
pi = [0 for i in range(T)]
########################################

t = 0   # stage
while t < T-1:
    t = t + 1
    # Sampling: Q(j|i) is density function of N(i,1) #
    pi_star = norm.rvs(loc=pi[t - 1], scale=sigma, size=1, random_state=None)  # 状态转移进行随机抽样
    ##################################################
    alpha = min(1, (norm_dist_prob(pi_star[0]) / norm_dist_prob(pi[t - 1])))   # alpha值

    u = random.uniform(0, 1)
    if u < alpha:
        pi[t] = pi_star[0]
    else:
        pi[t] = pi[t - 1]


plt.scatter(pi, norm.pdf(pi, loc=3, scale=2),label='Target Distribution')
num_bins = 50
plt.hist(pi, num_bins, normed=1, facecolor='red',label='Samples Distribution', alpha=0.7)
plt.legend()
plt.show()