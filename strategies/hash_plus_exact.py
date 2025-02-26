from strategies.constants import m, n, num_coins, x
import numpy as np
import bisect
import math

# HASH_PLUS_SAMPLE
def g(rewards, coins, alpha, beta, c_pos):
  # find g value 
  largest =  float('-inf')
  for j in range(c_pos):
    # not j - 1 because j starts at 0
    current = -1 * x * j + (1+rewards[j]) * math.exp(-1 * (1-beta) * (1-alpha) * coins[j])
    largest = max(current, largest)
  return largest

def h(c_0, r_0, i_min, alpha, beta):
  return -1 * x * i_min + r_0 * math.exp(-1 * (1-alpha) * (1-beta) * c_0)

def sim(D, alpha, beta, l):
  D.sort()
  c = [0]*num_coins # coin values
  r = [0]*num_coins # (expected) reward values
  F = [0]*n # final distribution
  
  # make n draws from our distribution
  for i in range(n):
    draw = 0
    # step 1: draw c1 from exp(alpha)
    c[0] = np.random.exponential(scale=(1/alpha))

    # step 2: calculate ci for all i > 1
    for k in range(1, num_coins):
      c[k] = c[k-1] + np.random.exponential(scale=(1/alpha))
    
    # step 3: for all i >= 1, draw r_i from D iid
    for k in range(num_coins):
      r[k] = D[np.random.randint(low=0, high=n)]

    #TODO do some handling for beta = 0
    if beta > 0:
      # draw a c_0 and r_0
      c_0 = np.random.exponential(scale=(1/(beta * (1-alpha))))
      r_0 = D[np.random.randint(low=0, high=n)]
    
      # find an i_min s.t. c_{i_min} < c_0 < _{i_min + 1}
      i_min = bisect.bisect_left(c, c_0) if beta != 0 else num_coins

      # calculate g(c_0, c_-0, r_-0)
      # if i_min = 0 that means that there is no i s.t. c_i < c_0. 
      # Therefore the adversary has no coins that would win over Beta portion of the network
      # so we should definitely allow the network's coin to win (rather, we HAVE to) --> g= 0
      g_val = 0
      if i_min > 0:
        g_val = g(r, c, alpha, beta, i_min) 

      h_val = h(c_0, r_0, i_min, alpha, beta)
      # print(g_val,h_val)
      draw = max(h_val, g_val) - l - alpha
      F[i] = draw
    else:
      F[i] = g(r,c,alpha, beta, num_coins) - l - alpha

  return F
