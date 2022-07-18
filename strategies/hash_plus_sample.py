from strategies.constants import m, n, num_coins, x
import numpy as np
import bisect
import math
from multiprocessing import Pool
from functools import partial
import time

# TODO need to code in the honest strategy here (ie. where beta = 0)
memoized_c0 = None

# HASH_PLUS_SAMPLE
def g(rewards, coins, alpha, beta, c_pos):
  # marginal reward for revealing the coins
  # ie., -1 * x * j (not (j-1) because j starts at 0)
  reveal_rewards = np.arange(c_pos) * -1 * x
  g_vals = reveal_rewards + (1 + rewards[:c_pos]) * np.exp(coins[:c_pos] * -1 * (1-beta) * (1-alpha))
  return np.max(g_vals)

def h(c_0, r_0, i_min, alpha, beta):
  return  -1 * x * i_min + r_0 * math.exp(-1 * (1-alpha) * (1-beta) * c_0)


def draw_wrapper(D, alpha, beta, l,i):
  global memoized_c0
  c = np.zeros(num_coins)# coin values
  r = np.zeros(num_coins)# (expected) reward values

  # step 1: draw c1 from exp(alpha)
  c[0] = np.random.exponential(scale=(1/alpha))

  # step 2: calculate ci for all i > 1
  for k in range(1, num_coins):
    c[k] = c[k-1] + np.random.exponential(scale=(1/alpha))
  
  # step 3: for all i >= 1, draw r_i from D iid
  for k in range(num_coins):
    r[k] = D[np.random.randint(low=0, high=n)]

  output_sum = 0
  h_sums = 0
  i_min = 0

  # step 4: estimate over m draws for c_0
  for w in range(m):
    c_0 = 0 
    g_val = 0
    
    if beta == 0:
      g_val = g(r,c,alpha, beta, num_coins)
      k = len(D)
      my_val = g_val * k
      output_sum += max(g_val * k/(m * n), 0)
      # TODO this has to be fixed for beta
    else:
      c_0 = np.random.exponential(scale=(1/(beta * (1-alpha))))
      # c_0 =memoized_c0[w]

      # find an i_min s.t. c_{i_min} < c_0 < _{i_min + 1}
      i_min = bisect.bisect_left(c, c_0) if beta != 0 else num_coins

      # calculate g(c_0, c_-0, r_-0)
      # if i_min = 0 that means that there is no i s.t. c_i < c_0. 
      # Therefore the adversary has no coins that would win over Beta portion of the network
      # so we should definitely allow the network's coin to win (rather, we HAVE to) --> g= 0
      g_val = 0
      if i_min > 0:
        g_val = g(r, c, alpha, beta, i_min) 

      # calculate r_h value
      r_h = (g_val + x * i_min) * math.exp((1-alpha)*(1-beta)*c_0)

      # find our k value
      k = np.count_nonzero(np.array(D) < r_h)

      # get inner integral approximation
      # h_sums = 0
      # for j in range(k, len(D)):
      #   h_sums += h(c_0, D[j], i_min, alpha, beta)
      h_sums = np.sum(np.where(D >= r_h, -1 * x * i_min + D * np.exp(-1 * (1-alpha) * (1-beta) * c_0), 0.0))

      output_sum += max(0, (g_val * k + h_sums) / (m*n))
      
  draw = output_sum - alpha - l
  return draw


def sim(D, alpha, beta, l, num_cores):
  global memoized_c0
  c = np.zeros(num_coins)# coin values
  r = np.zeros(num_coins)# (expected) reward values
  F = np.zeros(n)  # final distribution
  # start_time = time.time()
  # if memoized_c0 is None:
  #   memoized_c0 = np.zeros(m)
  #   for i in range(m):
  #     memoized_c0[i] = np.random.exponential(scale=(1/(beta * (1-alpha))))

  draw_func = partial(draw_wrapper, D, alpha, beta, l)
  with Pool(num_cores) as p:
    new_F = p.map(draw_func, F)
  # end_time = time.time()
  # print(end_time-start_time)
  # print(new_F)
  return np.array(new_F)
  
  """
  # make n draws from our distribution
  for i in range(n):
    # step 1: draw c1 from exp(alpha)
    c[0] = np.random.exponential(scale=(1/alpha))

    # step 2: calculate ci for all i > 1
    for k in range(1, num_coins):
      c[k] = c[k-1] + np.random.exponential(scale=(1/alpha))
    
    # step 3: for all i >= 1, draw r_i from D iid
    for k in range(num_coins):
      r[k] = D[np.random.randint(low=0, high=n)]

    output_sum = 0
    cats_sum = 0
    h_sums = 0
    i_min = 0
    # step 4: estimate over m draws for c_0
    for _ in range(m):
      c_0 = 0 
      g_val = 0
      
      if beta == 0:
        g_val = g(r,c,alpha, beta, num_coins)
        k = len(D)
        my_val = g_val * k
        output_sum += g_val * k
      else:
        c_0 = np.random.exponential(scale=(1/(beta * (1-alpha))))

        # find an i_min s.t. c_{i_min} < c_0 < _{i_min + 1}
        i_min = bisect.bisect_left(c, c_0) if beta != 0 else num_coins

        # calculate g(c_0, c_-0, r_-0)
        # if i_min = 0 that means that there is no i s.t. c_i < c_0. 
        # Therefore the adversary has no coins that would win over Beta portion of the network
        # so we should definitely allow the network's coin to win (rather, we HAVE to) --> g= 0
        g_val = 0
        if i_min > 0:
          g_val = g(r, c, alpha, beta, i_min) 

        # calculate r_h value
        r_h = (g_val + x * i_min) * math.exp((1-alpha)*(1-beta)*c_0)

        # find our k value
        k = np.count_nonzero(np.array(D) < r_h)

        # get inner integral approximation
        # h_sums = 0
        # for j in range(k, len(D)):
        #   h_sums += h(c_0, D[j], i_min, alpha, beta)
        h_sums = np.sum(np.where(D >= r_h, -1 * x * i_min + D * np.exp(-1 * (1-alpha) * (1-beta) * c_0), 0.0))

        my_val = g_val * k + h_sums
        output_sum += g_val * k + h_sums 
        
    draw = output_sum / (m * n) - alpha - l
    F[i] = draw
  """
  return F
