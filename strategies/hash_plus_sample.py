from strategies.constants import m, n, num_coins, x
import numpy as np
import bisect
import math

# TODO need to code in the honest strategy here (ie. where beta = 0)

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
  return  -1 * x * i_min + r_0 * math.exp(-1 * (1-alpha) * (1-beta) * c_0)

def cats(alpha, beta, c, r, D, c0):
  if beta == 0:
    c0 = 500*(1/(1-alpha)) # very large number so it can't win

  # let pos such that c_pos < c0 < c_pos+1
  pos_c = bisect.bisect_left(c, c0) if beta != 0 else num_coins

  # compute g = adversary's best reward
  g = 0
  if pos_c != 0:
    g = np.max([np.exp(c[s]*(alpha-1)*(1-beta))*(1+r[s]) - x*s for s in range(pos_c)])

  # approximate P[r0 < r^h threshold]
  r_thresh = (g + x*pos_c) * np.exp((1-alpha)*(1-beta)*c0)
  k = np.count_nonzero(D < r_thresh)

  # approximate integral over PDF of r
  r_pdf = np.sum(np.where(D >= r_thresh, D*np.exp((alpha-1.0)*(1.0-beta)*c0) - x*pos_c, 0.0))
  
  return (k, pos_c, g,r_pdf, g*k + r_pdf)

def sim(D, alpha, beta, l):
  D.sort()
  c = [0]*num_coins # coin values
  r = [0]*num_coins # (expected) reward values
  F = np.zeros(n) # final distribution

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
        h_sums = 0
        for j in range(k, len(D)):
          h_sums += h(c_0, D[j], i_min, alpha, beta)

        my_val = g_val * k + h_sums
        output_sum += g_val * k + h_sums 
        
      # add catherine's 
      cats_k, cats_i_min, cats_g, r_pdf, cats_val = cats(alpha ,beta, c, r, D, c_0)
      if abs(h_sums - r_pdf) > 0.0001:
        print("HSUMS ERROR")
        print(h_sums, r_pdf)

      if cats_g - k*g_val > 0.0001:
        print("GVAL ERROR")
        print(g_val, cats_g)

      if abs(my_val - cats_val) > 0.0001:
        print("VAL ERROR")
        print(k, cats_k, i_min, cats_i_min)
        print(h_sums, k*g_val, my_val)
        print(r_pdf, cats_g, cats_val)
        print(my_val, cats_val)
      cats_sum += cats_val

    if abs(output_sum- cats_sum) > 0.0001:
      print("ERROR")
      print("\toutput_sum", output_sum)
      print("\tcats_sum", cats_sum)
    draw = output_sum / (m * n) - alpha - l
    F[i] = draw

  return F
