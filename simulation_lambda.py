from distutils.log import debug
import math
import numpy as np
from scipy import misc
import argparse
# from sqlalchemy import all_, false

"""Setup"""

# Number of coins
num_coins = 100
# Length of distribution
n = 10000
# Number of draws for c_0
m = 10000
# Portion of the (1-alpha) network that we know
beta = 0.5
# reward for revealing one coin
x = 1

# if lambda = alpha then expected reward of honest should be 0
# b/c the honest strategy gets alpha per round
# however, in new simulations we are also subtracting alpha per round 
# for the honest strategy in this new format (since we subtract (lambda + alpha))
# we want to subtract a total of alpha per round. 
# so lambda = 0 is the same as subtracting alpha per round

def g(rewards, coins, alpha, c_0):
  # find i s.t. c_i < c_0 < c_{i+1}
  i = 1
  for j in range(len(coins)):
    if coins[j] < c_0:
      i = j
  
  # find g value 
  largest =  float('-inf')
  for j in range(len(coins)-1):
    # not j - 1 because j starts at 0
    current = -1 * x * j + (1+rewards[j]) * math.exp(-1 * (1-beta) * (1-alpha) * coins[j])
    largest = max(current, largest)
  return largest

def find_k(draws, r_h):
  # print(draws[-100:])
  # print(r_h)
  for i in range(n):
    if draws[i] > r_h:
      return i
  return n

def h(c_0, r_0, i, alpha):
  return -1 * x * i + r_0 * math.exp(-1 * (1-alpha) * (1-beta) * c_0)

def sum_greater_r_h(k, i, D, c_0, alpha):
  total = 0
  for j in range(k, n):
    total += h(c_0, D[j], i, alpha)
  return total

"""Update rule for OPTPLUS strategy"""
def optplus_sim(D, alpha, l):
  D.sort()
  c = [0]*num_coins # coin values
  r = [0]*num_coins # (expected) reward values
  F = [0]*n # final distribution

  # make n draws from our distribution
  for j in range(n):
    # print(">>>>>>>>j=", j)
    # step 1: draw c1 from exp(alpha)
    c[0] = np.random.exponential(scale=(1/alpha))

    # step 2: calculate ci for all i > 1
    for k in range(1, num_coins):
      c[k] = c[k-1] + np.random.exponential(scale=(1/alpha))
    
    # step 3: for all i >= 1, draw r_i from D iid
    for k in range(num_coins):
      r[k] = D[np.random.randint(low=0, high=n)]
    
    # step 4: estimate over m draws for c_0
    output_sum = 0
    for i in range(m):
      # print(">>>>>>>>>>i=", i)
      # draw our c_0 
      c_0 = np.random.exponential(scale=(1/(beta * (1-alpha))))

      # calculate g(c_0, c_-0, r_-0)
      g_val = g(r, c, alpha, c_0) 

      # calculate r_h value
      r_h = (g_val + x * i) * math.exp((1-alpha)*(1-beta)*c_0)

      # count number of draws that were < r_h
      lower_count = sum(reward < r_h for reward in r)

      # find our k value
      k = find_k(D, r_h)
      if k == n:
        # there is no r_i that is > r_h
        inner_int = 0
      else:
        inner_int = sum_greater_r_h(k, i, D, c_0, alpha)

      # add everything together for this iteration
      output_sum += g_val * lower_count + (1/(n-k+1) * inner_int)

    draw = output_sum / (m * n) - alpha - l
    F[j] = draw

  return F

# Simulates expected rewards for given alpha and lambda.
def simulate(alpha, l, strategy_sim, difference, times, debug_flag):
  D_honest = np.zeros(n)
  
  # run first iteration
  F0 = strategy_sim(D_honest, alpha, l)

  if debug_flag:
    print("FIRST ROUND EXPECTED REWARDS: ", np.average(F0))

  # run sim until expectation doesn't change by _difference_ for _times_ times in a row
  bestF = F0
  lastavg = np.average(F0)
  currentavg = 0
  runs = 1
  runs_in_range = 0
  while abs(lastavg - currentavg) > difference or runs_in_range < times:
    nowF = strategy_sim(bestF, alpha, l)
    lastavg = np.average(bestF)
    currentavg = np.average(nowF)
    bestF = nowF
    runs += 1

    if abs(lastavg - currentavg) < difference:
      runs_in_range += 1
    elif abs(lastavg - currentavg) > difference and runs_in_range > 0:
      runs_in_range = 0
    
    if debug_flag:
      print("RUN #: ", runs)
      print("CURR AVG: ", currentavg)
      print("RUN IN RANGE #: ", runs_in_range)

  return bestF

# Run simulation for alpha = 0.1 to 1.0 in increments of 0.01.
def gather_data(strategy_sim, difference, times, output_file_name, debug_flag):
  distributions = np.zeros((100, n+1))

  for num in range(1, 100): # multiples of 1/100
    alpha = num / 100
    if debug_flag: print("ALPHA: ", alpha)

    # sanity check: reward should be 0 for lambda = alpha
    # l = alpha
    # bestF = simulate(alpha, l, strategy_sim, difference, times, debug_flag=True)
    # if abs(np.average(bestF)) > 0.01: return # Sanity check failed
    # if debug_flag: print("Passed sanity check")

    # binary search for lambda
    l = 0  # same as mid
    start = 0
    end = 1

    while (start <= end):
      l = (start + end)/2
      if debug_flag: print("Lambda: ", l)

      bestF = simulate(alpha, l, strategy_sim, difference, times, debug_flag)
      reward = np.average(bestF)

      if debug_flag: print("Reward for ", l, ": ", reward)
      
      # arbitrary error diff
      if abs(reward) < 0.01:
        break

      if reward > 0:
        start = l
      else:
        end = l
      
    if debug_flag: print("LAST lambda was ", l)

    # Save the best distribution and corresponding lambda
    distributions[num-1][0] = l
    distributions[num-1][1:] = bestF
    np.save(output_file_name, distributions)

# Run simulation starting at alpha = start_alpha to 1.0 in increments of 0.01.
def gather_data_from_given_start(distributions, strategy_sim, start_alpha, difference, times, output_file_name, debug_flag):  
  for num in range(int(start_alpha*100), 100): # multiples of 1/100
    alpha = num / 100
    
    if debug_flag: print("ALPHA: ", alpha)
    
    # binary search for lambda
    l = 0  # same as mid
    start = 0
    end = 1

    while (start <= end):
      l = (start + end)/2
      if debug_flag: print("Lambda: ", l)

      bestF = simulate(alpha, l, strategy_sim, difference, times, debug_flag)
      reward = np.average(bestF)

      if debug_flag: print("Reward for ", l, ": ", reward)
      
      # arbitrary error diff
      if abs(reward) < 0.01:
        break

      if reward > 0:
        start = l
      else:
        end = l
      
    if debug_flag: print("LAST lambda was ", l)

    # Save the best distribution and corresponding lambda
    distributions[num-1][0] = l
    distributions[num-1][1:] = bestF
    np.save(output_file_name, distributions)
    

def run_once(alpha, l, difference, times, debug_flag):
  print("RUNNING ONCE")
  if debug_flag: print("ALPHA: ", alpha)
  if debug_flag: print("Lambda: ", l)

  r = simulate(alpha, l, optplus_sim, difference, times, debug_flag)

  if debug_flag: print("Reward for ", l, ": ", r)

def main():
  parser = argparse.ArgumentParser()

  parser.add_argument('--run_once', action='store_true')
  parser.add_argument('--alpha', type=float, required=False)
  parser.add_argument('--l', type=float, required=False)
  args = parser.parse_args()

  if args.run_once:
    run_once(args.alpha, args.l, difference=0.01, times=4, debug_flag=True)
  else:
  # gather_data(optplus_sim, difference=0.01, times=4, output_file_name="OPTPLUS distributions 0.01 to 1.0.npy", debug_flag=True)
    distributions = np.load("OPTPLUS distributions 0.01 to 1.0.npy")
    gather_data_from_given_start(distributions, optplus_sim, start_alpha=0.33, 
    difference=0.01, times=4, output_file_name="OPTPLUS distributions 0.01 to 1.0.npy", debug_flag=True)


if __name__ == "__main__":
    main()
