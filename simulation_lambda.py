import math
import numpy as np
from scipy import misc
from sqlalchemy import all_, false

"""Setup"""

# Number of coins
i = 100
# Length of distribution
x = 10000

"""Update rule for OPTPLUS strategy"""
def optplus_sim(D, alpha, l):
  c = [0]*i # coin values
  r = [0]*i # (expected) reward values
  F = [0]*x # final distribution

  for j in range(x):
    # step 1: draw c1 from exp(alpha)
    c[0] = np.random.exponential(scale=(1/alpha))

    # step 2: calculate ci for all i > 1
    for k in range(1, i):
      c[k] = c[k-1] + np.random.exponential(scale=(1/alpha))
    
    # step 3: for all i >= 1, draw r_i from D iid
    for k in range(i):
      r[k] = D[np.random.randint(low=0, high=x)]
    
    # step 4: draw reward for taking network's coin as seed
    r0 = D[np.random.randint(low=0, high=x)]

    # step 5: output reward
    output_sum = 0
    best_so_far = r[0]
    all_winners = True
    for k in range(i-1):
      prob = math.exp(-1*(1-alpha)*c[k]) - math.exp(-1*(1-alpha)*c[k+1])
      max_reward = max(best_so_far, r[k])
      opt_reward = max(max_reward+1-l, r0-l)
      if opt_reward == (r0-l):
        all_winners = False
        # output_sum -= l
        # print("Losing ", opt_reward, " vs Winning ", max_reward+1-l)
      output_sum += prob*opt_reward

      best_so_far = max_reward

    output_sum = output_sum-l if all_winners else output_sum
    # if not all_winners:
    #   output_sum -= l
    #   print("subtracted lambda")

    F[j] = output_sum

  return F

# Simulates expected rewards for given alpha and lambda.
def simulate(alpha, l, strategy_sim, difference, times, debug_flag):
  D_honest = np.zeros(x)
  
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
  distributions = np.zeros((100, x+1))

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
    

def main():
  # gather_data(optplus_sim, difference=0.01, times=4, output_file_name="OPTPLUS distributions 0.01 to 1.0.npy", debug_flag=True)
  
  distributions = np.load("OPTPLUS distributions 0.01 to 1.0.npy")
  gather_data_from_given_start(distributions, optplus_sim, start_alpha=0.33, 
    difference=0.01, times=4, output_file_name="OPTPLUS distributions 0.01 to 1.0.npy", debug_flag=True)


if __name__ == "__main__":
    main()
