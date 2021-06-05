import math
import numpy as np
from scipy import misc

"""Setup"""

# Number of coins
i = 100 
# Length of distribution
x = 10000

"""Strategies for simulation"""

# Update rule for OPT strategy
def opt_sim(D, alpha):
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

    # step 4: output sum calculation 
    output_sum = math.exp(-1*(1-alpha)*c[0])
    best_so_far = r[0]
    for k in range(i-1):
      prob = math.exp(-1*(1-alpha)*c[k]) - math.exp(-1*(1-alpha)*c[k+1])
      max_reward = max(best_so_far, r[k])
      output_sum += prob*max_reward

      best_so_far = max_reward

    F[j] = output_sum

  return F

# Update rule for COIN strategy
def coin_sim(D, alpha):
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

    # step 4: output coin with maximum expected value
    exp_vals = np.multiply(np.exp(np.multiply(c, alpha-1)), np.add(r, 1))
    max_coin = np.max(exp_vals)

    F[j] = max_coin

  return F

# Run simulation for alpha = 0.1 to 1.0 in increments of 0.01.
def gather_data(strategy_sim, difference, times, output_file_name, debug_flag):
  distributions = np.zeros((100, x))

  for num in range(1, 100): # multiples of 1/100
    alpha = num / 100

    D_honest = np.zeros(x)
    
    # run first iteration
    F0 = strategy_sim(D_honest, alpha)

    if debug_flag:
      print("ALPHA: ", alpha)
      print("FIRST ROUND EXPECTED REWARDS: ", np.average(F0))

    # run sim until expectation doesn't change by *difference*, *times* times in a row
    bestF = F0
    lastavg = np.average(F0)
    currentavg = 0
    runs = 1
    runs_in_range = 0
    while abs(lastavg - currentavg) > difference or runs_in_range < times:
      nowF = strategy_sim(bestF, alpha)
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

    distributions[num-1] = bestF

    np.save(output_file_name, distributions)

# Run simulation starting at alpha = start_alpha to 1.0 in increments of 0.01.
def gather_data_from_given_start(distributions, strategy_sim, start_alpha, difference, times, output_file_name, debug_flag):
  for num in range(start_alpha*100, 100): # multiples of 1/100
    alpha = num / 100

    D_honest = np.zeros(x)
    
    # run first iteration
    F0 = strategy_sim(D_honest, alpha)

    if debug_flag:
      print("ALPHA: ", alpha)
      print("FIRST EXPECT: ", np.average(F0))

    # run sim until expectation doesn't change by *difference*, *times* times in a row
    bestF = F0
    lastavg = np.average(F0)
    currentavg = 0
    runs = 1
    runs_in_range = 0
    while abs(lastavg - currentavg) > difference or runs_in_range < times:
      nowF = strategy_sim(bestF, alpha)
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

    distributions[num-1] = bestF

    np.save(output_file_name, distributions)
    

def main():
  # def gather_data(strategy_sim, difference, times, output_file_name, debug_flag):

  # Example usage for optimal strategy. 
  # Simulation will run until the expected rewards does not change by more than 0.01 for at least 4 times.

  gather_data(opt_sim, difference=0.01, times=4, output_file_name="optimal simulation distributions.npy", debug_flag=False)

  # Example usage for gather_data_from_given_start.
  # Should load distributions from an existing npy file, ideally the same name as your output file name.
  # Will start from a given start_alpha. 
  # Because gather_data takes a long time, if work stops in the middle this function can be used to restart work.

  # distributions = np.load("coin simulation distributions.npy")
  # gather_data_from_given_start(distributions, coin_sim, start_alpha=0.86, 
  #   difference=0.001, times=3, output_file_name="coin simulation distributions.npy", debug_flag=False)

if __name__ == "__main__":
    main()
