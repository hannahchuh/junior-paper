from distutils.log import debug
import math
import numpy as np
from scipy import misc
import argparse
from strategies import hash_plus_sample
from strategies.constants import m, n, x, num_coins
# from sqlalchemy import all_, false


# if lambda = alpha then expected reward of honest should be 0
# b/c the honest strategy gets alpha per round
# however, in new simulations we are also subtracting alpha per round 
# for the honest strategy in this new format (since we subtract (lambda + alpha))
# we want to subtract a total of alpha per round. 
# so lambda = 0 is the same as subtracting alpha per round

# Simulates expected rewards for given alpha and lambda.
def simulate(alpha, beta, l, strategy_sim, difference, times, debug_flag):
  D_honest = np.zeros(n)
  
  # run first iteration
  F0 = strategy_sim(D_honest, alpha, beta, l)

  if debug_flag:
    print("FIRST ROUND EXPECTED REWARDS: ", np.average(F0))

  # run sim until expectation doesn't change by _difference_ for _times_ times in a row
  bestF = F0
  lastavg = np.average(F0)
  currentavg = 0
  runs = 1
  runs_in_range = 0
  while abs(lastavg - currentavg) > difference or runs_in_range < times:
    nowF = strategy_sim(bestF, alpha, beta, l)
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
def gather_data(strategy_sim, difference, beta, times, output_file_name, debug_flag):
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

      bestF = simulate(alpha, beta, l, strategy_sim, difference, times, debug_flag)
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
def gather_data_from_given_start(distributions, strategy_sim, start_alpha, beta, difference, times, output_file_name, debug_flag):  
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

      bestF = simulate(alpha, beta, l, strategy_sim, difference, times, debug_flag)
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
    

def run_once(alpha, beta, l, difference, times, debug_flag):
  print("RUNNING ONCE")
  if debug_flag: print("ALPHA: ", alpha)
  if debug_flag: print("Lambda: ", l)

  r = simulate(alpha, beta, l, hash_plus_sample.sim, difference, times, debug_flag)

  if debug_flag: print("Reward for ", l, ": ", r)

def main():
  parser = argparse.ArgumentParser()

  parser.add_argument('--run_once', action='store_true')
  parser.add_argument('--alpha', type=float, required=False)
  parser.add_argument('--l', type=float, required=False)
  args = parser.parse_args()

  if args.run_once:
    run_once(args.alpha, args.l, beta=0.5, difference=0.01, times=4, debug_flag=True)
  else:
  # gather_data(optplus_sim, difference=0.01, beta=0.5, times=4, output_file_name="OPTPLUS distributions 0.01 to 1.0.npy", debug_flag=True)
    distributions = np.load("OPTPLUS distributions 0.01 to 1.0.npy")
    gather_data_from_given_start(distributions, hash_plus_sample.sim, start_alpha=0.2, beta=0.5,
    difference=0.01, times=4, output_file_name="OPTPLUS distributions 0.01 to 1.0.npy", debug_flag=True)


if __name__ == "__main__":
    main()
