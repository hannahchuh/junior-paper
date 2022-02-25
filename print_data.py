import math
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

def display(loaded_distributions):
  es = np.zeros(100)
  for i in range(1, 100):
    alpha = i/100
    D = loaded_distributions[i-1,:]
    es[i] = np.average(D)

    print(alpha)
    print(es[i])
  
  return es

def main():

  # distributions = np.load("FINAL distributions 0.01 to 1.0.npy") # 10000, OPT, violet
  # distributions = np.load("COIN distributions 0.01 to 1.0.npy") # 10000, COIN, red

  distributions = np.load("OPTPLUS distributions 0.01 to 1.0.npy") # 10000, COIN, red

  display(distributions)


if __name__ == "__main__":
    main()

