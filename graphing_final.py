import math
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

def calculate_es(loaded_distributions, s_D):
  es = np.zeros(100)
  for i in range(1, 101):
    alpha = i/100
    es_alpha = alpha
    D = loaded_distributions[i-1,:]
    sorted_D = sorted(D)
    es_alpha += sorted_D[0]*alpha
    for j in range(1, s_D):
      index = j - 1
      interval = sorted_D[index+1] - sorted_D[index]
      prob = alpha - (((1-alpha)*alpha*j)/(s_D - alpha*j))
      es_alpha += (interval * prob)
    es[i-1] = es_alpha
  return es

def graph_distribution(loaded_distributions, s_D, strategy, color, labels_flag):
  es = calculate_es(loaded_distributions, s_D)

  alphas = [*range(1, 101)]
  alphas = np.divide(alphas, 100)

  # plot E^S
  fig = plt.figure()
  ax = fig.add_subplot(111)

  ax.plot(alphas[:100], es[:100], color=color)
  ax.legend([strategy])

  if labels_flag:
    for xy in zip(alphas, es):
      # if (xy[0]*100) % 10 == 0:
      #   ax.annotate('(%s, %.4f)' % xy, xy=xy, textcoords='data') 
      
      if xy[0] == 0.99:
        ax.annotate('(%s, %s)' % xy, xy=xy, textcoords='data') 

  plt.xlabel("alpha")
  plt.ylabel("expected reward (E^S)")
  plt.show()

  # plot E^S / (1 + E^S)
  prop_es = np.divide(es, np.add(1, es))

  fig = plt.figure()
  ax = fig.add_subplot(111)

  ax.plot(alphas[:99], prop_es[:99], color=color)
  ax.plot(alphas[:99], alphas[:99], color="blue")
  ax.legend([strategy, 'HONEST'])
  plt.xlabel("alpha")
  plt.ylabel("proportional reward (E^S/(1 + E^S))")
  plt.show()

def main():

  # distributions = np.load("FINAL distributions 0.01 to 1.0.npy") # 10000, OPT, violet
  distributions = np.load("March updated distributions 0.01 to 1.0.npy") # 10000, COIN, red

  graph_distribution(distributions, s_D=10000, strategy="COIN", color="red", labels_flag=True)


if __name__ == "__main__":
    main()

