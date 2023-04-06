import numpy as np
import matplotlib.pyplot as pyplot
from numpy.random import default_rng
import os

def data_init(map_depth,
        dist_params,
        class_size = 10):

    generator = default_rng()

  dataset = np.array([[generator.normal(loc = mean,
                     scale = sigma,
                     size = (map_depth)) for i in range(class_size)]
                      for mean, sigma in dist_params])

  return dataset.reshape((class_size*len(dist_params), -1))

def weights_init(size,
         map_depth,
         dist_sigma = 1.0):

  generator = default_rng()
  weights = generator.normal(scale = dist_sigma, size = (size, size, map_depth))
  return weights

def update_area(time,
        max_size):

  if time == 1:
    return max_size
  else:
    return int(max_size//np.log(time))


def kohonen_rule(weights,
         input_vector,
         position,
         update_scale,
         learning_rate = 0.1):


  xa, ya = np.subtract(position, update_scale//2)
  xb, yb = np.add(position, update_scale//2)%weights.shape[0]

  if xa > xb:
    xb *= -1
  if ya > yb:
    yb *= -1

  ty = np.arange(np.min((ya,yb)), np.max((ya,yb)))
  tx = np.arange(np.min((xa,xb)), np.max((xa,xb)))

  for i in ty:
    for j in tx:
      weights[i,j,:] += learning_rate * np.subtract(input_vector, weights[i,j,:])

  return weights

def distance(input_vector,
       weights):

  return -np.sum(np.abs(input_vector - weights))

def normalize(array):

  return (array - np.min(array))/(np.max(array)-np.min(array))

def print_activations(mesh_coords,
            activations,
            time,
            size,
            path = './out/activations'):

  if path.split('/')[1] not in os.listdir('.'):
    os.mkdir(path.split('/')[1])

  if path.split('/')[2] not in os.listdir('./'+path.split('/')[1]):
    os.mkdir(path)

  figure, axis = pyplot.subplots()

  mesh = axis.pcolormesh(mesh_coords[0], mesh_coords[1], activations.reshape((size, size)))
  figure.colorbar(mesh)
  figure.savefig(path + '/snapshot' + str(time) + '.png', format='png')

  pyplot.close(figure)

def print_map(weights,
        dataset,
        time,
        size,
        path = './out/clusters'):

  if path.split('/')[1] not in os.listdir('.'):
    os.mkdir(path.split('/')[1])

  if path.split('/')[2] not in os.listdir('./'+path.split('/')[1]):
    os.mkdir(path)

  figure, axis = pyplot.subplots()

  axis.scatter(weights[:, 0], weights[:, 1])

  axis.scatter(dataset[:, 0], dataset[:, 1],
        color = 'red',
        marker='x')

  figure.savefig(path+ '/snapshot'+str(time) + '.png',
        format='png')

  pyplot.close(figure)

def main():
  size = 100
  map_depth = 2

  dataset = normalize(data_init(map_depth, [(-10, 3), (20,3)], 10))

  weights = normalize(weights_init(size, map_depth))

  activations = np.zeros((size, size))
  mesh_coords = np.meshgrid([i for i in range(0,size)], [i for i in range(0,size)])

  print('*'*80)
  print('\n')

  for time in range(1, size):
    new_weights = weights
    for example in np.random.default_rng().choice(dataset, 1):

      activations = np.fromiter([distance(example, weight)
                for weight in weights.reshape(-1, 2)], np.float)

      activations = normalize(activations)
      avg_activation = np.average(activations)

      new_weights = kohonen_rule(weights,
                example,
                (np.argmax(activations)//size, np.argmax(activations)%size),
                update_area(time, weights.shape[0]),
                0.1)

    weights = new_weights

    print(avg_activation)
    print('\n')

    print_activations(mesh_coords, activations, time, size)
    print_map(weights, dataset, time, size)

if __name__ == "__main__":
  main()
