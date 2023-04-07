import os
import numpy as np
from matplotlib import pyplot
from numpy.random import default_rng

def data_init(map_depth,
          dist_params,
          class_size = 10):
    """Generate dummy data from normal distribution with given `dist_params`.
    """
    generator = default_rng()

    dataset = np.array([[generator.normal(loc = mean,
                       scale = sigma,
                       size = (map_depth)) for i in range(class_size)]
                        for mean, sigma in dist_params])

    return dataset.reshape((class_size*len(dist_params), -1))

def weights_init(size,
           map_depth,
           dist_sigma = 1.0):
    """Initialize weights randomly with given sigma.
    """
    generator = default_rng()
    weights = generator.normal(scale = dist_sigma, size = (size, size, map_depth))
    return weights

def update_area(time,
          max_size):
    """Get size of the updated area based on time.
    """

    return int(max_size-time)


def kohonen_rule(weights,
           input_vector,
           position,
           update_scale,
           learning_rate = 0.1):

    # Compute two points defining updated aread
    xa, ya = np.subtract(position, update_scale//2)
    xb, yb = np.add(position, update_scale//2)
    # Generate list of updated indices
    ty = np.arange(np.min((ya,yb)), np.max((ya,yb))) % weights.shape[0]
    tx = np.arange(np.min((xa,xb)), np.max((xa,xb))) % weights.shape[0]
    tx, ty = np.meshgrid(tx, ty, indexing='ij')

    # Update all weights
    weights[tx, ty, :] += learning_rate * (input_vector - weights[tx, ty, :])

    return weights

def distance(input_vector,
         weights):
    """Simple euclidean norm
    """
    return np.sqrt(np.sum(np.square(input_vector - weights)))

def normalize(array):
    """Min-max normalization
    """
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
            path = './out/clusters'):

    if path.split('/')[1] not in os.listdir('.'):
        os.mkdir(path.split('/')[1])

    if path.split('/')[2] not in os.listdir('./'+path.split('/')[1]):
        os.mkdir(path)

    figure, axis = pyplot.subplots()

    axis.scatter(weights[:, 0], weights[:, 1])

    axis.scatter(dataset[:, 0], dataset[:, 1],
                 color = 'red', marker='x')

    figure.savefig(path+ '/snapshot'+str(time) + '.png',
                   format='png')

    pyplot.close(figure)

def main():
    size = 100
    map_depth = 2

    dataset = normalize(data_init(map_depth, [(-10, 3), (20,3)], 10))

    weights = normalize(weights_init(size, map_depth))

    mesh_coords = np.meshgrid([i for i in range(0,size)], [i for i in range(0,size)])

    print('*'*80)
    print('\n')

    for time in range(1, size):
        for example in np.random.default_rng().choice(dataset, 1):

            activations = np.fromiter([distance(example, weights[i][j])
                  for i in range(size) for j in range(size)], np.float32)

            activations = normalize(activations)
            avg_activation = np.average(activations)

            new_weights = kohonen_rule(weights=weights,
                  input_vector=example,
                  position=(np.argmin(activations)//size, np.argmin(activations)%size),
                  update_scale=update_area(time, weights.shape[0]),
                  learning_rate = (size - time)/size)

        weights = new_weights.copy()

        print(avg_activation)
        print('\n')

        print_activations(mesh_coords, activations, time, size)
        print_map(weights, dataset, time)

if __name__ == "__main__":
    main()
