import numpy as np
import timeit
import pandas as pd
import matplotlib.pyplot as plt

def createWeightsMask(noRows, noCols):
    """
    noRows = input size
    noCols = output size
    """
    np.random.seed(1)
    mask_weights = np.random.rand(noRows, noCols)
    prob = 1 - (1 * (noRows + noCols)) / (noRows * noCols)  # normal tp have 8x connections
    mask_weights[mask_weights < prob] = 0
    mask_weights[mask_weights >= prob] = 1
    noParameters = np.sum(mask_weights)
    print (prob,noParameters, noRows*noCols)
    return [noParameters,mask_weights]


def bench_all():
    setup_naive = """\
import numpy as np
from __main__ import createWeightsMask
import sparseBiparteDenseMat
from scipy import sparse

input_size = {}
output_size = {}

np.random.seed(1)
input_example = np.random.rand(input_size)
noParameters, mask = createWeightsMask(output_size,input_size)

print("type:" + str(type(mask)))
print("memory footprint:" + str(mask.data.nbytes))"""

    setup_scipy = """\
import numpy as np
from __main__ import createWeightsMask
import sparseBiparteDenseMat
from scipy import sparse

input_size = {}
output_size = {}

np.random.seed(1)
input_example = np.random.rand(input_size)
noParameters, mask = createWeightsMask(output_size,input_size)
mask = sparse.{}(mask)
print("type:" + str(type(mask)))
print("memory footprint:" + str(mask.data.nbytes))"""

    setup_compressed = """\
import numpy as np
from __main__ import createWeightsMask
import sparseBiparteDenseMat
from scipy import sparse

input_size = {}
output_size = {}

np.random.seed(1)
input_example = np.random.rand(input_size)
noParameters, mask = createWeightsMask(input_size,output_size)
neuron_connectivity = mask.astype(int).sum(axis = 1).astype(np.int)
input_ex_compressed = np.empty(mask.sum().astype(np.int))
compressed_mask, augmentation = sparseBiparteDenseMat.transform_graph(mask)
activations_decompressed = np.empty(output_size)
print('sparse_bupartite')"""

    sparseBiparteExperiment = """\
sparseBiparteDenseMat.transform_input(input_example, mask, neuron_connectivity, input_ex_compressed)
sparseBiparteDenseMat.compressed_multiply(input_ex_compressed, compressed_mask)
sparseBiparteDenseMat.decompress_output(input_ex_compressed, augmentation, activations_decompressed)"""

    cols = ['naive', 'bsr', 'coo', 'csc', 'csr', 'dia', 'dok', 'lil', 'proposal']
    experiments = [2, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225, 256, 289, 324, 361, 400, 441, 484, 529, 576, 625, 676, 729, 784, 841, 900, 961, 1024]
    bench_restult = pd.DataFrame(columns=cols, index = experiments)
    print(bench_restult.loc[2, 'naive'])

    for n in experiments:
        bench_restult.loc[n,'naive'] = timeit.timeit("mask.dot(input_example)", setup=setup_naive.format(n,n))
        bench_restult.loc[n,'bsr'] = timeit.timeit("mask.dot(input_example)", setup=setup_scipy.format(n,n,"bsr_matrix"))
        bench_restult.loc[n,'coo'] = timeit.timeit("mask.dot(input_example)", setup=setup_scipy.format(n,n,"coo_matrix"))
        bench_restult.loc[n,'csc'] = timeit.timeit("mask.dot(input_example)", setup=setup_scipy.format(n,n,"csc_matrix"))
        bench_restult.loc[n,'csr'] = timeit.timeit("mask.dot(input_example)", setup=setup_scipy.format(n,n,"csr_matrix"))
        bench_restult.loc[n,'dia'] = timeit.timeit("mask.dot(input_example)", setup=setup_scipy.format(n,n,"dia_matrix"))
        #bench_restult.loc[n,'dok'] = timeit.timeit("mask.dot(input_example)", setup=setup_scipy.format(n,n,"dok_matrix"))
        bench_restult.loc[n,'lil'] = timeit.timeit("mask.dot(input_example)", setup=setup_scipy.format(n,n,"lil_matrix"))
        bench_restult.loc[n,'proposal'] = timeit.timeit(sparseBiparteExperiment, setup=setup_compressed.format(n,n))

    bench_restult.to_csv('benchmark_results.csv')
    print(bench_restult)
    bench_restult.plot()
    plt.legend(loc='best')
    plt.show()


if __name__ == "__main__":
    bench_all()

