import numpy as np
import sparseBiparteDenseMat
import timeit



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

def bench_scipy():   
    setup = """\
import numpy as np
from __main__ import createWeightsMask
from scipy import sparse
import time

input_size = 200
output_size = 300

np.random.seed(1)
input_example = mask_weights = np.random.rand(input_size)
foo, mask = createWeightsMask(output_size,input_size)
mask = mask.astype(int)
neuron_connectivity = mask.sum(axis = 0).astype(np.int)
foo = np.empty(mask.sum().astype(np.int) + mask.shape[1])
{}
print("type:" + str(type(mask)))
print("memory footprint:" + str(mask.data.nbytes))"""
    
    print(timeit.timeit("mask.dot(input_example)", setup=setup))
    #print(timeit.timeit("mask.dot(input_example)", setup=setup.format("mask = sparse.bsr_matrix(mask)", number = 1000)))
    #print(timeit.timeit("mask.dot(input_example)", setup=setup.format("mask = sparse.coo_matrix(mask)", number = 1000)))
    #print(timeit.timeit("mask.dot(input_example)", setup=setup.format("mask = sparse.csc_matrix(mask)", number = 1000)))
    print(timeit.timeit("mask.dot(input_example)", setup=setup.format("mask = sparse.csr_matrix(mask)")))
    #print(timeit.timeit("mask.dot(input_example)", setup=setup.format("mask = sparse.dia_matrix(mask)", number = 1000)))
    #print(timeit.timeit("mask.dot(input_example)", setup=setup.format("mask = sparse.dok_matrix(mask)")))
    #print(timeit.timeit("mask.dot(input_example)", setup=setup.format("mask = sparse.lil_matrix(mask)")))



def test_sparseBiparteDenseMat():
    setup = """\
import numpy as np
from __main__ import createWeightsMask
import sparseBiparteDenseMat
from scipy import sparse

input_size = 200
output_size = 300
    
np.random.seed(1)
input_example = np.random.rand(input_size)
noParameters, mask = createWeightsMask(input_size,output_size)
neuron_connectivity = mask.astype(int).sum(axis = 1).astype(np.int)
input_ex_tansformed = np.empty(mask.sum().astype(np.int))
sparse_mask, augmentation = sparseBiparteDenseMat.transform_graph(mask)
output_sparse = np.zeros(output_size)
"""

    sparseBiparteExperiment = """\
input_ex_tansformed = sparseBiparteDenseMat.transform_input(input_example, mask, neuron_connectivity, input_ex_tansformed)
output_dense = np.multiply(np.transpose(input_ex_tansformed), sparse_mask)
sparseBiparteDenseMat.sparsify_output(output_dense, augmentation, output_size, noParameters, output_sparse)
"""


    print("SparseBiparte")
    print(timeit.timeit(sparseBiparteExperiment, setup=setup))
    #print("repeat")
    #print(timeit.timeit("np.repeat(input_example, neuron_connectivity)", setup=setup))



if __name__ == "__main__":
    test_sparseBiparteDenseMat()
    bench_scipy()

