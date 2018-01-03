import numpy as np
cimport numpy as np
cimport cython
#import time


@cython.boundscheck(False)
cpdef compressed_multiply(np.ndarray[np.float_t, ndim=1] input_ex_tansformed, np.ndarray[np.float_t, ndim=1] sparse_mask):
    cdef unsigned int i

    for i in range(0,input_ex_tansformed.shape[0]):
        input_ex_tansformed[i] = <np.float_t>(input_ex_tansformed[i] * sparse_mask[i])

@cython.boundscheck(False)
cpdef single_step_multiply(np.ndarray[np.float_t, ndim=1] input_ex, np.ndarray[np.float_t, ndim=1] compressed_mask, np.ndarray[np.int_t, ndim=1] neuron_connectivity,  np.ndarray[np.float_t, ndim=1] output_compressed):
    cdef unsigned int input_len = input_ex.shape[0]
    cdef unsigned int index = 0
    cdef unsigned int input_neuron_nr = 0

    for input_neuron_nr in range(input_len):
        for c in range(<unsigned int>(neuron_connectivity[input_neuron_nr])):
            output_compressed[index] = <np.float_t>(input_ex[input_neuron_nr] * compressed_mask[index])
            index = <unsigned int>(index + 1)


@cython.boundscheck(False)
cpdef decompress_output(np.ndarray[np.float_t, ndim=1] output_compressed, np.ndarray[np.int_t, ndim=1] augmentation, np.ndarray[np.float_t, ndim=1] output_decompressed):
    cdef unsigned int i = 0
    cdef unsigned int j = 0
    cdef float value

    for i in range(0, output_decompressed.shape[0]):
        output_decompressed[i] = 0

    i = 0
    for i in range(0, output_compressed.shape[0]):
        j = augmentation[i]
        value = output_compressed[i]
        output_decompressed[j] = <np.float_t>(output_decompressed[j] + value)


@cython.boundscheck(False)
cpdef transform_graph(np.ndarray[np.float_t, ndim=2] graph):
    cdef unsigned int input_size = graph.shape[0]
    cdef unsigned int output_size = graph.shape[1]
    cdef unsigned int no_parameters = np.count_nonzero(graph)
    cdef np.ndarray[np.float_t, ndim=1] transformed_graph = np.empty(no_parameters, dtype=float)
    cdef np.ndarray[np.int_t, ndim=1] augmentation = np.empty(no_parameters, dtype=int)
    cdef unsigned int i = 0
    cdef unsigned int j = 0
    cdef unsigned int k = 0

    for i in range(0, input_size):
        for j in range(0, output_size):
            if graph[i,j] > 0:
                transformed_graph[k] = graph[i,j]
                augmentation[k] = j
                k = <unsigned int>(k + 1)

    return transformed_graph, augmentation

@cython.boundscheck(False)
cpdef transform_input(np.ndarray[np.float_t, ndim=1] input_ex, np.ndarray[np.float_t, ndim=2] graph, np.ndarray[np.int_t, ndim=1] neuron_connectivity,  np.ndarray[np.float_t, ndim=1] transformed_input):
    cdef unsigned int input_len = graph.shape[0]
    cdef unsigned int start_index = 0
    cdef unsigned int end_index, index, index_2
    cdef unsigned int x = 0

    for index in range(0, input_len):
        #get indexes
        end_index = <unsigned int>(start_index + neuron_connectivity[index])
        
        # Transform input
        for index_2 in range(start_index, end_index):
            transformed_input[index_2] = input_ex[x]

        start_index = end_index
        x = <unsigned int>(x + 1)