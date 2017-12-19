import numpy as np
cimport numpy as np
cimport cython
#import time

cpdef sparsify_output(np.ndarray[np.float_t, ndim=1] dense_output, np.ndarray[np.int_t, ndim=1] augmentation, unsigned int output_size, unsigned int noParameters, np.ndarray[np.float_t, ndim=1] output_sparse):
    cdef unsigned int i = 0
    cdef unsigned int j = 0
    cdef float value


    for i in range(0, output_size):
        output_sparse[i] = 0

    i = 0
    for i in range(0, noParameters):
        j = augmentation[i]
        value = dense_output[i]
        output_sparse[j] = output_sparse[j] + value

    return output_sparse


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


cpdef transform_input(np.ndarray[np.float_t, ndim=1] input_ex, np.ndarray[np.float_t, ndim=2] graph, np.ndarray[np.int_t, ndim=1] neuron_connectivity,  np.ndarray[np.float_t, ndim=1] transformed_input):
#    t0 = time.time()
    cdef unsigned int input_len = graph.shape[0]
    cdef unsigned int start_index = 0
    cdef unsigned int end_index, index, index_2
    cdef unsigned int x = 0

#    start_index = 0
    for index in range(0, input_len):
        #get indexes
        end_index = <unsigned int>(start_index + neuron_connectivity[index])
        
        # Transform input
        for index_2 in range(start_index, end_index):
            transformed_input[index_2] = input_ex[x]

        start_index = end_index
        x = <unsigned int>(x + 1)
        
    return transformed_input
        
#    t4 = time.time()
#    
#    print("Hoi")
#    print(t1 - t0)
#    print(t2-t1)
#    print(t3-t2)
#    print(t4-t3)
#    time.sleep(1)
#    