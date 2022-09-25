from pennylane import numpy as np
from sklearn.metrics import mutual_info_score as MIS

#这里面放一些做kernel常用的函数





def multiclass_target_alignment(
    X,
    Y,
    K,
    n_class,
):
    """Kernel-target alignment between kernel and labels."""

    T = np.zeros((len(Y), len(Y)))
    for i in range(len(Y)):
        for j in range(len(Y)):
            if(Y[i] == Y[j]):
                T[i,j] = 1
            else:
                T[i,j] = -1 / (n_class - 1)


    inner_product = np.sum(K * T)
    norm = np.sqrt(np.sum(K * K) * np.sum(T * T))
    inner_product = inner_product / norm

    return inner_product

def multiclass_target_alignment_origin(
    X,
    Y,
    kernel,
    n_class,
    assume_normalized_kernel=False,
):
    """Kernel-target alignment between kernel and labels."""

    K = qml.kernels.square_kernel_matrix(
        X,
        kernel,
        assume_normalized_kernel=assume_normalized_kernel,
    )


    T = np.zeros((len(Y), len(Y)))
    for i in range(len(Y)):
        for j in range(len(Y)):
            if(Y[i] == Y[j]):
                T[i,j] = 1
            else:
                T[i,j] = -1 / (n_class - 1)

    inner_product = np.sum(K * T)
    norm = np.sqrt(np.sum(K * K) * np.sum(T * T))
    inner_product = inner_product / norm

    return inner_product

def target_alignment_with_kernels(
    K1,
    K2,
):
    """Kernel-target alignment between kernel and kernel."""

    inner_product = np.sum(K1 * K2)
    norm = np.sqrt(np.sum(K1 * K1) * np.sum(K2 * K2))
    inner_product = inner_product / norm

    return inner_product

def get_MI_between_data_and_labels(X, Y):
    temp_MI = []
    for i in range(np.shape(X)[1]):
        temp_MI.append(MIS(X[:,i], Y)) 
    
    return np.mean(temp_MI)


def get_kernel_similarity(iter_number, kernel_list):
    kernel_similarity_list = np.ones(len(kernel_list))
    kernel_1 = kernel_list[iter_number]
    for i in range(iter_number+1, len(kernel_list)):
        kernel_2 = kernel_list[i]
        kernel_similarity = target_alignment_with_kernels(kernel_1, kernel_2)
        kernel_similarity_list[i] = kernel_similarity
    
    return kernel_similarity_list


