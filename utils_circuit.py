import pennylane as qml
from pennylane import numpy as np



#这里面放一些做kernel常用的函数

def reshape_circuit_list(curcuit_code_list, n_qubits):
    final_curcuit_code_list = []
    for i in range(len(curcuit_code_list)):
        curcuit_code = curcuit_code_list[i]
        curcuit_code = np.reshape(curcuit_code, (n_qubits,3), order='F')
        final_curcuit_code_list.append(curcuit_code)

    return final_curcuit_code_list

def generate_circuit_code(wires):
    circuit_code = []
    for i in range(3):
        for j in range(wires):
            if(i == 0):
                circuit_code.append(random.randint(0,2))
            else:
                circuit_code.append(random.randint(0,1))

    return circuit_code


def kernel_layer(x, wires, curcuit_code):
    """Building block of the embedding ansatz"""
    for j, wire in enumerate(wires):
        for i in range(np.shape(curcuit_code)[1]):
            if i == 0:
                if(curcuit_code[j,i] == 0):
                    qml.RX(x[j], wires=[wire])
                elif(curcuit_code[j,i] == 1):
                    qml.RY(x[j], wires=[wire])
                elif(curcuit_code[j,i] == 2):
                    qml.RZ(x[j], wires=[wire])
            elif i == 1:
                if(curcuit_code[j,i] == 1):
                    qml.Hadamard(wires=[wire])
            elif i == 2:
                if(curcuit_code[j,i] == 1):
                    if j == len(wires) - 1:
                        qml.CNOT(wires = [j, 0])
                    else:
                        qml.CNOT(wires = [j, j+1])

