import numpy as np

def labels(input,labels):
    output=[]
    for k in range(len(labels)):
        for i in range(len(input)):
            for j in range(3):
                if input[i][j] == labels[k]:
                    input[i] = np.array([0, 0, 0])


    for i in range(len(input)):
        if np.sum(input[i] == np.array([0, 0, 0])) != 3:
            output.append(input[i])

    return np.array(output)
