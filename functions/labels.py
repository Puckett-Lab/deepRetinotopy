import numpy as np

def labels(input,labels):
    faces_indexes = np.array([])
    for j in range(len(labels)):
        faces_indexes = np.concatenate((faces_indexes, np.where(input == labels[j])[0]), axis=0)

    faces = []
    for i in range(len(faces_indexes)):
        faces.append(input[int(faces_indexes[i])])

    return np.reshape(faces,(len(faces),3))
