# project-2.py

# import libraries
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import urllib.request
import io
import gzip

DATA_URL = 'https://snap.stanford.edu/data/facebook_combined.txt.gz'

def getFileData(url: str) -> list:
    response = urllib.request.urlopen(url)
    compressed_file = io.BytesIO(response.read())
    decompressed_file = gzip.GzipFile(fileobj=compressed_file)
    # read file to edges list
    edges: list = list()
    while True:
        line = decompressed_file.readline()
        if not line: break # no more lines to read
        # parse line string
        line = str(line.decode("utf-8")).strip()
        edge_str_data = line.split(' ')
        point_1 = int(edge_str_data[0])
        point_2 = int(edge_str_data[1])
        edge = (point_1, point_2)
        edges.append(edge)
    decompressed_file.close()
    compressed_file.close()
    return edges

# edges_arr = getFileData(DATA_URL)

def vectorDotProduct(v1, v2):
    # Validate vectors
    if len(v1) != len(v2):
        raise ValueError("Cannot preform DOT product on vectors of different sizes")
    dot_product = 0
    for i in range(len(v1)): dot_product += v1[i] * v2[i]
    return dot_product

def transposeMatrix(matrix: np.ndarray):
    # validate matrix data type
    if not isinstance(matrix, np.ndarray):
        raise TypeError("`matrix` must be type `np.ndarray`")
    transposed_matrix = np.ndarray((matrix.shape[1], matrix.shape[0]), dtype=matrix.dtype)
    for col_index in range(matrix.shape[1]):
        transposed_matrix[col_index] = matrix[:,col_index]
    return transposed_matrix

def powerIteration(matrix: np.ndarray, p: np.array):
    p_i = np.array([0] * p.shape[0], dtype=float)
    for index in range(p.shape[0]):
        p_i[index] = vectorDotProduct(matrix[index,:], p)
    greatest_value = p_i[0]
    for i in range(len(p_i)):
        i_val = p_i[i]
        if i_val > greatest_value:
            greatest_value = i_val
    for i in range(len(p_i)):
        p_i[i] = p_i[i] / greatest_value
    return p_i

def verticesPrestigeCentrality(matrix: np.ndarray):
    # validate matrix data type
    if not isinstance(matrix, np.ndarray):
        raise TypeError("`matrix` must be type `np.ndarray`")
    p_0 = np.array([1] * matrix.shape[1])
    p_1 = powerIteration(matrix, p_0)
    return p_1

m = np.ndarray(shape=(5, 5), dtype=int)
m[0] = np.array([0,1,0,0,0])
m[1] = np.array([0,0,1,0,0])
m[2] = np.array([1,0,0,1,1])
m[3] = np.array([1,0,0,0,0])
m[4] = np.array([0,0,1,0,0])

verticesPrestigeCentrality(m)

# END
