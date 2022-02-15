# import pandas as pd
import numpy as np
# from sklearn import datasets
# import matplotlib.pyplot as plt
import math

# iris_data = datasets.load_iris()
# print(iris_data['DESCR'])

# data = iris_data['data']

# print("the type is", type(data))
# print("the shape is", data.shape)
# print("the len is", len(data))
# print("the numver of dimensions is", np.ndim(data))
# print("the first 5 entries are:\n", data[:5,])

# features = iris_data['feature_names']

# plt.scatter(data[:,0], data[:,1])
# plt.xlabel(features[0])
# plt.ylabel(features[1])


# X1
# red, blue, yellow
# X2
# yes, no
# X3
# north, south, east, west

data    = np.chararray((7, 3), itemsize=6)

data[0] = np.array(['red',    'yes',  'north'])
data[1] = np.array(['blue',   'no',   'south'])
data[2] = np.array(['yellow', 'no',   'east'])
data[3] = np.array(['yellow', 'no',   'west'])
data[4] = np.array(['red',    'yes',  'north'])
data[5] = np.array(['yellow', 'yes',  'north'])
data[6] = np.array(['blue',   'no',   'west'])

# print(data)

# target = data
# np.unique(target)
# x2 = target[:,1]


# counts = [sum(x2==b'yes'), sum(x2==b'no')]
# target_names = ['yes' ,' no']
# plt.bar(target_names, counts)
# plt.ylabel('Number of Occurrences')
# plt.xlabel("'Yes' or 'No' value")
# plt.title('Yes/No Bar Graph')


# Question 2
# --------------------------------------------------------------------

oneHotData = np.ndarray((7, 9))

for row in range(data.shape[0]):
    row_array = data[row,:]
    color       = row_array[0]
    yes_no      = row_array[1]
    direction   = row_array[2]
    # empty row for encoded data
    hot_data_row = np.array([0] * 9)
    # encode color
    if color == b'red': hot_data_row[0] = 1
    elif color == b'blue': hot_data_row[1] = 1
    elif color == b'yellow': hot_data_row[2] = 1
    # encode yes/no
    if yes_no == b'yes': hot_data_row[3] = 1
    else: hot_data_row[4] = 1
    # encode direction (north, south, east, and west)
    if   direction == b'north': hot_data_row[5] = 1
    elif direction == b'south': hot_data_row[6] = 1
    elif direction == b'east': hot_data_row[7] = 1
    elif direction == b'west': hot_data_row[8] = 1
    # set row values on encoded matrix
    oneHotData[row] = hot_data_row

print("Question 2: \n", oneHotData, '\n')

# Question 3 - Euclidean Distance
# --------------------------------------------------------------------

def euclideanDistance(m, row1Num, row2Num):
    x1 = m[row1Num,:]
    x2 = m[row2Num,:]
    answer = 0
    for i in range(m.shape[1]):
        answer += (x1[i] - x2[i]) ** 2
    return math.sqrt(answer)

print("Question 3.) - Euclidean Distance (x2 and x7): ", euclideanDistance(oneHotData, 1, 6), '\n')

# Question 4 - cosine similarity (cosine of the angle)
# --------------------------------------------------------------------

def cosineOfRows(m, row1Num, row2Num):
    x1 = m[row1Num,:]
    x2 = m[row2Num,:]
    top = 0
    bottom_left = 0
    bottom_right = 0
    for i in range(m.shape[1]):
        top += x1[i] * x2[i]
        bottom_left += x1[i] ** 2
        bottom_right += x2[i] ** 2
    return top / (math.sqrt(bottom_left) * math.sqrt(bottom_right))

print("Cosine x2 and x7: ", cosineOfRows(oneHotData, 1, 6))

# Question 5 - Hamming Distance
# --------------------------------------------------------------------

def hammingDistance(m, row1Num, row2Num):
    x1 = m[row1Num,:]
    x2 = m[row2Num,:]
    sum = 0
    for i in range(m.shape[1]):
        if (x1[i] == 1 and x2[i] == 0) or (x1[i] == 0 and x2[i] == 1):
            sum += 1
    return sum

print("Hamming Distance (XOR) of x2 and x7: ", hammingDistance(oneHotData, 1, 6))

# Question 6 - Jaccard similarity
# --------------------------------------------------------------------

def jaccardSimilarity(m, row1Num, row2Num):
    x1 = m[row1Num,:]
    x2 = m[row2Num,:]
    top = 0
    bottom = 0
    for i in range(m.shape[1]):
        # top
        if (x1[i] == x2[i]): top += 1
        # bottom
        if (x1[i] != x2[i]): bottom += 1
    return (top / bottom)
print("Jaccard similarity of x2 and x7: ", jaccardSimilarity(oneHotData, 1, 6))

# Question 7 - Multi-Dimensional Mean
# --------------------------------------------------------------------

def multiDimensionalMean(m):
    mean_data = [0] * m.shape[1]
    for row in range(m.shape[0]):
        for col in range(m.shape[1]):
            mean_data[col] += m[row, col]
    for col in range(len(mean_data)):
        mean_data[col] = mean_data[col] / m.shape[0]
    return mean_data
print("Multi-Dimensional Mean of Y: {!s}".format(multiDimensionalMean(oneHotData)))

# Question 8 - Variance
# --------------------------------------------------------------------

def coVariance(m, col1Num, col2Num):
    col1 = m[:,col1Num]
    col2 = m[:,col2Num]
    answer = 0
    for i in range(m.shape[0]):
        answer += (col1[i] - col1.mean()) * (col2[i] - col2.mean())
    return answer / (m.shape[0] - 1)
print("Estimated variance of the first column of Y: {!s}".format(coVariance(oneHotData, 0, 0)))

# We can double check this by:
print('Double check with np built in function: {!s}'.format(np.var(oneHotData[:,0], ddof=1)))

# Question 9 - z-score
# --------------------------------------------------------------------

def zScore(m):
    z_score = np.ndarray(m.shape)
    for row in range(z_score.shape[0]):
        for col in range(z_score.shape[1]):
            z_score[row, col] = 0

            x_ij = m[row, col]
            mean = m[:,col].mean()
            div = math.sqrt(coVariance(m, col, col))

            z_score[row, col] = (x_ij - mean) / div

    return z_score

print('\n',zScore(oneHotData))