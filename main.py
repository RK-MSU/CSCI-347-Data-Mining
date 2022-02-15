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

    hot_data_row = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])

    color = row_array[0]
    yes_no = row_array[1]
    direction = row_array[2]
    
    if color == b'red': hot_data_row[0] = 1
    elif color == b'blue': hot_data_row[1] = 1
    elif color == b'yellow': hot_data_row[2] = 1

    if yes_no == b'yes': hot_data_row[3] = 1
    else: hot_data_row[4] = 1

    if direction == b'north': hot_data_row[5] = 1
    elif direction == b'south': hot_data_row[6] = 1
    elif direction == b'east': hot_data_row[7] = 1
    elif direction == b'west': hot_data_row[8] = 1
    
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
    pass

print("Question 3.) - Euclidean Distance (x2 and x7): ", euclideanDistance(oneHotData, 1, 6), '\n')

# Question 4 - cosine similarity (cosine of the angle)
# --------------------------------------------------------------------

# data = np.ndarray((7, 9))

# #                  |       |     |           |
# data[0] = np.array([1, 0, 0, 1, 0, 1, 0, 0, 0])
# #                  |       |     |           |
# data[1] = np.array([0, 1, 0, 0, 1, 0, 1, 0, 0])
# #                  |       |     |           |
# data[2] = np.array([0, 0, 1, 0, 1, 0, 0, 1, 0])
# #                  |       |     |           |
# data[3] = np.array([0, 0, 1, 0, 1, 0, 0, 0, 1])
# #                  |       |     |           |
# data[4] = np.array([1, 0, 0, 1, 0, 1, 0, 0, 0])
# #                  |       |     |           |
# data[5] = np.array([0, 0, 1, 1, 0, 1, 0, 0, 0])
# #                  |       |     |           |
# data[6] = np.array([0, 1, 0, 0, 1, 0, 0, 0, 1])


x2 = oneHotData[1,:]
x7 = oneHotData[6,:]

print('X2: ',x2)
print('X7: ', x7)


def oneHotDistance():
    answer = 0
    for i in range(len(x2)):
        a = x2[i]
        b = x7[i]
        answer += (a-b) ** 2
    return math.sqrt(answer)

print("One Hot Distance: ", oneHotDistance())


def cosineX2andX7():
    top = 0
    for i in range(len(x2)):
        top += x2[i] * x7[i]
    bottom_left = 0
    bottom_right = 0
    for i in range(len(x2)):
        bottom_left += x2[i] ** 2
        bottom_right += x7[i] ** 2
    bottom_left = math.sqrt(bottom_left)
    bottom_right = math.sqrt(bottom_right)
    answer = top / (bottom_left * bottom_right)
    return answer
print("Cosine x2 and x7: ", cosineX2andX7())


def xorOfx2andx7():
    sum = 0
    for i in range(len(x2)):
        if (x2[i] == 1 and x7[i] == 0) or (x2[i] == 0 and x7[i] == 1):
            sum += 1
    return sum

print("XOR of x2 and x7", xorOfx2andx7())


def JACCARDOfx2andx7():
    top = 0
    bottom = 0
    for i in range(len(x2)):
        # top
        if (x2[i] == x7[i]): top += 1
        # bottom
        if (x2[i] != x7[i]): bottom += 1
    return (top / bottom)

print("JACCARDOfx2andx7():", JACCARDOfx2andx7())


# def multiMean():
#     multi_mean = []
#     for col in range(data.shape[1]): # rows
#         multi_mean[col] = data[:,col].mean()
#         # for col in range(data.shape[1]): # cols
#             # multi_mean[row] 
#     return multi_mean

# print(multiMean())