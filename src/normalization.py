
import numpy as np
from variance import covariance
from mean import arrayMean
from numpy_data_operations import arrMin, arrMax

def rangeNormalize(m):
    # create normlized matrix based on shape of input matrix
    normalized_arr = np.ndarray(m.shape)
    # loop through input matrix rows
    for row_index in range(m.shape[0]):
        # loop through input matrix columns
        for col_index in range(m.shape[1]):
            # get current column array
            col_arr = m[:,col_index]
            # min value for current column
            col_min = arrMin(col_arr)
            # max value for current column
            col_max = arrMax(col_arr)
            # calculate current x_ij normalized value
            x_ij_norm_value = (m[row_index, col_index] - col_min) / (col_max - col_min)
            # set x_ij normalized value in normalized matrix
            normalized_arr[row_index, col_index] = x_ij_norm_value
    # return the normalized array
    return normalized_arr

def zScoreNormalize(m):
    # create normlized matrix based on shape of input matrix
    z_score = np.ndarray(m.shape)
    # loop through input matrix rows
    for row_index in range(m.shape[0]):
        # loop through input matrix columns
        for col_index in range(m.shape[1]):
            # get current column array
            col_arr = m[:,col_index]
            # calculate the standard devieation for the current column
            col_std_div = (covariance(col_arr)) ** (1/2)
            # calculate the column's mean
            col_mean = arrayMean(col_arr)
            # get the x_ij value from the imput matix
            x_ij = m[row_index, col_index]
            # calculate the x_ji z-score
            x_ij_zscore = (x_ij - col_mean) / col_std_div
            # set x_ij normalized value in normalized matrix
            z_score[row_index, col_index] = x_ij_zscore
    # return the normalized array
    return z_score