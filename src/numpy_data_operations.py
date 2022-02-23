import numpy as np

def arrMin(arr):
    # stored min value
    min_value = None
    # loop through array values
    for i in arr:
        # check if current val is less the stored min
        if min_value is None:
            min_value = i
        elif i < min_value:
            # update stored min value
            min_value = i
    # return min value in array
    return min_value

def arrMax(arr):
    # stored max value
    max_value = None
    # loop through array values
    for i in arr:
        # check if current val is less the stored max
        if max_value is None:
            max_value = i
        elif max_value < i:
            # update stored max value
            max_value = i
    # return main value in array
    return max_value