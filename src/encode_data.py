import numpy as np

def labelEncodeMatrix(m):
    # label encoded matrix
    encoded_matrix = np.ndarray((len(m), len(m[0])))
    # a dictionary for maintaining encoded column values
    # found in the matrix
    encode_cols_data = {}
    # loop through input matrix rows
    for row in range(len(m)):
        # initalize empty array for current row encoding
        encoded_row = [0] * len(m[row])
        # loop through input matrix columns
        for col in range(len(m[row])):
            # x_ij value
            cell = m[row][col]
            # update the (temp) encoded row[col] value
            encoded_row[col] = cell
            # we only care about strings, so continue is not a type string
            if not isinstance(cell, str): continue
            # We have a string value, so let's set up our
            # encode columns dictionary
            if col not in encode_cols_data: encode_cols_data[col] = list()
            # Check to see if the current x_ij has aleady been set in our
            # encode columns dictionary data
            # If it has not, we will add it
            if cell not in encode_cols_data[col]:
                # add x_ij string value to our encode column dictionary data 
                encode_cols_data[col].append(cell)
            # set the (temp) encoded row x_ij value with the corresponding
            # label encoded value from out columns dictionary
            encoded_row[col] = encode_cols_data[col].index(cell)
        # append (temp) encoded row to our label encoded matrix
        encoded_matrix[row] = np.array(encoded_row)
    # return label encoded matrix
    return encoded_matrix


# labelEncodeMatrix([
#     [ 0.2, 23,   5.7, 'A'],
#     [ 0.4,  1,   5.4, 'B'],
#     [ 1.8, 0.5,  5.2, 'C'],
#     [ 5.6, 50,   5.1, 'A'],
#     [-0.5, 34,   5.3, 'B'],
#     [ 0.4, 19,   5.4, 'C'],
#     [ 1.1, 11,   5.5, 'C'],
# ])