# dimensionality

from math import pi as PI, factorial

def volumeDim(radius : int, dimension : int):
    # Returns True if even
    def isEven(val : int):
        mod = val % 2
        if mod == 0:
            return True
        else:
            return False

    top = PI ** (dimension / 2)
    bottom = 0
    is_dimension_even = isEven(dimension)
    if is_dimension_even:
        temp_dimension = dimension / 2
        temp_bottom = 1
        while temp_dimension > 0:
            temp_bottom = temp_dimension * temp_bottom
            temp_dimension -= 1
        bottom = temp_bottom
    # if dim is odd
    else:
        bottom = (PI ** (1/2)) * ((factorial(dimension)) / (2 ** ((dimension + 1) / 2) ) )

    answer = (top / bottom)
    answer = (answer) * (radius ** dimension)

    return answer

# END
