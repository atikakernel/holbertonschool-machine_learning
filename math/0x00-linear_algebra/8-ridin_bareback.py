def matrix_shape(matrix):
    """calculates the shape of a matrix"""
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape


def matrix_transpose(matrix):
    """calculates the transpose of a matrix"""
    return[[matrix[col][row] for col in range(len(matrix))]
           for row in range(len(matrix[0]))]


def mat_mul(mat1, mat2):
    """return a matrix multiplication"""
    shape1 = matrix_shape(mat1)
    shape2 = matrix_shape(mat2)
    values = []
    if shape1[1] == shape2[0]:
        mul_rows = shape1[0]
        mul_cols = shape2[1]
        trans2 = matrix_transpose(mat2)
        for row_mat1 in mat1:
            for row_mat2 in trans2:
                values.append(sum([i*j for (i, j) in zip(row_mat1, row_mat2)]))
        return [values[idx:idx + mul_cols]
                for idx in range(0, len(values), mul_cols)]
    return None
