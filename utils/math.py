import numpy as np

# find rotation to world frame for first trajectory
def rot_between_vectors(a,b):
    ''' Compute the rotation matrix that rotates vector a to vector b.

    Parameters:
    a (numpy.ndarray): The source vector.
    b (numpy.ndarray): The target vector.

    Returns:
    numpy.ndarray: The rotation matrix that aligns vector a with vector b.
    '''
    # rotates a -> b
    def skew(vector):
        return np.array([[0, -vector[2], vector[1]], 
                        [vector[2], 0, -vector[0]], 
                        [-vector[1], vector[0], 0]])

    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    v = np.cross(a,b)
    c = np.dot(a,b)
    s = np.linalg.norm(v)

    R = np.eye(3) + skew(v) + np.linalg.matrix_power(skew(v),2)*((1-c)/s**2)

    return R

def get_z_zero_vec(vec):
    vec_ = vec.copy()
    vec_[2] = 0.0
    vec_ /= np.linalg.norm(vec_)
    return vec_