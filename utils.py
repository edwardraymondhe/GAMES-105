import numpy as np
from scipy.spatial.transform import Rotation as R

def get_unit_vector(vec):
    magnitude = np.linalg.norm(vec)
    if magnitude == 0:
        return vec
    else:
        return vec / magnitude

def get_angle_between_unit_vectors(a, b):
    dot_product = np.dot(a, b)
    return np.arccos(dot_product)

def get_rotation_matrix(a, b):
    a = get_unit_vector(a)
    b = get_unit_vector(b)
    u = get_unit_vector(np.cross(a, b))
    
    theta = get_angle_between_unit_vectors(a, b)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    matrix = np.array(
            [[0, -u[2], u[1]],
            [u[2], 0, -u[0]],
            [-u[1], u[0], 0]]
        )
    
    r = R.identity().as_matrix() + sin_theta * matrix + (1 - cos_theta) * matrix * matrix
    
    return r

def list_equal_within_tolerance(list1, list2):
    array1 = np.array(list1)
    array2 = np.array(list2)
    # 使用allclose方法判断两数组是否在指定精度内相等
    return np.allclose(array1, array2, atol=0.001)