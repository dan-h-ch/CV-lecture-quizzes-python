import cv2
import numpy as np

# Project a point from 3D to 2D using a matrix operation

# Given: Point p in 3-space [x y z], and focal length f
# Return: Location of projected point on 2D image plane [u v]


def project_point(p, f):
    # TODO: Define and apply projection matrix
    identity = np.identity(3).astype('float64')
    projection_matrix = np.zeros((3, 4))
    projection_matrix[:,:-1] = identity
    projection_matrix[2,2] = 1/f

    homogenous_coordinates = np.array(p)
    homogenous_coordinates = np.append(homogenous_coordinates, [1])
    homogenous_coordinates = homogenous_coordinates.reshape(4, 1)

    product = np.matmul(projection_matrix, homogenous_coordinates)

    answer = np.array([product[0,0]/product[2,0], product[1,0]/product[2,0]])

    return answer

# Test: Given point and focal length (units: mm)
p = np.array([[200, -900, 50]])
f = 50

print (project_point(p, f))