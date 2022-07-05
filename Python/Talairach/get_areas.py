import numpy as np
import scipy.io as sio
import os

# read matlab file
def read_mat(filename):
    mat = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return mat
cwd = os.path.dirname(os.path.abspath(__file__))+'\\'
mat_data = read_mat(filename = cwd+'TDdatabase.mat')


def mni2cor(MNI_coords):
    """ 
    Converts MNI coordinates to atlas coordinates
    """
    MNI_coords = np.array([MNI_coords[0], MNI_coords[1], MNI_coords[2],1])
    T_ = np.array([[2, 0, 0, -92], [0, 2, 0, -128], [0, 0, 2, -74], [0, 0, 0, 1]])
    T_ = np.linalg.inv(T_).T
    return np.rint(np.dot(np.array([MNI_coords]), T_)).astype(np.int32)[0, :3]

def search_area(MNI_coords):
    index = mni2cor(MNI_coords)
    DB = mat_data['DB']
    areas = []
    for i in range(len(DB)):
        graylevel = DB[i].mnilist[index[0]-1, index[1]-1, index[2]-1]
        if graylevel == 0:
            areas.append('undefined')
            continue
        else:
            areas.append(DB[i].anatomy[graylevel-1])
    return areas

MNI_coords = np.array([2, 4, 60])
print(search_area(MNI_coords))
# print(mni2cor(MNI_coords))
# print(mat_data['DB'][0].mnilist[MNI_coords[0], MNI_coords[1], MNI_coords[2]])