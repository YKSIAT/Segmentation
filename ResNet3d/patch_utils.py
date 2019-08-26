import numpy as np


# def get_patch_from_array_around_ranch(image_array, patch_location_wh_plane, patch_location_d_dimensionality):
def get_patch_from_array_around_ranch(image_array, patch_location_wh_plane):
    """

    :param image_array:
    :param patch_location_wh_plane:
    :return:
    """

    shape = (240, 240, 155)  # image shape
    P_shape = (160, 160, 128)  # patch shape
    # Center_coordinate = (120, 120, 77)  # Center coordinate

    shape = shape
    patch_shape = P_shape
    Center_coordinate = [int(shape[0] / 2), int(shape[1] / 2), int(shape[2] / 2)]

    assert patch_location_wh_plane in ["00", "01", "02", "03", "04", "05", "06", "07", "cc"],\
        "patch_location mast be 0 or 1 or 2 or 3"
    # assert patch_location_d_dimensionality in ["front", "back"]
    assert (Center_coordinate[0] + int(patch_shape[0] / 2) < shape[0]) or \
           (Center_coordinate[0] - int(patch_shape[0] / 2) >= 0), "Out of size range"
    assert (Center_coordinate[1] + int(patch_shape[1] / 2) < shape[1]) or \
           (Center_coordinate[1] - int(patch_shape[1] / 2) >= 0), "Out of size range"
    assert (Center_coordinate[2] + int(patch_shape[2] / 2) < shape[0]) or \
           (Center_coordinate[2] - int(patch_shape[2] / 2) >= 0), "Out of size range"

    # if patch_location_d_dimensionality == "front":
    if patch_location_wh_plane == "00":
        patch_data = image_array[0:P_shape[0], shape[1] - P_shape[1]:shape[1], 0:P_shape[2]]
        return patch_data
    elif patch_location_wh_plane == "01":
        patch_data = image_array[shape[0] - P_shape[0]:shape[0], shape[1]-P_shape[1]:shape[1], 0:P_shape[2]]
        return patch_data
    elif patch_location_wh_plane == "02":
        patch_data = image_array[0:P_shape[0], 0:P_shape[1], 0:P_shape[2]]
        return patch_data
    elif patch_location_wh_plane == "03":
        patch_data = image_array[shape[0] - P_shape[0]:shape[0], 0:P_shape[1], 0:P_shape[2]]
        return patch_data
# elif patch_location_d_dimensionality == "back":
    elif patch_location_wh_plane == "04":
        patch_data = image_array[0:P_shape[0], shape[1]-P_shape[1]:shape[1], shape[2]-P_shape[2]:shape[2]]
        return patch_data
    elif patch_location_wh_plane == "05":
        patch_data = \
            image_array[shape[0] - P_shape[0]:shape[0], shape[1]-P_shape[1]:shape[1], shape[2]-P_shape[2]:shape[2]]
        return patch_data
    elif patch_location_wh_plane == "06":
        patch_data = image_array[0:P_shape[0], 0:P_shape[1], shape[2]-P_shape[2]:shape[2]]
        return patch_data
    elif patch_location_wh_plane == "07":
        patch_data = image_array[shape[0] - P_shape[0]:shape[0], 0:P_shape[1], shape[2]-P_shape[2]:shape[2]]
        return patch_data
    elif patch_location_wh_plane == "cc":
        patch_data = image_array[
                     Center_coordinate[0] - int(patch_shape[0] / 2):Center_coordinate[0] + int(patch_shape[0] / 2),
                     Center_coordinate[1] - int(patch_shape[1] / 2):Center_coordinate[1] + int(patch_shape[1] / 2),
                     Center_coordinate[2] - int(patch_shape[2] / 2):Center_coordinate[2] + int(patch_shape[2] / 2)]
        return patch_data


def fuse_array2complete_matrix(pre_array00, pre_array01, pre_array02, pre_array03,
                               pre_array04, pre_array05, pre_array06, pre_array07,
                               pre_array_center):
    """

    :param pre_array00:
    :param pre_array01:
    :param pre_array02:
    :param pre_array03:
    :param pre_array04:
    :param pre_array05:
    :param pre_array06:
    :param pre_array07:
    :param pre_array_center:

    :return:
    """

    fused_matrix_shape = [240, 240, 155]
    shape = (240, 240, 155)  # image shape
    P_shape = (160, 160, 128)  # patch shape
    Center_coordinate = (120, 120, 77)  # Center coordinate
    fused_matrix = np.zeros(fused_matrix_shape)

    fused_matrix[0:P_shape[0], shape[1]-P_shape[1]:shape[1], 0:P_shape[2]] = pre_array00
    fused_matrix[shape[0] - P_shape[0]:shape[0], shape[1]-P_shape[1]:shape[1], 0:P_shape[2]] = pre_array01
    fused_matrix[0:P_shape[0], 0:P_shape[1], 0:P_shape[2]] = pre_array02
    fused_matrix[shape[0] - P_shape[0]:shape[0], 0:P_shape[1], 0:P_shape[2]] = pre_array03
    fused_matrix[0:P_shape[0], shape[1]-P_shape[1]:shape[1], shape[2]-P_shape[2]:shape[2]] = pre_array04
    fused_matrix[shape[0] - P_shape[0]:shape[0], shape[1]-P_shape[1]:shape[1], shape[2]-P_shape[2]:shape[2]] \
        = pre_array05
    fused_matrix[0:P_shape[0], 0:P_shape[1], shape[2]-P_shape[2]:shape[2]] = pre_array06
    fused_matrix[shape[0] - P_shape[0]:shape[0], 0:P_shape[1], shape[2]-P_shape[2]:shape[2]] = pre_array07

    fused_matrix[Center_coordinate[0] - int(P_shape[0] / 2):Center_coordinate[0] + int(P_shape[0] / 2),
                 Center_coordinate[1] - int(P_shape[1] / 2):Center_coordinate[1] + int(P_shape[1] / 2),
                 Center_coordinate[2] - int(P_shape[2] / 2):Center_coordinate[2] + int(P_shape[2] / 2)] \
        = pre_array_center
    return fused_matrix
















































