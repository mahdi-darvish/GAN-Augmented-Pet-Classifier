import numpy as np
from . import image

def sample_bounding_box(size, landmarks, margin=0.1, bb_min=1):

    bb_old = image.get_bounding_box(landmarks)
    bb_old = [np.max((0, bb_old[0])), np.max((0, bb_old[1])), np.min((size[0] - 1, bb_old[2])), np.min((size[1] - 1, bb_old[3]))]
    bb_old_size = np.max((bb_old[2] - bb_old[0], bb_old[3] - bb_old[1]))
    img_size_min = int(np.min(size) * (1. + 2. * margin))
    bb_crop_size_min = int(bb_old_size * bb_min) + 1
    bb_crop_size_max = np.max((img_size_min - 1, bb_crop_size_min))
    bb_crop_size = np.random.random_integers(low=bb_crop_size_min, high=bb_crop_size_max)
    bb_crop_start_x_min = np.max((-int(margin * size[0]), bb_old[2] - bb_crop_size))
    bb_crop_start_x_max = np.max((0, bb_old[0] + int((1. - bb_min) * bb_old_size) + 1))
    bb_crop_start_x = np.random.random_integers(low=bb_crop_start_x_min, high=bb_crop_start_x_max)
    bb_crop_start_y_min = np.max((-int(margin * size[1]), bb_old[3] - bb_crop_size))
    bb_crop_start_y_max = np.max((0, bb_old[1] + int((1. - bb_min) * bb_old_size) + 1))
    bb_crop_start_y = np.random.random_integers(low=bb_crop_start_y_min, high=bb_crop_start_y_max)
    bb_crop = [bb_crop_start_x, bb_crop_start_y, bb_crop_start_x + bb_crop_size, bb_crop_start_y + bb_crop_size]
    return np.array(bb_crop)

def sample_bounding_box_scale_balanced(size, landmarks):

    bb_old = image.get_bounding_box(landmarks)
    bb_old_size = np.max((bb_old[2] - bb_old[0], bb_old[3] - bb_old[1]))
    bb_size_min = bb_old_size
    bb_size_max = 14 * bb_old_size
    if np.random.random_sample() > 0.5:
        val = np.random.beta(1.05, 30)
    else:
        val = 1 - np.random.beta(1, 10000)
    bb_crop_size = int(bb_size_min + val * (bb_size_max - bb_size_min))
    bb_crop_start_x = np.random.random_integers(low=np.max((0, bb_old[2] - bb_crop_size)),  high=np.max((0, bb_old[0] + 1)))
    bb_crop_start_y = np.random.random_integers(low=np.max((0, bb_old[3] - bb_crop_size)), high=np.max((0, bb_old[1] + 1)))
    bb_crop = [bb_crop_start_x, bb_crop_start_y, np.min((bb_crop_start_x + bb_crop_size, size[0] - 1)), np.min((bb_crop_start_y + bb_crop_size, size[1] - 1))]
    return np.array(bb_crop)


def sample_bounding_box_scale_balanced_black(landmarks):

    bb_min = 0.9
    bb_old = image.get_bounding_box(landmarks)
    bb_old_shape = np.array((bb_old[2] - bb_old[0], bb_old[3] - bb_old[1]))
    bb_old_size = np.max(bb_old_shape)
    margin = (1 - bb_min) / 2
    bb_old_min = np.round([bb_old[0] + bb_old_shape[0] * margin, bb_old[1] + bb_old_shape[1] * margin, bb_old[2] - bb_old_shape[0] * margin, bb_old[3] - bb_old_shape[1] * margin])
    scale = np.random.random_sample() * 0.94 + 0.08
    bb_crop_size = int(round(bb_old_size / scale))
    bb_crop_start_x = np.random.random_integers(low=bb_old_min[2] - bb_crop_size, high=bb_old_min[0] + 1)
    bb_crop_start_y = np.random.random_integers(low=bb_old_min[3] - bb_crop_size, high=bb_old_min[1] + 1)
    bb_crop_end_x = bb_crop_start_x + bb_crop_size
    bb_crop_end_y = bb_crop_start_y + bb_crop_size
    bb_crop = [bb_crop_start_x, bb_crop_start_y, bb_crop_end_x, bb_crop_end_y]
    return np.array(bb_crop)

def sample_bounding_box_landmarks(landmarks, margin=0.1, random_margin=0.1):

    bb = image.get_bounding_box(landmarks)
    margins = 2 * random_margin * np.random.random_sample(size=4) - random_margin + margin
    bb_size = np.max((bb[2] - bb[0], bb[3] - bb[1]))
    margins *= bb_size
    bb_crop = [bb[0] - margins[0], bb[1] - margins[1],bb[2] + margins[2], bb[3] + margins[3]]
    bb_crop_size = np.max((bb_crop[2] - bb_crop[0], bb_crop[3] - bb_crop[1]))
    bb_crop_center = [(bb_crop[2] + bb_crop[0]) / 2, (bb_crop[3] + bb_crop[1]) / 2]
    bb_crop = [bb_crop_center[0] - bb_crop_size / 2, bb_crop_center[1] - bb_crop_size / 2, bb_crop_center[0] + bb_crop_size / 2, bb_crop_center[1] + bb_crop_size / 2]
    return np.round(bb_crop).astype('int')
