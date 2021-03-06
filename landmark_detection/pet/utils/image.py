import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .general import image_size, eye_left, eye_right, L_mouth, ear_left, ear_right

def load(path):
    img = load_image(path)
    landmarks = load_landmarks(path + '.animal')
    return img, landmarks

def load_image(path):
    return Image.open(path)

def load_landmarks(path):
    with open(path, 'r') as animal:
        landmarks = np.array([float(i) for i in animal.readline().split()[1:]]).reshape((-1, 2))
    return landmarks

def save_landmarks(landmarks, path):
    with open(path, 'w') as animal:
        animal.write(' '.join([str(int(landmarks.shape[0]))] + [str(l) for l in landmarks.flatten()]))

def get_bounding_box(landmarks):
    return np.concatenate([np.min(landmarks, axis=0), np.max(landmarks, axis=0)])

def postprocess_bounding_box(bb, image_size, margin=0.1):
    ratio = float(image_size) / max(image_size)
    new_size = tuple(int(x * ratio) for x in image_size)
    x_diff = (image_size - new_size[0]) // 2
    y_diff = (image_size - new_size[1]) // 2
    bb -= np.array((x_diff, y_diff, x_diff, y_diff))
    bb /= ratio
    bb_size = np.max((bb[2] - bb[0], bb[3] - bb[1]))
    margin *= bb_size
    bb_crop = [bb[0] - margin, bb[1] - margin,bb[2] + margin, bb[3] + margin]
    bb_crop_size = np.max((bb_crop[2] - bb_crop[0], bb_crop[3] - bb_crop[1]))
    bb_crop_center = [(bb_crop[2] + bb_crop[0]) / 2, (bb_crop[3] + bb_crop[1]) / 2]
    bb_crop = [bb_crop_center[0] - bb_crop_size / 2, bb_crop_center[1] - bb_crop_size / 2, bb_crop_center[0] + bb_crop_size / 2, bb_crop_center[1] + bb_crop_size / 2]
    return np.round(bb_crop).astype('int')

def rotate(img, landmarks, angle, expand=True, sampling_method='random'):
    if angle in (0, 360):
        return img, landmarks
    radians = np.radians(angle)
    offset_x, offset_y = img.size[0] / 2, img.size[1] / 2
    adjusted_x = landmarks[:, 0] - offset_x
    adjusted_y = landmarks[:, 1] - offset_y
    cos_rad = np.cos(radians)
    sin_rad = np.sin(radians)
    qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
    qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y
    landmarks = np.array([qx, qy]).T
    old_size = img.size
    if angle == 90:
        img = img.transpose(Image.ROTATE_90)
    elif angle == 180:
        img = img.transpose(Image.ROTATE_180)
    elif angle == 270:
        img = img.transpose(Image.ROTATE_270)
    else:
        if sampling_method == 'random':
            sampling_method = np.random.choice([Image.NEAREST, Image.BILINEAR, Image.BICUBIC])
        img = img.rotate(angle, expand=expand, resample=sampling_method)
    landmarks[:, 0] += (img.size[0] - old_size[0]) / 2
    landmarks[:, 1] += (img.size[1] - old_size[1]) / 2
    return img, landmarks

def resize(img, landmarks, sampling_method='random'):
    old_size = img.size
    if old_size != (image_size, image_size):
        ratio = float(image_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        if sampling_method == 'random':
            sampling_method = np.random.choice([Image.NEAREST, Image.BOX, Image.BILINEAR, Image.HAMMING, Image.BICUBIC, Image.LANCZOS])
        old_img = img.resize(new_size, sampling_method)
        img = Image.new('RGB', (image_size, image_size))
        x_diff = (image_size - new_size[0]) // 2
        y_diff = (image_size - new_size[1]) // 2
        img.paste(old_img, (x_diff, y_diff))
        landmarks *= ratio
        landmarks += np.array((x_diff, y_diff))
    return img, landmarks

def flip(img, landmarks):
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    landmarks[:, 0] = img.size[0] - landmarks[:, 0]
    for a, b in ((eye_left, eye_right), (ear_left, ear_right)):
        tmp = landmarks[a].copy()
        landmarks[a] = landmarks[b]
        landmarks[b] = tmp
    return img, landmarks

def crop(img, landmarks, bounding_box):
    img = img.crop(bounding_box)
    landmarks -= bounding_box[:2]
    return img, landmarks

def draw_landmarks(img, landmarks, color='yellow', lines=True, lines_color='green', width=2):
    draw = ImageDraw.Draw(img)
    fnt = ImageFont.load_default()
    if lines:
        def draw_line(a, b):
            return draw.line((tuple(landmarks[a]), tuple(landmarks[b])), fill=lines_color, width=width)
        draw_line(eye_left, eye_right)
        draw_line(eye_right, L_mouth)
        draw_line(L_mouth, eye_left)
        draw_line(ear_left, eye_left)
        draw_line(ear_left, eye_right)
        draw_line(ear_right, eye_left)
        draw_line(ear_right, eye_right )
        draw_line(ear_right, ear_left)
    for i_lnd, lnd in enumerate(landmarks):
        draw.ellipse(((lnd[0] - width, lnd[1] - width), (lnd[0] + width, lnd[1] + width)), fill=color)
        draw.text((lnd[0] + width, lnd[1] + width), str(i_lnd), font=fnt, fill=color)

def save_image(img, path):
    img.save(path)

def save_with_landmarks(img, path, landmarks_truth=(), landmarks_predicted=()):
    img = img.copy()
    draw_landmarks(img, landmarks_predicted)
    draw_landmarks(img, landmarks_truth, color='red', lines=False)
    save_image(img, path)
