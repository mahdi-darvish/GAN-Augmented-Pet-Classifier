from keras.models import load_model
from PIL import Image
import pet.utils.general
import pet.utils.image
from keras.applications import mobilenet_v2
import numpy as np
from keras.utils.data_utils import get_file

base_url = None    # insert your weights here

class Predictor:
    def __init__(self, bbox_model_path=None, landmarks_model_path=None, lazy=True):
        self.bbox_model_path = bbox_model_path
        self.landmarks_model_path = landmarks_model_path
        self.lazy = lazy
        self.loaded = False
        self.bbox_model = None
        self.landmarks_model = None
        if not lazy:
            self.load_models()

    def load_models(self):
        if not self.loaded:
            if self.bbox_model_path is None:
                model_name = 'pet_bbox.h5'
                self.bbox_model_path = get_file(model_name, base_url + model_name, cache_subdir='models')
            if self.landmarks_model_path is None:
                model_name = 'pet_landmarks.h5'
                self.landmarks_model_path = get_file(model_name, base_url + model_name, cache_subdir='models')
            dummy_loss_fn = pet.utils.general.get_loss_fn('bbox', 'iou_and_mse_landmarks', 1e-5)
            custom_objects = pet.utils.general.get_custom_objects('iou_and_mse_landmarks', dummy_loss_fn)
            self.bbox_model = load_model(self.bbox_model_path, custom_objects=custom_objects)
            self.landmarks_model = load_model(self.landmarks_model_path, custom_objects=custom_objects)
            self.loaded = True

    def predict(self, img, dtype='float'):
        self.load_models()
        img_bbox, img_landmarks = img.copy(), img.copy()
        img_bbox, _ = pet.utils.image.resize(img_bbox, 0, sampling_method=Image.LANCZOS)
        x = np.expand_dims(mobilenet_v2.preprocess_input(np.asarray(img_bbox)), axis=0)
        bbox = self.bbox_model.predict(x, verbose=0)[0, :4]
        bbox = pet.utils.image.postprocess_bounding_box(bbox, img.size)
        img_landmarks, _ = pet.utils.image.crop(img_landmarks, 0, bbox)
        img_landmarks, _ = pet.utils.image.resize(img_landmarks, 0, sampling_method=Image.LANCZOS)
        x = np.expand_dims(mobilenet_v2.preprocess_input(np.asarray(img_landmarks)), axis=0)
        y_pred = self.landmarks_model.predict(x, verbose=0)[0]
        bb_size = np.max((bbox[2] - bbox[0], bbox[3] - bbox[1]))
        ratio = bb_size / pet.utils.general.IMG_SIZE
        predicted_landmarks = (y_pred * ratio).reshape((-1, 2)) + bbox[:2]
        if 'int' in dtype:
            predicted_landmarks = np.round(predicted_landmarks)
        return predicted_landmarks.astype(dtype)
