import os
import sys
from PIL import Image
import numpy as np
from pet import Predictor
from pet.animal_data_generator import animal_generator
import pet.utils.image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if __name__ == '__main__':
    data_path = os.path.join('..', '..', 'animal-dataset', 'data', 'clean')
    path_train = os.path.join(data_path, 'training')
    path_val_bbox = os.path.join(data_path, 'validation')
    path_test_bbox = os.path.join(data_path, 'test')
    path_val_landmarks = os.path.join(data_path, 'landmarks_validation')
    path_test_landmarks = os.path.join(data_path, 'landmarks_test')
    test_validation_args = dict(include_landmarks=False, batch_size=32, shuffle=False, flip_horizontal=False, rotate=False, rotate_90=False, rotate_n=0, crop_scale_balanced_black=False, crop_scale_balanced=False, sampling_method_resize=Image.LANCZOS)
    datagen_train_bbox = animal_generator(path=path_train, output_type='bbox', crop=False, **test_validation_args)
    datagen_val_bbox = animal_generator(path=path_val_bbox, output_type='bbox', crop=False, **test_validation_args)
    datagen_test_bbox = animal_generator(path=path_test_bbox, output_type='bbox', crop=False, **test_validation_args)
    datagen_train_landmarks = animal_generator(path=path_train, output_type='landmarks', crop=True, crop_landmarks_random_margin=0., **test_validation_args)
    datagen_val_landmarks = animal_generator(path=path_val_landmarks, output_type='landmarks', crop=False, **test_validation_args)
    datagen_test_landmarks = animal_generator(path=path_test_landmarks, output_type='landmarks', crop=False, **test_validation_args)
    predictor = Predictor(lazy=False)
    predictor.bbox_model.compile(optimizer='sgd', loss='mse', metrics=predictor.bbox_model.metrics + ['mae'])
    predictor.landmarks_model.compile(optimizer='sgd', loss='mse', metrics=predictor.landmarks_model.metrics + ['mae'])

    def bbox_mape(datagen):
        mapes = []
        for xs, ys in datagen:
            ys_pred = predictor.bbox_model.predict(xs, verbose=1)[:, :4]
            for y, y_pred in zip(ys, ys_pred):
                bbox_size = np.max(np.diff(y.reshape((-1, 2)), axis=0))
                mape = np.mean(np.abs(y - y_pred) / bbox_size * 100.)
                mapes.append(mape)
        return np.mean(mapes)

    def landmarks_mape(datagen):
        mapes = []
        for xs, ys in datagen:
            ys_pred = predictor.landmarks_model.predict(xs, verbose=1)
            for y, y_pred in zip(ys, ys_pred):
                bb = pet.utils.image.get_bounding_box(y.reshape((-1, 2)))
                face_size = np.max(np.diff(bb.reshape((-1, 2)), axis=0))
                mape = np.mean(np.abs(y - y_pred) / face_size * 100.)
                mapes.append(mape)
        return np.mean(mapes)

    results_train_bbox = predictor.bbox_model.evaluate_generator(datagen_train_bbox, verbose=1)
    results_train_bbox_mape = bbox_mape(datagen_train_bbox)
    results_val_bbox = predictor.bbox_model.evaluate_generator(datagen_val_bbox, verbose=1)
    results_val_bbox_mape = bbox_mape(datagen_val_bbox)
    results_test_bbox = predictor.bbox_model.evaluate_generator(datagen_test_bbox, verbose=1)
    results_test_bbox_mape = bbox_mape(datagen_test_bbox)
    results_train_landmarks = predictor.landmarks_model.evaluate_generator(datagen_train_landmarks, verbose=1)
    results_train_landmarks_mape = landmarks_mape(datagen_train_landmarks)
    results_val_landmarks = predictor.landmarks_model.evaluate_generator(datagen_val_landmarks, verbose=1)
    results_val_landmarks_mape = landmarks_mape(datagen_val_landmarks)
    results_test_landmarks = predictor.landmarks_model.evaluate_generator(datagen_test_landmarks, verbose=1)
    results_test_landmarks_mape = landmarks_mape(datagen_test_landmarks)
    print('train landmarks rmse %.2f' % np.sqrt(results_train_landmarks[0]))
    print('validation landmarks rmse %.2f' % np.sqrt(results_val_landmarks[0]))
    print('test landmarks rmse %.2f' % np.sqrt(results_test_landmarks[0]))
