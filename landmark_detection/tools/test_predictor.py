import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
import datetime
from collections import defaultdict
from pet import Predictor
import pet.utils.image
from pet.utils.general import eye_left, eye_right, L_mouth, ear_left, ear_right

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default=os.path.join('..', '..', 'animal-dataset', 'data', 'clean', 'test'))
    args = parser.parse_args()
    predictor = Predictor()
    output_dir = os.path.join('output', datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
    os.makedirs(output_dir)
    metrics = defaultdict(list)
    for img_filename in tqdm([f for f in os.listdir(args.data_path) if f[-4:] in ('.jpg', '.bmp', '.gif', '.png')]):
        img_path = os.path.join(args.data_path, img_filename)
        img, landmarks_truth = pet.utils.image.load(img_path)
        landmarks_predicted = predictor.predict(img)
        bb_truth = pet.utils.image.get_bounding_box(landmarks_truth)
        face_size = np.max(np.diff(bb_truth.reshape((-1, 2)), axis=0))

        def get_mape(a, b):
            return np.mean(np.abs((landmarks_truth[a: b + 1] - landmarks_predicted[a: b + 1]) / face_size * 100.))

        err = landmarks_truth - landmarks_predicted
        metrics['mae'].append(np.mean(np.abs(err)))
        metrics['mse'].append(np.mean(np.square(err)))
        metrics['mspe'].append(np.mean(np.square(err / face_size * 100.)))
        mape = np.mean(np.abs(err / face_size * 100.))
        metrics['mape'].append(mape)
        metrics['mape eyes'].append(get_mape(eye_right, eye_left))
        metrics['mape mouth'].append(get_mape(L_mouth, L_mouth))
        metrics['mape ears'].append(get_mape(ear_right, ear_left))
        output_filename = '%.9f_%s' % (mape, img_filename)
        output_path = os.path.join(output_dir, output_filename)
        pet.utils.image.save_with_landmarks(img, output_path, landmarks_truth, landmarks_predicted)
        pet.utils.image.save_landmarks(landmarks_predicted, output_path + '.animal')

    for name, vals in metrics.items():
        print('%s:\t%.2f' % (name, np.mean(vals)))
        if name.startswith('ms'):
            print('r%s:\t%.2f' % (name, np.sqrt(np.mean(vals))))
