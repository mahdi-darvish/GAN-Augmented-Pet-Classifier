import os
import argparse

from pet.utils import image
from pet.predictor import Predictor


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path')
	parser.add_argument('--landmarks_model_path')
    args = parser.parse_args()
    if args.pooling == 'None':
        args.pooling = None

img_path = args.img_path
landmarks_model_path = args.landmarks_model_path
img = image.load_image(img_path)
predictor = Predictor(landmarks_model_path=landmarks_model_path)
landmarks = predictor.predict(img)
image.save_landmarks(landmarks, img_path + '.cat')
image.draw_landmarks(img, landmarks)
image.save_image(img, img_path + '.ldmrk.png')	