import csv
import numpy as np
import keras.backend as Ker

image_size = 224
image_shape = (image_size, image_size, 3)
L_mouth = 2
ear_right = 3
ear_left = 4
eye_right = 0
eye_left = 1

def get_loss_fn(output_type, name, iou_and_mse_landmarks_ratio=None):
    if output_type == 'landmarks':
        if not name == 'mse':
            print('Loss_fn "%s" not available in landmarks training. Forcing loss_fn to be "mse".' % name)
        return 'mse'
    loss_fn_map = {'mse': 'mse',
                   'iou': iou_loss}
    if name == 'iou_and_mse_landmarks':
        assert iou_and_mse_landmarks_ratio is not None
        loss_fn = get_iou_and_mse_landmarks_loss(iou_and_mse_landmarks_ratio)
    else:
        loss_fn = loss_fn_map[name]

    return loss_fn

def get_custom_objects(loss_name=None, loss_fn=None):
    custom_objects = {'iou': iou, 'iou_loss': iou_loss}
    if loss_name == 'iou_and_mse_landmarks':
        custom_objects['iou_and_mse_landmarks_loss'] = loss_fn
    return custom_objects

def iou(y_true, y_pred):

    y_true = Ker.permute_dimensions(y_true, (1, 0))
    y_pred = Ker.permute_dimensions(y_pred, (1, 0))
    x_0 = Ker.max([Ker.gather(y_true, 0), Ker.gather(y_pred, 0)], axis=0)
    y_0 = Ker.max([Ker.gather(y_true, 1), Ker.gather(y_pred, 1)], axis=0)
    x_1 = Ker.min([Ker.gather(y_true, 2), Ker.gather(y_pred, 2)], axis=0)
    y_1 = Ker.min([Ker.gather(y_true, 3), Ker.gather(y_pred, 3)], axis=0)
    area_inter = Ker.clip(x_1 - x_0, 0, None) * Ker.clip(y_1 - y_0, 0, None)
    area_true = (Ker.gather(y_true, 2) - Ker.gather(y_true, 0)) * (Ker.gather(y_true, 3) - Ker.gather(y_true, 1))
    area_pred = (Ker.gather(y_pred, 2) - Ker.gather(y_pred, 0)) * (Ker.gather(y_pred, 3) - Ker.gather(y_pred, 1))
    iou_ = area_inter / (area_true + area_pred - area_inter)
    return Ker.mean(iou_, axis=-1)

def iou_loss(y_true, y_pred):
    return 1. - iou(y_true, y_pred)

def get_iou_and_mse_landmarks_loss(ratio):
    def iou_and_mse_landmarks_loss(y_true, y_pred):
        iou_ = iou_loss(y_true, y_pred)
        y_true = Ker.permute_dimensions(y_true, (1, 0))
        y_pred = Ker.permute_dimensions(y_pred, (1, 0))
        y_true = Ker.gather(y_true, np.arange(4, 12))
        y_pred = Ker.gather(y_pred, np.arange(4, 12))
        mse = Ker.mean(Ker.square(y_pred - y_true), axis=0)
        return iou_ + mse * ratio
    return iou_and_mse_landmarks_loss

def append_hp_result(path, exp_name, args, history, test_metrics, monitor, mode):
    try:
        with open(path, 'r') as f:
            csv_reader = csv.reader(f, delimiter=';', lineterminator='\n')
            header = next(csv_reader)
    except FileNotFoundError:
        header = ['exp_name'] + list(args) + ['best_ep'] + list(history) + list(test_metrics)
        with open(path, 'w') as f:
            csv_writer = csv.writer(f, delimiter=';', lineterminator='\n')
            csv_writer.writerow(header)
    if mode == 'max':
        best_ep = np.argmax(history[monitor])
    else:
        best_ep = np.argmin(history[monitor])
    pool = {k: v[best_ep] for k, v in history.items()}
    pool['best_ep'] = best_ep
    pool['exp_name'] = exp_name
    pool.update(test_metrics)
    row = [args[h] if h in args.keys() else pool[h] if h in pool else '' for h in header]
    with open(path, 'a') as f:
        csv_writer = csv.writer(f, delimiter=';', lineterminator='\n')
        csv_writer.writerow(row)
