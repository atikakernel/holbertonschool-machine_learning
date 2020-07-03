#!/usr/bin/env python3
"""
This file contain the Yolo class
"""
import tensorflow.keras as K
import numpy as np


class Yolo():
    """
    Yolo v3 algorithm
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        class constructor
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            data = f.read()
        self.class_names = data.split()
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def _sigmoid(self, x):
        """
        Perform sigmoid function of a vector.
        """
        return 1. / (1. + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """
        Process Outputs tuple of (boxes, box_confidences, box_class_probs)
        """

        boxes, box_confidences, box_class_probs = [], [], []

        for i in range(len(outputs)):
            ih, iw = image_size
            t_xy, t_wh, objectness, classes = np.split(outputs[i], (2, 4, 5),
                                                       axis=-1)

            box_confidences.append(_sigmoid(objectness))
            box_class_probs.append(_sigmoid(classes))

            grid_size = np.shape(outputs[i])[1]
            C_xy = np.meshgrid(range(grid_size), range(grid_size))
            C_xy = np.stack(C_xy, axis=-1)

            C_xy = np.expand_dims(C_xy, axis=2)
            b_xy = _sigmoid(t_xy) + C_xy

            b_xy = b_xy / grid_size

            inp = self.model.input_shape[1:3]
            b_wh = (np.exp(t_wh) / inp) * self.anchors[i]

            bx = b_xy[:, :, :, :1]
            by = b_xy[:, :, :, 1:2]
            bw = b_wh[:, :, :, :1]
            bh = b_wh[:, :, :, 1:2]

            x1 = (bx - bw / 2) * image_size[1]
            y1 = (by - bh / 2) * image_size[0]
            x2 = (bx + bw / 2) * image_size[1]
            y2 = (by + bh / 2) * image_size[0]

            boxes.append(np.concatenate([x1, y1, x2, y2], axis=-1))

        return boxes, box_confidences, box_class_probs
