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

    def process_outputs(self, outputs, image_size):
        """Process Darknet outputs
        returns
        boxes: the boundary boxes
        """
        boxes = [output[:, :, :, 0:4] for output in outputs]
        for oidx, output in enumerate(boxes):
            for y in range(output.shape[0]):
                for x in range(output.shape[1]):
                    centery = ((1 / (1 + np.exp(-output[y, x, :, 1])) + y)
                               / output.shape[0] * image_size[0])
                    centerx = ((1 / (1 + np.exp(-output[y, x, :, 0])) + x)
                               / output.shape[1] * image_size[1])
                    prior_resizes = self.anchors[oidx].astype(float)
                    prior_resizes[:, 0] *= (np.exp(output[y, x, :, 2])
                                            / 2 * image_size[1] /
                                            self.model.input.shape[1].value)
                    prior_resizes[:, 1] *= (np.exp(output[y, x, :, 3])
                                            / 2 * image_size[0] /
                                            self.model.input.shape[2].value)
                    output[y, x, :, 0] = centerx - prior_resizes[:, 0]
                    output[y, x, :, 1] = centery - prior_resizes[:, 1]
                    output[y, x, :, 2] = centerx + prior_resizes[:, 0]
                    output[y, x, :, 3] = centery + prior_resizes[:, 1]
        box_confidences = [1 / (1 + np.exp(-output[:, :, :, 4, np.newaxis]))
                           for output in outputs]
        box_class_probs = [1 / (1 + np.exp(-output[:, :, :, 5:]))
                           for output in outputs]
        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filter box outputs into more compact forms and remove results under
        class threshold.
        """
        all_boxes = np.concatenate([boxs.reshape(-1, 4) for boxs in boxes])
        class_probs = np.concatenate([probs.reshape(-1,
                                                    box_class_probs[0].
                                                    shape[-1])
                                      for probs in box_class_probs])
        all_classes = class_probs.argmax(axis=1)
        all_confidences = (np.concatenate([conf.reshape(-1)
                                           for conf in box_confidences])
                           * class_probs.max(axis=1))
        thresh_idxs = np.where(all_confidences < self.class_t)
        return (np.delete(all_boxes, thresh_idxs, axis=0),
                np.delete(all_classes, thresh_idxs),
                np.delete(all_confidences, thresh_idxs))
