#!/usr/bin/env python3
"""
This file contain the Yolo class
"""
import tensorflow.keras as K
import numpy as np
import cv2
import glob
import os


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

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """Perform non max suppression
        """
        tmp_boxes = []
        tmp_classes = []
        tmp_scores = []

        for clase in np.unique(box_classes):
            indexes = np.where(box_classes == clase)
            boxes_ofclas = filtered_boxes[indexes]
            classes_ofclas = box_classes[indexes]
            scores_ofclas = box_scores[indexes]

            x1 = boxes_ofclas[:, 0]
            y1 = boxes_ofclas[:, 1]
            x2 = boxes_ofclas[:, 2]
            y2 = boxes_ofclas[:, 3]

            areas = (x2 - x1 + 1) * (y2 - y1 + 1)
            order = scores_ofclas.argsort()[::-1]

            keep = []
            while order.size > 0:
                i = order[0]
                keep.append(i)
                xx1 = np.maximum(x1[i], x1[order[1:]])
                yy1 = np.maximum(y1[i], y1[order[1:]])
                xx2 = np.minimum(x2[i], x2[order[1:]])
                yy2 = np.minimum(y2[i], y2[order[1:]])

                w = np.maximum(0.0, xx2 - xx1 + 1)
                h = np.maximum(0.0, yy2 - yy1 + 1)
                inter = w * h
                ovr = inter / (areas[i] + areas[order[1:]] - inter)

                inds = np.where(ovr <= self.nms_t)[0]
                order = order[inds + 1]

            tmp_boxes.append(boxes_ofclas[keep])
            tmp_classes.append(classes_ofclas[keep])
            tmp_scores.append(scores_ofclas[keep])

        boxes_predic = np.concatenate(tmp_boxes, axis=0)
        classes_predic = np.concatenate(tmp_classes, axis=0)
        scores_predic = np.concatenate(tmp_scores, axis=0)

        return boxes_predic, classes_predic, scores_predic

    @staticmethod
    def load_images(folder_path):
        """ This method load an image """
        images = [cv2.imread(file)
                  for file in glob.glob(folder_path + '/*.jpg')]
        paths = [file for file in glob.glob(folder_path + '/*.jpg')]
        return images, paths

    def preprocess_images(self, images):
        """ Preprocess images to posterior use with darknet"""
        resized = []
        dims = self.model.input_shape[1:3]
        for img in images:
            resized.append(cv2.resize(img, dims,
                                      interpolation=cv2.INTER_CUBIC) / 255)
        shapes = []
        for img in images:
            shapes.append(img.shape[:2])
        return np.array(resized), np.array(shapes)

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """
            image: a numpy.ndarray containing an unprocessed image
            boxes: a numpy.ndarray containing the boundary boxes for the image
            box_classes: a numpy.ndarray containing the class indices
                for each box
        """
        img = image
        for i, box in enumerate(boxes):
            color_ln = (255, 0, 0)
            color_tx = (0, 0, 255)
            start_point = (int(box[0]), int(box[3]))
            end_point = (int(box[2]), int(box[1]))
            start_text = (int(box[0]), int(box[1])-5)
            img = cv2.rectangle(img,
                                start_point,
                                end_point,
                                color_ln,
                                thickness=2)
            img = cv2.putText(img,
                              self.class_names[box_classes[i]] +
                              " " + "{:.2f}".format(box_scores[i]),
                              start_text,
                              cv2.FONT_HERSHEY_SIMPLEX,
                              0.5,
                              color_tx,
                              thickness=1,
                              lineType=cv2.LINE_AA,
                              bottomLeftOrigin=False)
        cv2.imshow(file_name, image)
        key = cv2.waitKey(0)
        if key == ord('s'):
            if not os.path.exists('detections'):
                os.makedirs('detections')
            os.chdir('detections')
            cv2.imwrite(file_name, img)
            os.chdir('../')
        cv2.destroyAllWindows()

    def predict(self, folder_path):
        """
        Object detect over all images in a folder
        """
        originals, file_names = self.load_images(folder_path)
        images, origin_sizes = self.preprocess_images(originals)
        predicts = self.model.predict(images)
        for idx in range(predicts[0].shape[0]):
            outputs = [predict[idx, ...] for predict in predicts]
            processed = self.process_outputs(outputs, origin_sizes[idx])
            threshed = self.filter_boxes(*processed)
            non_maxed = self.non_max_suppression(*threshed)
            self.show_boxes(originals[idx], *non_maxed, file_names[idx])
        outnames = ['detections/' + os.path.split(file)[1]
                    for file in file_names]
        return non_maxed, outnames
