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

    def nms(self, bc, thresh, scores):
        """
        Function that computes the index
        Sorted index score for non max supression
        """
        x1 = bc[:, 0]
        y1 = bc[:, 1]
        x2 = bc[:, 2]
        y2 = bc[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

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
            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
        return keep

    @staticmethod
    def load_images(folder_path):
        """
        Function to load images using cv2
        """
        images = []
        images_paths = glob.glob(folder_path + '/*', recursive=False)
        for i in images_paths:
            image = cv2.imread(i)
            images.append(image)
        return images, images_paths

    def preprocess_images(self, images):
        """
        Resize and rescale images.
        """
        image_sizes = np.zeros((len(images), 2))
        target_height = self.model.input.shape[1].value
        target_width = self.model.input.shape[2].value
        proc_images = np.zeros((len(images), target_height, target_width, 3))
        for idx, image in enumerate(images):
            image_sizes[idx][0] = image.shape[0]
            image_sizes[idx][1] = image.shape[1]
            proc_images[idx] = cv2.resize(image,
                                          (target_height, target_width),
                                          interpolation=cv2.INTER_CUBIC)
            proc_images[idx] -= proc_images[idx].min()
            proc_images[idx] /= proc_images[idx].max()
        return proc_images, image_sizes.astype(int)

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
