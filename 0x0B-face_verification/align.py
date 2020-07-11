#!/usr/bin/env python3
"""
Face Align script
"""

import numpy as np
import dlib
import cv2


class FaceAlign:
    """
    Class FaceAlign
    """
    def __init__(self, shape_predictor_path):
        self.detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(shape_predictor_path)

    def detect(self, image):
        """
        Detects a face in an image
        """
        rects = self.detector(image)

        if not len(rects):
            return dlib.get_rect(image)

        max_area = [None, 0]
        for rect in rects:
            if rect.area() > max_area[1]:
                max_area = [rect, rect.area()]
        return max_area[0]

    def find_landmarks(self, image, detection):
        """ numpy.ndarray of shape (p, 2)containing the
                 landmark points, or None on failure
        """
        try:
            shape = self.shape_predictor(image, detection)
            coord = np.zeros((shape.num_parts, 2), dtype="int")
            for i in range(0, shape.num_parts):
                coord[i] = (shape.part(i).x, shape.part(i).y)

            return coord
        except Exception:
            return None

    def align(self, image, landmark_indices, anchor_points, size=96):
        """ numpy.ndarray of shape (size, size, 3) containing the aligned
        image, or None if no face is detected
        """
        rect = self.detect(image)
        shape = self.find_landmarks(image, rect)
        landmark = shape[landmark_indices]
        M = cv2.getAffineTransform(landmark.astype(np.float32),
                                   (anchor_points * size).astype(np.float32))

        alg = cv2.warpAffine(image, M, (size, size))
        return alg
