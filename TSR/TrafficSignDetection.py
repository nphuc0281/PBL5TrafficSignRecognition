import torch
import cv2
from .Camera import Camera


class TrafficSignDetection:
    """
    Class implement YOLOv5 model to detect traffic sign from camera
    """

    def __init__(self, out_file, path_to_weight, cam_mode=None):
        self.model = self.load_model(path_to_weight)
        self.classes = self.model.names
        print(self.classes)
        self.out_file = out_file
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.cam = Camera(cam_mode)
        self.camera = self.cam.cam
        print('Device used', self.device)

    @staticmethod
    def load_model(path_to_weight):
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=path_to_weight)
        return model

    def score_frame(self, frame):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]

        for i in range(n):
            row = cord[i]
            if row[4] >= 0.3:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

        return frame
