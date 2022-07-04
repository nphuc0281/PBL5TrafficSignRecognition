import torch
import cv2
from .Camera import Camera


class TrafficSignDetection:
    def __init__(self, path_to_weight, cam_mode=None):
        self.model = self.load_model(path_to_weight)
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.cam = Camera(cam_mode)
        self.camera = self.cam.cam
        print('Device used', self.device)

    @staticmethod
    def load_model(path_to_weight):
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=path_to_weight)
        return model

    def score_frame(self, frame):
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def class_to_label(self, x):
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        new_labels = []
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.3:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, bgr, 2)
                new_labels.append(self.class_to_label(labels[i]))

        return frame, new_labels
