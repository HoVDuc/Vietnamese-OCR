import cv2
import numpy as np
import matplotlib.pyplot as plt
import pyclipper
from shapely.geometry import Polygon

class PostProcess:

    def __init__(self) -> None:
        pass

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [
            points[index_1], points[index_2], points[index_3], points[index_4]
        ]
        return box, min(bounding_box[1])
    
    def unclip(self, box):
        poly = Polygon(box)
        distance = poly.area * self.unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def post_processing(self):
        boxes = []
        for contour in self.contours:
            contour = np.array([contour])
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.002 * peri, True)
            points = approx.reshape((-1, 2))
            if len(points) < 4:
                continue
            box = self.unclip(points).reshape((-1, 1, 2))
            box, _ = self.get_mini_boxes(box)
            box = np.array(box)
            box = box.astype("int32")
            boxes.append(box)
    
        return np.array(boxes)
    
    def cut_img(self, img, contour):
        x1, x2, x3, x4 = contour
        x = min(x1[0], x4[0])
        y = min(x1[1], x2[1])
        w = max(x2[0], x3[0]) - x
        h = max(x4[1], x3[1]) - y
        img = img[y:y+h, x:x+w]
        
        return img if np.any(img) else None
        
    def get_img(self):
        imgs = []
        for contour in self.contours:
            im = self.cut_img(self.image, contour)
            if not im is None:
                imgs.append(im)
        return imgs
    
    def __call__(self, image, contours, unclip_ratio=0.6):
        self.image = image
        self.contours = contours
        self.unclip_ratio = unclip_ratio

        self.contours = self.post_processing()
        return self.get_img()
