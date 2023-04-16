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
    
    def get_size(self, contour):
        x1, x2, x3, x4 = contour
        x = min(x1[0], x4[0])
        y = min(x1[1], x2[1])
        w = max(x2[0], x3[0]) - x
        h = max(x4[1], x3[1]) - y
        return h, w
        
    def get_img(self):
        imgs = []
        for contour in self.contours:
            im = self.cut_img(self.image, contour)
            if not im is None:
                imgs.append(im)
        return imgs
    
    def reorder(self, points):
        points = points.reshape((4, 2))
        new_points = np.zeros((4, 1, 2), dtype=np.int32)
        add = points.sum(1)
        new_points[0] = points[np.argmin(add)]
        new_points[3] = points[np.argmax(add)]
        diff = np.diff(points, axis=1)
        new_points[1] = points[np.argmin(diff)]
        new_points[2] = points[np.argmax(diff)]
        return new_points
    
    def get_Perspective(self, img, masked, location, size: tuple):
        """Takes original image as input"""
        height, width = size
        pts1 = np.float32(location)
        pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
        
        # Apply Perspective Transform Algorithm
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        result = cv2.warpPerspective(masked, matrix, (width, height))
        return result
    
    def __call__(self, image, contours, unclip_ratio=0.6):
        self.image = image
        self.contours = contours
        self.unclip_ratio = unclip_ratio

        self.contours = self.post_processing()
        imgs = []
        for contour in self.contours:
            size = self.get_size(contour)
            location = self.reorder(contour)
            imgWrap = self.get_Perspective(self.image, self.image, location, size)
            imgs.append(imgWrap)

        return imgs
