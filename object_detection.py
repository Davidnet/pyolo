"""
File that contains the class for initialazing the fast object detector
"""

import sys, os, shutil, subprocess
import numpy as np
import pyyolo
import cv2
import tfinterface as ti

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
URL_PATH = "gs://vision-198622-production-models/object-detection/tiny-yolo/tiny-yolo.weights"


class TinyYOLODetector(ti.estimator.getters.FileGetter):
    """Class detection for YOLO objects in general"""
    def __init__(self, weightfile, thresh=0.24, hier_thresh=0.5, *args, **kwargs):
        #darknet_path = './darknet'
        darknet_path = os.path.join(DIR_PATH, "darknet")
        #datacfg = 'cfg/coco.data'
        datacfg = os.path.join(darknet_path, "cfg", "coco.data")
        #cfgfile = 'cfg/tiny-yolo.cfg'
        cfgfile = os.path.join(darknet_path, "cfg", "tiny-yolo.cfg")
        #weightfile = '../tiny-yolo.weights'
        self.thresh = thresh
        self.hier_thresh = hier_thresh
        pyyolo.init(darknet_path, datacfg, cfgfile, weightfile)



    def _distance_approximator(self, box):
        """
        Function to calculate distance
        """
        ymin, xmin, ymax, xmax = box
        mid_x = (xmax + xmin) / (2*self.w)
        mid_y = (ymax + ymin) / (2*self.h)  # TODO: use mid_y
        apx_distance = round((1 - (xmax - xmin)) ** 4, 1)
        return apx_distance

    def _parser_raw_output(self, raw_output):
        """
        Convert YOLO predictions into visual tools conventions
        """
        keys = ["top", "left", "bottom", "right"]
        preds = raw_output.get("prob")
        id_class = raw_output.get("class")
        box = list(map(raw_output.get, keys))
        distance = self._distance_approximator(box)
        return dict(box=box,
                    distance=distance,
                    id=len(id_class),
                    name=id_class,
                    prob=preds
                   )

    def _single_predict(self, img_np):
        """Prediction function"""
        img_np = img_np.transpose(2, 0, 1)
        self.c, self.h, self.w = img_np.shape
        data = img_np.ravel()/255.0
        data = np.ascontiguousarray(data, dtype=np.float32)
        raw_outputs = pyyolo.detect(self.w, self.h, self.c, data, self.thresh, self.hier_thresh)
        return list(map(self._parser_raw_output, raw_outputs))

    def predict(self, image=None):
        """ Batch Prediction """
        if image is None:
            raise ValueError("Keyword argument image was not found")
        else:
            return list(map(self._single_predict, image))[0]

    def __del__(self):
        pyyolo.cleanup()
