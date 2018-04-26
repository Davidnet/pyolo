"""
File that contains the class for initialazing the fast object detector
"""
import sys, os
import numpy as np
import pyyolo
import cv2

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

class TinyYOLODetector(object):
    def __init__(self, thresh=0.24, hier_thresh=0.5):
        #darknet_path = './darknet'
        darknet_path = os.path.join(DIR_PATH, "darknet")
        #datacfg = 'cfg/coco.data'
        datacfg = os.path.join(darknet_path, "cfg", "coco.data")
        #cfgfile = 'cfg/tiny-yolo.cfg'
        cfgfile = os.path.join(darknet_path, "cfg", "tiny-yolo.cfg")
        #weightfile = '../tiny-yolo.weights'
        weightfile = os.path.join(DIR_PATH, "tiny-yolo.weights")
        self.thresh = thresh
        self.hier_thresh = hier_thresh
        pyyolo.init(darknet_path, datacfg, cfgfile, weightfile)

    def _parser_raw_output(self, raw_output):
        """
        Convert YOLO predictions into visual tools conventions
        """
        keys = ["top", "left", "bottom", "right"]
        preds = raw_output.get("prob")
        id_class = raw_output.get("class")
        return dict(box=list(map(raw_output.get, keys)),
                    id=len(id_class),
                    name=id_class,
                    prob=preds
                   )

    def _single_predict(self, img_np):
        """Prediction function"""
        img_np = img_np.transpose(2, 0, 1)
        c, h, w = img_np.shape
        data = img_np.ravel()/255.0
        data = np.ascontiguousarray(data, dtype=np.float32)
        raw_outputs = pyyolo.detect(w, h, c, data, self.thresh, self.hier_thresh)
        return list(map(self._parser_raw_output, raw_outputs))

    def predict(self, image):
        """ Batch Prediction """
        return list(map(self._single_predict, image))[0] 

    def __del__(self):
        pyyolo.cleanup()
