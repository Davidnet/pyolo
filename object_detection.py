"""
File that contains the class for initialazing the fast object detector
"""

import numpy as np
import pyyolo
import cv2


class FastObjectDetector(object):
    def __init__(self, ):
        darknet_path = './darknet'
        datacfg = 'cfg/coco.data'
        cfgfile = 'cfg/tiny-yolo.cfg'
        weightfile = '../tiny-yolo.weights'
        self.thresh = 0.24
        self.hier_thresh = 0.5
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

    def predict(self, img_np):
        """Prediction function"""
        img_np = img_np.transpose(2, 0, 1)
        c, h, w = img.shape
        data = img_np.ravel()/255.0
        data = np.ascontiguousarray(data, dtype=np.float32)
        raw_outputs = pyyolo.detect(w, h, c, data, self.thresh, self.hier_thresh)
        return list(map(_parser_raw_output, raw_outputs))


    def __del__(self):
        pyyolo.cleanup()
