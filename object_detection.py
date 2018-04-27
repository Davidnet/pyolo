"""
File that contains the class for initialazing the fast object detector
"""

import sys, os, shutil, subprocess
import numpy as np
import pyyolo
import cv2

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
URL_PATH = "gs://vision-198622-production-models/object-detection/tiny-yolo/tiny-yolo.weights"

class TinyYOLODetector(object):
    """Class detection for TinyYolo"""
    @staticmethod
    def get(url=URL_PATH, *args, **kwargs):
        rm = kwargs.pop("rm", False)

        hash_name = str(hash(url)).replace("-", "0")
        filename = os.path.basename(url)

        home_path = os.path.expanduser('~')
        model_dir_base = os.path.join(home_path, ".local", "tfinterface", "frozen_graphs", hash_name)
        model_path = os.path.join(model_dir_base, filename)

        if os.path.exists(model_dir_base):

            is_empty = len(os.listdir(model_dir_base)) == 0
            files_counts = [ os.path.join(dp, f) for dp, _, filenames in os.walk(model_dir_base) for f in filenames if f.endswith(".gstmp") ]
            temp_files = len(files_counts) > 0

            if rm or is_empty or temp_files:
                shutil.rmtree(model_dir_base, ignore_errors = True)

        if not os.path.exists(model_dir_base):
            os.makedirs(model_dir_base)

            cmd = "gsutil -m cp -R {source_folder} {dest_folder}".format(
                source_folder = url,
                dest_folder = model_dir_base,
            )

            # print("CMD: {}".format(cmd))

            subprocess.check_call(
                cmd,
                stdout = subprocess.PIPE, shell = True,
            )
        return YOLODetector(model_path, *args, **kwargs)

class YOLODetector(object):
    """Class detection for YOLO objects in general"""
    def __init__(self, weightfile, thresh=0.24, hier_thresh=0.5):
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
        mid_x = (xmax + xmin) / 2
        mid_y = (ymax + ymin) / 2  # TODO: use mid_y
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
