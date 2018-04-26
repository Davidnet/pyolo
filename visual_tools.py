import pyyolo
import numpy as np
import sys
import cv2
import utils
from time import time

darknet_path = './darknet'
datacfg = 'cfg/coco.data'
cfgfile = 'cfg/tiny-yolo.cfg'
weightfile = '../tiny-yolo.weights'
filename = darknet_path + '/data/person.jpg'
thresh = 0.24
hier_thresh = 0.5

def converter_api(raw_outputs):
    xmin = raw_outputs.get("left")
    xmax = raw_outputs.get("right")
    ymin = raw_outputs.get("top")
    ymax = raw_outputs.get("bottom")
    preds = raw_outputs.get("prob")
    id = raw_outputs.get("class")
    predictions = dict(box = [ymin, xmin, ymax, xmax], id = len(id), name = id)  
    return predictions

def all_converter(outputs):
    predictions = []
    for output in outputs:
        predictions.append(converter_api(output))
    return predictions
    


pyyolo.init(darknet_path, datacfg, cfgfile, weightfile)
# camera 
print('----- test python API using a camera')
cap = cv2.VideoCapture(1)
while True:
        ret, img = cap.read()
        original_img = img
	# ret_val, img = cam.read()
	img = img.transpose(2,0,1)
	c, h, w = img.shape[0], img.shape[1], img.shape[2]
	# print w, h, c 
	data = img.ravel()/255.0
	data = np.ascontiguousarray(data, dtype=np.float32)
        tick = time()
	outputs = pyyolo.detect(w, h, c, data, thresh, hier_thresh)	
        tock = time() - tick
        print('tock: ', tock)
        predictions = all_converter(outputs)
        frame = utils.visualize_boxes_on_image(original_img, predictions, show_distances = False)
        cv2.imshow('frame', frame) 
        cv2.waitKey(1)

cv2.destroyAllWindows()

        

                
                
# free model
pyyolo.cleanup()
