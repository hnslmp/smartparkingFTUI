import os
import sys
import telebot

sys.path.append(os.path.abspath(os.path.join('Mask_RCNN')))
os.chdir('Mask_RCNN')
API_KEY = '2038935012:AAEUSpqHqqsg3JdRTBhhQR98xwMlph3JS7E'
bot = telebot.TeleBot(API_KEY)

cwd = os.getcwd()
print("Current working directory: {0}".format(cwd))

import numpy as np
import cv2
import mrcnn.config
import mrcnn.utils
from mrcnn.model import MaskRCNN
from pathlib import Path
# from google.colab.patches import cv2_imshow
import pickle5 as pickle

from shapely.geometry import box
from shapely.geometry import Polygon as shapely_poly
from IPython.display import clear_output, Image, display, HTML
import io
import base64

class Config(mrcnn.config.Config):
    NAME = "coco_pretrained_model_config"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 81

config = Config()
config.display()


ROOT_DIR = Path(".")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

if not os.path.exists(COCO_MODEL_PATH):
    mrcnn.utils.download_trained_weights(COCO_MODEL_PATH)

model = MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=Config())

model.load_weights(COCO_MODEL_PATH, by_name=True)

# if not os.path.exists("./data"):
#     os.makedirs("./data")

VIDEO_SOURCE = "data/parkiranft_Trim.mp4"
PARKING_REGIONS = "data/regionparkiranft.p"
with open(PARKING_REGIONS, 'rb') as f:
    parked_car_boxes = pickle.load(f)

def get_car_boxes(boxes, class_ids):
    car_boxes = []

    for i, box in enumerate(boxes):
        # If the detected object isn't a car / truck, skip it
        if class_ids[i] in [3, 8, 6]:
            car_boxes.append(box)

    return np.array(car_boxes)

def compute_overlaps(parked_car_boxes, car_boxes):
    
    new_car_boxes = []
    for box in car_boxes:
        y1 = box[0]
        x1 = box[1]
        y2 = box[2]
        x2 = box[3]
        
        p1 = (x1, y1)
        p2 = (x2, y1)
        p3 = (x2, y2)
        p4 = (x1, y2)
        new_car_boxes.append([p1, p2, p3, p4])
    
    overlaps = np.zeros((len(parked_car_boxes), len(new_car_boxes)))
    for i in range(len(parked_car_boxes)):
        for j in range(car_boxes.shape[0]):
            pol1_xy = parked_car_boxes[i]
            pol2_xy = new_car_boxes[j]
            polygon1_shape = shapely_poly(pol1_xy)
            polygon2_shape = shapely_poly(pol2_xy)

            polygon_intersection = polygon1_shape.intersection(polygon2_shape).area
            polygon_union = polygon1_shape.union(polygon2_shape).area
            IOU = polygon_intersection / polygon_union
            overlaps[i][j] = IOU

    return overlaps

def arrayShow (imageArray):
    ret, png = cv2.imencode('.png', imageArray)
    encoded = base64.b64encode(png)
    return Image(data=encoded.decode('ascii'))

alpha = 0.6
video_capture = cv2.VideoCapture(VIDEO_SOURCE)
# video_capture = cv2.VideoCapture(0)
# cnt=0

video_FourCC    = cv2.VideoWriter_fourcc('M','J','P','G')
video_fps       = video_capture.get(cv2.CAP_PROP_FPS)
video_size      = (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
out = cv2.VideoWriter("out.avi", video_FourCC, video_fps, video_size)

#info_message = 'Halo! Berikut informasi ketersediaan parkir di Universitas Indonesia :\n1. Gedung Dekanat FTUI\n\t\t- Tersedia : {}'.format(SLOT_PARKIR_KOSONG)
info_message = ''

@bot.message_handler(commands=['info'])
def info(message):
  bot.send_message(message.chat.id, info_message)

while video_capture.isOpened():
    success, frame = video_capture.read()
    
    overlay = frame.copy()
    if not success:
        break

    rgb_image = frame[:, :, ::-1]
    results = model.detect([rgb_image], verbose=0)

    car_boxes = get_car_boxes(results[0]['rois'], results[0]['class_ids'])
    overlaps = compute_overlaps(parked_car_boxes, car_boxes)
    cnt=0
    #print(overlaps)
    for parking_area, overlap_areas in zip(parked_car_boxes, overlaps):  
        max_IoU_overlap = np.max(overlap_areas)
        if max_IoU_overlap < 0.15:
            cv2.fillPoly(overlay, [np.array(parking_area)], (71, 27, 92))
            free_space = True
            cnt += 1   
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    print(cnt)
    out.write(frame)

    clear_output(wait=True)
    img = arrayShow(frame)
    display(img)
    info_message = 'Halo! Berikut informasi ketersediaan parkir di Universitas Indonesia :\n1. Gedung Dekanat FTUI\n\t\t- Tersedia : {}'.format(cnt)
    bot.polling()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
out.release()
cv2.destroyAllWindows()