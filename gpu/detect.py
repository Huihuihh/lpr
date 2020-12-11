from argparse import ArgumentParser
from time import time
import tensorflow as tf
from tensorflow.python.platform import gfile
import sys
import os
from cv2 import dnn
import cv2
import finemapping as fm
import numpy as np
from PIL import Image, ImageDraw, ImageFont


fontC = ImageFont.truetype("Font/platech.ttf", 38, 0)  # 加载中文字体，38表示字体大小，0表示unicode编码


inWidth = 480
inHeight = 640
inScaleFactor = 0.007843
meanVal = 127.5
net = dnn.readNetFromCaffe("model/MobileNetSSD_test.prototxt","model/lpr.caffemodel")
net.setPreferableBackend(dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(dnn.DNN_TARGET_OPENCL)
provinces={'<Beijing>': '京', '<Shanghai>': '沪', '<Tianjin>': '津', '<Chongqing>': '渝', '<Hebei>': '冀', '<Shanxi>': '晋', '<InnerMongolia>': '蒙', '<Jilin>': '吉', '<Heilongjiang>': '黑', '<Jiangsu>': '苏', '<Zhejiang>': '浙', '<Anhui>': '皖', '<Fujian>': '闽', '<Jiangxi>': '赣', '<Shandong>': '鲁', '<Henan>': '豫', '<Hubei>': '鄂', '<Hunan>': '湘', '<Guangdong>': '粤', '<Guangxi>': '桂', '<Hainan>': '琼', '<Sichuan>': '川', '<Guizhou>': '贵', '<Yunnan>': '云', '<Tibet>': '藏', '<Shaanxi>': '陕', '<Gansu>': '甘', '<Qinghai>': '青', '<Ningxia>': '宁', '<Xinjiang>': '新', '<Liaoning>': '辽', '<police>': '警', '<HongKong>': '港', 'Macau': '澳'}
r_vocab={0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: '<Anhui>', 11: '<Beijing>', 12: '<Chongqing>', 13: '<Fujian>', 14: '<Gansu>', 15: '<Guangdong>', 16: '<Guangxi>', 17: '<Guizhou>', 18: '<Hainan>', 19: '<Hebei>', 20: '<Heilongjiang>', 21: '<Henan>', 22: '<HongKong>', 23: '<Hubei>', 24: '<Hunan>', 25: '<InnerMongolia>', 26: '<Jiangsu>', 27: '<Jiangxi>', 28: '<Jilin>', 29: '<Liaoning>', 30: '<Macau>', 31: '<Ningxia>', 32: '<Qinghai>', 33: '<Shaanxi>', 34: '<Shandong>', 35: '<Shanghai>', 36: '<Shanxi>', 37: '<Sichuan>', 38: '<Tianjin>', 39: '<Tibet>', 40: '<Xinjiang>', 41: '<Yunnan>', 42: '<Zhejiang>', 43: '<police>', 44: 'A', 45: 'B', 46: 'C', 47: 'D', 48: 'E', 49: 'F', 50: 'G', 51: 'H', 52: 'I', 53: 'J', 54: 'K', 55: 'L', 56: 'M', 57: 'N', 58: 'O', 59: 'P', 60: 'Q', 61: 'R', 62: 'S', 63: 'T', 64: 'U', 65: 'V', 66: 'W', 67: 'X', 68: 'Y', 69: 'Z', 70: '_', -1: ''}

def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("--model", help="Path to an .pb file with a trained model.", required=True, type=str)
    return parser

def detect(frame):

    frame_resized = cv2.resize(frame, (inWidth, inHeight)); # 将原图缩放到指定高宽

    heightFactor = frame.shape[0] / inHeight;  # 计算高度缩放比例
    widthFactor = frame.shape[1] / inWidth;

    blob = dnn.blobFromImage(frame_resized, inScaleFactor, (inWidth, inHeight), meanVal) # 读入图片
    net.setInput(blob)
    detections = net.forward()   # 定位车牌

    cols = frame_resized.shape[1]  # 缩放后图片宽度
    rows = frame_resized.shape[0]

    res_set = []
    plates = []
    xLeftBottoms = []
    yLeftBottoms = []
    xRightTops = []
    yRightTops = []
    # 循环遍历定位到的车牌
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2:   # 车牌定位置信度大于指定值
            class_id = int(detections[0, 0, i, 1])

            xLeftBottom = int(detections[0, 0, i, 3] * cols) # 被实际检测图中车牌框左上点横坐标
            yLeftBottom = int(detections[0, 0, i, 4] * rows)
            xRightTop = int(detections[0, 0, i, 5] * cols)  # 被实际检测图中车牌框右下点横坐标
            yRightTop = int(detections[0, 0, i, 6] * rows)

            xLeftBottom_ = int(widthFactor * xLeftBottom); # 原图中车牌框左上点横坐标
            yLeftBottom_ = int(heightFactor * yLeftBottom);
            xRightTop_ = int(widthFactor * xRightTop);
            yRightTop_ = int(heightFactor * yRightTop);
            # 适当扩大车牌定位框
            h = yRightTop_ - yLeftBottom_
            w = xRightTop_ - xLeftBottom_
            yLeftBottom_ -= int(h * 0.5)
            yRightTop_ += int(h * 0.5)
            xLeftBottom_ -= int(w * 0.14)
            xRightTop_ += int(w * 0.14)

            image_sub = frame[yLeftBottom_:yRightTop_,xLeftBottom_:xRightTop_] # 截取原图车牌定位区域

            # 必须调整车牌到统一大小
            plate = image_sub
            if plate.shape[0] > 36:
                plate = cv2.resize(image_sub, (136, 36 * 2))
            else:
                plate = cv2.resize(image_sub, (136, 36 ))

            # 精定位，倾斜校正
            image_rgb = fm.findContoursAndDrawBoundingBox(plate)
            # 车牌左右边界修正
            #image_rgb = fv.finemappingVertical(image_rgb)
            plates.append(image_rgb)
            #plates.append(plate)
            xLeftBottoms.append(xLeftBottom_)
            yLeftBottoms.append(yLeftBottom_)
            xRightTops.append(xRightTop_)
            yRightTops.append(yRightTop_)
    return plates, xLeftBottoms, yLeftBottoms, xRightTops, yRightTops

def keymap_replace(
        string: str,
        mappings: dict,
        lower_keys=False,
        lower_values=False,
        lower_string=False,
    ) -> str:
    """Replace parts of a string based on a dictionary.
    This function takes a string a dictionary of
    replacement mappings. For example, if I supplied
    the string "Hello world.", and the mappings
    {"H": "J", ".": "!"}, it would return "Jello world!".
    Keyword arguments:
    string       -- The string to replace characters in.
    mappings     -- A dictionary of replacement mappings.
    lower_keys   -- Whether or not to lower the keys in mappings.
    lower_values -- Whether or not to lower the values in mappings.
    lower_string -- Whether or not to lower the input string.
    """
    replaced_string = string.lower() if lower_string else string
    for character, replacement in mappings.items():
        replaced_string = replaced_string.replace(
            character.lower() if lower_keys else character,
            replacement.lower() if lower_values else replacement
        )
    return replaced_string

def decode_beams(vals, r_vocab):
  beams_list = []
  for val in vals:
    decoded_number = ''
    for code in val:
      decoded_number += r_vocab[code]
    beams_list.append(decoded_number)
  return beams_list

def main():
    args = build_argparser().parse_args()

    soft_config = tf.ConfigProto(allow_soft_placement=True)
    soft_config.gpu_options.allow_growth = True
    sess = tf.Session(config=soft_config)
    sess.run(tf.global_variables_initializer())
    # Load the model
    model = args.model
    model_exp = os.path.expanduser(model)
    with gfile.FastGFile(model_exp,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

    capture =cv2.VideoCapture(0)
    cv2.namedWindow('camera', 1)

    k = 0
    count = 3
    timer=0
    while cv2.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        if (k == 0):
            plates, xLeftBottoms, yLeftBottoms, xRightTops, yRightTops = detect(frame)
            if (len(plates)==0):
                cv2.imshow('camera', frame)
            else:
                for i, plate in enumerate(plates):
                    in_frame = cv2.resize(plates[i], (94, 24))
                    in_frame = cv2.cvtColor(in_frame, cv2.COLOR_BGR2RGB)
                    in_frame = np.float32(in_frame)
                    in_frame = np.multiply(in_frame, 1.0/255.0)

                    input = tf.get_default_graph().get_tensor_by_name("input:0")
                    output = tf.get_default_graph().get_tensor_by_name("d_predictions:0")
                    results = sess.run(output, feed_dict={input: [in_frame]})

                    decoded_lp = decode_beams(results, r_vocab)[0]
                    decoded_lp = keymap_replace(decoded_lp, provinces)
                    cv2.rectangle(frame, (xLeftBottoms[i], yLeftBottoms[i]), (xRightTops[i], yRightTops[i]), (255, 178, 50), 2)
                    img = Image.fromarray(frame)
                    draw = ImageDraw.Draw(img)
                    draw.text((xLeftBottoms[i] + 1, yLeftBottoms[i] - 28), decoded_lp, (0, 0, 255), font=fontC)
                    imagex = np.array(img)
                    cv2.imshow('camera', imagex)
        k = k + 1
        k = k % count

    capture.release()
    cv2.destroyWindow('camera')


if __name__ == '__main__':
    sys.exit(main() or 0)
