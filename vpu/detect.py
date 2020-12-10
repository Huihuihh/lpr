#!/usr/bin/env python3
#
# Copyright (C) 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

from argparse import ArgumentParser
import logging as log
import sys
from time import time
import os
from cv2 import dnn
import cv2
from openvino.inference_engine import IENetwork, IEPlugin
import numpy as np
import finemapping as fm
from PIL import Image, ImageDraw, ImageFont


fontC = ImageFont.truetype("Font/platech.ttf", 38, 0)  # 加载中文字体，38表示字体大小，0表示unicode编码


inWidth = 480
inHeight = 640
inScaleFactor = 0.007843
meanVal = 127.5
net = dnn.readNetFromCaffe("model/MobileNetSSD_test.prototxt","model/lpr.caffemodel")
net.setPreferableBackend(dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(dnn.DNN_TARGET_OPENCL)
num_classes=71
provinces={'<Beijing>': '京', '<Shanghai>': '沪', '<Tianjin>': '津', '<Hebei>': '冀', '<Shanxi>': '晋', '<InnerMongolia>': '蒙', '<Jilin>': '吉', '<Heilongjiang>': '黑', '<Jiangsu>': '苏', '<Zhejiang>': '浙', '<Anhui>': '皖', '<Fujian>': '闽', '<Jiangxi>': '赣', '<Shandong>': '鲁', '<Henan>': '豫', '<Hubei>': '鄂', '<Hunan>': '湘', '<Guangdong>': '粤', '<Guangxi>': '桂', '<Hainan>': '琼', '<Sichuan>': '川', '<Guizhou>': '贵', '<Yunnan>': '云', '<Tibet>': '藏', '<Shaanxi>': '陕', '<Gansu>': '甘', '<Qinghai>': '青', '<Ningxia>': '宁', '<Xinjiang>': '新', '<Liaoning>': '辽', '<Chongqing>': '渝', '<police>': '警', '<HongKong>': '港', 'Macau': '澳'}
vocab={'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '<Anhui>': 10, '<Beijing>': 11, '<Chongqing>': 12, '<Fujian>': 13, '<Gansu>': 14, '<Guangdong>': 15, '<Guangxi>': 16, '<Guizhou>': 17, '<Hainan>': 18, '<Hebei>': 19, '<Heilongjiang>': 20, '<Henan>': 21, '<HongKong>': 22, '<Hubei>': 23, '<Hunan>': 24, '<InnerMongolia>': 25, '<Jiangsu>': 26, '<Jiangxi>': 27, '<Jilin>': 28, '<Liaoning>': 29, '<Macau>': 30, '<Ningxia>': 31, '<Qinghai>': 32, '<Shaanxi>': 33, '<Shandong>': 34, '<Shanghai>': 35, '<Shanxi>': 36, '<Sichuan>': 37, '<Tianjin>': 38, '<Tibet>': 39, '<Xinjiang>': 40, '<Yunnan>': 41, '<Zhejiang>': 42, '<police>': 43, 'A': 44, 'B': 45, 'C': 46, 'D': 47, 'E': 48, 'F': 49, 'G': 50, 'H': 51, 'I': 52, 'J': 53, 'K': 54, 'L': 55, 'M': 56, 'N': 57, 'O': 58, 'P': 59, 'Q': 60, 'R': 61, 'S': 62, 'T': 63, 'U': 64, 'V': 65, 'W': 66, 'X': 67, 'Y': 68, 'Z': 69, '_': 70}
r_vocab={0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: '<Anhui>', 11: '<Beijing>', 12: '<Chongqing>', 13: '<Fujian>', 14: '<Gansu>', 15: '<Guangdong>', 16: '<Guangxi>', 17: '<Guizhou>', 18: '<Hainan>', 19: '<Hebei>', 20: '<Heilongjiang>', 21: '<Henan>', 22: '<HongKong>', 23: '<Hubei>', 24: '<Hunan>', 25: '<InnerMongolia>', 26: '<Jiangsu>', 27: '<Jiangxi>', 28: '<Jilin>', 29: '<Liaoning>', 30: '<Macau>', 31: '<Ningxia>', 32: '<Qinghai>', 33: '<Shaanxi>', 34: '<Shandong>', 35: '<Shanghai>', 36: '<Shanxi>', 37: '<Sichuan>', 38: '<Tianjin>', 39: '<Tibet>', 40: '<Xinjiang>', 41: '<Yunnan>', 42: '<Zhejiang>', 43: '<police>', 44: 'A', 45: 'B', 46: 'C', 47: 'D', 48: 'E', 49: 'F', 50: 'G', 51: 'H', 52: 'I', 53: 'J', 54: 'K', 55: 'L', 56: 'M', 57: 'N', 58: 'O', 59: 'P', 60: 'Q', 61: 'R', 62: 'S', 63: 'T', 64: 'U', 65: 'V', 66: 'W', 67: 'X', 68: 'Y', 69: 'Z', 70: '_', -1: ''}


def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("--model", help="Path to an .xml file with a trained model.", required=True, type=str)
    parser.add_argument("--cpu_extension",
                        help="MKLDNN (CPU)-targeted custom layers. "
                             "Absolute path to a shared library with the kernels implementation", type=str, default=None)
    parser.add_argument("--plugin_dir", help="Path to a plugin folder", type=str, default=None)
    parser.add_argument("--device",
                        help="Specify the target device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device specified (CPU by default)", default="CPU",
                        type=str)
    return parser


def load_ir_model(model_xml, device, plugin_dir, cpu_extension):
    model_bin = os.path.splitext(model_xml)[0] + ".bin"

    # initialize plugin
    log.info("Initializing plugin for %s device...", device)
    plugin = IEPlugin(device=device, plugin_dirs=plugin_dir)
    if cpu_extension and 'CPU' in device:
      plugin.add_cpu_extension(cpu_extension)

    # read IR
    log.info("Reading IR...")
    net = IENetwork(model=model_xml, weights=model_bin)

    if "CPU" in device:
      supported_layers = plugin.get_supported_layers(net)
      not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
      if not_supported_layers:
        log.error("Following layers are not supported by the plugin for specified device %s:\n %s",
                  device, ', '.join(not_supported_layers))
        log.error("Please try to specify cpu extensions library path in sample's command line parameters using "
                  "--cpu_extension command line argument")
        sys.exit(1)

    # input / output check
    assert len(net.inputs.keys()) == 1, "LPRNet must have only single input"
    assert len(net.outputs) == 1, "LPRNet must have only single output topologies"

    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    log.info("Loading IR to the plugin...")
    exec_net = plugin.load(network=net)
    shape = net.inputs[input_blob].shape # pylint: disable=E1136
    del net

    return exec_net, plugin, input_blob, out_blob, shape

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
            plates.append(image_rgb)
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

def decode_ie_output(vals, r_vocab):
  vals = vals.flatten()
  decoded_number = ''
  for val in vals:
    if val < 0:
      break
    decoded_number += r_vocab[val]
  return decoded_number

def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    exec_net, plugin, input_blob, out_blob, shape = load_ir_model(args.model, args.device,
                                                                  args.plugin_dir, args.cpu_extension)
    n_batch, channels, height, width = shape

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
                    in_frame = cv2.resize(plates[i], (width, height))
                    in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
                    in_frame = in_frame.reshape((n_batch, channels, height, width))

                    detect_time = []
                    t0 = time()
                    result = exec_net.infer(inputs={input_blob: in_frame})
                    detect_time.append((time() - t0) * 1000)
                    print('detect_time is {} ms'.format(np.average(np.asarray(detect_time))))

                    lp_code = result[out_blob][0]
                    lp_number = decode_ie_output(lp_code, r_vocab)
                    lp_number = keymap_replace(lp_number, provinces)
                    cv2.rectangle(frame, (xLeftBottoms[i], yLeftBottoms[i]), (xRightTops[i], yRightTops[i]), (255, 178, 50), 2)
                    img = Image.fromarray(frame)
                    draw = ImageDraw.Draw(img)
                    draw.text((xLeftBottoms[i] + 1, yLeftBottoms[i] - 28), lp_number, (0, 0, 255), font=fontC)
                    imagex = np.array(img)
                    cv2.imshow('camera', imagex)
        k = k + 1
        k = k % count
    capture.release()
    cv2.destroyWindow('camera')


if __name__ == '__main__':
    sys.exit(main() or 0)
