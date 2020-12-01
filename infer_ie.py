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

from __future__ import print_function
from argparse import ArgumentParser
import logging as log
import sys
import os
from cv2 import dnn
import cv2
from openvino.inference_engine import IENetwork, IEPlugin
from lpr.trainer import decode_ie_output
import numpy as np
from tfutils.helpers import load_module
from hyperlpr_py3 import pipline as pp
from PIL import Image, ImageDraw, ImageFont


fontC = ImageFont.truetype("Font/platech.ttf", 38, 0)  # 加载中文字体，38表示字体大小，0表示unicode编码


inWidth = 480
inHeight = 640
WHRatio = inWidth / float(inHeight)
inScaleFactor = 0.007843
meanVal = 127.5
plateTypeName = ["蓝", "黄", "绿", "白", "黑 "]
net = dnn.readNetFromCaffe("/home/awcloud/Desktop/code/lpr/model/MobileNetSSD_test.prototxt","/home/awcloud/Desktop/code/lpr/model/lpr.caffemodel")
#net = dnn.readNetFromCaffe("/home/awcloud/Desktop/code/caffe/models/VGGNet/VOC0712/SSD_300x300/deploy.prototxt","/home/awcloud/Desktop/code/caffe/VGG_VOC0712_SSD_300x300_iter_55500.caffemodel")
net.setPreferableBackend(dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(dnn.DNN_TARGET_OPENCL)


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
    parser.add_argument('--config', help='Path to a config.py', required=True)
    parser.add_argument('--output', help='Output image')
    parser.add_argument('--input_image', help='Image with license plate')
    return parser


def display_license_plate(number, license_plate_img):
    size = cv2.getTextSize(number, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    text_width = size[0][0]
    text_height = size[0][1]

    height, width, _ = license_plate_img.shape
    license_plate_img = cv2.copyMakeBorder(license_plate_img, 0, text_height + 10, 0,
                                           0 if text_width < width else text_width - width,
                                           cv2.BORDER_CONSTANT, value=(255, 255, 255))
    #cv2.putText(license_plate_img, number, (0, height + text_height + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    cv2.putText(license_plate_img, number, (0, height + text_height + 5), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255))

    return license_plate_img

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

def rotate(image, degree):
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
    # 将图像旋转180度
    M = cv2.getRotationMatrix2D(center, degree, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def detect(frame):

    frame_resized = cv2.resize(frame, (inWidth, inHeight)); # 将原图缩放到指定高宽
    #cv2.imshow("test", frame_resized);
    #cv2.waitKey(0);

    heightFactor = frame.shape[0] / inHeight;  # 计算高度缩放比例
    widthFactor = frame.shape[1] / inWidth;

    blob = dnn.blobFromImage(frame_resized, inScaleFactor, (inWidth, inHeight), meanVal) # 读入图片
    net.setInput(blob)
    detections = net.forward()   # 定位车牌
    # print("车牌定位时间:", time.time() - t0)

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
           # print("y1:",yLeftBottom_, "y2:",yRightTop_, "x1:",xLeftBottom_, "x2:", xRightTop_)  # 输出车牌在原图中位置信息
            # 适当扩大车牌定位框
            h = yRightTop_ - yLeftBottom_
            w = xRightTop_ - xLeftBottom_
            yLeftBottom_ -= int(h * 0.5)
            yRightTop_ += int(h * 0.5)
            xLeftBottom_ -= int(w * 0.14)
            xRightTop_ += int(w * 0.14)

           # cv2.rectangle(frame, (xLeftBottom_-2, yLeftBottom_-2), (xRightTop_+2, yRightTop_+2),(0, 0,255))    #车牌位置绘制红色边框

            image_sub = frame[yLeftBottom_:yRightTop_,xLeftBottom_:xRightTop_] # 截取原图车牌定位区域

            # 必须调整车牌到统一大小
            plate = image_sub
            # print(plate.shape[0],plate.shape[1])
            if plate.shape[0] > 36:
                plate = cv2.resize(image_sub, (136, 36 * 2))
            else:
                plate = cv2.resize(image_sub, (136, 36 ))
          #  cv2.imshow("test", plate)
          #  cv2.waitKey(0)
            # 判断车牌颜色
            plate_type = pp.td.SimplePredict(plate)
            plate_color = plateTypeName[plate_type]

            if (plate_type > 0) and (plate_type < 5):
                plate = cv2.bitwise_not(plate)


            # 精定位，倾斜校正
            image_rgb = pp.fm.findContoursAndDrawBoundingBox(plate)
           # cv2.imshow("test", image_rgb);
           # cv2.waitKey(0)
            # 车牌左右边界修正
            image_rgb = pp.fv.finemappingVertical(image_rgb)
            plates.append(image_rgb)
            xLeftBottoms.append(xLeftBottom_)
            yLeftBottoms.append(yLeftBottom_)
            xRightTops.append(xRightTop_)
            yRightTops.append(yRightTop_)
    return plates, xLeftBottoms, yLeftBottoms, xRightTops, yRightTops
            #img_to_display = image_rgb.copy()
            #in_frame = cv2.resize(image_rgb, (width, height))
            #in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            #in_frame = in_frame.reshape((n_batch, channels, height, width))

            #result = exec_net.infer(inputs={input_blob: in_frame})
            #lp_code = result[out_blob][0]
            #lp_number = decode_ie_output(lp_code, cfg.r_vocab)
            #lp_number = keymap_replace(lp_number, provinces)
            #print('Output: {}'.format(lp_number))
            ##img_to_display = display_license_plate(lp_number, img_to_display)
            #cv2.rectangle(frame, (xLeftBottom_, yLeftBottom_), (xRightTop_, yRightTop_), (255, 178, 50), 2)
            #img = Image.fromarray(frame)
            #draw = ImageDraw.Draw(img)
            #draw.text((xLeftBottom_ + 1, yLeftBottom_ - 28), lp_number, (0, 0, 255), font=fontC)
            #imagex = np.array(img)
            ##cv2.putText(imagex, lp_number, (xLeftBottom_, yLeftBottom_), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
            #cv2.imshow('tttt', imagex)
        #else:
        #    img = Image.fromarray(frame)
        #    imagex = np.array(img)
        #    cv2.imshow('tttt', imagex)

provinces={'<Beijing>': '京', '<Shanghai>': '沪', '<Tianjin>': '津', '<Sichuan>': '渝', '<Hebei>': '冀', '<Shanxi>': '晋', '<InnerMongolia>': '蒙', '<Jilin>': '吉', '<Heilongjiang>': '黑', '<Jiangsu>': '苏', '<Zhejiang>': '浙', '<Anhui>': '皖', '<Fujian>': '闽', '<Jiangxi>': '赣', '<Shandong>': '鲁', '<Henan>': '豫', '<Hubei>': '鄂', '<Hunan>': '湘', '<Guangdong>': '粤', '<Guangxi>': '桂', '<Hainan>': '琼', '<Sichuan>': '川', '<Guizhou>': '贵', '<Yunnan>': '云', '<Tibet>': '藏', '<Shaanxi>': '陕', '<Gansu>': '甘', '<Qinghai>': '青', '<Ningxia>': '宁', '<Xinjiang>': '新', '<Liaoning>': '辽'}

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

def imreadex(filename):
    return cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_COLOR)


def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    cfg = load_module(args.config)
    exec_net, plugin, input_blob, out_blob, shape = load_ir_model(args.model, args.device,
                                                                  args.plugin_dir, args.cpu_extension)
    n_batch, channels, height, width = shape

    cv2.namedWindow('tttt', 0)
    cv2.resizeWindow("tttt", 640, 480)
    #image = cv2.imread(args.input_image)
    #cap = cv2.VideoCapture(args.input_image)
    #hasFrame, frame = cap.read()
    cap = cv2.VideoCapture('/home/awcloud/Desktop/code/lpr/test.mp4')
    #frame = '/home/awcloud/Desktop/code/License_plate_recognition/Test/京AD77972.jpg'
    #frame = '/home/awcloud/Desktop/code/caffe/plate.jpg'
    while cv2.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        frame = rotate(frame, -90)
        #cv2.imshow('tttt', frame)
        plates, xLeftBottoms, yLeftBottoms, xRightTops, yRightTops = detect(frame)
        if (len(plates)==0):
            cv2.imshow('tttt', frame)
        else:
            for i, plate in enumerate(plates):
                in_frame = cv2.resize(plates[i], (width, height))
                in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
                in_frame = in_frame.reshape((n_batch, channels, height, width))

                result = exec_net.infer(inputs={input_blob: in_frame})
                lp_code = result[out_blob][0]
                lp_number = decode_ie_output(lp_code, cfg.r_vocab)
                lp_number = keymap_replace(lp_number, provinces)
                print('Output: {}'.format(lp_number))
                #img_to_display = display_license_plate(lp_number, img_to_display)
                cv2.rectangle(frame, (xLeftBottoms[i], yLeftBottoms[i]), (xRightTops[i], yRightTops[i]), (255, 178, 50), 2)
                img = Image.fromarray(frame)
                draw = ImageDraw.Draw(img)
                draw.text((xLeftBottoms[i] + 1, yLeftBottoms[i] - 28), lp_number, (0, 0, 255), font=fontC)
                imagex = np.array(img)
                #cv2.putText(imagex, lp_number, (xLeftBottom_, yLeftBottom_), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
                cv2.imshow('tttt', imagex)
        #card_imgs, boxs = tect(frame, exec_net, input_blob, out_blob, width, height, n_batch, channels, cfg)

      #if type(frame) == type(""):
      #    img = imreadex(frame)
      #else:
      #    img = frame
      #pic_hight, pic_width = img.shape[:2]
      #resize_rate = MAX_WIDTH / pic_width
      #if (len(card_imgs) == 0):
      #    cv2.imshow('tttt', frame) 
      #else:
      #    for i, card_img in enumerate(card_imgs):
      #        bbox = boxs[i]
      #        # 左上点横坐标
      #        xLeftBottom = int(bbox[0][0])
      #        yLeftBottom = int(bbox[0][1])
      #        # 右下点横坐标
      #        xRightTop = int(bbox[2][0])
      #        yRightTop = int(bbox[2][1])
      #        #print('bbox[0][0] is', int(bbox[0][0]))
      #        if pic_width > MAX_WIDTH:
      #            xLeftBottom_ = int(xLeftBottom / resize_rate)
      #            yLeftBottom_ = int(yLeftBottom / resize_rate)
      #            xRightTop_ = int(xRightTop / resize_rate)
      #            yRightTop_ = int(yRightTop / resize_rate)
      #        else:
      #            xLeftBottom_ = xLeftBottom
      #            yLeftBottom_ = yLeftBottom
      #            xRightTop_ = xRightTop
      #            yRightTop_ = yRightTop
      #        in_frame = cv2.resize(card_img, (width, height))
      #        in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
      #        in_frame = in_frame.reshape((n_batch, channels, height, width))

      #        result = exec_net.infer(inputs={input_blob: in_frame})
      #        lp_code = result[out_blob][0]
      #        lp_number = decode_ie_output(lp_code, cfg.r_vocab)
      #        print('Output: {}'.format(lp_number))
      #  #img_to_display = display_license_plate(lp_number, img_to_display)
      #  #bbox = np.int0(bbox)
      #  #oldimg = cv2.drawContours(pic, (600,560), 0, (0, 0, 255), 2)
      #        cv2.rectangle(frame, (xLeftBottom_-1, yLeftBottom_+1), (xRightTop_+1, yRightTop_-1), (0, 255, 0), 2)
      #        cv2.putText(frame, lp_number, (xLeftBottom_, yLeftBottom_), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
      #        cv2.imshow('tttt', frame)
      #if card_imgs is []:
      #    return
#  while True:
#      hasFrame, image = cap.read()
#      #image = rotate(image, -90)
#      frame_resized = cv2.resize(image, (inWidth, inHeight)); # 将原图缩放到指定高宽
#  #cv2.imshow("test", frame_resized);
#  #cv2.waitKey(0);
#
#      heightFactor = image.shape[0] / inHeight;  # 计算高度缩放比例
#      widthFactor = image.shape[1] / inWidth;
#
#      blob = dnn.blobFromImage(frame_resized, inScaleFactor, (inWidth, inHeight), meanVal) # 读入图片
#      net.setInput(blob)
#      detections = net.forward()   # 定位车牌
#  # print("车牌定位时间:", time.time() - t0)
#
#      cols = frame_resized.shape[1]  # 缩放后图片宽度
#      rows = frame_resized.shape[0]
#
#      res_set = []
#      # 循环遍历定位到的车牌
#      for i in range(detections.shape[2]):
#          global image_rgb
#          confidence = detections[0, 0, i, 2]
#          if confidence > 0.2:   # 车牌定位置信度大于指定值
#              class_id = int(detections[0, 0, i, 1])
#
#              xLeftBottom = int(detections[0, 0, i, 3] * cols) # 被实际检测图中车牌框左上点横坐标
#              yLeftBottom = int(detections[0, 0, i, 4] * rows)
#              xRightTop = int(detections[0, 0, i, 5] * cols)  # 被实际检测图中车牌框右下点横坐标
#              yRightTop = int(detections[0, 0, i, 6] * rows)
#
#              xLeftBottom_ = int(widthFactor * xLeftBottom); # 原图中车牌框左上点横坐标
#              yLeftBottom_ = int(heightFactor * yLeftBottom);
#              xRightTop_ = int(widthFactor * xRightTop);
#              yRightTop_ = int(heightFactor * yRightTop);
#         # print("y1:",yLeftBottom_, "y2:",yRightTop_, "x1:",xLeftBottom_, "x2:", xRightTop_)  # 输出车牌在原图中位置信息
#          # 适当扩大车牌定位框
#              h = yRightTop_ - yLeftBottom_
#              w = xRightTop_ - xLeftBottom_
#              yLeftBottom_ -= int(h * 0.5)
#              yRightTop_ += int(h * 0.5)
#              xLeftBottom_ -= int(w * 0.14)
#              xRightTop_ += int(w * 0.14)
#
#             # cv2.rectangle(frame, (xLeftBottom_-2, yLeftBottom_-2), (xRightTop_+2, yRightTop_+2),(0, 0,255))    #车牌位置绘制红色边框
#
#              image_sub = image[yLeftBottom_:yRightTop_,xLeftBottom_:xRightTop_] # 截取原图车牌定位区域
#
#              # 必须调整车牌到统一大小
#              plate = image_sub
#              #print(plate.shape[0],plate.shape[1])
#              if plate.shape[0] > 36:
#                  plate = cv2.resize(image_sub, (136, 36 * 2))
#              else:
#                  plate = cv2.resize(image_sub, (136, 36 ))
#              plate_type = pp.td.SimplePredict(plate)
#              plate_color = plateTypeName[plate_type]
#              if (plate_type > 0) and (plate_type < 5):
#                  plate = cv2.bitwise_not(plate)
#              image_rgb = pp.fm.findContoursAndDrawBoundingBox(plate)
#              image_rgb = pp.fv.finemappingVertical(image_rgb)
#              #cv2.imwrite('/home/awcloud/Desktop/lpr_image/test1.jpg', image_rgb)
#      #else:
#      #    image_rgb = frame_resized
#
#    image = cv2.imread(args.input_image)
#    img_to_display = image.copy()
#    in_frame = cv2.resize(image, (width, height))
#    in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
#    in_frame = in_frame.reshape((n_batch, channels, height, width))
#
#    result = exec_net.infer(inputs={input_blob: in_frame})
#    lp_code = result[out_blob][0]
#    lp_number = decode_ie_output(lp_code, cfg.r_vocab)
#    print('Output: {}'.format(lp_number))
#    img_to_display = display_license_plate(lp_number, img_to_display)
#    #cv2.rectangle(image, (xLeftBottom_, yLeftBottom_), (xRightTop_, yRightTop_), (255, 178, 50), 2)
#    #cv2.putText(frame, lp_number, (xLeftBottom_, yLeftBottom_), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#  #image = display_license_plate(lp_number, image)
#    if args.output:
#      cv2.imwrite(args.output, img_to_display)
#    else:
#      cv2.imshow('tttt', img_to_display)
#      cv2.waitKey(0)

    #del exec_net
    #del plugin
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    sys.exit(main() or 0)
