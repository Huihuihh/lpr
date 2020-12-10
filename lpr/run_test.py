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
import tensorflow as tf
import numpy as np
import cv2

from lpr.trainer import decode_beams


r_vocab={0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: '<Anhui>', 11: '<Beijing>', 12: '<Chongqing>', 13: '<Fujian>', 14: '<Gansu>', 15: '<Guangdong>', 16: '<Guangxi>', 17: '<Guizhou>', 18: '<Hainan>', 19: '<Hebei>', 20: '<Heilongjiang>', 21: '<Henan>', 22: '<HongKong>', 23: '<Hubei>', 24: '<Hunan>', 25: '<InnerMongolia>', 26: '<Jiangsu>', 27: '<Jiangxi>', 28: '<Jilin>', 29: '<Liaoning>', 30: '<Macau>', 31: '<Ningxia>', 32: '<Qinghai>', 33: '<Shaanxi>', 34: '<Shandong>', 35: '<Shanghai>', 36: '<Shanxi>', 37: '<Sichuan>', 38: '<Tianjin>', 39: '<Tibet>', 40: '<Xinjiang>', 41: '<Yunnan>', 42: '<Zhejiang>', 43: '<police>', 44: 'A', 45: 'B', 46: 'C', 47: 'D', 48: 'E', 49: 'F', 50: 'G', 51: 'H', 52: 'I', 53: 'J', 54: 'K', 55: 'L', 56: 'M', 57: 'N', 58: 'O', 59: 'P', 60: 'Q', 61: 'R', 62: 'S', 63: 'T', 64: 'U', 65: 'V', 66: 'W', 67: 'X', 68: 'Y', 69: 'Z', 70: '_', -1: ''}

def load_graph(frozen_graph_filename):
  with tf.gfile.GFile(frozen_graph_filename, 'rb') as file:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(file.read())
  with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def)
  return graph

def display_license_plate(number, license_plate_img):
  size = cv2.getTextSize(number, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
  text_width = size[0][0]
  text_height = size[0][1]

  height, width, _ = license_plate_img.shape
  license_plate_img = cv2.copyMakeBorder(license_plate_img, 0, text_height + 10, 0,
                                         0 if text_width < width else text_width - width,
                                         cv2.BORDER_CONSTANT, value=(255, 255, 255))
  cv2.putText(license_plate_img, number, (0, height + text_height + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

  return license_plate_img

def build_argparser():
  parser = ArgumentParser()
  parser.add_argument('--model', help='Path to frozen graph file with a trained model.', required=True, type=str)
  parser.add_argument('--output', help='Output image')
  parser.add_argument('--input', help='Image with license plate')
  return parser


def main():
  args = build_argparser().parse_args()

  graph = load_graph(args.model)

  image = cv2.imread(args.input)
  img = cv2.resize(image, (94, 24))
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img = np.float32(img)
  img = np.multiply(img, 1.0/255.0)

  input = graph.get_tensor_by_name("import/input:0")
  output = graph.get_tensor_by_name("import/d_predictions:0")

  with tf.Session(graph=graph) as sess:
    results = sess.run(output, feed_dict={input: [img]})
    print(results)

    decoded_lp = decode_beams(results, r_vocab)[0]
    print(decoded_lp)

    img_to_display = display_license_plate(decoded_lp, image)

    if args.output:
      cv2.imwrite(args.output, img_to_display)
    else:
      cv2.imshow('License Plate', img_to_display)
      cv2.waitKey(0)


if __name__ == "__main__":
  main()
