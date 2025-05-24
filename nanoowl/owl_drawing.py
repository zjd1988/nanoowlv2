# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import PIL.Image
import PIL.ImageDraw
import cv2
from .owl_predictor import OwlDecodeOutput
import matplotlib.pyplot as plt
import numpy as np
from typing import List


def get_colors(count: int):
    cmap = plt.cm.get_cmap("rainbow", count)
    colors = []
    for i in range(count):
        color = cmap(i)
        color = [int(255 * value) for value in color]
        colors.append(tuple(color))
    return colors


def draw_ch_text(image, text, position, font_path, font_size, color):
    """
    在 OpenCV 图像上写中文字符
    :param image: OpenCV 图像
    :param text: 要显示的中文文本
    :param position: 文本的左下角位置 (x, y)
    :param font_path: 字体文件路径
    :param font_size: 字体大小
    :param color: 文本颜色 (B, G, R)
    :return: 包含中文文本的 OpenCV 图像
    """
    # 将 OpenCV 图像转换为 PIL 图像
    pil_image = PIL.Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = PIL.ImageDraw.Draw(pil_image)
    # 加载字体
    font = PIL.ImageFont.truetype(font_path, font_size)
    # text_width, text_height = draw.textsize(text, font=font)
    # 获取文本的边界框
    bbox = draw.textbbox((0, 0), text, font=font)
    # 计算文本的宽度和高度
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    text_image = PIL.Image.fromarray(np.zeros((text_height, text_width, 3), dtype=np.uint8))
    text_draw = PIL.ImageDraw.Draw(text_image)
    text_draw.text((0, 0), text, font=font, fill=color)
    text_roi_image = cv2.cvtColor(np.array(text_image), cv2.COLOR_RGB2BGR)
    offset_x = position[0]
    offset_y = position[1]
    start_x = offset_x
    end_x = text_width + offset_x
    start_y = offset_y
    end_y = text_height + offset_y
    ori_h, ori_w, _ = image.shape
    if end_x > ori_w:
        end_x = ori_w
    if end_y > ori_h:
        end_y = ori_h
    if end_y - start_y + 1 < text_height:
        start_y = end_y - text_height
    if end_x - start_x + 1 < text_width:
        start_x = end_x - text_width
    image[start_y:end_y, start_x:end_x, :] = text_roi_image
    return image, position


def draw_owl_output(image, output: OwlDecodeOutput, text: List[str], draw_text=True):
    is_pil = not isinstance(image, np.ndarray)
    if is_pil:
        image = np.array(image)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75
    colors = get_colors(len(text))
    num_detections = len(output.labels)

    for i in range(num_detections):
        box = output.boxes[i]
        label_index = int(output.labels[i])
        box = [int(x) for x in box]
        pt0 = (box[0], box[1])
        pt1 = (box[2], box[3])
        cv2.rectangle(
            image,
            pt0,
            pt1,
            colors[label_index],
            4
        )
        if draw_text:
            offset_y = 12
            offset_x = 0
            label_text = text[label_index] + ' ' + f'{output.scores[i]:.2f}'
            font_path = "/data1/zhaojd-a/github_codes/clip_owl_prj/nanoowlv2/assets/simfang.ttf"
            image, _ = draw_ch_text(image, 
                label_text, 
                (box[0] + offset_x, box[1] + offset_y),
                font_path,
                20,
                colors[label_index])
            # cv2.putText(
            #     image,
            #     label_text,
            #     (box[0] + offset_x, box[1] + offset_y),
            #     font,
            #     font_scale,
            #     colors[label_index],
            #     2,# thickness
            #     cv2.LINE_AA
            # )
    if is_pil:
        image = PIL.Image.fromarray(image)
    return image
