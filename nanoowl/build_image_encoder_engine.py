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
import sys
import os
sys.path.append("/data1/zhaojd-a/github_codes/clip_owl_prj/nanoowlv2")

import argparse
from nanoowl.owl_predictor import OwlPredictor


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("output_path", type=str)
    parser.add_argument("--onnx_path", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="google/owlvit-base-patch32")
    parser.add_argument("--fp16_mode", type=bool, default=True)
    parser.add_argument("--onnx_opset", type=int, default=16)
    parser.add_argument("--also_run_trtexec", action="store_true")
    args = parser.parse_args()
    
    predictor = OwlPredictor(
        model_name=args.model_name
    )

    predictor.build_image_encoder_engine(
        args.output_path,
        onnx_path=args.onnx_path,
        fp16_mode=args.fp16_mode,
        onnx_opset=args.onnx_opset,
        also_run_trtexec=args.also_run_trtexec
    )