## 下载模型文件
```
分别下载owlv2、AltCLIP、chinese-clip-vit-large-patch14模型
python download_llm.py
```

## 准备镜像
```
docker pull nvcr.io/nvidia/tensorrt:23.10-py3
docker pull nvcr.io/nvidia/tritonserver:23.10-py3
docker pull nvcr.io/nvidia/deepstream:7.0-triton-multiarch
docker pull bluenviron/mediamtx:latest-ffmpeg
```

## owlv2模型onnx转换
```
python -m nanoowl.build_image_encoder_engine data/owlv2_engine_models/owlv2_image_encoder_large_patch14.engine \
    --model_name $PWD/models/google/owlv2-large-patch14-ensemble \
    --onnx_path $PWD/data/owlv2_onnx_models/owlv2_image_encoder_large_patch14.onnx
```

## owlv2 onnx模型engine转换
```
docker run --rm -it --gpus="device=0" --shm-size 32g -v $PWD/nanoowlv2:/workspace/nanoowlv2 nvcr.io/nvidia/tensorrt:23.10-py3 /bin/bash

/usr/src/tensorrt/bin/trtexec --onnx=data/owlv2_onnx_models/owlv2_image_encoder_large_patch14.onnx \
    --saveEngine=data/owlv2_engine_models/owlv2_image_encoder_large_patch14.engine \
    --fp16 --shapes=image:1x3x1008x1008
```

## AltCLIP onnx模型engine转换
```
从网盘下载onnx模型到data目录下，保存为altclip_text.onnx
docker run --rm -it --gpus="device=0" --shm-size 32g -v $PWD/nanoowlv2:/workspace/nanoowlv2 nvcr.io/nvidia/tensorrt:23.10-py3 /bin/bash

/usr/src/tensorrt/bin/trtexec --onnx=data/altclip_onnx_models/altclip_text.onnx \
    --saveEngine=data/altclip_engine_models/altclip_text.engine \
    --fp16 --shapes=input_ids:1x77,attention_mask:1x77
```

## chinese-clip-vit-large-patch14模型onnx转换
```
git clone https://github.com/OFA-Sys/Chinese-CLIP.git

<!-- 创建虚拟环境，并安装依赖 -->
cd Chinese-CLIP
python -m pip install onnx==1.13.0 onnxruntime-gpu==1.13.1 onnxmltools==1.11.1
python -m pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install -r requirements.txt 

export PYTHONPATH=${PYTHONPATH}:$PWD/cn_clip
checkpoint_path=$PWD/../models/AI-ModelScope/chinese-clip-vit-large-patch14/clip_cn_vit-l-14.pt

python cn_clip/deploy/pytorch_to_onnx.py \
       --model-arch ViT-L-14 \
       --pytorch-ckpt-path ${checkpoint_path} \
       --save-onnx-path data/cn_clip_onnx_models \
       --convert-text --convert-vision

<!-- 或者直接拉取镜像 -->
docker pull cnstark/pytorch:1.12.0-py3.9.12-cuda11.6.2-ubuntu20.04
docker run --rm -it --gpus="device=0" --shm-size 32g -v $PWD/models:/workspace/Chinese-CLIP/models -v $PWD/Chinese-CLIP:/workspace/Chinese-CLIP cnstark/pytorch:1.12.0-py3.9.12-cuda11.6.2-ubuntu20.04 /bin/bash

cd /workspace/Chinese-CLIP
python -m pip install onnx==1.13.0 onnxruntime-gpu==1.13.1 onnxmltools==1.11.1
python -m pip install -r requirements.txt 

export PYTHONPATH=${PYTHONPATH}:$PWD/cn_clip
checkpoint_path=$PWD/models/AI-ModelScope/chinese-clip-vit-large-patch14/clip_cn_vit-l-14.pt

python cn_clip/deploy/pytorch_to_onnx.py \
       --model-arch ViT-L-14 \
       --pytorch-ckpt-path ${checkpoint_path} \
       --save-onnx-path data/cn_clip_onnx_models \
       --convert-text --convert-vision

```

## chinese-clip-vit-large-patch14 onnx模型归一化处理
```
<!-- python环境跟chinese-clip-vit-large-patch14保持一致 -->
git clone https://github.com/zjd1988/tiny_torch_trainer.git
cd tiny_torch_trainer
<!-- 修改转换脚本process_clip_v2.py的模型输入和输出路径 -->
playground/model_convertor/models/cnclip/process_clip_v2.py
执行即可python playground/model_convertor/models/cnclip/process_clip_v2.py

```

## chinese-clip-vit-large-patch14 onnx模型engine转换
```
docker run --rm -it --gpus="device=0" --shm-size 32g -v $PWD/nanoowlv2:/workspace/nanoowlv2 nvcr.io/nvidia/tensorrt:23.10-py3 /bin/bash

<!-- visual model -->
/usr/src/tensorrt/bin/trtexec --onnx=data/cnclip_onnx_models/ViT-L-14.img.deploy.onnx --saveEngine=data/cnclip_engine_models/cnclip_visual.engine --fp16

<!-- text model -->
/usr/src/tensorrt/bin/trtexec --onnx=data/cnclip_onnx_models/ViT-L-14.txt.fp32.onnx --saveEngine=data/cnclip_engine_models/cnclip_text.engine --fp16
```

## tritonserver模型加载测试
```
<!-- 将转换后的engine文件拷贝triton_models指定位置 -->
cp data/cn_clip_engine_models/cnclip_text.engine triton_models/cnclip_text/1/model.plan
cp data/cn_clip_engine_models/cnclip_visual.engine triton_models/cnclip_visual/1/model.plan
cp data/owlv2_engine_models/owlv2_visual.engine triton_models/owlv2_visual/1/model.plan
cp data/altclip_engine_models/altclip_text.engine triton_models/altclip_text/1/model.plan

docker run --rm -it --gpus="device=0" --shm-size 32g -p 9001:9001-v $PWD/triton_models:/workspace/triton_models \
    nvcr.io/nvidia/tensorrt:23.10-py3 /bin/bash
or 
docker run --rm -it --gpus all --shm-size 32g -p 9001:9001-v $PWD/triton_models:/workspace/triton_models \
    nvcr.io/nvidia/tensorrt:23.10-py3 /bin/bash

tritonserver --model-repository=/workspace/models --grpc-port=9991
or
tritonserver --model-repository=/workspace/models --grpc-port=9991 --model-control-mode=explicit
```

## 推流拉流测试
```
<!-- 启动mediamtx流服务器 -->
docker run --rm -it -v $PWD/test_videos:/workspace/test_videos -e MTX_RTSPTRANSPORTS=tcp -p 18554:8554 bluenviron/mediamtx:latest-ffmpeg

<!-- 进入流服务器镜像 -->
docker exec -it xxxxx /bin/sh

<!-- 推流 -->
ffmpeg -re -stream_loop -1 -i sample_1080p_h264.mp4 -c copy -f rtsp rtsp://127.0.0.1:8554/mystream

<!-- 拉流 -->
使用vlc软件拉流: rtsp://xxx.xxx.xxx.xxx:18554/mystream
使用gst-paly拉流: gst-play-1.0 rtsp://localhost:18554/mystream
使用ffmpeg拉流: ffmpeg -i rtsp://localhost:18554/mystream -c copy output.mp4
```

