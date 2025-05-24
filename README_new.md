## 下载模型文件
```
分别下载owlv2和AltCLIP模型
python download_llm.py

```

## 准备镜像
```
docker pull nvcr.io/nvidia/tensorrt:23.10-py3
docker pull nvcr.io/nvidia/tensorrt:23.10-py3
```

## owlv2模型onnx转换
```
python -m nanoowl.build_image_encoder_engine data/owlv2_image_encoder_large_patch14.engine \
    --model_name $PWD/models/google/owlv2-large-patch14-ensemble --onnx_path $PWD/data/owlv2_image_encoder_large_patch14.onnx
```

## owlv2 onnx模型engine转换
```
docker run -it --gpus="device=0" --shm-size 32g -v $PWD/nanoowlv2:/workspace/nanoowlv2 nvcr.io/nvidia/tensorrt:23.10-py3 /bin/bash

/usr/src/tensorrt/bin/trtexec --onnx=data/owlv2_image_encoder_large_patch14.onnx \
    --saveEngine=data/owlv2_image_encoder_large_patch14.engine --fp16 --shapes=image:1x3x1008x1008
```

## AltCLIP模型engine转换
```
从网盘下载onnx模型到data目录下，保存为altclip_model.onnx

docker run -it --gpus="device=0" --shm-size 32g -v $PWD/nanoowlv2:/workspace/nanoowlv2 nvcr.io/nvidia/tensorrt:23.10-py3 /bin/bash

/usr/src/tensorrt/bin/trtexec --onnx=data/altclip_model.onnx --saveEngine=data/altclip_model.engine --fp16 \
    --shapes=input_ids:1x77,attention_mask:1x77
```


## 模型测试
```

```

