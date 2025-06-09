import PIL
import time
import torch
import numpy as np
import onnxruntime as ort
import tritonclient.grpc as grpcclient
from transformers import AltCLIPProcessor
from nanoowl.owl_drawing import draw_owl_output
from nanoowl.owl_predictor import OwlPredictor


def test_altclip_tritonserver_perf(altclip_pt_path, triton_url, test_count):
    # prepare inputs
    altclip_processor = AltCLIPProcessor.from_pretrained(altclip_pt_path)
    text = [r"一只猫头鹰",]
    text_input = altclip_processor(text=text, return_tensors="np", padding='max_length', max_length=77, truncation=True)
    input_ids = text_input['input_ids']
    attention_mask = text_input['attention_mask']

    model_name = "altclip_text"
    triton_client = grpcclient.InferenceServerClient(url=triton_url)
    input_ids_tensor = grpcclient.InferInput("input_ids", input_ids.shape, "INT32")
    input_ids_tensor.set_data_from_numpy(input_ids.astype(np.int32))
    attention_mask_tensor = grpcclient.InferInput("attention_mask", attention_mask.shape, "INT32")
    attention_mask_tensor.set_data_from_numpy(attention_mask.astype(np.int32))
    inputs = [input_ids_tensor, attention_mask_tensor]
    outputs = [grpcclient.InferRequestedOutput("query_embed")]

    # warmup
    response = triton_client.infer(model_name, inputs, outputs=outputs)

    # test
    start = time.time()
    count = 0
    while count < test_count:
        response = triton_client.infer(model_name, inputs, outputs=outputs)
        count += 1
    end = time.time()
    print(f"{model_name} test_count {test_count} average infer time is {(end-start)/test_count}")


def test_owlv2_tritonserver_perf(owlv2_pt_path, triton_url, test_count):
    # prepare inputs
    owl_predictor = OwlPredictor(owlv2_pt_path, image_encoder_engine=None, no_roi_align=True)
    image = PIL.Image.open("assets/owl_glove_small.jpg")
    text = [r"一只猫头鹰",]
    text_encodings = owl_predictor.encode_text(text)
    pixel_values = owl_predictor.processor(images=image, return_tensors="np").pixel_values
    query_embeds = text_encodings.text_embeds.detach().cpu().numpy()

    model_name = "owlv2_visual"
    triton_client = grpcclient.InferenceServerClient(url=triton_url)
    pixel_values_tensor = grpcclient.InferInput("pixel_values", pixel_values.shape, "FP32")
    pixel_values_tensor.set_data_from_numpy(pixel_values)
    query_embeds_tensor = grpcclient.InferInput("query_embeds", query_embeds.shape, "FP32")
    query_embeds_tensor.set_data_from_numpy(query_embeds)
    inputs = [pixel_values_tensor, query_embeds_tensor]
    outputs = [grpcclient.InferRequestedOutput("target_class_logits"), 
        grpcclient.InferRequestedOutput("objectnesses"),
        grpcclient.InferRequestedOutput("boxes")]

    # warmup
    response = triton_client.infer(model_name, inputs, outputs=outputs)

    # test
    start = time.time()
    count = 0
    while count < test_count:
        response = triton_client.infer(model_name, inputs, outputs=outputs)
        count += 1
    end = time.time()
    print(f"{model_name} test_count {test_count} average infer time is {(end-start)/test_count}")


def compare_altclip_onnx_with_tritonserver(altclip_pt_path, altclip_onnx_path, triton_url):
    altclip_processor = AltCLIPProcessor.from_pretrained(altclip_pt_path)
    text = [r"一只猫头鹰",]
    text_input = altclip_processor(text=text, return_tensors="np", padding='max_length', max_length=77, truncation=True)
    # get onnxruntime infer result
    input_ids = text_input['input_ids']
    attention_mask = text_input['attention_mask']
    text_encode_session = ort.InferenceSession(altclip_onnx_path)
    onnx_inputs = {}
    onnx_inputs["input_ids"] = input_ids
    onnx_inputs["attention_mask"] = attention_mask
    onnx_outputs = ["query_embed",]
    onnx_result = text_encode_session.run(onnx_outputs, onnx_inputs)[0]

    # get tritonserver infer result
    model_name = "altclip_text"
    triton_client = grpcclient.InferenceServerClient(url=triton_url)
    input_ids_tensor = grpcclient.InferInput("input_ids", input_ids.shape, "INT32")
    input_ids_tensor.set_data_from_numpy(input_ids.astype(np.int32))
    attention_mask_tensor = grpcclient.InferInput("attention_mask", attention_mask.shape, "INT32")
    attention_mask_tensor.set_data_from_numpy(attention_mask.astype(np.int32))
    inputs = [input_ids_tensor, attention_mask_tensor]
    outputs = [grpcclient.InferRequestedOutput("query_embed")]
    response = triton_client.infer(model_name, inputs, outputs=outputs)
    grpc_result = response.as_numpy("query_embed")
    cmp_result = np.allclose(grpc_result, onnx_result, atol=0.001)
    print(f"{model_name} onnx result compare with triton result is {cmp_result}")


def compare_owlv2_onnx_with_tritonserver(owlv2_pt_path, owlv2_onnx_path, triton_url):
    owl_predictor = OwlPredictor(owlv2_pt_path, image_encoder_engine=None, no_roi_align=True)
    image = PIL.Image.open("assets/owl_glove_small.jpg")
    text = [r"一只猫头鹰",]
    text_encodings = owl_predictor.encode_text(text)
    pixel_values = owl_predictor.processor(images=image, return_tensors="np").pixel_values
    query_embeds = text_encodings.text_embeds.detach().cpu().numpy()

    # get onnxruntime infer result
    owlv2_session = ort.InferenceSession(owlv2_onnx_path)
    onnx_inputs = {}
    onnx_inputs["pixel_values"] = pixel_values
    onnx_inputs["query_embeds"] = query_embeds
    onnx_outputs = ["target_class_logits", "objectnesses", "boxes"]
    onnx_result = owlv2_session.run(onnx_outputs, onnx_inputs)[0]

    # get tritonserver infer result
    model_name = "owlv2_visual"
    triton_client = grpcclient.InferenceServerClient(url=triton_url)
    pixel_values_tensor = grpcclient.InferInput("pixel_values", pixel_values.shape, "FP32")
    pixel_values_tensor.set_data_from_numpy(pixel_values)
    query_embeds_tensor = grpcclient.InferInput("query_embeds", query_embeds.shape, "FP32")
    query_embeds_tensor.set_data_from_numpy(query_embeds)
    inputs = [pixel_values_tensor, query_embeds_tensor]
    outputs = [grpcclient.InferRequestedOutput("target_class_logits"), 
        grpcclient.InferRequestedOutput("objectnesses"),
        grpcclient.InferRequestedOutput("boxes")]
    response = triton_client.infer(model_name, inputs, outputs=outputs)
    grpc_result = response.as_numpy("target_class_logits")
    cmp_result = np.allclose(grpc_result, onnx_result, atol=0.01)
    # cmp_result = np.max(np.abs(grpc_result - onnx_result) / np.abs(grpc_result)) < 0.0001
    print(f"{model_name} onnx result compare with triton result is {cmp_result}")


def test_owlv2_pt(owlv2_pt_path, save_path):
    owl_predictor = OwlPredictor(owlv2_pt_path, image_encoder_engine=None, no_roi_align=True)
    image = PIL.Image.open("assets/owl_glove_small.jpg")
    text = ["an owl", "a glove"]
    # text = ["an owl",]
    # text = ["a glove",]
    text_encodings = owl_predictor.encode_text(text)
    output = owl_predictor.predict_pt(image=image, text=text, text_encodings=text_encodings, threshold=0.05, nms_threshold=0.3)
    print(output)
    image = draw_owl_output(image, output, text=text, draw_text=True)
    image.save(save_path)


def test_owlv2_onnx(owlv2_pt_path, owlv2_onnx_path, save_path):
    owl_predictor = OwlPredictor(owlv2_pt_path, image_encoder_engine=None, no_roi_align=True)
    image = PIL.Image.open("assets/owl_glove_small.jpg")
    text = ["an owl", "a glove"]
    # text = ["an owl",]
    # text = ["a glove",]
    text_encodings = owl_predictor.encode_text(text)
    session = ort.InferenceSession(owlv2_onnx_path)
    output = owl_predictor.predict_with_onnx(image=image, text=text, text_encodings=text_encodings, onnx_session=session, threshold=0.05, nms_threshold=0.3)
    print(output)
    image = draw_owl_output(image, output, text=text, draw_text=True)
    image.save(save_path)


def test_owlv2_tritonserver(owlv2_pt_path, triton_url, save_path):
    # prepare inputs
    owl_predictor = OwlPredictor(owlv2_pt_path, image_encoder_engine=None, no_roi_align=True)
    image = PIL.Image.open("assets/owl_glove_small.jpg")
    text = ["an owl",]
    text_encodings = owl_predictor.encode_text(text)
    pixel_values = owl_predictor.processor(images=image, return_tensors="np").pixel_values
    query_embeds = text_encodings.text_embeds.detach().cpu().numpy()

    model_name = "owlv2_visual"
    triton_client = grpcclient.InferenceServerClient(url=triton_url)
    pixel_values_tensor = grpcclient.InferInput("pixel_values", pixel_values.shape, "FP32")
    pixel_values_tensor.set_data_from_numpy(pixel_values)
    query_embeds_tensor = grpcclient.InferInput("query_embeds", query_embeds.shape, "FP32")
    query_embeds_tensor.set_data_from_numpy(query_embeds)
    inputs = [pixel_values_tensor, query_embeds_tensor]
    outputs = [grpcclient.InferRequestedOutput("target_class_logits"), 
        grpcclient.InferRequestedOutput("objectnesses"),
        grpcclient.InferRequestedOutput("boxes")]
    response = triton_client.infer(model_name, inputs, outputs=outputs)

    infer_results = []
    target_class_logits = response.as_numpy("target_class_logits")
    objectnesses = response.as_numpy("objectnesses")
    boxes = response.as_numpy("boxes")
    infer_results.append(target_class_logits)
    infer_results.append(objectnesses)
    infer_results.append(boxes * max(image.size))
    output = owl_predictor.decode_onnx(infer_results, threshold=0.1, nms_threshold=0.3, class_based_nms=True)
    print(output)
    image = draw_owl_output(image, output, text=text, draw_text=True)
    image.save(save_path)


if __name__ == "__main__":
    altclip_pt_path = "/data1/zhaojd-a/github_codes/clip_owl_prj/nanoowlv2/models/BAAI/AltCLIP"
    altclip_onnx_path = "/data1/zhaojd-a/github_codes/clip_owl_prj/nanoowlv2/data/altclip_model.onnx"
    owlv2_pt_path = "/data1/zhaojd-a/github_codes/clip_owl_prj/nanoowlv2/models/google/owlv2-large-patch14-ensemble"
    owlv2_onnx_path = "/data1/zhaojd-a/github_codes/clip_owl_prj/nanoowlv2/data/owl_image_encoder_large_patch14_new.onnx"
    triton_url = "127.0.0.1:9991"
    # compare_altclip_onnx_with_tritonserver(altclip_pt_path, altclip_onnx_path, triton_url)
    # test_altclip_tritonserver_perf(altclip_pt_path, triton_url, 100)
    # compare_owlv2_onnx_with_tritonserver(owlv2_pt_path, owlv2_onnx_path, triton_url)
    # test_owlv2_tritonserver_perf(owlv2_pt_path, triton_url, 100)
    # test_owlv2_pt(owlv2_pt_path, "owlv2_pt_result.jpg")
    # test_owlv2_onnx(owlv2_pt_path, owlv2_onnx_path, "owlv2_onnx_result.jpg")
    test_owlv2_tritonserver(owlv2_pt_path, triton_url, "owlv2_tritonserver_result.jpg")