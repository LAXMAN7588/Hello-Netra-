import numpy as np
import onnxruntime as ort

DET = r"D:\blind\HelloNetraOCR\app\src\main\assets\det\ch_PP-OCRv5_det_mobile.onnx"
REC = r"D:\blind\HelloNetraOCR\app\src\main\assets\rec\en_PP-OCRv5_rec_mobile.onnx"
DICT = r"D:\blind\HelloNetraOCR\app\src\main\assets\rec\ppocrv5_en_dict.txt"

print("========== DETECTOR ==========")

det_session = ort.InferenceSession(
    DET,
    providers=["CPUExecutionProvider"]
)

print("Providers:", det_session.get_providers())

det_input = det_session.get_inputs()[0]
det_output = det_session.get_outputs()[0]

print("Input name:", det_input.name)
print("Input declared shape:", det_input.shape)
print("Output name:", det_output.name)
print("Output declared shape:", det_output.shape)

# Test with 640x640 float32 input
det_x = np.zeros((1, 3, 640, 640), dtype=np.float32)

det_result = det_session.run(
    [det_output.name],
    {det_input.name: det_x}
)[0]

print("Runtime output shape:", det_result.shape)
print("Runtime output dtype:", det_result.dtype)
print("Runtime output min:", float(det_result.min()))
print("Runtime output max:", float(det_result.max()))
print("Runtime output mean:", float(det_result.mean()))


print("\n========== RECOGNIZER ==========")

rec_session = ort.InferenceSession(
    REC,
    providers=["CPUExecutionProvider"]
)

print("Providers:", rec_session.get_providers())

rec_input = rec_session.get_inputs()[0]
rec_output = rec_session.get_outputs()[0]

print("Input name:", rec_input.name)
print("Input declared shape:", rec_input.shape)
print("Output name:", rec_output.name)
print("Output declared shape:", rec_output.shape)

# Standard PP-OCR recognition test size
rec_x = np.zeros((1, 3, 48, 320), dtype=np.float32)

rec_result = rec_session.run(
    [rec_output.name],
    {rec_input.name: rec_x}
)[0]

print("Runtime output shape:", rec_result.shape)
print("Runtime output dtype:", rec_result.dtype)
print("Runtime output min:", float(rec_result.min()))
print("Runtime output max:", float(rec_result.max()))
print("Runtime output mean:", float(rec_result.mean()))

print("\n========== DICTIONARY ==========")

with open(DICT, "r", encoding="utf-8") as f:
    chars = [line.rstrip("\r\n") for line in f]

print("Dictionary entries:", len(chars))
print("First 20 entries:", chars[:20])
print("Expected model classes:", len(chars) + 2)

print("\nDone.")