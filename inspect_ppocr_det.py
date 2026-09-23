import onnx

path = r"D:\blind\HelloNetraOCR\app\src\main\assets\det\ch_PP-OCRv5_det_mobile.onnx"

model = onnx.load(path)

print("=== PP-OCRv5 DETECTOR ===")
print("IR version:", model.ir_version)

print("\nOpsets:")
for opset in model.opset_import:
    print("  domain:", opset.domain or "ai.onnx", "version:", opset.version)

print("\nInputs:")
for x in model.graph.input:
    t = x.type.tensor_type
    dims = []
    for d in t.shape.dim:
        if d.dim_value:
            dims.append(str(d.dim_value))
        elif d.dim_param:
            dims.append(d.dim_param)
        else:
            dims.append("?")
    print(f"  {x.name}: shape={dims}, elem_type={t.elem_type}")

print("\nOutputs:")
for x in model.graph.output:
    t = x.type.tensor_type
    dims = []
    for d in t.shape.dim:
        if d.dim_value:
            dims.append(str(d.dim_value))
        elif d.dim_param:
            dims.append(d.dim_param)
        else:
            dims.append("?")
    print(f"  {x.name}: shape={dims}, elem_type={t.elem_type}")