import onnx

model_path = r"D:\blind\HelloNetraOCR\app\src\main\assets\rec\en_PP-OCRv5_rec_mobile.onnx"
dict_path = r"D:\blind\HelloNetraOCR\app\src\main\assets\rec\ppocrv5_en_dict.txt"

model = onnx.load(model_path)

print("========== MODEL METADATA ==========")

metadata = {
    p.key: p.value
    for p in model.metadata_props
}

print("Metadata keys:")
for k in metadata:
    print(" ", repr(k))

print("\nCharacter metadata:")
for key in ("character", "characters", "character_dict", "character_dict_path"):
    if key in metadata:
        value = metadata[key]
        print("FOUND:", key)
        print("Length:", len(value))
        print("First 200 chars:")
        print(repr(value[:200]))
        print("Last 100 chars:")
        print(repr(value[-100:]))

print("\n========== EXTERNAL DICTIONARY ==========")

with open(dict_path, "r", encoding="utf-8") as f:
    chars = [line.rstrip("\r\n") for line in f]

print("Dictionary length:", len(chars))
print("First 30:", chars[:30])
print("Last 30:", chars[-30:])

if "character" in metadata:
    embedded = metadata["character"]

    # Try common representations
    embedded_lines = embedded.splitlines()

    print("\nEmbedded splitlines:", len(embedded_lines))
    print("First 30 embedded:", embedded_lines[:30])

    if embedded_lines == chars:
        print("\nRESULT: EXACT MATCH")
    else:
        print("\nRESULT: DOES NOT MATCH")

        max_len = min(len(embedded_lines), len(chars))

        for i in range(max_len):
            if embedded_lines[i] != chars[i]:
                print(
                    "First mismatch at index:",
                    i,
                    "embedded=",
                    repr(embedded_lines[i]),
                    "external=",
                    repr(chars[i]),
                )
                break
else:
    print("\nNo 'character' metadata found in this ONNX model.")

print("\nDone.")