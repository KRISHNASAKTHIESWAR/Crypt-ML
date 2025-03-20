# with open("models/MobileFaceNet.onnx.encrypted", "rb") as f:
#     print(f.read(20).hex())

with open("models/MobileFaceNet.onnx.encrypted", "rb") as f:
    print(f.read(100).decode(errors="ignore"))
