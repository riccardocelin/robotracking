import onnx

model = onnx.load("model.onnx")
print([i.name for i in model.graph.input])
print([o.name for o in model.graph.output])