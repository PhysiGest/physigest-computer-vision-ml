import onnx


def inspect_onnx_model(model_path):
    # Load the ONNX model
    model = onnx.load(model_path)

    # Check the model's input
    print("Model Input Information:")
    for input in model.graph.input:
        print(f"Name: {input.name}")
        print(f"Shape: {[dim.dim_value for dim in input.type.tensor_type.shape.dim]}")
        print(f"Type: {input.type.tensor_type.elem_type}\n")

    # Check the model's output
    print("Model Output Information:")
    for output in model.graph.output:
        print(f"Name: {output.name}")
        print(f"Shape: {[dim.dim_value for dim in output.type.tensor_type.shape.dim]}")
        print(f"Type: {output.type.tensor_type.elem_type}\n")


# Specify the path to your ONNX model
model_path = "yolov8s-seg.onnx"
inspect_onnx_model(model_path)
