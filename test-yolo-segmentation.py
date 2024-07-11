import onnxruntime as ort
import numpy as np
import cv2


def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    # Resize the image to the input size of the model
    image = cv2.resize(image, (640, 640))
    # Convert the image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Normalize the image to [0, 1]
    image = image.astype(np.float32) / 255.0
    # Transpose the image to CHW format
    image = np.transpose(image, (2, 0, 1))
    # Add a batch dimension
    image = np.expand_dims(image, axis=0)
    return image


def run_inference(model_path, image_path):
    # Preprocess the image
    input_image = preprocess_image(image_path)

    # Load the ONNX model
    session = ort.InferenceSession(model_path)

    # Get input and output names
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]

    # Run inference
    outputs = session.run(output_names, {input_name: input_image})

    return outputs


# Specify the paths
model_path = "yolov8s-seg.onnx"
image_path = "./bus.jpg"

# Run inference and get the output
outputs = run_inference(model_path, image_path)

# Print the output
for i, output in enumerate(outputs):
    print(f"Output {i}: {output.shape}")
    print(output)
