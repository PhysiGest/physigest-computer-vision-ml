from ultralytics import YOLO
import os


def extract_and_save_labels(model_path, output_file):
    # Load the model
    model = YOLO(model_path)

    # Extract the class names
    class_names = model.names

    # Sort the class names by index
    sorted_names = [class_names[i] for i in range(len(class_names))]

    # Save the labels to a text file
    with open(output_file, "w") as f:
        for name in sorted_names:
            f.write(f"{name}\n")

    return sorted_names


def main():
    # Path to your .pt file
    pt_file_path = "yolov8s-seg.pt"

    # Output file path
    output_file = "./yolo-seg-classes.txt"

    # Extract and save labels
    labels = extract_and_save_labels(pt_file_path, output_file)

    if labels:
        print(f"Labels extracted and saved to {output_file}")
        print(f"Number of labels: {len(labels)}")
        print("\nExample usage:")
        print("with open('./yolo_classes.txt', 'r') as f:")
        print("    self.labels = [line.strip() for line in f.readlines()]")
        print("print(f'Number of labels: {len(self.labels)}')")
        print("# To get a label: self.labels[class_id]")
    else:
        print("Unable to extract labels. Please check your model file.")


if __name__ == "__main__":
    main()
