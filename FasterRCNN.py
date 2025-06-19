import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import numpy as np
import os
import cv2 as cv

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib.pyplot as plt
import torchvision.transforms as T


def plot_detections(image_path, boxes, labels, scores, threshold=0.5):
    image = np.array(Image.open(image_path).convert("RGB"))
    plt.figure(figsize=(5, 5))
    # plt.imshow(image)
    ax = plt.gca()

    # Define a color for each label (assuming labels are from 1 to num_classes)
    colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()

    for box, label, score in zip(boxes, labels, scores):
        if score >= threshold:
            color = colors[label]
            xmin, ymin, xmax, ymax = box
            rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, edgecolor=color, linewidth=2)
            ax.add_patch(rect)
            ax.text(xmin, ymin, f'{label}: {score:.2f}', bbox=dict(facecolor=color, alpha=0.5), fontsize=10,
                    color='white')

    # plt.axis('off')
    # plt.show()
    return cv.resize(image, (256, 256))


# Load the model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

num_classes = 4  # 1 class (Paddy Plant) + background

# Get the number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# Replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Set the model to evaluation mode
model.eval()


# Function to load and preprocess the image
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([T.ToTensor()])
    return transform(image).unsqueeze(0)  # Add batch dimension


def Model_RCNN():
    path = './Dataset'
    out_dir = os.listdir(path)
    for i in range(len(out_dir)):
        in_dir = path + '/' + out_dir[i]
        folder_name = os.listdir(in_dir)
        for j in range(len(folder_name)):
            Image_Name = in_dir + '/' + folder_name[j]
            # Example image path
            image_path = Image_Name

            # Load and preprocess the image
            image = load_image(image_path)

            # Perform object detection
            with torch.no_grad():
                prediction = model(image)

            # Extracting bounding boxes, labels, and scores
            boxes = prediction[0]['boxes'].cpu().numpy()
            labels = prediction[0]['labels'].cpu().numpy()
            scores = prediction[0]['scores'].cpu().numpy()

            # Visualize the detections
            plot_detections(image_path, boxes, labels, scores)

            return image_path




