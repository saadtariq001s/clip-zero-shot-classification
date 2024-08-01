# CLIP Zero-Shot Image Classification

This repository contains a notebook demonstrating zero-shot image classification using the CLIP (Contrastive Language–Image Pretraining) model. The notebook uses the `transformers` library to load a pre-trained CLIP model and the `gradio` library to create an interactive web interface for predicting image labels.

## Overview

The notebook showcases two main functionalities:

1. **Zero-Shot Image Classification with CLIP:**
   - Loads the CLIP model and processor from the `transformers` library.
   - Uses a pre-defined set of labels to predict the most probable label for a given image.

2. **Interactive Web Interface with Gradio:**
   - Provides a web interface where users can upload an image and enter a label to get the predicted probability.

## Installation

To run the code, you'll need to install the required packages. You can install them using the following command:

```sh
pip install transformers gradio pillow
```

## Usage

### Zero-Shot Image Classification

The notebook includes a script to classify an image using the CLIP model. It reads an image and a set of labels, processes them, and outputs the probabilities for each label.

Example:

```python
from transformers import CLIPModel, AutoProcessor
from PIL import Image

# Load the model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")

# Load the image and define labels
image = Image.open("kittens.jpeg")
labels = ["a photo of a cat", "a photo of a dog"]

# Process the inputs and get the output
inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
outputs = model(**inputs)

# Calculate probabilities
probs = outputs.logits_per_image.softmax(dim=1)[0]
for i in range(len(labels)):
    print(f"label: {labels[i]} - probability of {probs[i].item():.4f}")
```

### Gradio Web Interface

The notebook also sets up a Gradio interface for interactive use. It allows users to upload an image and input a label to get the predicted probability.

Example usage:

```python
import gradio as gr
from transformers import CLIPModel, AutoProcessor
from PIL import Image

# Load the model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")

def predict(image, label):
    inputs = processor(text=label, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)[0]
    return f"Label: {label} - Probability: {probs.item():.4f}"

# Create a Gradio interface
gradio_app = gr.Interface(
    fn=predict,
    inputs=[gr.Image(type="pil"), gr.Textbox(label="Label")],
    outputs="text",
    title="CLIP Image Classification",
    description="Label an image with a caption and get the predicted probability.",
)

# Launch the Gradio app
gradio_app.launch()
```

## Note

- The script might output warnings about unused or unrecognized arguments like `padding`. These can typically be ignored.

## License

This project is licensed under the MIT License.
