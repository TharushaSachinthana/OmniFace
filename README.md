# OmniFace: Comprehensive Face Detection and Profiling System

OmniFace is an advanced multi-attribute classification system leveraging deep learning to detect and classify faces based on demographic attributes such as age, gender, and ethnicity. It combines state-of-the-art computer vision techniques with a robust neural network architecture.

---

## Key Features

- **Face Detection**: Utilizes pre-trained models for high-accuracy face detection.
- **Demographic Classification**: Predicts age, gender, and ethnicity for each detected face.
- **Multi-Output Neural Network**: Handles multiple outputs (age, gender, ethnicity) with shared feature extraction layers.
- **Customizable Inference**: Easily adapt the model for custom datasets or tasks.
- **Efficient Data Pipeline**: Simplifies dataset preprocessing and augmentation.

---

## Project Structure

```plaintext
OmniFace/
├── data/
│   └── dataset/          # Placeholder for datasets
├── notebooks/
│   ├── 17-10-24_BoundingBoxGen.ipynb   # Bounding box generator notebook
│   ├── final_multimodel.ipynb          # Training and evaluation notebook
│   ├── inference.ipynb                 # Inference notebook for custom images
├── src/
│   ├── train.py                        # Training script
│   ├── evaluate.py                     # Evaluation script
│   ├── predict.py                      # Custom image prediction
│   ├── model.py                        # Model architecture definition
│   ├── data_preprocessing.py           # Dataset loading and preprocessing
│   ├── bounding_box.py                 # Bounding box generator module
├── tests/
│   ├── test_train.py                   # Unit tests for training pipeline
│   ├── test_model.py                   # Unit tests for model architecture
├── requirements.txt                    # Dependencies
├── README.md                           # Project documentation
└── LICENSE                             # Project license
```

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- NVIDIA GPU with CUDA (optional for faster training)
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

---

## Usage

### 1. Dataset Preparation

- Download the [UTKFace Dataset](https://susanqq.github.io/UTKFace/) and [WIDER Face Dataset](https://universe.roboflow.com/large-benchmark-datasets/wider-face-ndtcz).
- Place datasets in the `data/dataset/` directory.

### 2. Training the Model

Train the multi-output model using the following command:
```bash
python src/train.py
```

### 3. Evaluation

Evaluate model performance on validation data:
```bash
python src/evaluate.py
```

### 4. Custom Inference

Generate predictions for custom images:
```bash
python src/predict.py --image_path /path/to/image.jpg
```

---

## Results

The system outputs a structured profile for each detected face, including:
- **Age**: Predicted as an integer (e.g., 25).
- **Gender**: Predicted as "Male" or "Female".
- **Ethnicity**: One of the predefined categories (e.g., Asian, African, etc.).

### Example

**Input Image:**
![Example Input](https://via.placeholder.com/150)

**Predicted Output:**
```json
{
  "Age": 25,
  "Gender": "Male",
  "Ethnicity": "Asian"
}
```

---

## Testing

Run unit tests to ensure the codebase is functional:
```bash
pytest tests/
```

---

## Future Enhancements

- **Real-Time Analysis**: Implement live video processing.
- **Emotion Detection**: Add emotional state classification.
- **Mobile Deployment**: Develop a lightweight version for mobile devices.
- **Explainability**: Integrate model interpretability tools.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Authors

- **M.K.T.S. Thilakarathna (2020/E/159)**
- **A.B. Weerakoon (2020/E/169)**

---

We welcome contributions and feedback to improve OmniFace. Feel free to open issues or submit pull requests!
