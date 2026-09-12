
# AI-Based Suspicious QR Code Detection - ML + DL

This version contains both Machine Learning and Deep Learning.

## ML
Random Forest analyzes decoded QR content and URL/text features.

## DL
A small Convolutional Neural Network (CNN) analyzes the QR image.

## Workflow

QR Image
  -> CNN (Deep Learning)
  -> OpenCV QR Decoder
  -> Feature Extraction
  -> Random Forest (Machine Learning)
  -> Combined AI Result
  -> Streamlit

## Run on Windows

Open VS Code Terminal in this folder.

1. Create environment:
python -m venv venv

2. Activate:
venv\Scripts\activate

3. Install:
pip install -r requirements.txt

4. Train ML:
python train_model.py

5. Train CNN:
python train_dl_model.py

6. Start app:
streamlit run app.py

## Note about the CNN

The included CNN is a simple educational demonstration trained on generated QR-like images so the project remains easy to run without a large image dataset. For a final research/college project, replace it with a properly labeled real QR image dataset.

The system is an educational prototype, not a production cybersecurity scanner.
