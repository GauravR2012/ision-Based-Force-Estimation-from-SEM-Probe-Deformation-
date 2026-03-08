# Vision-Based Force Estimation from SEM Probe Deformation

A computer vision and deep learning framework for estimating micro/nanoscale forces by analyzing probe deformation in Scanning Electron Microscope (SEM) video data.

---

## Overview

Measuring forces at the micro–nano scale is challenging because traditional sensors such as strain gauges and piezoresistive sensors introduce noise, complexity, or intrusive instrumentation. This project proposes a **vision-based force sensing system** that estimates applied force by analyzing deformation of a micro-cantilever probe in SEM video frames.

The system uses computer vision techniques and deep learning models to learn the mapping between **probe deformation features and applied force values**, enabling accurate force estimation without physical force sensors.

---

## Pipeline

![Pipeline](assets/pipeline.png)

The full system pipeline consists of the following stages:

1. **Frame Extraction**
2. **Probe Detection (YOLOv5)**
3. **Probe Segmentation (SAM2)**
4. **Image Processing & Contour Extraction**
5. **Geometric Feature Extraction**
6. **Machine Learning Force Prediction**

---

## Features Extracted

The following geometric and spatial features are extracted from probe contours:

* centroid coordinates (cx, cy)
* bounding box parameters
* contour area and perimeter
* aspect ratio
* extent
* eccentricity
* orientation angle
* Hu moments
* **tip deflection (most important feature)**

These features represent probe deformation caused by applied forces.

---

## Models Used

Three machine learning architectures were evaluated:

* **Feedforward Neural Network (FNN)**
* **Long Short-Term Memory (LSTM)**
* **Transformer Model**

The **Transformer model achieved the best prediction performance** due to its ability to capture temporal relationships in deformation patterns.

---

## Example Results

Predicted force vs ground truth:

![Prediction](assets/force_prediction.png)

Training curves comparison:

![Training Curves](assets/training_curves.png)

---

## Repository Structure

```
vision-based-force-estimation
│
├── README.md
├── requirements.txt
│
├── src
│   ├── frame_extraction.py
│   ├── segmentation_processing.py
│   ├── feature_extraction.py
│   ├── dataset_builder.py
│   ├── train_models.py
│   └── inference.py
│
├── assets
│   ├── pipeline.png
│   ├── training_curves.png
│   └── force_prediction.png
│
├── demo
│   └── probe_deflection.gif
│
└── sample_data
    └── example_frame.png
```

---

## Installation

Clone the repository

```
git clone https://github.com/YOUR_USERNAME/vision-based-force-estimation
cd vision-based-force-estimation
```

Install dependencies

```
pip install -r requirements.txt
```

---

## Running the Pipeline

Example workflow:

```
python src/frame_extraction.py
python src/feature_extraction.py
python src/train_models.py
```

---

## Applications

* Nanomanipulation inside SEM
* Nanorobotics research
* MEMS/NEMS testing
* AFM probe characterization
* Bio-cellular force analysis

---

## Author

**Gaurav Ramteke**
Indian Institute of Technology Kharagpur

---

## License

MIT License
