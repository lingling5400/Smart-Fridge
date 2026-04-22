# Smart Refrigerator AI System

📅 Duration: Feb 2026 – Apr 2026

## Project Overview

This project aims to reduce household food waste by using AI-based image recognition to automatically classify the freshness of food ingredients stored in a refrigerator.

The system detects different types of ingredients (fresh vs. spoiled), generates a real-time inventory list of fridge contents, and integrates a Large Language Model (LLM) to provide personalized meal and recipe suggestions, including vegetarian-friendly options.

---

## Key Features

- Food ingredient detection (fresh vs. spoiled)
- Automatic refrigerator inventory management
- AI-powered meal and recipe recommendations
- Vegetarian diet support through LLM integration

---

## Technologies Used

### Deep Learning & AI
- YOLOv11 (Ultralytics) for object detection
- PyTorch / TensorFlow for model development and tuning
- Custom dataset labeling for fruits, vegetables, and meat products
- Data augmentation techniques (rotation, brightness adjustment, etc.)

### Model Training & Optimization
- Hyperparameter tuning (epochs, image size, etc.)
- Performance evaluation and stability analysis
- Baseline model achieved:
  - mAP50: ~0.87
  - mAP50-95: ~0.70

---

## System Architecture

### Frontend
- React 18
- TypeScript
- Tailwind CSS

The frontend displays detection results and integrates with backend inference outputs in a separated architecture.

---

## Deployment

- GitHub for version control
- Render platform for Web Demo deployment

---

## Project Goal

To build an intelligent household assistant that combines computer vision and language models to improve food management efficiency and reduce food waste in daily life.

---

## Dataset Source

- Roboflow Public Dataset  
  https://roboflow.com/

The dataset used in this project was obtained from Roboflow, including annotated images of fruits, vegetables, and meat products for object detection and freshness classification tasks.


## Knowledge Distillation Framework

- YOLO Distiller (Base Implementation)  
  https://github.com/danielsyahputra/yolo-distiller

This project is partially based on the YOLO Distiller framework. It has been adapted and modified to support custom teacher-student training and experimental settings for object detection tasks in this project.

The original implementation has been modified and extended to support:
- Custom teacher-student training pipeline
- Integration with YOLOv11 models
- Experimental adjustments for loss configuration and training settings

---

## License

This project is licensed based on the YOLO Distiller repository.

- Original License Reference:  
  https://github.com/danielsyahputra/yolo-distiller

The licensing of this project follows and complies with the original repository’s license terms. Please refer to the original source for detailed license information.

---
## Live Demo / Result Showcase

A web-based demo is available for viewing the final system output and model inference results:

- https://github.com/darke45678-dev/AI-fridge-3/tree/gh-pages

This page is used to display the visualization of the trained model outputs and the final system demonstration.
