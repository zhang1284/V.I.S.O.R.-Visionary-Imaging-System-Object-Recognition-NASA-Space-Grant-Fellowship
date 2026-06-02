# V.I.S.O.R. — Visionary Imaging System Object Recognition

[![NASA Space Grant Fellowship](https://img.shields.io/badge/Fellowship-NASA%20Space%20Grant-blue.svg)](https://www.nespacegrant.org/)
[![Language](https://img.shields.io/badge/Language-Java-orange.svg)](https://www.oracle.com/java/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**V.I.S.O.R.** is an advanced computer vision and deep learning research platform developed under the auspices of the **NASA Space Grant Fellowship**. The system is purpose-built to pioneer autonomous object detection, multi-class image classification, and high-fidelity semantic segmentation within unstructured, high-contrast, or extreme extraterrestrial environments. 

By leveraging structured, high-performance Java architectures, V.I.S.O.R. addresses critical operational challenges in aerospace imaging, including cosmic radiation artifact mitigation, dynamic lens flare compensation, and real-time edge-computing constraint management.

---

## 🛰️ Core Research Focus & Capabilities

* **Autonomous Object Detection:** Real-time identification, bounding, and tracking of orbital hardware, space debris, and surface features.
* **Semantic Terrain Segmentation:** Surface-mapping algorithms designed to distinguish complex planetary regolith, craters, and structural hazards to assist autonomous robotic navigation.
* **Environmental Robustness:** Custom image filtering pipelines engineered to reconstruct signals corrupted by cosmic ray strikes, harsh direct solar illumination, and extreme shadow contrast.

---

## 🛠️ System Architecture & Data Pipeline

The V.I.S.O.R. architecture decouples data ingestion, preprocessing, model inference, and performance telemetry to maintain optimal processing efficiency:

```text
  ┌─────────────────┐      ┌──────────────────────┐      ┌──────────────────┐
  │  Raw Imagery    │ ───> │ Preprocessing Engine │ ───> │  Neural Network  │
  │  (Space/Sensor) │      │ (Noise/Flare Filter) │      │  (Inference Core)│
  └─────────────────┘      └──────────────────────┘      └──────────────────┘
                                                                   │
                                                                   ▼
  ┌─────────────────┐      ┌──────────────────────┐      ┌──────────────────┐
  │  Telemetry &    │ <─── │  Post-Processing     │ <─── │ Prediction Maps  │
  │  Metrics (IoU)  │      │  (Bounding/Masking)  │      │ & Class Tensors  │
  └─────────────────┘      └──────────────────────┘      └──────────────────┘
