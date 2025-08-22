# Grape Vision : A Computer Vision pipeline for estimating berry volume and tracking growth kinetics over time

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)

A comprehensive computer vision pipeline to accurately estimate grape berry volume from images and track their growth over time using YOLO and innovative volume estimation methods.

---

![An example of output from an image analysis](docs/images/Before_After.png) 
*A demonstration of the `analyze` command. The left panel shows the input image, and the right panel shows the final output with detected berries and pearls overlaid with their fitted ellipses and ID numbers.*

---

### **Project Context**

This project is the refactored and improved version of a computer vision pipeline originally developed during my research internships at the Institut Agro in Montpellier, France. Its goal is to provide a practical and efficient tool for grape berry phenotyping using modern object detection models.

My work during this period also contributed to a co-authored scientific publication, which explores an alternative methodology for a similar task. The paper is provided here for those interested in the broader academic context of this research area.

**Title**: Ripening dynamics revisited: an automated method to track the development of asynchronous berries on time-lapse images \
**Published in**: Plant Methods, 2023 \
[[**Click here to view the publication**](https://plantmethods.biomedcentral.com/articles/10.1186/s13007-023-01125-8)]


---

### **Features**

This tool provides a flexible command-line interface (CLI) to handle various computer vision tasks:

*   🍇 **Detection (`detect`):** Performs object detection for "berries" and "pearls" on single images or entire folders, saving results as standard YOLO label files and visualized images.
*   🔬 **Analysis (`analyze`):** Runs the full pipeline, from detection and segmentation to volume estimation, outputting detailed CSV results and ellipse visualizations for each image.
*   ⏳ **Tracking (`track`):** Tracks individual berries across a time-series sequence of images, generating a complete tracking history and optionally saving individual crop sequences for each berry.
*   📈 **Time-Series Analysis (`track-analyze`):** Combines tracking and volume analysis to measure berry growth over time, providing both a master CSV file and individual results for each tracked berry.
*   ✂️ **Crop Analysis (`analyze-crops`):** Runs the volume estimation pipeline on pre-existing folders of cropped berries, allowing for analysis after manual track correction.

---

### **Tech Stack**

*   **Python 3.11+**
*   **PyTorch & YOLOv11** for detection, segmentation, and tracking.
*   **OpenCV** for image processing and manipulation.
*   **NumPy & Pandas** for data handling and analysis.
*   **Scipy, Shapely & Scikit-image** for geometric calculations.

---

### **Installation**

Follow these steps to set up the project environment. It is highly recommended to use a virtual environment.

<details>
<summary><strong>For Conda users (Recommended)</strong></summary>

This is the most reliable way to replicate the development environment.

```bash
# Clone the repository
git clone https://github.com/mcafier/grape-vision.git
cd grape-vision

# Create the environment from the file
conda env create -f environment.yml

# Activate the new environment
conda activate grape-vision
```

</details>

<details>
<summary><strong>For Pip users</strong></summary>
This method uses the standard pip and venv tools.

```bash
# Clone the repository and create a virtual environment
git clone https://github.com/mcafier/grape-vision.git
cd grape-vision
python -m venv venv
source venv/bin/activate  # on macOS/Linux
# .\venv\Scripts\activate  # on Windows

# Install the required packages
pip install -r requirements.txt
```
</details>

### **Usage**

The entire pipeline is controlled via `main.py` from your terminal. The `--output` argument is optional and will default to a "Results" folder next to your input if not provided.

**1. Detection Only**
*   **Use Case:** You want to use the trained models to find berries in your images and get the raw bounding boxes.
*   **Command:**
    ```bash
    python main.py detect --input path/to/your/images --output path/to/your/output
    ```
*   **Output:** Creates standard YOLO `.txt` label files and visualized images with bounding boxes.

**2. Full Analysis on Single Images**
*   **Use Case:** You want to calculate the volume of all berries in one or more images.
*   **Command:**
    ```bash
    python main.py analyze --input path/to/your/images --output path/to/your/output
    ```
*   **Output:** A detailed `.csv` file with volume and ellipse data, plus visualized images with numbered ellipses.

**3. Tracking Only**
*   **Use Case:** You have a time-series of images and want to track each berry, saving the raw tracking data and cropped images of each berry's journey.
*   **Command:**
    ```bash
    python main.py track --input path/to/your/image_sequence --output path/to/your/output
    ```
*   **Output:** A `tracking_summary.json` file, YOLO label files, and folders of cropped images for each tracked berry.

**4. Full Time-Series Analysis**
*   **Use Case:** You want to track berries AND calculate their volume in every frame to measure growth.
*   **Command:**
    ```bash
    python main.py track-analyze --input path/to/your/image_sequence --output path/to/your/output
    ```
*   **Output:** A master CSV with all volume data over time, plus folders for each berry containing visualized crops and an individual volume-over-time CSV.

**5. Analysis on Pre-Tracked Crops**
*   **Use Case:** You have already run tracking and perhaps manually corrected the crop folders. Now you only want to run volume estimation on them.
*   **Command:**
    ```bash
    python main.py analyze-crops --input path/to/your/tracked_crops_folder --output path/to/your/output
    ```
*   **Output:** The same output as the `track-analyze` command, but it skips the tracking step.

---

### **Configuration**

All pipeline parameters can be adjusted in the `config.yaml` file without changing the code. This includes model paths, confidence thresholds, and analysis methods.

```yaml
# -----------------------------------------------------------------------------
# Master Configuration for the Grape Vision Pipeline
#
# This file controls all model paths, thresholds, and operational parameters.
# -----------------------------------------------------------------------------

# --- Model Metadata ---
# This section defines data about the models themselves.
model_data:
  class_names: ['Berry', 'Pearl'] #do not modify this

# --- Model Paths ---
# Paths to the trained model weight files (.pt).
# These can be absolute paths or relative to the project's root directory.
models:
  # Model used for initial detection of objects.
  detection_weights: "models/yolov11l_detect.pt"
  
  # Model used for precise segmentation of objects
  segmentation_weights: "models/yolov11l_seg.pt"     #models/YOLOV8L_03-08-23.pt

# --- Component Settings ---
# Fine-tune the behavior of each component in the pipeline.

detection_settings:
  # Detections with a confidence score below this value will be ignored.
  confidence_threshold: 0.5

segmentation_settings:
  # Confidence threshold for the segmentation model.
  confidence_threshold: 0.4

tracking_settings:
  # Path to the specialized tracker configuration file.
  # This file controls the low-level tracking algorithm (e.g., BoT-SORT, ByteTrack).
  tracker_config_file: "configs/tracker.yaml"

# --- Analysis & Output Settings ---
# Control the analysis methods and what gets saved.

analysis_settings:
  # The method used to calculate volume from a mask.
  # Available options: 'ransac', 'convex_hull'
  volume_estimation_method: "convex_hull"

  ransac:
    min_samples: 5 #Use a minimum of 5, as ellipses have 5 parameters
    residual_threshold: 2.0  # Use floats for thresholds
    max_trials: 100

  # Settings specific to time-series (video/image sequence) analysis.
  time_series:
    # If true, the 'track' command will create folders for each tracked
    # berry containing its cropped images over time.
    create_crop_folders: true

visualizer_settings:
  #Controls the the thickness of the ellipses displayed, and the font size of the IDs 
  line_thickness: 3
  font_scale: 1
```

---