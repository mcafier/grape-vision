# src/detector.py
from ultralytics import YOLO
import numpy as np

class Detector:
    """
    Handles object detection using a YOLO model, with flexible output modes.
    Can either return detection data in memory or save results directly to files
    using the underlying ultralytics library functionality.
    """

    def __init__(self, model_weights_path: str):
        """
        Initializes the Detector by loading the YOLO model.

        Args:
            model_weights_path (str): Path to the trained YOLO detection model weights.
        """
        self.model = YOLO(model_weights_path)
        print(f"Detector initialized with model: {model_weights_path}")

    def detect(self,
               image_source: (str | np.ndarray),
               confidence_threshold: float = 0.5,
               save: bool = False,
               save_txt: bool = False,
               output_project_dir: str = None,
               output_run_name: str = None) -> list:
        """
        Performs detection on a single image or an image path.

        This method ALWAYS returns a list of detections in memory.
        If save=True, it ALSO tells the ultralytics engine to save its standard
        outputs to disk.

        Args:
            image_source (str | np.ndarray): Path to the image or the image itself as a NumPy array.
            confidence_threshold (float): The confidence threshold for detection.
            save (bool): If True, save images with bounding boxes.
            save_txt (bool): If True, save bounding box data as YOLO format .txt files.
            output_project_dir (str): The parent directory for saving results (ultralytics' 'project' arg).
            output_run_name (str): The specific folder name for this run (ultralytics' 'name' arg).

        Returns:
            list: A list of detection dictionaries. Each dictionary contains:
                  {'box': [x, y, w, h], 'class_id': int}
                  The box is in the xywh format.
        """
        results = self.model.predict(
            source=image_source,
            conf=confidence_threshold,
            save=save,
            save_txt=save_txt,
            project=output_project_dir,
            name=output_run_name,
            verbose=False # Keeps the console clean
        )

        detections = []
        # We always process the results to return them, even if saving is enabled.
        # This makes the method's behavior consistent.
        for box in results[0].boxes:
            # Get coordinates in xywh format and convert to integers
            xywh_coords = [int(coord) for coord in box.numpy().xywh[0]]

            detections.append({
                'box': xywh_coords,   # center_x, center_y, width, height
                'class_id': box.cls.int().item(),
                'class_name': results[0].names[box.cls.int().item()]
            })

        return detections