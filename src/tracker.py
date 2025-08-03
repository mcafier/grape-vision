# src/tracker.py
from ultralytics import YOLO
from collections import defaultdict
import os

class Tracker:
    """Handles object tracking in image sequences using a YOLO model."""

    def __init__(self, model_weights_path: str, tracker_config_file: str):
        """
        Initializes the Tracker.

        Args:
            model_weights_path (str): Path to the trained YOLO model (same as detector).
            tracker_config_file (str): Path to the tracker configuration file (e.g., 'botsort.yaml').
        """
        self.model = YOLO(model_weights_path)
        tracker_config_file = tracker_config_file
        print(f"Tracker initialized with model '{model_weights_path}' and configuration file {tracker_config_file}")

    def track_sequence(self,
                       image_folder_path: str,
                       #tracker_config_file: str,
                       save_txt: bool = False,
                       output_dir: str = None,
                       run_name: str = None) -> dict:
        """
        Processes an image sequence folder and tracks objects.

        Args:
            image_folder_path (str): Path to the input folder of sequential images.
            tracker_config_file (str): Path to the tracker configuration (e.g., 'botsort.yaml').
            save_txt (bool): If True, save tracking results as YOLO format .txt files.
            output_dir (str): Parent directory for saving results (ultralytics' 'project' arg).
            run_name (str): Specific folder name for this run (ultralytics' 'name' arg).

        Returns:
            A dictionary where keys are track_ids and values are lists of detections.
            e.g., { 1: [{'frame_name': 'img_001.jpg', 'box_xywh': [x,y,w,h], 'class_id': 0}, ...], ... }
        """
        tracking_results = defaultdict(list)

        results_generator = self.model.track(
            source=image_folder_path,
            #tracker=tracker_config_file,
            persist=True,  # Crucial for tracking across a sequence of images
            save_txt=save_txt,
            project=output_dir,
            name=run_name,
            stream=True,
            verbose=False
        )

        for results in results_generator:
            if results.boxes.id is None:
                continue

            frame_name = os.path.basename(results.path)
            track_ids = results.boxes.id.cpu().numpy().astype(int).tolist()
            boxes_xywh = results.boxes.xywh.cpu().numpy().astype(int).tolist()
            #class_ids = results.boxes.cls.cpu().numpy().astype(int)
            class_names = [results.names[cls.item()] for cls in results.boxes.cls.int()]

            for track_id, box, class_name in zip(track_ids, boxes_xywh, class_names):
                tracking_results[track_id].append({
                    'frame_name': frame_name,
                    'box_xywh': list(box),
                    'class_name': class_name
                })

        return dict(tracking_results)