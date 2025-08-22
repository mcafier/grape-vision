import os
import cv2
import pandas as pd
import yaml
import json
from glob import glob
from pathlib import Path
from tqdm import tqdm


from src.detector import Detector
from src.segmenter import Segmenter
from src.volume_estimator import VolumeEstimator
from src.visualizer import Visualizer
from src.tracker import Tracker

class ImageAnalysisPipeline:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # We need the class names from the model for labeling
        self.class_names = self.config['model_data']['class_names'] # e.g., ['Berry', 'Pearl']

        # Initialize all components that might be needed
        self.detector = Detector(self.config['models']['detection_weights'])
        self.segmenter = Segmenter(self.config['models']['segmentation_weights'])
        self.volume_estimator = VolumeEstimator(self.config['analysis_settings'])

        #Grab the visualizer config and initialize it
        viz_config = self.config.get('visualizer_settings', {})
        self.visualizer = Visualizer(
        line_thickness=viz_config.get('line_thickness', 2),
        font_scale=viz_config.get('font_scale', 0.8)
        )

    # --- TASK 1: DETECTION ONLY ---
    def run_detection_only(self, input_path: str, output_dir: str = None):
        print("--- Running in Detection-Only Mode ---")

        run_name = "detection_output"
        #if output_dir is None:             #this should be handled in my main
        #    output_dir=input_path
        # The detector now handles all the file I/O!
        # It takes the input path directly.
        self.detector.detect(
            image_source=str(input_path),
            confidence_threshold=self.config['detection_settings']['confidence_threshold'],
            save=True,
            save_txt=True,
            output_project_dir=str(output_dir),
            output_run_name=run_name
        )
        print("Detection complete. Files saved.")

    # --- TASK 2: FULL ANALYSIS ---
    def run_full_analysis(self, input_path: str, output_dir: str):
        print("--- Running in Full-Analysis Mode ---")

        if self.volume_estimator.method == 'ransac':
            print("Info: RANSAC method is selected. Analysis may take a while for each image.")

        # ... logic to find all image files ...
        image_files = []

        if input_path.is_dir():
            print(f"Input is a directory. Scanning for images in: {input_path}")
            valid_extensions = ('.png', '.jpg', '.jpeg')
            # Scan the directory for files with valid extensions
            all_files = os.listdir(input_path)
            image_f = [f for f in all_files if f.lower().endswith(valid_extensions)]
            image_files = [input_path / f for f in sorted(image_f)] # Use pathlib's / operator

            # If source is a single image
        elif input_path.suffix.lower() in ('.png', '.jpg', '.jpeg'):
            image_files.append(input_path)
                        
        # ... the analysis begins here ...
        for img_path in tqdm(image_files, desc="Analyzing images"):
            original_img = cv2.imread(str(img_path)) #We have to convert the Path object to a string
            detections = self.detector.detect(
                image_source=original_img,
                confidence_threshold=self.config['detection_settings']['confidence_threshold']
            )
            
            all_results = []
            berry_counter = 1 # We'll use this berry counter to act as an ID for each berry / pearl

            for det in detections:
                class_name = det['class_name']

                # Extracting the crop from the image
                x, y, w, h = det['box']
                # Increasing height and width of the box slightly (10%)
                w = int(1.1 * w) 
                h = int(1.1 * h) 
                start_x, start_y = x - w//2, y - h//2  # x and y are the center coordinates of the box
                
                # Make sure the new coordinates do not go out of the bounds of the image
                start_x = max(0, start_x)
                start_y = max(0, start_y)
                end_x = min(original_img.shape[1], start_x + w)
                end_y = min(original_img.shape[0], start_y + h)

                crop_img = original_img[start_y:end_y, start_x:end_x]

                # Run segmentation and volume estimation
                contour, _ = self.segmenter.segment(crop_img)
                if contour is None:
                    continue
                
                # Get volume. We can specify method here.
                ellipse, volume = self.volume_estimator.estimate(contour)
                if ellipse is None:
                    continue

                # Adjust ellipse back to original image coordinates
                ellipse_center_original = (int(ellipse[0][0] + start_x), int(ellipse[0][1] + start_y))
                adjusted_ellipse = (ellipse_center_original, ellipse[1], ellipse[2])

                # Store result with a unique ID for this image
                all_results.append({
                    'id': berry_counter,
                    'class_name': class_name,
                    'volume_voxels': volume,
                    'ellipse_params': adjusted_ellipse,
                    'original_bbox': det['box']  # I'm using xywh, might create problems later
                })
                berry_counter += 1
            
            # Now save everything for this image
            img_basename = img_path.stem #same as "os.path.splitext(os.path.basename(img_path))[0]" but with pathlib method
            img_output_dir = output_dir / img_basename
            os.makedirs(img_output_dir, exist_ok=True)
            
            # Save CSV file
            df = pd.DataFrame(all_results)
            df.to_csv(img_output_dir / "analysis_results.csv", index=False)

            # Save visualization with ellipses and IDs
            viz_img = self.visualizer.draw_analysis_results(original_img, all_results)
            cv2.imwrite(str(img_output_dir / "analysis_viz.jpg"), viz_img)


class TimeSeriesPipeline:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.class_names = self.config['model_data']['class_names']


        # Initialize all components
        self.tracker = Tracker(self.config['models']['detection_weights'], self.config['tracking_settings']['tracker_config_file'])
        self.segmenter = Segmenter(self.config['models']['segmentation_weights'])
        self.volume_estimator = VolumeEstimator(self.config['analysis_settings'])

        #Grab the visualizer config and initialize it
        viz_config = self.config.get('visualizer_settings', {})
        self.visualizer = Visualizer(
        line_thickness=viz_config.get('line_thickness', 2),
        font_scale=viz_config.get('font_scale', 0.8)
        )
        
    # --- PRIVATE HELPER: Creates crop folders from tracking data ---
    def _create_crop_folders(self, input_dir: Path, output_dir: Path, tracking_data: dict):
        print("Creating individual berry crop folders...")

        # Create the main parent directory for all crops
        tracked_crops_parent_dir = output_dir / "tracked_crops"
        tracked_crops_parent_dir.mkdir(parents=True, exist_ok=True)

        for track_id, detections in tracking_data.items():
            # Assume first detection gives the class name for the whole track
            class_name = detections[0]['class_name']
            
            berry_folder_name = f"{class_name}_{track_id}"
            berry_dir = tracked_crops_parent_dir / berry_folder_name
            berry_dir.mkdir(exist_ok=True) # The pathlib way to create a directory

            for det in detections:
                original_img_path = input_dir / det['frame_name']
                frame = cv2.imread(str(original_img_path))

                # Defensive check in case an image is missing from the sequence
                if frame is None:
                    print(f"Warning: Could not read image file {original_img_path}. Skipping crop.")
                    continue
                
                x, y, w, h = det['box_xywh']
                # Increasing height and width of the box slightly (10%)
                w = int(1.1 * w) 
                h = int(1.1 * h) 
                start_x, start_y = x - w//2, y - h//2  # x and y are the center coordinates of the box
                
                # Make sure the new coordinates do not go out of the bounds of the image
                start_x = max(0, start_x)
                start_y = max(0, start_y)
                end_x = min(frame.shape[1], start_x + w)
                end_y = min(frame.shape[0], start_y + h)

                crop_img = frame[start_y:end_y, start_x:end_x]
                
                crop_filename = f"{class_name}_{track_id}_{det['frame_name']}"
                output_crop_path = berry_dir / crop_filename
                cv2.imwrite(str(output_crop_path), crop_img)

    # --- Use Case 1: TRACKING ONLY ---
    def run_tracking_only(self, input_dir: str, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        print("--- Running in Tracking-Only Mode ---")
        
        tracking_data = self.tracker.track_sequence(
            image_folder_path=str(input_dir),
           # tracker_config_file=self.config['tracking_settings']['tracker_config_file'],
            save_txt=True, # Always save labels for this mode
            output_dir=str(output_dir),
            run_name="yolo_track_output"
        )
        
        # Save the structured tracking data to a JSON file
        summary_path = output_dir / "tracking_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(tracking_data, f, indent=4)
        print(f"Tracking summary saved to {summary_path}")

        # Optionally create crop folders based on config
        if self.config['analysis_settings']['time_series']['create_crop_folders']:
            self._create_crop_folders(input_dir, output_dir, tracking_data)

    # --- Use Case 2: TRACKING + ANALYSIS ---
    def run_full_analysis(self, input_dir: str, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        print("--- Running in Full Time-Series Analysis Mode ---")

        if self.volume_estimator.method == 'ransac':
            print("Info: RANSAC method is selected. Analysis may take a while for each image.")
        
        tracking_data = self.tracker.track_sequence(
            image_folder_path=str(input_dir),
            #tracker_config_file=self.config['tracking_settings']['tracker_config_file'],
            save_txt=False, # Don't save labels for this mode
            output_dir=str(output_dir),
            run_name="yolo_track_output"
        )

        # Save tracking summary JSON
        summary_path = output_dir / "tracking_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(tracking_data, f, indent=4)
        print(f"Tracking summary saved to {summary_path}")
        
        all_volume_results = []
        
        # This is the core analysis loop
        for track_id, detections in tqdm(tracking_data.items(), desc="Analyzing tracked berries"):
            viz_crop_dir = output_dir / "visualized_crops" / f"{detections[0]['class_name']}_{track_id}"
            os.makedirs(viz_crop_dir, exist_ok=True)

            single_berry_results = []
                    
            for det in detections:
                # ... (load original image, crop it) ...
                original_img_path = input_dir / det['frame_name']
                frame = cv2.imread(str(original_img_path))
                
                x, y, w, h = det['box_xywh']
                # Increasing height and width of the box slightly (10%)
                w = int(1.1 * w) 
                h = int(1.1 * h) 
                start_x, start_y = x - w//2, y - h//2  # x and y are the center coordinates of the box
                
                # Make sure the new coordinates do not go out of the bounds of the image
                start_x = max(0, start_x)
                start_y = max(0, start_y)
                end_x = min(frame.shape[1], start_x + w)
                end_y = min(frame.shape[0], start_y + h)

                crop_img = frame[start_y:end_y, start_x:end_x]

                # ... (run segmenter, volume_estimator on crop_img) ...
                # Run segmentation and volume estimation
                contour, _ = self.segmenter.segment(crop_img)
                if contour is None:
                    continue
                
                # Get volume.
                ellipse, volume = self.volume_estimator.estimate(contour)
                if ellipse is None:
                    continue

                if volume is not None:
                    all_volume_results.append({
                        'image_name': det['frame_name'],
                        'track_id': track_id,
                        'class_name': det['class_name'],
                        'ellipse_params': str(ellipse), # Convert to string for CSV
                        'volume_voxels': volume
                    })
                
                # Dict for single berry CSVs
                    single_berry_results.append({
                        'frame_name': det['frame_name'],
                        'volume_voxels': volume
                    })
                
                # ... (use visualizer to draw ellipse on crop) ...
                viz_img = self.visualizer.draw_ellipse(crop_img, ellipse)
                # ... (save visualized crop to viz_crop_dir) ...
                cv2.imwrite(str(viz_crop_dir / det['frame_name']), viz_img)
            
            # Output single berry CSVs
            if single_berry_results: # Only save if we have data
                df_berry = pd.DataFrame(single_berry_results)
                csv_path_berry = viz_crop_dir / "volume_over_time.csv"
                df_berry.to_csv(csv_path_berry, index=False)

        # Save the final volume results to a single CSV
        df = pd.DataFrame(all_volume_results)
        df.to_csv(output_dir / "volume_analysis_results.csv", index=False)
        print("Full analysis complete. Results saved.")
        print("A CSV for each berry has been saved in its respective visualized_crops folder")

    # --- Use Case 3: ANALYSIS ON PRE-TRACKED CROPS ---
    def run_analysis_on_crops(self, input_dir: str, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        print("--- Running Analysis on Pre-Cropped Folders ---")

        if self.volume_estimator.method == 'ransac':
            print("Info: RANSAC method is selected. Analysis may take a while for each image.")
        
        all_volume_results = []
        folders_to_process = []
        
        # Define valid, case-insensitive extensions once
        valid_extensions = ('.jpg', '.jpeg', '.png')

        # --- Intelligently determine which folders to process ---
        # Check if the input_dir itself contains images (single track case)
        if any(f.lower().endswith(valid_extensions) for f in os.listdir(input_dir)):
            print("Single track folder detected. Processing images directly.")
            folders_to_process = [input_dir]
            # Create a more descriptive output directory for the single case
            input_folder_name = input_dir.name
            output_dir = output_dir / f"{input_folder_name}_analysis"
            os.makedirs(output_dir, exist_ok=True)
        else:
            # Otherwise, look for subdirectories (multi-track case)
            print("Multiple track folders detected. Looking for subdirectories.")
            folders_to_process = [p for p in input_dir.glob('*') if p.is_dir()]

        if not folders_to_process:
            print("Warning: No valid crop folders or images found in the input directory.")
            return # Exit early

        for folder_path in tqdm(folders_to_process, desc="Analyzing tracked berries"):
            folder_name = folder_path.name
            viz_crop_dir = output_dir / "visualized_crops" / folder_name
            os.makedirs(viz_crop_dir, exist_ok=True)
            
            image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)])
            image_paths = [os.path.join(folder_path, f) for f in image_files]

            single_berry_results = []

            
            for img_path in image_paths:
                img_name = os.path.basename(img_path)
                crop_img = cv2.imread(str(img_path))

                # ... (run segmenter, volume_estimator on crop_img) ...
                # Run segmentation and volume estimation
                contour, _ = self.segmenter.segment(crop_img)
                if contour is None:
                    continue
                
                # Get volume
                ellipse, volume = self.volume_estimator.estimate(contour)
                if ellipse is None:
                    continue
                
                if volume is not None:
                    all_volume_results.append({
                        'image_name': os.path.basename(img_path),
                        'track_id': folder_name, # Use folder name as ID
                        'ellipse_params': str(ellipse), # Convert to string for CSV
                        'volume_voxels': volume
                    })

                                    # Dict for single berry CSVs
                    single_berry_results.append({
                        'frame_name': os.path.basename(img_path),
                        'volume_voxels': volume
                    })

                # ... (save visualized crop to viz_crop_dir) ...
                viz_img = self.visualizer.draw_ellipse(crop_img, ellipse)
                cv2.imwrite(str(viz_crop_dir / img_name), viz_img)

              # Output single berry CSVs
            if single_berry_results: # Only save if we have data
                df_berry = pd.DataFrame(single_berry_results)
                csv_path_berry = viz_crop_dir / "volume_over_time.csv"
                df_berry.to_csv(csv_path_berry, index=False)
        
        # Save the final volume results to a single CSV
        df = pd.DataFrame(all_volume_results)
        df.to_csv(output_dir / "volume_analysis_results.csv", index=False)
        print("Full analysis complete. Results saved.")
        print("A CSV for each berry has been saved in its respective visualized_crops folder")
