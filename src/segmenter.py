# src/segmenter.py
from ultralytics import YOLO
from shapely.geometry import Point, Polygon
import cv2
import numpy as np


class Segmenter:
    """A class to handle berry segmentation using a YOLO model."""
    
    def __init__(self, model_weights_path: str):
        """
        Initializes the Segmenter.
        
        Args:
            model_weights_path (str): Path to the trained YOLO segmentation model weights.
        """
        print(f"Initializing Segmenter with model: {model_weights_path}")
        self.model = YOLO(model_weights_path)

    def _get_center_mask(self, yolo_results):
        """
        Private helper method to find the centermost mask.
        """
        try:
            masks_xy = yolo_results[0].masks.xy
            masks_data = yolo_results[0].masks.numpy().data
            image_height, image_width = yolo_results[0].orig_shape
        except (AttributeError, IndexError):
            return None, None # No masks found in results

        if not masks_xy:
            return None, None

        center_point = Point(image_width / 2, image_height / 2)
        
        for contour, mask_binary in zip(masks_xy, masks_data):
            mask_polygon = Polygon(contour)
            if mask_polygon.contains(center_point):
                # We found it! Return the contour and the binary mask data.
                return contour, mask_binary
        
        # If no mask contained the center, we didn't find a suitable one.
        return None, None


    def segment(self, INPUT):
        """
        Runs segmentation on a single image and returns the centermost mask.
        
        Args:
            image (np.ndarray): The input image (likely a crop of a berry).
            
        Returns:
            A tuple of (contour, binary_mask) for the centermost mask, or (None, None) if not found.
        """
        # Run YOLO prediction
        results = self.model.predict(source=INPUT, verbose=False) # verbose=False cleans up console output
        
        # Find and return the centermost mask using our helper method
        contour, binary_mask = self._get_center_mask(results)

        if contour is not None:
            # Clean up the contour data type so that it's correct coordinates
            contour = np.round(contour).astype(int)
        
        return contour, binary_mask