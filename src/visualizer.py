# src/visualizer.py
import cv2
import numpy as np

class Visualizer:
    """A utility class for drawing results on images."""
    
    def __init__(self, line_thickness=2, font_scale=0.8):
        self.line_thickness = line_thickness
        self.font_scale = font_scale
        self.colors = {
            'Berry': (255, 0, 0),    # Blue for berries
            'Pearl': (255, 165, 0),  # Orange for pearls
            'default': (0, 255, 0)   # Green for others
        }


    def draw_bounding_boxes(self, image: np.ndarray, detections: list) -> np.ndarray:
        """
        Draws bounding boxes and labels on an image.
        Args:
            image: The image to draw on.
            detections: A list of detection dicts, e.g., 
                        [{'box': [x1,y1,x2,y2], 'class_name': 'berry', 'conf': 0.9}, ...]
        """
        viz_img = image.copy()
        for det in detections:
            x1, y1, x2, y2 = det.boxes.numpy().xyxy[0]
            class_name = det.get('class_name', 'default') # .get() is safer
            color = self.colors.get(class_name, self.colors['default'])
            
            cv2.rectangle(viz_img, (x1, y1), (x2, y2), color, self.line_thickness)
            
            label = f"{class_name} {det.get('conf', 0):.2f}"
            cv2.putText(viz_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        self.font_scale, color, self.line_thickness)
        return viz_img

    def draw_ellipse(self, image: np.ndarray, parameters: tuple) -> np.ndarray:
        """
        Draws a fitted ellipse on an image.
        Args:
            image: The image to draw on.
            parameters: A tuple of ellipse parameters, e.g.,
                     ((x,y), (majorAxis, minorAxis), angle)
        """
        viz_img = image.copy()
        color = (255, 0, 0) # Blue
        width = 2 # The width of the ellipse

        cv2.ellipse(viz_img, parameters, color, width)
        return viz_img




    def draw_analysis_results(self, image: np.ndarray, results: list) -> np.ndarray:
        """
        Draws fitted ellipses and ID numbers on an image.
        Args:
            image: The image to draw on.
            results: A list of analysis dicts, e.g.,
                     [{'id': 1, 'ellipse': ellipse_obj, 'class_name': 'berry'}, ...]
        """
        viz_img = image.copy()
        for res in results:
            class_name = res.get('class_name', 'default')
            color = self.colors.get(class_name, self.colors['default'])
            
            # Draw the ellipse
            cv2.ellipse(viz_img, res['ellipse_params'], color, self.line_thickness)
            
            # Draw the ID number at the center of the ellipse
            center = (int(res['ellipse_params'][0][0]), int(res['ellipse_params'][0][1]))
            cv2.putText(viz_img, str(res['id']), center, cv2.FONT_HERSHEY_SIMPLEX,
                        self.font_scale, color, self.line_thickness, cv2.LINE_AA)
        return viz_img