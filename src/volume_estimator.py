# src/volume_estimator.py
import cv2
import numpy as np
import random
from scipy.spatial import ConvexHull
from scipy import spatial

class VolumeEstimator:
    """Calculates berry volume from a mask contour using different methods."""

    def __init__(self, method: str = 'convex_hull'):
        """
        Initializes the VolumeEstimator.
        Args:
            method (str): The method to use, e.g., 'ransac' or 'convex_hull'.
        """
        if method not in ['ransac', 'convex_hull']:
            raise ValueError(f"Unknown estimation method: {method}")
        self.method = method
        print(f"VolumeEstimator initialized with method: {self.method}")
    
    def _get_volume_from_ellipse(self, ellipse):
        """Calculates volume from ellipse parameters."""
        major_axis = max(ellipse[1]) / 2
        minor_axis = min(ellipse[1]) / 2
        return (4/3) * np.pi * major_axis * (minor_axis**2)

    def _detect_outliers(self, contour_data):
        """Private helper for outlier detection (from Methode_B)."""
        # Calculate convex hull
        hull = ConvexHull(contour_data)
        hull_points = contour_data[hull.vertices]
        # Compute pairwise distances
        pairwise_distances = spatial.distance.cdist(hull_points, hull_points)
        # Compute the mean distance to the 10 nearest neighbors for each point
        mean_distances = np.mean(np.sort(pairwise_distances)[:, 1:11], axis=1)
        sorted_distances = np.sort(mean_distances)[::-1]  # Sort in descending order
        # Compute standard deviation for each step and find the jump
        std_values = [np.std(sorted_distances[:i]) for i in range(2, len(hull_points))]
        std_jump = np.argmax(np.diff(std_values)) +1
        # Select the points before the jump
        selected_points = mean_distances >= sorted_distances[std_jump]
        # Select the points before the jump
        inlier_points_boolean = mean_distances < sorted_distances[std_jump]
        
        # Extract the inlier coordinates
        inlier_coords = hull_points[inlier_points_boolean]
        
        # Check the ratio of inliers to total points
        # si moins de 85% de points ont survécu, il n'y avait probablement pas de outliers en réalité
        if len(inlier_coords) / len(hull_points) < 0.85:
            return hull_points
        
        return inlier_coords


    #
    def _estimate_with_ransac(self, contour: np.ndarray):
        """Performs RANSAC ellipse fitting (from Methode_A)."""
        # ... (Your RANSAC loop from Methode_A goes here) ...
        # This part is complex, let's assume it returns `best_model`
        nzero = [tuple(p) for p in contour]
        best_model = None
        best_score = 0
        iters = 7500 # Should be in config!
        sample_size = 8 # Should be in config!

        for _ in range(iters):
            try:
                sample = np.array(random.sample(nzero, sample_size))
                ellipse = cv2.fitEllipse(sample)
                # ... score the ellipse ...
                # (This is a simplified version of your scoring logic)
                score = cv2.pointPolygonTest(contour, ellipse[0], True)
                if score > best_score:
                    best_score = score
                    best_model = ellipse
            except:
                continue # RANSAC sample might not form an ellipse
        
        if best_model is None:
            return None, None
        
        volume = self._get_volume_from_ellipse(best_model)
        return best_model, volume

    def _estimate_with_convex_hull(self, contour: np.ndarray):
        """Performs convex hull ellipse fitting (from Methode_B)."""
        inliers = self._detect_outliers(contour)
        if len(inliers) < 5: # Not enough points to fit an ellipse
            return None, None
            
        ellipse = cv2.fitEllipse(inliers)
        volume = self._get_volume_from_ellipse(ellipse)
        return ellipse, volume

    def estimate(self, contour: np.ndarray):
        """
        Main estimation method. Takes a contour and returns ellipse and volume.
        
        Returns:
            A tuple of (ellipse_object, volume), or (None, None) if failed.
        """
        if self.method == 'ransac':
            return self._estimate_with_ransac(contour)
        elif self.method == 'convex_hull':
            return self._estimate_with_convex_hull(contour)