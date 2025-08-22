# src/volume_estimator.py
import cv2
import numpy as np
import random
from scipy.spatial import ConvexHull
from scipy import spatial
from skimage.measure import EllipseModel, ransac


class VolumeEstimator:
    """Calculates berry volume from a mask contour using different methods."""

    def __init__(self, analysis_config: dict):
        """
        Initializes the VolumeEstimator.
        Args:
            analysis_config (dict): A dictionary of analysis settings,
                                    typically from the main config file.
        """
        self.method = analysis_config.get('volume_estimation_method', 'convex_hull')
        
        # If using RANSAC, load its specific parameters from the config
        if self.method == 'ransac':
            ransac_config = analysis_config.get('ransac', {})
            # Use .get() for each parameter to provide safe defaults
            self.ransac_min_samples = ransac_config.get('min_samples', 5)
            self.ransac_residual_threshold = ransac_config.get('residual_threshold', 2.0)
            self.ransac_max_trials = ransac_config.get('max_trials', 100)
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
        # If less than 85% of points survived, there might not have been any outliers 
        if len(inlier_coords) / len(hull_points) < 0.85:
            return hull_points
        
        return inlier_coords


    def _estimate_with_ransac(self, contour: np.ndarray):

        # Guard clause to prevent errors on small contours
        if contour.shape[0] < self.ransac_min_samples:
            return None, None
        
        try:
            # The EllipseModel is created here, not outside the function.
            model_robust, inliers = ransac(
                contour, 
                EllipseModel, 
                min_samples=self.ransac_min_samples,
                residual_threshold=self.ransac_residual_threshold,
                max_trials=self.ransac_max_trials
            )
        except Exception:
            return None, None # Catch any unexpected error from ransac
        
        if model_robust is None:
            return None, None

        c_x, c_y, axis_1, axis_2, angle = model_robust.params
        axis_maj = max(axis_1, axis_2)
        axis_min = min(axis_1, axis_2)
        center = (int(c_x), int(c_y))
        axis = (int(axis_maj*2), int(axis_min*2))
        angle = int(np.rad2deg(angle))

        ellipse = (center, axis, angle)
        volume = self._get_volume_from_ellipse(ellipse)
        #print("One more berry done.")
        return ellipse, volume

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