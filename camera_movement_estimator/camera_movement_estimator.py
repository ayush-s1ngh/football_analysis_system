import pickle
import cv2
import numpy as np
import sys
import os

# Get the absolute path of the project root directory
# This ensures consistent imports regardless of where the script is run from
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the project root directory to Python's module search path
sys.path.append(project_root)

from utils import measure_distance, measure_xy_distance

class CameraMovementEstimator:
    """
    A class to estimate and track camera movement in video frames using optical flow.
    Uses Lucas-Kanade algorithm to track feature points and estimate camera motion.
    """

    def __init__(self, frame):
        """
        Initialize the camera movement estimator with parameters for feature detection
        and optical flow calculation.

        Args:
            frame (np.array): Initial video frame to setup feature detection
        """
        # Minimum distance threshold for considering camera movement
        self.minimum_distance = 5

        # Parameters for Lucas-Kanade optical flow
        self.lk_params = dict(
            winSize=(15, 15),      # Size of the search window
            maxLevel=2,            # Number of pyramid levels
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)  # Termination criteria
        )

        # Convert first frame to grayscale for feature detection
        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Create mask to focus feature detection on specific regions
        # Only detect features in the leftmost and center-right regions of the frame
        mask_features = np.zeros_like(first_frame_grayscale)
        mask_features[:, 0:20] = 1         # Left edge of frame
        mask_features[:, 900:1050] = 1     # Center-right region

        # Parameters for good features to track
        self.features = dict(
            maxCorners=100,        # Maximum number of corners to detect
            qualityLevel=0.3,      # Minimum quality level for corner detection
            minDistance=3,         # Minimum distance between corners
            blockSize=7,           # Size of window for corner detection
            mask=mask_features     # Mask for limiting feature detection regions
        )

    def add_adjust_positions_to_tracks(self, tracks, camera_movement_per_frame):
        """
        Adjust object positions to account for camera movement.

        Args:
            tracks (dict): Dictionary containing tracking information for all objects
            camera_movement_per_frame (list): List of camera movement vectors [x, y] for each frame
        """
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position']
                    camera_movement = camera_movement_per_frame[frame_num]
                    # Subtract camera movement from position to get adjusted coordinates
                    position_adjusted = (
                        position[0] - camera_movement[0],
                        position[1] - camera_movement[1]
                    )
                    tracks[object][frame_num][track_id]['position_adjusted'] = position_adjusted

    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):
        """
        Calculate camera movement between consecutive frames using optical flow.

        Args:
            frames (list): List of video frames
            read_from_stub (bool): Whether to read cached results
            stub_path (str): Path to cached results file

        Returns:
            list: Camera movement vectors [x, y] for each frame
        """
        # Try to load cached results if requested
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        # Initialize camera movement list with zero movement for all frames
        camera_movement = [[0, 0]] * len(frames)

        # Prepare first frame for optical flow
        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)

        # Process each subsequent frame
        for frame_num in range(1, len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            
            # Calculate optical flow to track feature points
            new_features, _, _ = cv2.calcOpticalFlowPyrLK(
                old_gray, frame_gray, old_features, None, **self.lk_params
            )

            # Find the maximum movement among all tracked features
            max_distance = 0
            camera_movement_x, camera_movement_y = 0, 0

            for i, (new, old) in enumerate(zip(new_features, old_features)):
                new_features_point = new.ravel()
                old_features_point = old.ravel()

                # Calculate distance between old and new feature positions
                distance = measure_distance(new_features_point, old_features_point)
                if distance > max_distance:
                    max_distance = distance
                    # Get x, y components of the movement
                    camera_movement_x, camera_movement_y = measure_xy_distance(
                        old_features_point, new_features_point
                    )
            
            # Update camera movement if it exceeds minimum threshold
            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [camera_movement_x, camera_movement_y]
                # Detect new features for next frame
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)

            old_gray = frame_gray.copy()
        
        # Cache results if stub_path provided
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_movement, f)

        return camera_movement
    
    def draw_camera_movement(self, frames, camera_movement_per_frame):
        """
        Visualize camera movement by drawing movement values on frames.

        Args:
            frames (list): List of video frames
            camera_movement_per_frame (list): List of camera movement vectors [x, y] for each frame

        Returns:
            list: Frames with camera movement information drawn on them
        """
        output_frames = []

        for frame_num, frame in enumerate(frames):
            frame = frame.copy()

            # Create semi-transparent overlay for text background
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (500, 100), (255, 255, 255), -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)

            # Draw camera movement values
            x_movement, y_movement = camera_movement_per_frame[frame_num]
            frame = cv2.putText(
                frame,
                f"Camera Movement X: {x_movement:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                3
            )
            frame = cv2.putText(
                frame,
                f"Camera Movement Y: {y_movement:.2f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                3
            )

            output_frames.append(frame)

        return output_frames