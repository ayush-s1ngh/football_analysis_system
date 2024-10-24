import cv2
import sys
import os

# Get the absolute path of the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the project root directory to sys.path to enable imports from parent directory
sys.path.append(project_root)

# Import utility functions for distance measurement and position calculation
from utils import measure_distance, get_foot_position

class SpeedAndDistance_Estimator():
    """
    A class to estimate and visualize speed and distance metrics for tracked objects in video frames.
    Primarily used for analyzing movement patterns in sports analytics.
    """
    
    def __init__(self):
        """
        Initialize the estimator with default parameters.
        frame_window: Number of frames to consider for speed calculation
        frame_rate: Frames per second of the video feed
        """
        self.frame_window = 5  # Calculate speed over 5 frames
        self.frame_rate = 24   # Assuming 24 FPS video
    
    def add_speed_and_distance_to_tracks(self, tracks):
        """
        Calculate and add speed and cumulative distance information to tracking data.
        
        Args:
            tracks (dict): Nested dictionary containing tracking information for different objects
                         Format: {object_type: {frame_num: {track_id: {track_info}}}}
        
        Updates tracks in-place, adding 'speed' and 'distance' fields to each track.
        """
        # Dictionary to store cumulative distance for each object and track ID
        total_distance = {}

        # Iterate through each object type in tracks (excluding ball and referees)
        for object, object_tracks in tracks.items():
            if object == "ball" or object == "referees":
                continue 
            
            # Get total number of frames for this object
            number_of_frames = len(object_tracks)
            
            # Process frames in windows for speed calculation
            for frame_num in range(0, number_of_frames, self.frame_window):
                # Calculate last frame in current window
                last_frame = min(frame_num + self.frame_window, number_of_frames-1)

                # Process each track in the current frame
                for track_id, _ in object_tracks[frame_num].items():
                    # Skip if track doesn't exist in last frame of window
                    if track_id not in object_tracks[last_frame]:
                        continue

                    # Get start and end positions for distance calculation
                    start_position = object_tracks[frame_num][track_id]['position_transformed']
                    end_position = object_tracks[last_frame][track_id]['position_transformed']

                    # Skip if either position is None
                    if start_position is None or end_position is None:
                        continue
                    
                    # Calculate distance covered in this window
                    distance_covered = measure_distance(start_position, end_position)
                    
                    # Calculate time elapsed in seconds
                    time_elapsed = (last_frame - frame_num) / self.frame_rate
                    
                    # Calculate speed in meters per second and convert to km/h
                    speed_meteres_per_second = distance_covered / time_elapsed
                    speed_km_per_hour = speed_meteres_per_second * 3.6

                    # Initialize distance tracking for new objects/tracks
                    if object not in total_distance:
                        total_distance[object] = {}
                    
                    if track_id not in total_distance[object]:
                        total_distance[object][track_id] = 0
                    
                    # Update cumulative distance
                    total_distance[object][track_id] += distance_covered

                    # Update speed and distance for all frames in current window
                    for frame_num_batch in range(frame_num, last_frame):
                        if track_id not in tracks[object][frame_num_batch]:
                            continue
                        tracks[object][frame_num_batch][track_id]['speed'] = speed_km_per_hour
                        tracks[object][frame_num_batch][track_id]['distance'] = total_distance[object][track_id]
    
    def draw_speed_and_distance(self, frames, tracks):
        """
        Draw speed and distance information on video frames.
        
        Args:
            frames (list): List of video frames to annotate
            tracks (dict): Tracking data containing speed and distance information
        
        Returns:
            list: Annotated frames with speed and distance information
        """
        output_frames = []
        
        # Process each frame
        for frame_num, frame in enumerate(frames):
            # Process each object type
            for object, object_tracks in tracks.items():
                if object == "ball" or object == "referees":
                    continue 
                
                # Process each track in current frame
                for _, track_info in object_tracks[frame_num].items():
                    # Check if speed information exists
                    if "speed" in track_info:
                        speed = track_info.get('speed', None)
                        distance = track_info.get('distance', None)
                        
                        if speed is None or distance is None:
                            continue
                       
                        # Get position for text placement
                        bbox = track_info['bbox']
                        position = get_foot_position(bbox)
                        position = list(position)
                        position[1] += 40  # Offset text 40 pixels below foot position

                        # Convert position to integers for drawing
                        position = tuple(map(int, position))
                        
                        # Draw speed and distance text
                        cv2.putText(frame, f"{speed:.2f} km/h", position, 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
                        cv2.putText(frame, f"{distance:.2f} m", 
                                  (position[0], position[1]+20),  # Offset distance text below speed
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
            
            output_frames.append(frame)
        
        return output_frames