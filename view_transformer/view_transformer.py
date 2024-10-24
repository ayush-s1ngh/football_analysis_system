import numpy as np 
import cv2

class ViewTransformer():
    """
    A class to transform pixel coordinates from camera perspective to top-down court view.
    Used for converting player positions from video coordinates to actual court positions.
    """
    
    def __init__(self):
        """
        Initialize the transformer with court dimensions and perspective mapping points.
        
        Court dimensions are in meters:
        - Width: 68 meters
        - Length: 23.32 meters
        
        Pixel vertices represent four points in the camera view (source coordinates).
        Target vertices represent corresponding points in top-down view (destination coordinates).
        """
        # Define court dimensions in meters
        court_width = 68
        court_length = 23.32

        # Define pixel coordinates of court corners in the camera view
        # Format: [bottom-left, top-left, top-right, bottom-right]
        self.pixel_vertices = np.array([[110, 1035],  # Bottom-left corner
                                      [265, 275],    # Top-left corner
                                      [910, 260],    # Top-right corner
                                      [1640, 915]])  # Bottom-right corner
        
        # Define corresponding court coordinates in meters for top-down view
        # Creates a rectangle representing the court
        self.target_vertices = np.array([
            [0, court_width],          # Bottom-left (0,68)
            [0, 0],                    # Top-left (0,0)
            [court_length, 0],         # Top-right (23.32,0)
            [court_length, court_width] # Bottom-right (23.32,68)
        ])

        # Convert vertices to float32 for OpenCV compatibility
        self.pixel_vertices = self.pixel_vertices.astype(np.float32)
        self.target_vertices = self.target_vertices.astype(np.float32)

        # Calculate perspective transformation matrix
        self.persepctive_trasnformer = cv2.getPerspectiveTransform(
            self.pixel_vertices, 
            self.target_vertices
        )

    def transform_point(self, point):
        """
        Transform a single point from pixel coordinates to court coordinates.
        
        Args:
            point (numpy.array): Point coordinates in pixel space [x,y]
        
        Returns:
            numpy.array or None: Transformed coordinates in meters if point is inside court,
                               None if point is outside court boundaries
        """
        # Convert point to integer coordinates for polygon test
        p = (int(point[0]), int(point[1]))
        
        # Check if point lies within court boundaries
        # Returns: 1 if inside, 0 if on edge, -1 if outside
        is_inside = cv2.pointPolygonTest(self.pixel_vertices, p, False) >= 0 
        
        # Return None if point is outside court boundaries
        if not is_inside:
            return None

        # Reshape point for perspective transform
        # Format required by cv2.perspectiveTransform: (n_points, 1, 2)
        reshaped_point = point.reshape(-1, 1, 2).astype(np.float32)
        
        # Apply perspective transformation
        tranform_point = cv2.perspectiveTransform(
            reshaped_point,
            self.persepctive_trasnformer
        )
        
        # Reshape result back to (n_points, 2)
        return tranform_point.reshape(-1, 2)

    def add_transformed_position_to_tracks(self, tracks):
        """
        Transform all tracked positions from pixel space to court coordinates.
        
        Args:
            tracks (dict): Nested dictionary containing tracking information
                         Format: {object_type: {frame_num: {track_id: {track_info}}}}
        
        Updates tracks in-place, adding 'position_transformed' field to each track.
        Position_transformed contains court coordinates in meters.
        """
        # Iterate through each object type (players, ball, etc.)
        for object, object_tracks in tracks.items():
            # Iterate through each frame
            for frame_num, track in enumerate(object_tracks):
                # Iterate through each tracked object in the frame
                for track_id, track_info in track.items():
                    # Get position from tracking info
                    position = track_info['position_adjusted']
                    position = np.array(position)
                    
                    # Transform position to court coordinates
                    position_trasnformed = self.transform_point(position)
                    
                    # Convert numpy array to list if transformation was successful
                    if position_trasnformed is not None:
                        position_trasnformed = position_trasnformed.squeeze().tolist()
                    
                    # Add transformed position to tracking data
                    tracks[object][frame_num][track_id]['position_transformed'] = position_trasnformed