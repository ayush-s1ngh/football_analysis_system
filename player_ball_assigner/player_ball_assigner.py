# Standard library imports for path handling and system configuration
import sys
import os

# Get the absolute path of the project root directory
# This ensures consistent imports regardless of where the script is run from
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the project root directory to Python's module search path
# This allows importing project modules from any subdirectory
sys.path.append(project_root)

# Import utility functions for bounding box calculations
from utils.bbox_utils import get_center_of_bbox, measure_distance

class PlayerBallAssigner:
    """
    A class to determine which player (if any) has possession of the ball
    based on proximity calculations between players and the ball.
    """

    def __init__(self):
        """
        Initialize the PlayerBallAssigner with a maximum distance threshold.
        The threshold of 70 pixels determines how close a player needs to be
        to the ball to be considered in possession of it.
        """
        self.max_player_ball_distance = 70
    
    def assign_ball_to_player(self, players, ball_bbox):
        """
        Determine which player is closest to the ball and likely has possession.
        Checks both left and right feet positions of each player.

        Args:
            players (dict): Dictionary of player information, keyed by player ID
                          Each player should have a 'bbox' key with coordinates
            ball_bbox (list): Bounding box coordinates of the ball [x1, y1, x2, y2]

        Returns:
            int: ID of the player closest to the ball, or -1 if no player is within
                the maximum allowed distance threshold
        """
        # Get the center coordinates of the ball
        ball_position = get_center_of_bbox(ball_bbox)

        # Initialize variables to track the closest player
        miniumum_distance = 99999  # Large initial value
        assigned_player = -1       # Default value indicating no player assigned

        # Check each player's distance to the ball
        for player_id, player in players.items():
            player_bbox = player['bbox']

            # Calculate distances from both feet (left and right edges of bbox)
            # Using the bottom of the bounding box (player_bbox[-1]) for y-coordinate
            distance_left = measure_distance((player_bbox[0], player_bbox[-1]), ball_position)
            distance_right = measure_distance((player_bbox[2], player_bbox[-1]), ball_position)
            
            # Use the shorter of the two distances (closest foot to ball)
            distance = min(distance_left, distance_right)

            # Update the assigned player if this player is within range
            # and closer than any previously checked player
            if distance < self.max_player_ball_distance:
                if distance < miniumum_distance:
                    miniumum_distance = distance
                    assigned_player = player_id

        return assigned_player