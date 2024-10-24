import sys
import os
# Get the absolute path of the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the project root directory to sys.path
sys.path.append(project_root)

from utils.bbox_utils import get_center_of_bbox, get_bbox_width, get_foot_position
from ultralytics import YOLO
import supervision as sv
import pickle
import numpy as np
import pandas as pd
import cv2


class Tracker:
    """
    A class for tracking and visualizing objects (players, referees, ball) in video frames.
    Uses YOLO model for object detection and ByteTrack for object tracking.
    """

    def __init__(self, model_path):
        """
        Initialize the tracker with a YOLO model.

        Args:
            model_path (str): Path to the YOLO model weights
        """
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def add_position_to_tracks(self, tracks):
        """
        Add position information to tracked objects. For the ball, uses center position;
        for players/referees, uses foot position.

        Args:
            tracks (dict): Dictionary containing tracking information for all objects
        """
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position

    def interpolate_ball_positions(self, ball_positions):
        """
        Interpolate missing ball positions using pandas interpolation.
        Fills gaps in ball tracking data for smoother visualization.

        Args:
            ball_positions (list): List of dictionaries containing ball position data

        Returns:
            list: Interpolated ball positions
        """
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # Interpolate missing values and fill any remaining gaps
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def detect_frames(self, frames):
        """
        Perform object detection on video frames in batches.

        Args:
            frames (list): List of video frames to process

        Returns:
            list: YOLO detection results for each frame
        """
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        """
        Get tracking information for all objects (players, referees, ball) in video frames.
        Can read from cached results (stub) or perform new detection and tracking.

        Args:
            frames (list): List of video frames
            read_from_stub (bool): Whether to read cached results
            stub_path (str): Path to cached results file

        Returns:
            dict: Tracking information for all objects
        """
        # Try to load cached results if requested
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            # Convert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert goalkeeper to player object for consistent tracking
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            # Update tracks with new detections
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            # Initialize empty dictionaries for current frame
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            # Process tracked objects (players and referees)
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
                
                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}
            
            # Process ball detections (not tracked, assumes single ball)
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        # Cache results if stub_path provided
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks
    
    def draw_ellipse(self, frame, bbox, color, track_id=None):
        """
        Draw an ellipse under tracked objects to indicate their position.
        Optionally adds ID number for tracked objects.

        Args:
            frame (np.array): Video frame to draw on
            bbox (list): Bounding box coordinates [x1, y1, x2, y2]
            color (tuple): RGB color for the ellipse
            track_id (int, optional): ID number to display

        Returns:
            np.array: Frame with drawn ellipse
        """
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        # Draw shadow ellipse
        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        # Draw ID number if provided
        if track_id is not None:
            rectangle_width = 40
            rectangle_height = 20
            x1_rect = x_center - rectangle_width//2
            x2_rect = x_center + rectangle_width//2
            y1_rect = (y2 - rectangle_height//2) + 15
            y2_rect = (y2 + rectangle_height//2) + 15

            cv2.rectangle(frame,
                         (int(x1_rect), int(y1_rect)),
                         (int(x2_rect), int(y2_rect)),
                         color,
                         cv2.FILLED)
            
            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

        return frame

    def draw_triangle(self, frame, bbox, color):
        """
        Draw a triangle above objects to indicate special status (e.g., ball possession).

        Args:
            frame (np.array): Video frame to draw on
            bbox (list): Bounding box coordinates [x1, y1, x2, y2]
            color (tuple): RGB color for the triangle

        Returns:
            np.array: Frame with drawn triangle
        """
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x, y],
            [x-10, y-20],
            [x+10, y-20],
        ])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame
    
    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        """
        Draw ball possession statistics for both teams on the frame.

        Args:
            frame (np.array): Video frame to draw on
            frame_num (int): Current frame number
            team_ball_control (np.array): Array indicating ball possession by team

        Returns:
            np.array: Frame with drawn statistics
        """
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Calculate possession percentages
        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]
        team_1 = team_1_num_frames/(team_1_num_frames+team_2_num_frames)
        team_2 = team_2_num_frames/(team_1_num_frames+team_2_num_frames)

        # Draw possession statistics
        cv2.putText(frame, f"Team 1 Ball Control: {team_1*100:.2f}%", (1400, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2*100:.2f}%", (1400, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control):
        """
        Draw all annotations (players, referees, ball, statistics) on video frames.

        Args:
            video_frames (list): List of video frames
            tracks (dict): Tracking information for all objects
            team_ball_control (np.array): Array indicating ball possession by team

        Returns:
            list: Annotated video frames
        """
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Draw players with team colors and ball possession indicators
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

                if player.get('has_ball', False):
                    frame = self.draw_triangle(frame, player["bbox"], (0, 0, 255))

            # Draw referees with yellow color
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))
            
            # Draw ball with green triangle
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0, 255, 0))

            # Draw ball possession statistics
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_video_frames.append(frame)

        return output_video_frames