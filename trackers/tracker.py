import sys
import os

# Get the absolute path of the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the project root directory to sys.path to ensure modules from the project root can be imported
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
    Tracker class to manage object detection, tracking, and annotations within video frames.
    It uses the YOLO model for object detection, ByteTrack for tracking, and applies custom annotations.
    """

    def __init__(self, model_path):
        """
        Initializes the Tracker with a YOLO model and ByteTrack tracker.

        Args:
            model_path (str): Path to the YOLO model weights.
        """
        self.model = YOLO(model_path)  # Load YOLO model for detection
        self.tracker = sv.ByteTrack()  # Initialize ByteTrack for object tracking

    def add_position_to_tracks(self, tracks):
        """
        Adds object positions to the tracks based on bounding box coordinates.
        If the object is the ball, the center of the bounding box is used as its position.
        Otherwise, the foot position is calculated for other objects.

        Args:
            tracks (dict): Dictionary containing object tracks for players, referees, and ball.
        """
        for object_type, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object_type == 'ball':
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object_type][frame_num][track_id]['position'] = position

    def interpolate_ball_positions(self, ball_positions):
        """
        Interpolates missing ball positions across frames using bounding box values.

        Args:
            ball_positions (list): List of ball positions across frames.

        Returns:
            list: Interpolated ball positions in the form of bounding boxes.
        """
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # Interpolate missing bounding box values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def detect_frames(self, frames):
        """
        Performs object detection on a batch of video frames.

        Args:
            frames (list): List of video frames to detect objects in.

        Returns:
            list: List of detection results for the frames.
        """
        batch_size = 20  # Define batch size for YOLO predictions
        detections = []  # Store all detections
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        """
        Tracks objects such as players, referees, and the ball across video frames.
        Tracks can be loaded from a stub (cached file) if provided.

        Args:
            frames (list): List of video frames.
            read_from_stub (bool): If True, load tracks from a stub file.
            stub_path (str): Path to the stub file to load/save tracks.

        Returns:
            dict: Dictionary containing tracked positions for players, referees, and ball.
        """
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        # Detect objects in frames
        detections = self.detect_frames(frames)

        # Initialize tracking dictionary
        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        # Process detections and track objects
        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            # Convert detection to supervision format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert "goalkeeper" class to "player"
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            # Track objects using ByteTrack
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            # Update tracks with detected objects
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}

                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            # Track ball object separately
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        # Save tracks to stub file if provided
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        """
        Draws an ellipse around a bounding box, typically used for annotating players or referees.

        Args:
            frame (np.array): The video frame to annotate.
            bbox (list): Bounding box of the object.
            color (tuple): Color for the ellipse.
            track_id (int): Optional track ID to annotate.

        Returns:
            np.array: Annotated video frame.
        """
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        # Draw an ellipse on the frame
        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        # Optionally, draw the track ID
        if track_id is not None:
            rectangle_width = 40
            rectangle_height = 20
            x1_rect = x_center - rectangle_width // 2
            x2_rect = x_center + rectangle_width // 2
            y1_rect = (y2 - rectangle_height // 2) + 15
            y2_rect = (y2 + rectangle_height // 2) + 15

            cv2.rectangle(frame, (int(x1_rect), int(y1_rect)), (int(x2_rect), int(y2_rect)), color, cv2.FILLED)
            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(frame, f"{track_id}", (int(x1_text), int(y1_rect + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        return frame

    def draw_triangle(self, frame, bbox, color):
        """
        Draws a triangle at the top of a bounding box, typically used to annotate the ball or special objects.

        Args:
            frame (np.array): The video frame to annotate.
            bbox (list): Bounding box of the object.
            color (tuple): Color for the triangle.

        Returns:
            np.array: Annotated video frame.
        """
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x, y],
            [x - 10, y - 20],
            [x + 10, y - 20],
        ])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        """
        Draws a semi-transparent rectangle showing the percentage of ball control for each team.

        Args:
            frame (np.array): The video frame to annotate.
            frame_num (int): Current frame number.
            team_ball_control (np.array): Array representing ball control over time.

        Returns:
            np.array: Annotated video frame.
        """
        # Draw a semi-transparent rectangle
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Calculate ball control percentage for each team
        team_ball_control_till_frame = team_ball_control[:frame_num + 1]
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 2].shape[0]
        team_1 = team_1_num_frames / (team_1_num_frames + team_2_num_frames)
        team_2 = team_2_num_frames / (team_1_num_frames + team_2_num_frames)

        # Display ball control percentage
        cv2.putText(frame, f"Team 1 Ball Control: {team_1 * 100:.2f}%", (1400, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2 * 100:.2f}%", (1400, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control):
        """
        Draws annotations on the video frames including player, referee, and ball positions, as well as team ball control.

        Args:
            video_frames (list): List of video frames.
            tracks (dict): Tracked object positions for players, referees, and ball.
            team_ball_control (np.array): Array representing ball control over time.

        Returns:
            list: Annotated video frames.
        """
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))  # Default red if no team color
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

                if player.get('has_ball', False):
                    frame = self.draw_triangle(frame, player["bbox"], (0, 0, 255))  # Mark the player with ball possession

            # Draw Referees
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))  # Yellow for referees

            # Draw Ball
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0, 255, 0))  # Green for the ball

            # Draw Team Ball Control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_video_frames.append(frame)

        return output_video_frames
