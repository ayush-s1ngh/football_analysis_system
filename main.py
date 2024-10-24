from utils import save_video, read_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
import cv2
import numpy as np

def main():
    """
    Main function that processes a sports video through multiple analysis steps:
    1. Object tracking (players, ball)
    2. Camera movement compensation
    3. Perspective transformation
    4. Speed and distance calculations
    5. Team assignment
    6. Ball possession tracking
    """
    
    # Step 1: Read input video frames
    video_frames = read_video('input_videos/8fd33_4.mp4')

    # Step 2: Initialize object tracker with pre-trained model
    tracker = Tracker('models/best.pt')

    # Step 3: Get object tracks (players, ball, referees)
    # Can read from pre-computed stub file for faster development
    tracks = tracker.get_object_tracks(video_frames,
                                     read_from_stub=True,
                                     stub_path='stubs/track_stubs.pkl')
    
    # Step 4: Add center position to all tracked objects
    tracker.add_position_to_tracks(tracks)

    # Step 5: Initialize and apply camera movement estimation
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames,
        read_from_stub=True,
        stub_path='stubs/camera_movement_stub.pkl'
    )
    
    # Step 6: Adjust object positions based on camera movement
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # Step 7: Transform pixel coordinates to actual court coordinates
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)
    
    # Step 8: Interpolate missing ball positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Step 9: Calculate speed and distance metrics for players
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Step 10: Initialize team assignment
    team_assigner = TeamAssigner()
    # Analyze first frame to determine team colors
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])
    
    # Step 11: Assign team information to all players in all frames
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            # Get team assignment based on player appearance
            team = team_assigner.get_player_team(video_frames[frame_num],   
                                               track['bbox'],
                                               player_id)
            
            # Add team information to tracking data
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # Step 12: Track ball possession
    player_assigner = PlayerBallAssigner()
    team_ball_control = []  # List to store which team has ball possession
    
    # Analyze each frame for ball possession
    for frame_num, player_track in enumerate(tracks['players']):
        # Get ball position
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        # Determine which player (if any) has the ball
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        # Update ball possession information
        if assigned_player != -1:
            # Mark player as having ball
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            # Record which team has possession
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            # If no clear possession, maintain previous state
            team_ball_control.append(team_ball_control[-1])

    # Convert ball control list to numpy array for easier processing
    team_ball_control = np.array(team_ball_control)

    # Step 13: Draw various annotations on video frames
    # Add object tracking annotations
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    # Add camera movement visualization
    output_video_frames = camera_movement_estimator.draw_camera_movement(
        output_video_frames, 
        camera_movement_per_frame
    )

    # Add speed and distance information
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

    # Step 14: Save final annotated video
    save_video(output_video_frames, 'output_videos/output_video.avi')

if __name__ == '__main__':
    main()