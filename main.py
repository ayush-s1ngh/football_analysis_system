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
    Main function to process a football match video. It tracks objects, assigns players to teams, 
    estimates camera movement, and calculates speed and distance, all while generating visual annotations
    on the video.
    """
    
    # Step 1: Read the video frames from an input file
    video_frames = read_video('input_videos/8fd33_4.mp4')

    # Step 2: Initialize the tracker for object tracking (players, ball, referees)
    tracker = Tracker('models/best.pt')

    # Step 3: Obtain the object tracks (players, referees, ball) from the video frames
    # The tracks can be read from a stub if already saved
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl')
    
    # Step 4: Add positions (e.g., center or foot position) to each tracked object
    tracker.add_position_to_tracks(tracks)

    # Step 5: Initialize and estimate camera movement across frames
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                              read_from_stub=True,
                                                                              stub_path='stubs/camera_movement_stub.pkl')
    
    # Step 6: Adjust object positions in tracks based on camera movement
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # Step 7: Transform object positions to the bird's-eye view (top-down view of the field)
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)
    
    # Step 8: Interpolate ball positions to ensure smooth tracking even when some frames are missing ball detection
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Step 9: Estimate speed and distance for all objects in the tracks
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Step 10: Assign teams to players by analyzing their jersey colors
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])
    
    # Loop over each frame and assign teams and colors to players
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],   
                                                 track['bbox'],
                                                 player_id)
            # Store the team and the corresponding team color for each player
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # Step 11: Assign ball possession to players and determine which team controls the ball
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    
    # Loop through each frame to determine the player in possession of the ball
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        # If a player has the ball, mark it and record the team in possession
        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            # If no player has the ball, maintain possession by the last team
            team_ball_control.append(team_ball_control[-1])

    # Convert ball control history into a NumPy array for easier manipulation
    team_ball_control = np.array(team_ball_control)

    # Step 12: Draw annotations on the video frames (player tracks, ball positions, team control, etc.)
    ## Draw object tracks (players, ball, referees) and team ball control information
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    ## Draw camera movement annotations on the video frames
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)

    ## Draw speed and distance information on the video frames
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

    # Step 13: Save the processed video with annotations to an output file
    save_video(output_video_frames, 'output_videos/output_video.avi')

if __name__ == '__main__':
    main()
