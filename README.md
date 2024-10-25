Football Analysis System
This project is a comprehensive analysis tool for processing football match videos, identifying players, tracking ball possession, and visualizing camera movement, player speed, and team control statistics.

Features
Object Detection and Tracking: Uses YOLO for detecting and tracking players, referees, and the ball across video frames.
Camera Movement Compensation: Implements optical flow to estimate and adjust for camera movement, ensuring stable tracking data.
Perspective Transformation: Converts pixel-based positions to actual court coordinates for spatial accuracy.
Speed and Distance Calculation: Computes speed and cumulative distance covered by each player.
Team Assignment: Clusters players into teams based on jersey colors.
Ball Possession Tracking: Determines which player is in possession of the ball based on proximity.
Installation
Clone the Repository:

bash
Copy code
git clone <YOUR_REPOSITORY_LINK>
cd football_analysis_system
Install Requirements:

bash
Copy code
pip install -r requirements.txt
Model Weights: Download YOLO model weights and place them in the models directory as best_yolov8.pt.

Usage
Run the Analysis: Execute the main.py script to process a video and generate an annotated output:

bash
Copy code
python main.py
Input Video: Place the input video file in the input_videos folder. Update the file path in main.py if necessary.

Output: The processed video with annotations will be saved to output_videos/output_video.avi.

Project Structure
main.py: Main script orchestrating the analysis steps.
trackers/:
tracker.py: Detects and tracks objects (players, referees, ball).
team_assigner.py: Assigns team information based on jersey colors.
camera_movement_estimator.py: Estimates and adjusts for camera movement.
view_transformer.py: Transforms player positions from camera perspective to court view.
speed_and_distance_estimator.py: Calculates speed and distance for players.
player_ball_assigner.py: Tracks ball possession for each frame.
utils/: Contains helper functions for video handling and bounding box calculations.
Example Workflow
Load Video Frames: Reads video into frames.
Object Detection: Identifies players, referees, and ball.
Camera Stabilization: Adjusts tracking data to neutralize camera movement.
Coordinate Transformation: Converts player positions to top-down court view.
Team Classification: Assigns team colors and IDs based on clustering.
Ball Possession: Identifies which player has possession in each frame.
Annotations: Adds visual data (tracking, speed, ball possession) to frames.
Save Output Video: Outputs the annotated video.
Future Improvements
Real-time Processing: Integrate live processing capabilities for streaming applications.
Enhanced Object Detection Models: Explore more specialized models for better detection accuracy.
License
This project is licensed under the MIT License.

Acknowledgments
Special thanks to the YOLO model developers and open-source contributors for making computer vision accessible.
