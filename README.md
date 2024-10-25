# Football Analysis System

This project is a comprehensive tool for analyzing football match videos, allowing the tracking of players, ball possession, camera movement, and various team statistics. The system can generate annotated videos with detailed data visualizations.The project uses a machine learning model, YOLO (You Only Look Once), to detect objects in video frames. YOLO is known for its speed and efficiency in detecting multiple objects in a single pass, which is crucial in real-time or near-real-time analysis of video. The project also uses ByteTrack for tracking these detected objects across frames, so each player, referee, and the ball can be uniquely identified over time.

## Features

1. **Object Detection and Tracking**: Utilizes YOLO to detect and track players, referees, and the ball.
2. **Camera Movement Compensation**: Estimates and adjusts for camera movement using optical flow to maintain stable tracking data.
3. **Perspective Transformation**: Converts pixel coordinates to actual court positions for spatial accuracy.
4. **Speed and Distance Calculation**: Calculates the speed and distance covered by each player throughout the game.
5. **Team Assignment**: Clusters players into teams based on jersey colors.
6. **Ball Possession Tracking**: Identifies which player is in possession of the ball based on proximity.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/ayush-s1ngh/football_analysis/tree/main
   cd football_analysis_system
2. **Install Requirements**:
    ```bash
    pip install -r requirements.txt
3. **Model Weights**: 
    Download YOLO model weights after running training/model_training and place them in the models directory as best_yolov8.pt.

## Usage

1. **Run the Analysis**: Execute the main.py script to process a video and generate an annotated output.
    ```bash
    python main.py
2. **Input Video**: Place the input video file in the input_videos folder. Update the file path in main.py if necessary.

3. **Output**: The processed video with annotations will be saved to output_videos/output_video.avi.

## Example Workflow
- **Load Video Frames**: Reads video into frames.
- **Object Detection**: Identifies players, referees, and ball.
- **Camera Stabilization**: Adjusts tracking data to neutralize camera movement.
- **Coordinate Transformat**ion: Converts player positions to top-down court view.
- **Team Classification**: Assigns team colors and IDs based on clustering.
- **Ball Possession**: Identifies which player has possession in each frame.
- **Annotations**: Adds visual data (tracking, speed, ball possession) to frames.
- **Save Output Vide**o: Outputs the annotated video.

## Future Improvements
- **Real-time Processing**: Integrate live processing capabilities for streaming applications.
- **Enhanced Object Detection** Models: Explore more specialized models for better detection accuracy.

## Acknowledgements
- Abdullah Tarek for [this](https://github.com/abdullahtarek/football_analysis) repository and video tutorial.
- Special thanks to the YOLO model developers and open-source contributors for making computer vision accessible.