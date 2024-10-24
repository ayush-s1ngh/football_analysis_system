import cv2

def read_video(video_path):
    """
    Read a video file and convert it to a list of frames.
    
    Args:
        video_path (str): Path to the input video file
        
    Returns:
        list: List of numpy arrays, where each array represents a video frame
              Format of each frame: Height x Width x Channels (BGR)
    """
    # Create a VideoCapture object to read the video
    cap = cv2.VideoCapture(video_path)
    
    # Initialize empty list to store video frames
    frames = []
    
    # Read frames until end of video
    while True:
        # Read a single frame
        # ret: Boolean indicating if frame was successfully read
        # frame: numpy array containing the frame data
        ret, frame = cap.read()
        
        # Break loop if frame reading failed (end of video)
        if not ret:
            break
            
        # Add frame to our list
        frames.append(frame)
        
    return frames

def save_video(output_video_frames, output_video_path):
    """
    Save a list of frames as a video file.
    
    Args:
        output_video_frames (list): List of numpy arrays representing video frames
                                  All frames must have the same dimensions
        output_video_path (str): Path where the output video will be saved
    
    Notes:
        - Uses XVID codec for compression
        - Frame rate is set to 24 FPS
        - Video dimensions are automatically determined from the first frame
    """
    # Define the codec using VideoWriter_fourcc
    # XVID is a popular codec that provides good compression
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    # Create VideoWriter object
    # Parameters:
    # - output_video_path: Path where video will be saved
    # - fourcc: Codec used for compression
    # - 24: Frame rate (FPS)
    # - (width, height): Frame dimensions from first frame
    out = cv2.VideoWriter(
        output_video_path, 
        fourcc, 
        24, 
        (output_video_frames[0].shape[1],  # Width
         output_video_frames[0].shape[0])   # Height
    )
    
    # Write each frame to the video file
    for frame in output_video_frames:
        out.write(frame)
    
    # Release the VideoWriter to close the output file
    out.release()