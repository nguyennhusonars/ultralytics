import argparse
import os
import re
import tempfile
import time
from collections import defaultdict
from glob import glob

import cv2
import numpy as np
from mss import mss
from pytube import YouTube

from ultralytics import YOLO

# Paths to your models
DETECTION_MODEL_PATH = "/home/sonnn/Documents/ultralytics/runs/detect/yolo8n_vehicle_1903/weights/best.pt"
CLASSIFICATION_MODEL_PATH = "/home/sonnn/Documents/ultralytics/vehicle-color-cls/yolov8n-color25/weights/best.pt"

# Create output directory and subdirectories for each color
OUTPUT_DIR = "/home/sonnn/Downloads/data_color_semisupervised"
# color_labels = ['beige', 'black', 'blue', 'brown', 'gold', 'green', 'grey',
#                 'orange', 'pink', 'purple', 'red', 'silver', 'tan', 'white', 'yellow']

color_labels = ["black", "blue", "brown", "green", "grey", "orange", "pink", "purple", "red", "white", "yellow"]

# Create main directory and subdirectories for each color
os.makedirs(OUTPUT_DIR, exist_ok=True)
for label in color_labels:
    os.makedirs(os.path.join(OUTPUT_DIR, label), exist_ok=True)

# Video file extensions to process
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm")

# Load models
detector = YOLO(DETECTION_MODEL_PATH)
classifier = YOLO(CLASSIFICATION_MODEL_PATH)

# Store classified colors and attempts for each track ID
track_colors = {}
track_attempts = defaultdict(int)  # Track number of classification attempts
track_last_extraction = {}  # Track last extraction time/frame for each track ID
MAX_CLASSIFICATION_ATTEMPTS = 9999  # Maximum number of attempts to classify an object
MINIMUM_DISTANCE_FROM_EDGE = 0
MIN_WIDTH = 120  # Set your minimum width in pixels

# Extraction interval settings
DEFAULT_EXTRACT_INTERVAL_FRAMES = 0  # 0 means no frame-based extraction
DEFAULT_EXTRACT_INTERVAL_SECONDS = 0  # 0 means no time-based extraction
DEFAULT_DETECTION_CONF = 0.25  # Default confidence threshold for object detection

# Define color mapping for each label (BGR format for OpenCV)
label_color_map = {
    "beige": (220, 245, 245),
    "black": (0, 0, 0),
    "blue": (255, 0, 0),
    "brown": (42, 42, 165),
    "gold": (0, 215, 255),
    "green": (0, 255, 0),
    "grey": (128, 128, 128),
    "orange": (0, 140, 255),
    "pink": (203, 192, 255),
    "purple": (128, 0, 128),
    "red": (0, 0, 255),
    "silver": (192, 192, 192),
    "tan": (180, 180, 210),
    "white": (255, 255, 255),
    "yellow": (0, 255, 255),
}


def clean_youtube_url(url):
    """Clean and extract YouTube video ID from various URL formats."""
    # Remove any escape characters
    url = url.replace("\\", "")

    # Try to extract video ID using regex
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",  # Standard YouTube URL
        r"youtu\.be\/([0-9A-Za-z_-]{11})",  # Short YouTube URL
        r"([0-9A-Za-z_-]{11})",  # Just the video ID
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            video_id = match.group(1)
            return f"https://www.youtube.com/watch?v={video_id}"

    return None


def download_youtube_video(url, max_retries=3):
    """Download YouTube video with retry logic and proper error handling."""
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1} of {max_retries} to download video...")

            # Create YouTube object with custom options
            yt = YouTube(url, use_oauth=False, allow_oauth_cache=True)

            # Get the best quality progressive stream
            video = yt.streams.filter(progressive=True, file_extension="mp4").order_by("resolution").desc().first()

            if video is None:
                print("No suitable video stream found")
                return None

            # Create temporary file path
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, f"temp_video_{int(time.time())}.mp4")

            # Download with progress callback
            print(f"Downloading: {yt.title}")
            print(f"Resolution: {video.resolution}")
            video.download(filename=temp_path)
            print("Download completed successfully!")
            return temp_path

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e!s}")
            if attempt < max_retries - 1:
                print("Waiting before retrying...")
                time.sleep(2)  # Wait 2 seconds before retrying
            else:
                print("All download attempts failed")
                return None


def get_video_source(source):
    """Handle different video sources (file path, RTSP, or YouTube URL)."""
    if source.startswith(("http://", "https://")):
        if "youtube.com" in source or "youtu.be" in source:
            # Clean and parse the URL
            clean_url = clean_youtube_url(source)
            if clean_url is None:
                print("Invalid YouTube URL format")
                return None

            print(f"Processing YouTube URL: {clean_url}")

            try:
                # Try to download with specific codec preference
                yt = YouTube(clean_url)
                # Try to get a stream with h264 codec first
                video = (
                    yt.streams.filter(progressive=True, file_extension="mp4", type="video")
                    .order_by("resolution")
                    .desc()
                    .first()
                )

                if video is None:
                    print("No suitable video stream found")
                    return None

                # Create temporary file path
                temp_dir = tempfile.gettempdir()
                temp_path = os.path.join(temp_dir, f"temp_video_{int(time.time())}.mp4")

                print(f"Downloading: {yt.title}")
                print(f"Resolution: {video.resolution}")
                video.download(filename=temp_path)
                print("Download completed successfully!")
                return temp_path

            except Exception as e:
                print(f"Error downloading YouTube video: {e!s}")
                return None
        else:
            # Handle other URLs (like RTSP)
            return source
    else:
        # Handle local file paths
        return source


def open_video_capture(source):
    """Open video capture with fallback options for different codecs."""
    # Try different backend options
    backends = [
        cv2.CAP_FFMPEG,  # Try FFMPEG first
        cv2.CAP_ANY,  # Fallback to any available backend
    ]

    for backend in backends:
        try:
            cap = cv2.VideoCapture(source, backend)
            if cap.isOpened():
                # Try to read first frame to verify it works
                ret, _ = cap.read()
                if ret:
                    # Set some properties for better performance
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # Set buffer size
                    # For RTSP, set some additional properties
                    if source.startswith(("rtsp://", "rtmp://")):
                        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"H264"))
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Smaller buffer for RTSP
                    return cap
                cap.release()
        except Exception as e:
            print(f"Error opening video with backend {backend}: {e}")
            continue

    return None


def process_video(
    video_source,
    vehicle_classes=None,
    skip_frames=0,
    color_threshold=0.7,
    video_index=None,
    reset_tracking_interval=0,
    reset_tracking_interval_frames=0,
    extract_interval_frames=DEFAULT_EXTRACT_INTERVAL_FRAMES,
    extract_interval_seconds=DEFAULT_EXTRACT_INTERVAL_SECONDS,
    detection_conf=DEFAULT_DETECTION_CONF,
):
    """Process video with vehicle detection and color classification.

    Args:
        video_source: Path to video file, RTSP URL, or YouTube URL
        vehicle_classes: List of vehicle class IDs to detect (None for all)
        skip_frames: Number of frames to skip between processing (0 for no skipping)
        color_threshold: Minimum confidence threshold for color classification (0.0 to 1.0)
        video_index: Index of the video being processed (None if not available)
        reset_tracking_interval: Interval in seconds to reset tracking (0 means no periodic reset)
        reset_tracking_interval_frames: Interval in frames to reset tracking (0 means no periodic reset by frame)
        extract_interval_frames: Extract image every N frames for each track ID (0 means no frame-based extraction)
        extract_interval_seconds: Extract image every N seconds for each track ID (0 means no time-based extraction)
        detection_conf: Confidence threshold for object detection (0.0 to 1.0)
    """
    global track_colors, track_attempts, track_last_extraction, detector
    # Reset tracking state for each video
    track_colors = {}
    track_attempts = defaultdict(int)
    track_last_extraction = {}
    detector = YOLO(DETECTION_MODEL_PATH)  # Re-instantiate the detector to reset the tracker

    # Get the appropriate video source
    source = get_video_source(video_source)
    if source is None:
        print("Failed to get video source")
        return

    # Open video source with fallback options
    cap = open_video_capture(source)
    if cap is None:
        print(f"Error: Could not open video source: {video_source}")
        return

    frame_count = 0
    processed_frames = 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    reset_interval_frames = 0
    if reset_tracking_interval_frames > 0:
        reset_interval_frames = reset_tracking_interval_frames
    elif reset_tracking_interval > 0:
        reset_interval_frames = int(reset_tracking_interval * fps)

    # Create named window and resize it
    cv2.namedWindow("Vehicle Color Classification", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Vehicle Color Classification", 1280, 720)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video or error reading frame")
                break

            frame_count += 1
            # Reset tracking state every N frames if interval is set
            if reset_interval_frames > 0 and frame_count % reset_interval_frames == 0:
                print(
                    f"[INFO] Resetting tracking state at frame {frame_count} (interval {reset_interval_frames} frames)"
                )
                track_colors = {}
                track_attempts = defaultdict(int)
                track_last_extraction = {}
                detector = YOLO(DETECTION_MODEL_PATH)

            # Skip frames if specified
            if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
                continue

            processed_frames += 1
            frame_clone = frame.copy()

            # Detect vehicles with class filtering
            results = detector.track(
                frame_clone,
                persist=True,
                classes=vehicle_classes,  # None means all classes
                conf=detection_conf,  # Apply detection confidence threshold
            )

            # Check if we have valid detection results
            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id
                classes = results[0].boxes.cls

                # Convert to numpy arrays if they exist
                if ids is not None:
                    ids = ids.cpu().numpy().astype(int)
                if classes is not None:
                    classes = classes.cpu().numpy().astype(int)

                # Only process if we have all required data
                if len(boxes) > 0 and ids is not None and classes is not None:
                    current_time = frame_count / fps  # Current time in seconds

                    for box, track_id, cls_id in zip(boxes, ids, classes):
                        x1, y1, x2, y2 = map(int, box)
                        width = x2 - x1

                        if width < MIN_WIDTH:
                            continue

                        frame_height, frame_width = frame_clone.shape[:2]
                        if (
                            x1 <= MINIMUM_DISTANCE_FROM_EDGE
                            or y1 <= MINIMUM_DISTANCE_FROM_EDGE
                            or x2 >= (frame_width - MINIMUM_DISTANCE_FROM_EDGE)
                            or y2 >= (frame_height - MINIMUM_DISTANCE_FROM_EDGE)
                        ):
                            continue

                        results[0].names[cls_id]
                        should_extract = False

                        # Check if we should extract based on frame interval
                        if extract_interval_frames > 0:
                            last_frame = track_last_extraction.get(track_id, {}).get("frame", 0)
                            if frame_count - last_frame >= extract_interval_frames:
                                should_extract = True
                                if track_id not in track_last_extraction:
                                    track_last_extraction[track_id] = {"frame": frame_count, "time": current_time}
                                else:
                                    track_last_extraction[track_id]["frame"] = frame_count
                                    track_last_extraction[track_id]["time"] = current_time

                        # Check if we should extract based on time interval
                        if extract_interval_seconds > 0:
                            last_time = track_last_extraction.get(track_id, {}).get("time", 0)
                            if current_time - last_time >= extract_interval_seconds:
                                should_extract = True
                                if track_id not in track_last_extraction:
                                    track_last_extraction[track_id] = {"frame": frame_count, "time": current_time}
                                else:
                                    track_last_extraction[track_id]["time"] = current_time
                                    track_last_extraction[track_id]["frame"] = frame_count

                        # Always process if it's a periodic extraction or initial classification
                        if should_extract or (
                            track_id not in track_colors and track_attempts[track_id] < MAX_CLASSIFICATION_ATTEMPTS
                        ):
                            crop = frame_clone[y1:y2, x1:x2]
                            crop_resized = cv2.resize(crop, (128, 128))

                            cls_results = classifier(crop_resized)
                            color_label = cls_results[0].names[int(cls_results[0].probs.top1)]
                            confidence = float(cls_results[0].probs.top1conf)

                            if not should_extract:  # Only increment attempts for initial classification
                                track_attempts[track_id] += 1

                            timestamp_ms = int(current_time * 1000)

                            # For periodic extraction, always save and update color
                            # For initial classification, only save if confidence meets threshold
                            if should_extract or (confidence >= color_threshold and not should_extract):
                                if should_extract:
                                    # Update the stored color for periodic extractions
                                    track_colors[track_id] = (color_label, confidence)

                                # Generate filename with appropriate suffix
                                if should_extract:
                                    suffix = f"_extract_{timestamp_ms}ms"
                                else:
                                    suffix = f"_{timestamp_ms}ms"

                                if video_index is not None:
                                    filename = f"{confidence:.2f}_{track_id}_vid{video_index}{suffix}.jpg"
                                else:
                                    filename = f"{confidence:.2f}_{track_id}{suffix}.jpg"

                                save_path = os.path.join(OUTPUT_DIR, color_label, filename)
                                cv2.imwrite(save_path, crop)

                                if should_extract:
                                    print(
                                        f"Track {track_id} extracted at frame {frame_count} (time: {current_time:.1f}s) with color {color_label} (conf: {confidence:.2f})"
                                    )
                                else:
                                    print(
                                        f"Track {track_id} classified as {color_label} with confidence {confidence:.2f} (attempt {track_attempts[track_id]}) - SAVED"
                                    )

                        # Draw box and color label - always use current classification for periodic extractions
                        if should_extract:
                            # Use current classification for periodic extractions
                            color_label = cls_results[0].names[int(cls_results[0].probs.top1)]
                            confidence = float(cls_results[0].probs.top1conf)
                            status = f"extract_{frame_count}"
                        else:
                            # Use stored classification for non-extraction frames
                            color_info = track_colors.get(track_id, ("Unknown", 0.0))
                            color_label, confidence = color_info
                            status = (
                                "saved"
                                if track_id in track_colors
                                else f"{track_attempts[track_id]}/{MAX_CLASSIFICATION_ATTEMPTS}"
                            )

                        if color_label in label_color_map:
                            box_color = label_color_map[color_label]
                            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                            cv2.putText(
                                frame,
                                f"ID:{track_id} {color_label} ({confidence:.2f}) [{status}]",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                box_color,
                                2,
                            )

            # Add frame info to display
            cv2.putText(
                frame,
                f"Frame: {frame_count} (Processed: {processed_frames})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            # Display the frame
            cv2.imshow("Vehicle Color Classification", frame)

            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        # Clean up temporary file if it was a YouTube video
        if video_source.startswith(("http://", "https://")) and (
            "youtube.com" in video_source or "youtu.be" in video_source
        ):
            try:
                temp_path = os.path.join(tempfile.gettempdir(), "temp_video.mp4")
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception as e:
                print(f"Error cleaning up temporary file: {e}")


def get_video_files(folder_path):
    """Get all video files in folder and subfolders."""
    video_files = []
    for ext in VIDEO_EXTENSIONS:
        # Search in current folder and all subfolders
        pattern = os.path.join(folder_path, f"**/*{ext}")
        video_files.extend(glob(pattern, recursive=True))
    return sorted(video_files)  # Sort for consistent processing order


def process_folder(
    folder_path,
    vehicle_classes=None,
    skip_frames=0,
    color_threshold=0.7,
    reset_tracking_interval=0,
    reset_tracking_interval_frames=0,
    extract_interval_frames=DEFAULT_EXTRACT_INTERVAL_FRAMES,
    extract_interval_seconds=DEFAULT_EXTRACT_INTERVAL_SECONDS,
    detection_conf=DEFAULT_DETECTION_CONF,
):
    """Process all videos in a folder and its subfolders."""
    video_files = get_video_files(folder_path)

    if not video_files:
        print(f"No video files found in {folder_path}")
        return

    print(f"\nFound {len(video_files)} video files to process:")
    for i, video_file in enumerate(video_files, 1):
        print(f"{i}. {video_file}")

    print("\nProcessing videos...")
    for i, video_file in enumerate(video_files, 1):
        print(f"\nProcessing video {i}/{len(video_files)}: {video_file}")
        process_video(
            video_file,
            vehicle_classes,
            skip_frames,
            color_threshold,
            video_index=i,
            reset_tracking_interval=reset_tracking_interval,
            reset_tracking_interval_frames=reset_tracking_interval_frames,
            extract_interval_frames=extract_interval_frames,
            extract_interval_seconds=extract_interval_seconds,
            detection_conf=detection_conf,
        )
        print(f"Completed video {i}/{len(video_files)}")


def capture_screen(monitor_number=1, region=None):
    """Capture screen from specified monitor using mss.

    Args:
        monitor_number: Monitor number to capture (1 is primary monitor)
        region: Tuple of (left, top, width, height) for specific region to capture, or None for full monitor
    """
    with mss() as sct:
        if monitor_number >= len(sct.monitors):
            print(f"Error: Monitor {monitor_number} not found")
            return None

        # Get monitor info
        monitor = sct.monitors[monitor_number]

        if region is not None:
            # Validate region coordinates
            left = region[0]
            top = region[1]
            width = region[2]
            height = region[3]

            # Check if region is within monitor bounds
            if left < 0 or top < 0:
                print(f"Error: Region coordinates ({left}, {top}) cannot be negative")
                return None

            if left + width > monitor["width"] or top + height > monitor["height"]:
                print(
                    f"Error: Region ({left}, {top}, {width}, {height}) exceeds monitor bounds ({monitor['width']}x{monitor['height']})"
                )
                return None

            # Calculate absolute coordinates for the region
            abs_left = monitor["left"] + left
            abs_top = monitor["top"] + top

            # Create custom monitor dict for the region
            monitor = {"left": abs_left, "top": abs_top, "width": width, "height": height}

            print(f"Capturing region: ({left}, {top}, {width}, {height}) on monitor {monitor_number}")
            print(f"Absolute coordinates: ({abs_left}, {abs_top}, {width}, {height})")

        try:
            # Capture the specified monitor/region
            screenshot = sct.grab(monitor)
            # Convert to numpy array and BGR format for OpenCV
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            return frame
        except Exception as e:
            print(f"Error capturing screen: {e}")
            return None


def list_available_monitors():
    """List all available monitors and their information."""
    with mss() as sct:
        monitors = sct.monitors
        print("\nAvailable monitors:")
        # Skip monitor 0 as it's the combined monitor
        for i in range(1, len(monitors)):
            monitor = monitors[i]
            print(f"Monitor {i}:")
            print(f"  Position: ({monitor['left']}, {monitor['top']})")
            print(f"  Size: {monitor['width']}x{monitor['height']}")
            print(f"  Primary: {'Yes' if i == 1 else 'No'}")
        return len(monitors) - 1  # Return number of actual monitors (excluding combined monitor)


def process_screen(
    monitor_number=1,
    vehicle_classes=None,
    skip_frames=0,
    color_threshold=0.7,
    reset_tracking_interval=0,
    reset_tracking_interval_frames=0,
    extract_interval_frames=DEFAULT_EXTRACT_INTERVAL_FRAMES,
    extract_interval_seconds=DEFAULT_EXTRACT_INTERVAL_SECONDS,
    detection_conf=DEFAULT_DETECTION_CONF,
    region=None,
):
    """Process screen capture for vehicle detection and color classification.

    Args:
        monitor_number: Monitor number to capture (1 is primary monitor)
        vehicle_classes: List of vehicle class IDs to detect (None for all)
        skip_frames: Number of frames to skip between processing (0 for no skipping)
        color_threshold: Minimum confidence threshold for color classification (0.0 to 1.0)
        reset_tracking_interval: Interval in seconds to reset tracking (0 means no periodic reset)
        reset_tracking_interval_frames: Interval in frames to reset tracking (0 means no periodic reset by frame)
        extract_interval_frames: Extract image every N frames for each track ID (0 means no frame-based extraction)
        extract_interval_seconds: Extract image every N seconds for each track ID (0 means no time-based extraction)
        detection_conf: Confidence threshold for object detection (0.0 to 1.0)
        region: Tuple of (left, top, width, height) for specific region to capture, or None for full monitor
    """
    global track_colors, track_attempts, track_last_extraction, detector

    # List available monitors and verify selection
    num_monitors = list_available_monitors()
    if monitor_number < 1 or monitor_number > num_monitors:
        print(f"Error: Invalid monitor number. Please choose between 1 and {num_monitors}")
        return

    print(f"\nCapturing from Monitor {monitor_number}")

    # Validate region if provided
    if region is not None:
        monitor_info = mss().monitors[monitor_number]
        if region[0] < 0 or region[1] < 0:
            print("Error: Region coordinates cannot be negative")
            return
        if region[0] + region[2] > monitor_info["width"] or region[1] + region[3] > monitor_info["height"]:
            print(
                f"Error: Region ({region[0]}, {region[1]}, {region[2]}, {region[3]}) exceeds monitor bounds ({monitor_info['width']}x{monitor_info['height']})"
            )
            return
        print(f"Region: left={region[0]}, top={region[1]}, width={region[2]}, height={region[3]}")

    # Reset tracking state
    track_colors = {}
    track_attempts = defaultdict(int)
    track_last_extraction = {}
    detector = YOLO(DETECTION_MODEL_PATH)

    frame_count = 0
    processed_frames = 0
    fps = 30  # Assuming 30 FPS for screen capture
    reset_interval_frames = 0
    if reset_tracking_interval_frames > 0:
        reset_interval_frames = reset_tracking_interval_frames
    elif reset_tracking_interval > 0:
        reset_interval_frames = int(reset_tracking_interval * fps)

    try:
        while True:
            # Capture screen from specified monitor/region
            frame = capture_screen(monitor_number, region)
            if frame is None:
                print("Error capturing screen")
                break

            frame_count += 1
            if reset_interval_frames > 0 and frame_count % reset_interval_frames == 0:
                print(
                    f"[INFO] Resetting tracking state at frame {frame_count} (interval {reset_interval_frames} frames)"
                )
                track_colors = {}
                track_attempts = defaultdict(int)
                track_last_extraction = {}
                detector = YOLO(DETECTION_MODEL_PATH)

            if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
                continue

            processed_frames += 1
            frame_clone = frame.copy()

            # Detect vehicles with class filtering
            results = detector.track(
                frame_clone,
                persist=True,
                classes=vehicle_classes,
                conf=detection_conf,  # Apply detection confidence threshold
            )

            # Process detections (same as in process_video)
            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id
                classes = results[0].boxes.cls

                if ids is not None:
                    ids = ids.cpu().numpy().astype(int)
                if classes is not None:
                    classes = classes.cpu().numpy().astype(int)

                if len(boxes) > 0 and ids is not None and classes is not None:
                    current_time = frame_count / fps  # Current time in seconds

                    for box, track_id, cls_id in zip(boxes, ids, classes):
                        x1, y1, x2, y2 = map(int, box)
                        width = x2 - x1

                        if width < MIN_WIDTH:
                            continue

                        frame_height, frame_width = frame_clone.shape[:2]
                        if (
                            x1 <= MINIMUM_DISTANCE_FROM_EDGE
                            or y1 <= MINIMUM_DISTANCE_FROM_EDGE
                            or x2 >= (frame_width - MINIMUM_DISTANCE_FROM_EDGE)
                            or y2 >= (frame_height - MINIMUM_DISTANCE_FROM_EDGE)
                        ):
                            continue

                        results[0].names[cls_id]
                        should_extract = False

                        # Check if we should extract based on frame interval
                        if extract_interval_frames > 0:
                            last_frame = track_last_extraction.get(track_id, {}).get("frame", 0)
                            if frame_count - last_frame >= extract_interval_frames:
                                should_extract = True
                                if track_id not in track_last_extraction:
                                    track_last_extraction[track_id] = {"frame": frame_count, "time": current_time}
                                else:
                                    track_last_extraction[track_id]["frame"] = frame_count
                                    track_last_extraction[track_id]["time"] = current_time

                        # Check if we should extract based on time interval
                        if extract_interval_seconds > 0:
                            last_time = track_last_extraction.get(track_id, {}).get("time", 0)
                            if current_time - last_time >= extract_interval_seconds:
                                should_extract = True
                                if track_id not in track_last_extraction:
                                    track_last_extraction[track_id] = {"frame": frame_count, "time": current_time}
                                else:
                                    track_last_extraction[track_id]["time"] = current_time
                                    track_last_extraction[track_id]["frame"] = frame_count

                        # Always process if it's a periodic extraction or initial classification
                        if should_extract or (
                            track_id not in track_colors and track_attempts[track_id] < MAX_CLASSIFICATION_ATTEMPTS
                        ):
                            crop = frame_clone[y1:y2, x1:x2]
                            crop_resized = cv2.resize(crop, (128, 128))

                            cls_results = classifier(crop_resized)
                            color_label = cls_results[0].names[int(cls_results[0].probs.top1)]
                            confidence = float(cls_results[0].probs.top1conf)

                            if not should_extract:  # Only increment attempts for initial classification
                                track_attempts[track_id] += 1

                            timestamp_ms = int(current_time * 1000)

                            # For periodic extraction, always save and update color
                            # For initial classification, only save if confidence meets threshold
                            if should_extract or (confidence >= color_threshold and not should_extract):
                                if should_extract:
                                    # Update the stored color for periodic extractions
                                    track_colors[track_id] = (color_label, confidence)

                                # Generate filename with appropriate suffix
                                if should_extract:
                                    pass
                                else:
                                    pass

                                filename = f"{confidence:.2f}_{track_id}_screen_{timestamp_ms}ms.jpg"
                                save_path = os.path.join(OUTPUT_DIR, color_label, filename)
                                cv2.imwrite(save_path, crop)

                                if should_extract:
                                    print(
                                        f"Track {track_id} extracted at frame {frame_count} (time: {current_time:.1f}s) with color {color_label} (conf: {confidence:.2f})"
                                    )
                                else:
                                    print(
                                        f"Track {track_id} classified as {color_label} with confidence {confidence:.2f} (attempt {track_attempts[track_id]}) - SAVED"
                                    )

                        color_info = track_colors.get(track_id, ("Unknown", 0.0))
                        color_label, confidence = color_info

                        status = (
                            "saved"
                            if track_id in track_colors
                            else f"{track_attempts[track_id]}/{MAX_CLASSIFICATION_ATTEMPTS}"
                        )

                        if color_label in label_color_map:
                            box_color = label_color_map[color_label]
                            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                            cv2.putText(
                                frame,
                                f"ID:{track_id} {color_label} ({confidence:.2f}) [{status}]",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                box_color,
                                2,
                            )

            # Add frame info to display
            cv2.putText(
                frame,
                f"Frame: {frame_count} (Processed: {processed_frames})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            # Create named window and resize it
            cv2.namedWindow("Vehicle Color Classification", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Vehicle Color Classification", 1280, 720)
            cv2.imshow("Vehicle Color Classification", frame)

            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video for vehicle color classification")
    parser.add_argument(
        "source",
        nargs="?",
        help="Path to video file, folder, RTSP URL, or YouTube URL. Use --screen for screen capture mode.",
    )
    parser.add_argument("--screen", action="store_true", help="Capture and process from screen instead of video source")
    parser.add_argument(
        "--monitor", type=int, default=1, help="Monitor number to capture from (default: 1, primary monitor)"
    )
    parser.add_argument(
        "--region",
        type=int,
        nargs=4,
        metavar=("LEFT", "TOP", "WIDTH", "HEIGHT"),
        help="Region of monitor to capture (left, top, width, height in pixels)",
    )
    parser.add_argument(
        "--classes", type=int, nargs="+", help="Vehicle class IDs to detect (e.g., 2 for car, 3 for motorcycle)"
    )
    parser.add_argument(
        "--skip-frames", type=int, default=0, help="Number of frames to skip between processing (default: 0)"
    )
    parser.add_argument(
        "--color-threshold",
        type=float,
        default=0.7,
        help="Minimum confidence threshold for color classification (0.0 to 1.0, default: 0.7)",
    )
    parser.add_argument(
        "--detection-conf",
        type=float,
        default=DEFAULT_DETECTION_CONF,
        help="Confidence threshold for object detection (0.0 to 1.0, default: 0.25)",
    )
    parser.add_argument(
        "--reset-tracking-interval",
        type=float,
        default=0,
        help="Interval in seconds to reset tracking (0 means no periodic reset)",
    )
    parser.add_argument(
        "--reset-tracking-interval-frames",
        type=int,
        default=0,
        help="Interval in frames to reset tracking (0 means no periodic reset by frame)",
    )
    parser.add_argument(
        "--extract-interval-frames",
        type=int,
        default=DEFAULT_EXTRACT_INTERVAL_FRAMES,
        help="Extract image every N frames for each track ID (0 means no frame-based extraction)",
    )
    parser.add_argument(
        "--extract-interval-seconds",
        type=float,
        default=DEFAULT_EXTRACT_INTERVAL_SECONDS,
        help="Extract image every N seconds for each track ID (0 means no time-based extraction)",
    )
    args = parser.parse_args()

    # Print available classes
    print("\nAvailable vehicle classes:")
    for class_id, class_name in detector.names.items():
        print(f"  {class_id}: {class_name}")

    if args.screen:
        print("\nProcessing screen capture...")
        # Convert region argument to tuple if provided
        region = tuple(args.region) if args.region else None
        process_screen(
            args.monitor,
            args.classes,
            args.skip_frames,
            args.color_threshold,
            args.reset_tracking_interval,
            args.reset_tracking_interval_frames,
            args.extract_interval_frames,
            args.extract_interval_seconds,
            args.detection_conf,
            region,
        )
    elif args.source:
        # Check if source is a folder
        if os.path.isdir(args.source):
            print(f"\nProcessing folder: {args.source}")
            process_folder(
                args.source,
                args.classes,
                args.skip_frames,
                args.color_threshold,
                args.reset_tracking_interval,
                args.reset_tracking_interval_frames,
                args.extract_interval_frames,
                args.extract_interval_seconds,
                args.detection_conf,
            )
        else:
            # Process single video source
            print("\nProcessing single video source...")
            process_video(
                args.source,
                args.classes,
                args.skip_frames,
                args.color_threshold,
                reset_tracking_interval=args.reset_tracking_interval,
                reset_tracking_interval_frames=args.reset_tracking_interval_frames,
                extract_interval_frames=args.extract_interval_frames,
                extract_interval_seconds=args.extract_interval_seconds,
                detection_conf=args.detection_conf,
            )
    else:
        parser.print_help()
        print("\nError: Either provide a source or use --screen option")
