# Signal Sight

Signal Sight is an advanced AI-powered solution for automated traffic monitoring and enforcement. It leverages state-of-the-art object detection and tracking algorithms to detect and track vehicles in live or recorded video feeds, calculate their speeds, and enable enforcement mechanisms such as e-challan issuance for traffic violations.

## Features
- **Vehicle Detection and Tracking**: Utilizes the YOLOv8 model for real-time vehicle detection and tracking.
- **Speed Calculation**: Calculates vehicle speeds using perspective transformation and movement tracking over time.
- **Customizable Detection Zones**: Define specific regions for traffic monitoring using polygon zones.
- **High-Performance Video Processing**: Supports video annotation and saves output in MP4 format.
- **Interactive Display**: Provides a real-time annotated video feed, showcasing detected vehicles, tracking IDs, and calculated speeds.
- **Scalable and Modular**: Built with flexibility to integrate additional features like license plate recognition or direct violation reporting.

## Technologies Used
- **Python**: Core programming language.
- **OpenCV**: For video processing and perspective transformations.
- **YOLOv8**: State-of-the-art object detection framework.
- **Supervision**: For annotations, tracking, and enhancing video analysis.
- **ByteTrack**: Advanced multi-object tracking algorithm.

## How It Works
1. **Input**: Provide a video feed (`vehicles.mp4` by default) for analysis.
2. **Detection and Tracking**: The YOLO model identifies vehicles, and ByteTrack maintains consistent tracking across frames.
3. **Speed Estimation**: Calculates speed based on changes in position over time, leveraging perspective transformations.
4. **Output**: An annotated video with vehicle IDs and speeds is saved in the specified output directory.

## Requirements
- Python 3.8 or later
- Dependencies: Install via `pip install -r requirements.txt`

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Signal-Sight.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the YOLO model weights (e.g., `yolov8x.pt`) and place them in the project directory.

## Usage
1. Update the `SOURCE_VIDEO_PATH` in `main.py` with the path to your video file.
2. Run the script:
   ```bash
   python main.py
   ```
3. Press `q` to stop the video feed at any time.

## Future Enhancements
- Integrate license plate recognition for fine generation.
- Add real-time e-challan integration with traffic databases.
- Expand support for multi-camera systems and edge deployment.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for suggestions and improvements.
