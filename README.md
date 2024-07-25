# drods
Realtime object detect system

MobileCameraApp/
├── trained_images/               # Directory for storing trained images
├── dataset/                      # Directory for storing dataset images
├── recognition_data.db           # SQLite database
├── classifier.joblib             # Serialized KNN classifier (if exists)
├── main.py                       # Main application script
├── yolo_detection.py             # YOLO detection functions
├── gui_design.py                 # GUI design
├── requirements.txt              # List of dependencies
├── README.md                     # Project README file
└── LICENSE                       # License file (optional)

List all your dependencies in the requirements.txt file. Example:

Copy code
opencv-python-headless
face-recognition
torch
Pillow
joblib
pytesseract
matplotlib
# Mobile Camera App

This is a real-time object and face detection application using YOLOv5 and KNN classifier with a graphical user interface built with Tkinter.

## Features

- Real-time object and face detection
- Custom object recognition using KNN classifier
- Night vision mode
- Image capturing and tagging
- Model switching
- Dashboard for visualizing recognition statistics
- Export recognition reports

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/MobileCameraApp.git
    cd MobileCameraApp
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install the dependencies:
    ```sh
    pip install -r requirements.txt
    ```

4. Download and install Tesseract-OCR from [here](https://github.com/tesseract-ocr/tesseract). Add the Tesseract-OCR directory to your PATH.

## Usage

1. Ensure the directories `trained_images` and `dataset` exist in the project root.
2. Run the application:
    ```sh
    python main.py
    ```

## Files

- `main.py`: Main application script
- `yolo_detection.py`: YOLO detection functions
- `gui_design.py`: GUI design
- `requirements.txt`: List of dependencies
- `recognition_data.db`: SQLite database for storing recognition data
- `classifier.joblib`: Serialized KNN classifier

## Contributing

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -am 'Add some feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
Posting on GitHub
Create a Repository:

Go to GitHub and create a new repository.
Name it MobileCameraApp (or any other name you prefer).
Initialize Git:

In your project directory, initialize Git:
sh
Copy code
git init
Add Remote Repository:

Add the remote repository you created on GitHub:
sh
Copy code
git remote add origin https://github.com/yourusername/MobileCameraApp.git
Add and Commit Files:

Add all files to the repository and commit:
sh
Copy code
git add .
git commit -m "Initial commit"
Push to GitHub:

Push your local repository to GitHub:
sh
Copy code
git push -u origin master
Add .gitignore:

Add a .gitignore file to exclude unnecessary files from the repository. Example:
markdown
Copy code
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
venv/
recognition_data.db
classifier.joblib
trained_images/
dataset/
Example .gitignore
plaintext
Copy code
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
venv/
recognition_data.db
classifier.joblib
trained_images/
dataset/
This .gitignore ensures that Python cache files, virtual environments, the SQLite database, the trained classifier, and images are not included in the repository.
