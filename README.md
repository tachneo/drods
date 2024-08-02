
### Project Structure
```
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
```

### `requirements.txt`
List all your dependencies in the `requirements.txt` file. Example:
```
opencv-python-headless
face-recognition
torch
Pillow
joblib
pytesseract
matplotlib
```

### `README.md`
Create a `README.md` file to describe your project. Here's a template:

```markdown
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
    git clone https://github.com/drods/MobileCameraApp.git
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
```

### Posting on GitHub

1. **Create a Repository:**
   - Go to GitHub and create a new repository.
   - Name it `MobileCameraApp` (or any other name you prefer).

2. **Initialize Git:**
   - In your project directory, initialize Git:
     ```sh
     git init
     ```

3. **Add Remote Repository:**
   - Add the remote repository you created on GitHub:
     ```sh
     git remote add origin https://github.com/yourusername/MobileCameraApp.git
     ```

4. **Add and Commit Files:**
   - Add all files to the repository and commit:
     ```sh
     git add .
     git commit -m "Initial commit"
     ```

5. **Push to GitHub:**
   - Push your local repository to GitHub:
     ```sh
     git push -u origin master
     ```

6. **Add `.gitignore`:**
   - Add a `.gitignore` file to exclude unnecessary files from the repository. Example:
     ```
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
     ```

### Example `.gitignore`
```plaintext
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
```
![image](https://github.com/user-attachments/assets/ce56a747-e9b0-46c2-970a-7e5a3113f9f3)

This `.gitignore` ensures that Python cache files, virtual environments, the SQLite database, the trained classifier, and images are not included in the repository.
