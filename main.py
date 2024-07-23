import tkinter as tk
from tkinter import messagebox, filedialog, simpledialog
import cv2
from PIL import Image, ImageTk
import torch
import threading
import queue
import sqlite3
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import joblib
import pytesseract
import csv

# Ensure the trained_images directory exists
os.makedirs("trained_images", exist_ok=True)
os.makedirs("dataset", exist_ok=True)

# Load YOLOv5 model
model_options = {
    "YOLOv5s": "yolov5s",
    "YOLOv5m": "yolov5m",
    "YOLOv5l": "yolov5l",
    "YOLOv5x": "yolov5x"
}

# Initialize database
conn = sqlite3.connect('recognition_data.db')
cursor = conn.cursor()
cursor.execute('''
CREATE TABLE IF NOT EXISTS recognitions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT,
    name TEXT,
    datetime TEXT,
    image_path TEXT,
    model_number TEXT,
    remarks TEXT
)
''')
conn.commit()

class MobileCameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Mobile Camera App")
        self.root.geometry("1200x700")
        self.root.configure(bg='#2c3e50')
        
        self.sidebar_frame = tk.Frame(root, bg='#2c3e50', width=200)
        self.sidebar_frame.pack(side='left', fill='y')
        
        self.main_frame = tk.Frame(root, bg='#34495e')
        self.main_frame.pack(side='right', expand=True, fill='both')

        self.timer_label = tk.Label(self.sidebar_frame, text="", font=("Helvetica", 10), bg='#2c3e50', fg='white')
        self.timer_label.pack(padx=10, pady=5)

        self.camera_frame = tk.Label(self.main_frame, bg='#34495e')
        self.camera_frame.pack(padx=20, pady=20, expand=True, fill='both')
        
        self.control_frame = tk.Frame(self.main_frame, bg='#2c3e50')
        self.control_frame.pack(side='bottom', pady=10)

        self.start_button = tk.Button(self.control_frame, text="Start Camera", command=self.start_camera, bg='#27ae60', fg='white', font=("Helvetica", 10))
        self.start_button.grid(row=0, column=0, padx=5, pady=5)
        
        self.stop_button = tk.Button(self.control_frame, text="Stop Camera", command=self.stop_camera, bg='#e74c3c', fg='white', font=("Helvetica", 10))
        self.stop_button.grid(row=0, column=1, padx=5, pady=5)

        self.start_detection_button = tk.Button(self.control_frame, text="Start Detection", command=self.start_detection, bg='#3498db', fg='white', font=("Helvetica", 10))
        self.start_detection_button.grid(row=0, column=2, padx=5, pady=5)

        self.stop_detection_button = tk.Button(self.control_frame, text="Stop Detection", command=self.stop_detection, bg='#e74c3c', fg='white', font=("Helvetica", 10))
        self.stop_detection_button.grid(row=0, column=3, padx=5, pady=5)
        
        self.capture_button = tk.Button(self.control_frame, text="Capture & Tag", command=self.capture_image, bg='#3498db', fg='white', font=("Helvetica", 10))
        self.capture_button.grid(row=0, column=4, padx=5, pady=5)
        
        self.train_button = tk.Button(self.control_frame, text="Train Model", command=self.train_model, bg='#e67e22', fg='white', font=("Helvetica", 10))
        self.train_button.grid(row=0, column=5, padx=5, pady=5)

        self.switch_model_button = tk.Button(self.control_frame, text="Switch Model", command=self.switch_model, bg='#9b59b6', fg='white', font=("Helvetica", 10))
        self.switch_model_button.grid(row=0, column=6, padx=5, pady=5)

        self.night_vision_button = tk.Button(self.control_frame, text="Toggle Night Vision", command=self.toggle_night_vision, bg='#2c3e50', fg='white', font=("Helvetica", 10))
        self.night_vision_button.grid(row=0, column=7, padx=5, pady=5)

        self.ip_entry_label = tk.Label(self.control_frame, text="IP Address:", font=("Helvetica", 10), bg='#2c3e50', fg='white')
        self.ip_entry_label.grid(row=0, column=8, padx=5, pady=5)
        self.ip_entry = tk.Entry(self.control_frame, font=("Helvetica", 10))
        self.ip_entry.grid(row=0, column=9, padx=5, pady=5)
        self.ip_entry.insert(0, "192.168.1.2:8080")

        self.dashboard_button = tk.Button(self.sidebar_frame, text="Dashboard", command=self.show_dashboard, bg='#3498db', fg='white', font=("Helvetica", 12, "bold"))
        self.dashboard_button.pack(fill='x', padx=10, pady=10)
        
        self.real_time_button = tk.Button(self.sidebar_frame, text="Real-time View", command=self.show_realtime_view, bg='#3498db', fg='white', font=("Helvetica", 12, "bold"))
        self.real_time_button.pack(fill='x', padx=10, pady=10)

        self.date_filter_label = tk.Label(self.sidebar_frame, text="Filter by Date Range:", font=("Helvetica", 12, "bold"), bg='#2c3e50', fg='white')
        self.date_filter_label.pack(fill='x', padx=10, pady=5)

        self.start_date_entry = tk.Entry(self.sidebar_frame, font=("Helvetica", 12))
        self.start_date_entry.pack(fill='x', padx=10, pady=5)
        self.start_date_entry.insert(0, "Start Date (YYYY-MM-DD)")

        self.end_date_entry = tk.Entry(self.sidebar_frame, font=("Helvetica", 12))
        self.end_date_entry.pack(fill='x', padx=10, pady=5)
        self.end_date_entry.insert(0, "End Date (YYYY-MM-DD)")

        self.filter_button = tk.Button(self.sidebar_frame, text="Apply Filter", command=self.apply_filter, bg='#27ae60', fg='white', font=("Helvetica", 12, "bold"))
        self.filter_button.pack(fill='x', padx=10, pady=10)
        
        self.export_button = tk.Button(self.sidebar_frame, text="Export Report", command=self.export_report, bg='#e67e22', fg='white', font=("Helvetica", 12, "bold"))
        self.export_button.pack(fill='x', padx=10, pady=10)
        
        self.category_count_label = tk.Label(self.sidebar_frame, text="", font=("Helvetica", 12), bg='#2c3e50', fg='white')
        self.category_count_label.pack(fill='x', padx=10, pady=10)

        self.vid = None
        self.running = False
        self.detecting = False
        self.night_vision = False
        self.model = model_options["YOLOv5s"]
        self.model_loaded = torch.hub.load('ultralytics/yolov5', self.model, pretrained=True)
        self.camera_index = 0
        
        self.frame_queue = queue.Queue()
        self.image_queue = queue.Queue()
        
        self.knn_classifier = None
        self.load_classifier()

        self.update_timer()
        self.update_category_count()
        self.root.after(10, self.update_gui_frame)

    def load_classifier(self):
        if os.path.exists('classifier.joblib'):
            self.knn_classifier = joblib.load('classifier.joblib')
        else:
            self.knn_classifier = None

    def save_classifier(self):
        if self.knn_classifier is not None:
            joblib.dump(self.knn_classifier, 'classifier.joblib')

    def extract_features(self, img_path):
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (100, 100))
        return gray.flatten()

    def start_camera(self):
        ip = self.ip_entry.get()
        if self.camera_index == 0:
            self.vid = cv2.VideoCapture(f"http://{ip}/video")
        else:
            self.vid = cv2.VideoCapture(f"http://{ip}/video2")
        
        if not self.vid.isOpened():
            messagebox.showerror("Error", "Unable to access the camera. Please check the IP address.")
            return
        self.running = True
        threading.Thread(target=self.update_frame, daemon=True).start()
    
    def stop_camera(self):
        self.running = False
        if self.vid:
            self.vid.release()
            self.vid = None
        self.camera_frame.config(image='')

    def start_detection(self):
        self.detecting = True
        threading.Thread(target=self.process_frame, daemon=True).start()

    def stop_detection(self):
        self.detecting = False

    def switch_camera(self):
        self.camera_index = 1 - self.camera_index  # Toggle between 0 and 1
        if self.running:
            self.stop_camera()
            self.start_camera()

    def capture_image(self):
        if self.vid and self.running:
            ret, frame = self.vid.read()
            if ret:
                category = simpledialog.askstring("Input", "Enter category:")
                name = simpledialog.askstring("Input", "Enter name:")
                model_number = simpledialog.askstring("Input", "Enter model number:")
                remarks = simpledialog.askstring("Input", "Enter remarks:")
                if category and name:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"trained_images/{category}_{name}_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    messagebox.showinfo("Image Capture", f"Image captured and saved as {filename}")
                    cursor.execute("INSERT INTO recognitions (category, name, datetime, image_path, model_number, remarks) VALUES (?, ?, ?, ?, ?, ?)", (category, name, timestamp, filename, model_number, remarks))
                    conn.commit()
                    self.train_model()
                else:
                    messagebox.showerror("Error", "Category and name must be provided.")

    def prepare_dataset(self):
        images = []
        labels = []
        for root, dirs, files in os.walk("trained_images"):
            for file in files:
                if file.endswith(".jpg"):
                    path = os.path.join(root, file)
                    try:
                        label = file.split('_')[1]  # Use the name as label
                    except IndexError:
                        continue  # Skip files that do not have the expected format
                    features = self.extract_features(path)
                    images.append(features)
                    labels.append(label)
        return np.array(images), np.array(labels)

    def train_model(self):
        images, labels = self.prepare_dataset()
        if len(images) > 0:
            self.knn_classifier = CustomKNNClassifier()
            self.knn_classifier.fit(images, labels)
            self.save_classifier()
            messagebox.showinfo("Train Model", "Training completed!")
        else:
            messagebox.showinfo("Train Model", "No images to train.")

    def switch_model(self):
        model_name = simpledialog.askstring("Input", f"Enter model name ({', '.join(model_options.keys())}):")
        if model_name in model_options:
            self.model = model_options[model_name]
            self.model_loaded = torch.hub.load('ultralytics/yolov5', self.model, pretrained=True)
            messagebox.showinfo("Model Switch", f"Switched to {model_name}")
        else:
            messagebox.showerror("Error", "Invalid model name. Please enter a valid model name.")
    
    def toggle_night_vision(self):
        self.night_vision = not self.night_vision
        state = "enabled" if self.night_vision else "disabled"
        messagebox.showinfo("Night Vision", f"Night vision {state}")

    def update_frame(self):
        while self.running and self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                self.frame_queue.put(frame)
            time.sleep(0.03)  # Slight delay to reduce CPU usage
    
    def process_frame(self):
        while self.detecting:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                # Apply night vision enhancement if enabled
                if self.night_vision:
                    frame = self.apply_night_vision(frame)
                
                # Reduce frame size for faster processing
                frame_resized = cv2.resize(frame, (640, 480))

                # Add date and time to the frame
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(frame_resized, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

                # Check if we have a trained classifier
                if self.knn_classifier:
                    gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
                    gray_frame = cv2.resize(gray_frame, (100, 100)).flatten().reshape(1, -1)
                    prediction = self.knn_classifier.predict(gray_frame)
                    label = prediction[0]
                    conf = 1.0  # Confidence not calculated, assuming 100% for custom KNN
                    x1, y1, x2, y2 = 50, 50, 200, 200  # Placeholder values, no bounding box info from KNN
                    cv2.putText(frame_resized, f'{label} {conf:.2f}', (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    # Perform YOLO detection
                    results = self.model_loaded(frame_resized)
                    detections = results.pandas().xyxy[0]
                    
                    for _, row in detections.iterrows():
                        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                        label = row['name']
                        conf = row['confidence']
                        cv2.putText(frame_resized, f'{label} {conf:.2f}', (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Convert frame to ImageTk format
                cv2image = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image=img)
                
                # Put the image in the queue
                self.image_queue.put(imgtk)

    def apply_night_vision(self, frame):
        # Convert to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Apply histogram equalization
        equalized_frame = cv2.equalizeHist(gray_frame)
        # Convert back to BGR
        night_vision_frame = cv2.cvtColor(equalized_frame, cv2.COLOR_GRAY2BGR)
        return night_vision_frame
    
    def update_gui_frame(self):
        if not self.image_queue.empty():
            imgtk = self.image_queue.get()
            self.camera_frame.imgtk = imgtk
            self.camera_frame.config(image=imgtk)
        self.root.after(10, self.update_gui_frame)

    def update_timer(self):
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        self.timer_label.config(text=current_time)
        self.root.after(1000, self.update_timer)

    def update_category_count(self):
        cursor.execute("SELECT category, COUNT(*) FROM recognitions GROUP BY category")
        data = cursor.fetchall()
        counts = "\n".join([f"{row[0]}: {row[1]}" for row in data])
        self.category_count_label.config(text=counts)
        self.root.after(60000, self.update_category_count)  # Update every minute

    def apply_filter(self):
        start_date = self.start_date_entry.get()
        end_date = self.end_date_entry.get()

        for widget in self.main_frame.winfo_children():
            widget.destroy()
        
        cursor.execute("SELECT category, COUNT(*) FROM recognitions WHERE datetime BETWEEN ? AND ? GROUP BY category", (start_date, end_date))
        data = cursor.fetchall()
        categories = [row[0] for row in data]
        counts = [row[1] for row in data]

        fig, ax = plt.subplots()
        ax.bar(categories, counts, color='blue')
        ax.set_title('Recognitions by Category')
        ax.set_xlabel('Category')
        ax.set_ylabel('Count')

        canvas = FigureCanvasTkAgg(fig, master=self.main_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(expand=True, fill='both')

    def export_report(self):
        cursor.execute("SELECT * FROM recognitions")
        data = cursor.fetchall()
        with open("recognition_report.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([desc[0] for desc in cursor.description])  # write headers
            writer.writerows(data)
        messagebox.showinfo("Export Report", "Report exported as recognition_report.csv")

    def show_dashboard(self):
        for widget in self.main_frame.winfo_children():
            widget.destroy()
        
        cursor.execute("SELECT category, COUNT(*) FROM recognitions GROUP BY category")
        data = cursor.fetchall()
        categories = [row[0] for row in data]
        counts = [row[1] for row in data]

        fig, ax = plt.subplots()
        ax.bar(categories, counts, color='blue')
        ax.set_title('Recognitions by Category')
        ax.set_xlabel('Category')
        ax.set_ylabel('Count')

        canvas = FigureCanvasTkAgg(fig, master=self.main_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(expand=True, fill='both')

        # Add three small charts to the dashboard
        fig2, ax2 = plt.subplots()
        ax2.pie(counts, labels=categories, autopct='%1.1f%%')
        ax2.set_title('Category Distribution')
        canvas2 = FigureCanvasTkAgg(fig2, master=self.main_frame)
        canvas2.draw()
        canvas2.get_tk_widget().pack(side='left', expand=True, fill='both', padx=10, pady=10)

        fig3, ax3 = plt.subplots()
        ax3.plot(categories, counts, marker='o')
        ax3.set_title('Trend Over Time')
        ax3.set_xlabel('Category')
        ax3.set_ylabel('Count')
        canvas3 = FigureCanvasTkAgg(fig3, master=self.main_frame)
        canvas3.draw()
        canvas3.get_tk_widget().pack(side='left', expand=True, fill='both', padx=10, pady=10)

        fig4, ax4 = plt.subplots()
        ax4.barh(categories, counts, color='green')
        ax4.set_title('Horizontal Bar Chart')
        ax4.set_xlabel('Count')
        ax4.set_ylabel('Category')
        canvas4 = FigureCanvasTkAgg(fig4, master=self.main_frame)
        canvas4.draw()
        canvas4.get_tk_widget().pack(side='left', expand=True, fill='both', padx=10, pady=10)

    def show_realtime_view(self):
        for widget in self.main_frame.winfo_children():
            widget.destroy()

        self.camera_frame = tk.Label(self.main_frame, bg='#34495e')
        self.camera_frame.pack(padx=20, pady=20, expand=True, fill='both')
        
        self.control_frame = tk.Frame(self.main_frame, bg='#2c3e50')
        self.control_frame.pack(side='bottom', pady=10)

        self.start_button = tk.Button(self.control_frame, text="Start Camera", command=self.start_camera, bg='#27ae60', fg='white', font=("Helvetica", 10))
        self.start_button.grid(row=0, column=0, padx=5, pady=5)
        
        self.stop_button = tk.Button(self.control_frame, text="Stop Camera", command=self.stop_camera, bg='#e74c3c', fg='white', font=("Helvetica", 10))
        self.stop_button.grid(row=0, column=1, padx=5, pady=5)

        self.start_detection_button = tk.Button(self.control_frame, text="Start Detection", command=self.start_detection, bg='#3498db', fg='white', font=("Helvetica", 10))
        self.start_detection_button.grid(row=0, column=2, padx=5, pady=5)

        self.stop_detection_button = tk.Button(self.control_frame, text="Stop Detection", command=self.stop_detection, bg='#e74c3c', fg='white', font=("Helvetica", 10))
        self.stop_detection_button.grid(row=0, column=3, padx=5, pady=5)
        
        self.capture_button = tk.Button(self.control_frame, text="Capture & Tag", command=self.capture_image, bg='#3498db', fg='white', font=("Helvetica", 10))
        self.capture_button.grid(row=0, column=4, padx=5, pady=5)
        
        self.train_button = tk.Button(self.control_frame, text="Train Model", command=self.train_model, bg='#e67e22', fg='white', font=("Helvetica", 10))
        self.train_button.grid(row=0, column=5, padx=5, pady=5)

        self.switch_model_button = tk.Button(self.control_frame, text="Switch Model", command=self.switch_model, bg='#9b59b6', fg='white', font=("Helvetica", 10))
        self.switch_model_button.grid(row=0, column=6, padx=5, pady=5)

        self.night_vision_button = tk.Button(self.control_frame, text="Toggle Night Vision", command=self.toggle_night_vision, bg='#2c3e50', fg='white', font=("Helvetica", 10))
        self.night_vision_button.grid(row=0, column=7, padx=5, pady=5)

        self.ip_entry_label = tk.Label(self.control_frame, text="IP Address:", font=("Helvetica", 10), bg='#2c3e50', fg='white')
        self.ip_entry_label.grid(row=0, column=8, padx=5, pady=5)
        self.ip_entry = tk.Entry(self.control_frame, font=("Helvetica", 10))
        self.ip_entry.grid(row=0, column=9, padx=5, pady=5)
        self.ip_entry.insert(0, "192.168.1.2:8080")

class CustomKNNClassifier:
    def __init__(self, k=1):
        self.k = k
        self.data = None
        self.labels = None

    def fit(self, data, labels):
        self.data = data
        self.labels = labels

    def predict(self, data):
        predictions = []
        for point in data:
            distances = np.linalg.norm(self.data - point, axis=1)
            nearest_neighbor = np.argmin(distances)
            predictions.append(self.labels[nearest_neighbor])
        return predictions

if __name__ == "__main__":
    root = tk.Tk()
    app = MobileCameraApp(root)
    root.mainloop()
