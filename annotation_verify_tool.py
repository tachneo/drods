import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk, ImageDraw
import os

class AnnotationVerifier:
    def __init__(self, root):
        self.root = root
        self.root.title("Annotation Verifier Tool")
        self.root.geometry("1200x800")

        self.create_widgets()
        self.annotations = []
        self.image_index = 0

    def create_widgets(self):
        self.frame = ttk.Frame(self.root, padding=10)
        self.frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.frame, bg="white")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.sidebar = ttk.Frame(self.frame, padding=10)
        self.sidebar.pack(side=tk.RIGHT, fill=tk.Y)

        self.load_button = ttk.Button(self.sidebar, text="Load Images", command=self.load_images)
        self.load_button.pack(fill=tk.X, pady=5)

        self.prev_button = ttk.Button(self.sidebar, text="Previous", command=self.show_previous_image)
        self.prev_button.pack(fill=tk.X, pady=5)

        self.next_button = ttk.Button(self.sidebar, text="Next", command=self.show_next_image)
        self.next_button.pack(fill=tk.X, pady=5)

        self.status = tk.StringVar()
        self.status.set("Welcome to the Annotation Verifier Tool")
        self.statusbar = ttk.Label(self.root, textvariable=self.status, relief=tk.SUNKEN, anchor=tk.W)
        self.statusbar.pack(side=tk.BOTTOM, fill=tk.X)

    def load_images(self):
        file_paths = filedialog.askopenfilenames(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if file_paths:
            self.image_paths = list(file_paths)
            self.image_index = 0
            self.display_image(self.image_paths[self.image_index])

    def display_image(self, file_path):
        try:
            self.img = Image.open(file_path)
            self.scale_factor = min(self.canvas.winfo_width() / self.img.width, self.canvas.winfo_height() / self.img.height)
            self.img_resized = self.img.resize((int(self.img.width * self.scale_factor), int(self.img.height * self.scale_factor)), Image.LANCZOS)
            self.img_tk = ImageTk.PhotoImage(self.img_resized)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img_tk)
            self.file_path = file_path
            self.load_annotations()
            self.status.set(f"Loaded image: {os.path.basename(file_path)}")
        except Exception as e:
            self.status.set(f"Error displaying image: {e}")

    def load_annotations(self):
        try:
            self.annotations = []
            annotation_file = self.file_path.replace(".jpg", ".txt").replace(".jpeg", ".txt").replace(".png", ".txt")
            if os.path.exists(annotation_file):
                with open(annotation_file, "r") as f:
                    for line in f.readlines():
                        label, x_center, y_center, width, height = line.strip().split()
                        x_center, y_center, width, height = map(float, (x_center, y_center, width, height))
                        bbox = (x_center * self.img.width - width * self.img.width / 2,
                                y_center * self.img.height - height * self.img.height / 2,
                                x_center * self.img.width + width * self.img.width / 2,
                                y_center * self.img.height + height * self.img.height / 2)
                        self.annotations.append((label, bbox))
            self.update_image()
        except Exception as e:
            self.status.set(f"Error loading annotations: {e}")

    def update_image(self):
        try:
            img_copy = self.img_resized.copy()
            draw = ImageDraw.Draw(img_copy)
            for label, bbox in self.annotations:
                draw.rectangle(bbox, outline="blue")
                draw.text((bbox[0], bbox[1]), label, fill="blue")
            self.img_tk = ImageTk.PhotoImage(img_copy)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img_tk)
        except Exception as e:
            self.status.set(f"Error updating image: {e}")

    def show_previous_image(self):
        if self.image_index > 0:
            self.image_index -= 1
            self.display_image(self.image_paths[self.image_index])

    def show_next_image(self):
        if self.image_index < len(self.image_paths) - 1:
            self.image_index += 1
            self.display_image(self.image_paths[self.image_index])

if __name__ == "__main__":
    root = tk.Tk()
    app = AnnotationVerifier(root)
    root.mainloop()
