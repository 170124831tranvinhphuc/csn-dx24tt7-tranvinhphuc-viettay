import tkinter as tk
import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import center_of_mass
from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier

# 1. LOAD MNIST DATASET
print("Loading MNIST dataset... Please wait.")
mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
X = mnist.data / 255.0
y = mnist.target.astype(int)

# Use 20,000 samples for better coverage
X_train, y_train = X[:20000], y[:20000]

print("Training KNN Classifier...")
model = KNeighborsClassifier(n_neighbors=3, weights="distance")
model.fit(X_train, y_train)
print("Model Ready!")


# 2. HELPER: CENTER OF MASS ALIGNMENT
def center_digit(img):
    cy, cx = center_of_mass(img)
    if np.isnan(cy) or np.isnan(cx):
        return img
    rows, cols = img.shape
    shiftx = np.round(cols / 2.0 - cx)
    shifty = np.round(rows / 2.0 - cy)

    M = np.float32([[1, 0, shiftx], [0, 1, shifty]])
    centered_img = cv2.warpAffine(img, M, (cols, rows))
    return centered_img


# 3. GUI APPLICATION
class DigitRecognizerApp:

    def __init__(self, root):
        self.root = root
        self.root.title("MNIST Digit Recognizer - Enhanced KNN")

        self.canvas = tk.Canvas(root, width=280, height=280, bg="white")
        self.canvas.pack(pady=10)

        self.image = Image.new("L", (280, 280), 255)
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.draw_on_canvas)

        self.btn_predict = tk.Button(
            root,
            text="Predict",
            command=self.predict_digit,
            font=("Helvetica", 12, "bold"),
        )
        self.btn_predict.pack(side=tk.LEFT, padx=20, pady=10)

        self.btn_clear = tk.Button(
            root,
            text="Clear",
            command=self.clear_canvas,
            font=("Helvetica", 12, "bold"),
        )
        self.btn_clear.pack(side=tk.RIGHT, padx=20, pady=10)

        self.lbl_result = tk.Label(
            root, text="Draw a digit (0-9)", font=("Helvetica", 16)
        )
        self.lbl_result.pack(pady=15)

    def draw_on_canvas(self, event):
        x, y = event.x, event.y
        r = 10
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="black")
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill=0)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.lbl_result.config(text="Draw a digit (0-9)")

    def predict_digit(self):
        img_np = np.array(self.image)
        img_inverted = cv2.bitwise_not(img_np)

        coords = cv2.findNonZero(img_inverted)
        if coords is None:
            self.lbl_result.config(text="Please draw a digit first!")
            return

        # Crop bounding box
        x, y, w, h = cv2.boundingRect(coords)
        crop = img_inverted[y : y + h, x : x + w]

        # Make square frame with margin
        max_side = max(w, h) + 60
        square_img = np.zeros((max_side, max_side), dtype=np.uint8)
        offset_x = (max_side - w) // 2
        offset_y = (max_side - h) // 2
        square_img[offset_y : offset_y + h, offset_x : offset_x + w] = crop

        # Resize to 20x20 first, then pad to 28x28 (Standard MNIST processing)
        resized_20x20 = cv2.resize(
            square_img, (20, 20), interpolation=cv2.INTER_AREA
        )
        mnist_img = np.pad(resized_20x20, ((4, 4), (4, 4)), "constant")

        # Center using Center of Mass
        mnist_centered = center_digit(mnist_img)

        # Apply slight Gaussian Blur to match smooth MNIST lines
        mnist_smoothed = cv2.GaussianBlur(mnist_centered, (3, 3), 0)

        normalized = mnist_smoothed / 255.0
        input_data = normalized.reshape(1, -1)

        pred = model.predict(input_data)[0]
        self.lbl_result.config(text=f"Prediction: {pred}")


if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()
