import os
import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# ==============================================================================
# BƯỚC 1: HUẤN LUYỆN MÔ HÌNH KNN VỚI TẬP DỮ LIỆU MNIST
# ==============================================================================
print("1. Đang tải tập dữ liệu MNIST (vui lòng chờ trong giây lát)...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)

X = mnist.data / 255.0  # Chuẩn hóa giá trị pixel về dải [0, 1]
y = mnist.target.astype(int)

# Chia tập dữ liệu (Lấy 15,000 mẫu huấn luyện để tối ưu tốc độ dự đoán thời gian thực)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_sub = X_train[:15000]
y_train_sub = y_train[:15000]
X_test_sub = X_test[:2000]
y_test_sub = y_test[:2000]

print("2. Đang huấn luyện mô hình KNN (K = 3)...")
knn = KNeighborsClassifier(n_neighbors=3, metric='minkowski', p=2)
knn.fit(X_train_sub, y_train_sub)

# Đánh giá nhanh độ chính xác
y_pred = knn.predict(X_test_sub)
acc = accuracy_score(y_test_sub, y_pred)
print(f"--> Khởi tạo hoàn tất! Độ chính xác mô hình trên tập kiểm thử: {acc * 100:.2f}%\n")


# ==============================================================================
# BƯỚC 2: HÀM TIỀN XỬ LÝ ẢNH VẼ TAY (CĂN GIỮA & RESIZE VỀ 28x28)
# ==============================================================================
def preprocess_image(pil_image):
    """
    Chuyển ảnh vẽ từ Canvas về định dạng 28x28 pixels chuẩn MNIST (Chữ trắng nền đen, căn giữa)
    """
    # Chuyển ảnh PIL sang định dạng OpenCV (Grayscale)
    img = np.array(pil_image.convert('L'))

    # Tìm hộp bao (bounding box) của nét vẽ chữ số
    coords = cv2.findNonZero(img)
    if coords is None:
        # Nếu bảng vẽ trống
        return np.zeros((1, 784))

    x, y, w, h = cv2.boundingRect(coords)
    crop = img[y:y+h, x:x+w]

    # Tạo khung hình vuông nền đen để chứa chữ số (giữ nguyên tỷ lệ)
    max_side = max(w, h) + 20  # Thêm padding xung quanh
    square_img = np.zeros((max_side, max_side), dtype=np.uint8)

    # Đặt nét chữ vào giữa khung vuông
    offset_x = (max_side - w) // 2
    offset_y = (max_side - h) // 2
    square_img[offset_y:offset_y+h, offset_x:offset_x+w] = crop

    # Resize về kích thước chuẩn 28x28 pixels
    resized = cv2.resize(square_img, (28, 28), interpolation=cv2.INTER_AREA)

    # Chuẩn hóa dải giá trị pixel [0, 255] -> [0, 1]
    normalized = resized / 255.0

    # Duỗi phẳng ma trận (28, 28) thành vector (1, 784)
    return normalized.reshape(1, 784)


# ==============================================================================
# BƯỚC 3: XÂY DỰNG GIAO DIỆN ĐỒ HỌA (GUI) VỚI TKINTER
# ==============================================================================
class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Đồ Án CSN - Nhận Dạng Chữ Số Viết Tay (KNN)")
        self.root.geometry("480 x 580")
        self.root.resizable(False, False)

        # Tiêu đề ứng dụng
        self.title_label = tk.Label(
            root, text="NHẬN DẠNG CHỮ SỐ VIẾT TAY", 
            font=("Arial", 16, "bold"), fg="#1E3A8A"
        )
        self.title_label.pack(pady=10)

        self.sub_label = tk.Label(
            root, text="Vẽ một chữ số (0-9) vào ô bên dưới:", 
            font=("Arial", 11)
        )
        self.sub_label.pack(pady=2)

        # Tạo Canvas vẽ tay (Kích thước 280x280)
        self.canvas_size = 280
        self.canvas = tk.Canvas(
            root, width=self.canvas_size, height=self.canvas_size, 
            bg="black", cursor="cross"
        )
        self.canvas.pack(pady=10)

        # Tạo đối tượng PIL Image để lưu nét vẽ song song với Canvas
        self.image_pil = Image.new("L", (self.canvas_size, self.canvas_size), 0)
        self.draw = ImageDraw.Draw(self.image_pil)

        # Bắt sự kiện di chuyển chuột để vẽ
        self.canvas.bind("<B1-Motion>", self.paint)

        # Khung chứa nút bấm
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)

        self.btn_predict = tk.Button(
            btn_frame, text="Nhận Dạng", font=("Arial", 11, "bold"), 
            bg="#22C55E", fg="white", width=12, height=1, command=self.predict_digit
        )
        self.btn_predict.grid(row=0, column=0, padx=10)

        self.btn_clear = tk.Button(
            btn_frame, text="Xóa Bảng", font=("Arial", 11, "bold"), 
            bg="#EF4444", fg="white", width=12, height=1, command=self.clear_canvas
        )
        self.btn_clear.grid(row=0, column=1, padx=10)

        # Khung hiển thị kết quả
        self.result_label = tk.Label(
            root, text="Kết quả: --", 
            font=("Arial", 18, "bold"), fg="#0F172A"
        )
        self.result_label.pack(pady=10)

        self.proba_label = tk.Label(
            root, text="Độ tin cậy: --%", 
            font=("Arial", 11, "italic"), fg="#475569"
        )
        self.proba_label.pack()

    def paint(self, event):
        """Vẽ nét chữ lên Canvas và lưu vào PIL Image"""
        r = 10  # Bán kính độ dày nét vẽ
        x1, y1 = (event.x - r), (event.y - r)
        x2, y2 = (event.x + r), (event.y + r)

        # Vẽ lên giao diện Tkinter
        self.canvas.create_oval(x1, y1, x2, y2, fill="white", outline="white")
        # Vẽ vào bộ nhớ PIL Image
        self.draw.ellipse([x1, y1, x2, y2], fill=255)

    def clear_canvas(self):
        """Xóa sạch bảng vẽ để vẽ chữ số mới"""
        self.canvas.delete("all")
        self.image_pil = Image.new("L", (self.canvas_size, self.canvas_size), 0)
        self.draw = ImageDraw.Draw(self.image_pil)
        self.result_label.config(text="Kết quả: --", fg="#0F172A")
        self.proba_label.config(text="Độ tin cậy: --%")

    def predict_digit(self):
        """Tiền xử lý và đưa nét vẽ vào KNN dự đoán"""
        input_vector = preprocess_image(self.image_pil)

        # Kiểm tra nếu bảng vẽ bị trống
        if np.sum(input_vector) == 0:
            self.result_label.config(text="Hãy vẽ chữ số!", fg="#DC2626")
            return

        # Dự đoán chữ số và xác suất
        prediction = knn.predict(input_vector)[0]
        probabilities = knn.predict_proba(input_vector)[0]
        confidence = probabilities[prediction] * 100

        # Cập nhật kết quả lên giao diện
        self.result_label.config(text=f"Kết quả: {prediction}", fg="#16A34A")
        self.proba_label.config(text=f"Độ tin cậy: {confidence:.1f}%")


# ==============================================================================
# BƯỚC 4: CHẠY ỨNG DỤNG
# ==============================================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()
