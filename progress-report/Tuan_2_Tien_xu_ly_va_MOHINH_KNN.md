# BÁO CÁO TIẾN ĐỘ - TUẦN 2
**Đề tài:** Nhận dạng chữ số viết tay MNIST bằng mô hình KNN  
**Sinh viên thực hiện:** Trần Vĩnh Phúc (MSSV: 170124831)

## Nội dung công việc đã hoàn thành
- [x] Xây dựng các hàm tiền xử lý ảnh với OpenCV và SciPy:
  - Cắt khung ảnh chứa nét vẽ (Bounding Box Crop).
  - Đưa về khung vuông có padding lề chuẩn 28x28.
  - Dịch chuyển căn giữa ảnh theo trọng tâm (Center of Mass).
  - Làm mịn nét vẽ bằng Gaussian Blur.
- [x] Huấn luyện thành công mô hình KNN (`k=3`, `weights='distance'`) trên tập dữ liệu đã chuẩn hóa.

## Đánh giá tuần 2
- Thuật toán hoạt động ổn định, tốc độ huấn luyện nhanh.
