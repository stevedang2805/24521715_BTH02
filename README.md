# Bài thực hành 2: Mạng neural tích chập

MSSV: 24521715

- Bài 1: LeNet trên MNIST (`assignment-1.py`)
- Bài 2: GoogLeNet trên VinaFood21 (`assignment-2.py`, `models/googlenet.py`)
- Bài 3: ResNet-18 trên VinaFood21 (`assignment-3.py`, `models/resnet18.py`)
- Bài 4: Fine-tune ResNet50 pretrained `microsoft/resnet-50` từ HuggingFace trên VinaFood21 (`assignment-4.py`)

Tất cả dùng Adam, đánh giá bằng accuracy, precision, recall, F1 (macro). Log chạy đầy đủ nằm trong `run_colab.ipynb`.

## Chạy

```bash
pip install -r requirements.txt
python assignment-1.py
python assignment-2.py --data_dir ./dataset/VinaFood21
python assignment-3.py --data_dir ./dataset/VinaFood21
python assignment-4.py --data_dir ./dataset/VinaFood21
```

`dataset/VinaFood21` gồm 2 thư mục `train/` và `test/`, mỗi thư mục có 21 thư mục con theo tên món ăn.

Với VinaFood21, mình tách 10% tập train (chia đều theo lớp) làm tập dev để chọn checkpoint, tập test chỉ dùng để đánh giá lần cuối. Kết quả: train 9049 ảnh, dev 995 ảnh, test 6682 ảnh. Ảnh được resize về 224x224, chuẩn hoá theo mean/std của ImageNet, khi train có lật ngang ngẫu nhiên.

## Kết quả trên tập test

| Bài | Mô hình | Epoch | lr | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| 1 | LeNet (MNIST) | 10 | 1e-3 | 0.9840 | 0.9841 | 0.9837 | 0.9839 |
| 2 | GoogLeNet | 30 (tốt nhất: 26) | 1e-4 | 0.3957 | 0.3814 | 0.3519 | 0.3443 |
| 3 | ResNet-18 | 30 (tốt nhất: 30) | 1e-3 | 0.6152 | 0.6431 | 0.6001 | 0.6057 |
| 4 | ResNet50 pretrained | 10 (tốt nhất: 10) | 1e-4 | 0.8493 | 0.8481 | 0.8535 | 0.8497 |

Batch size 32, GPU L4. Precision, recall, F1 là macro trên 21 lớp. Kết quả theo từng lớp được lưu ở `checkpoints/assignment-X/classification_report.txt` khi chạy.

## Nhận xét

- GoogLeNet chưa hội tụ: đến epoch 30 train loss vẫn còn 1.58 và vẫn đang giảm. Mô hình không có BatchNorm, train từ đầu với lr = 1e-4 trên khoảng 9 nghìn ảnh nên học rất chậm. Nếu train thêm epoch hoặc tăng lr thì kết quả có thể còn tăng.
- ResNet-18 học nhanh hơn GoogLeNet nhiều nhờ BatchNorm và residual connection, nhưng bị overfit: train loss xuống 0.16 trong khi F1 trên dev chỉ khoảng 0.67.
- ResNet50 pretrained tốt hơn hẳn: chỉ sau 2 epoch F1 trên dev đã khoảng 0.69, sau 10 epoch đạt 0.85 trên test. Đặc trưng học từ ImageNet giúp ích nhiều khi dữ liệu ít.
- `ca-ri` là lớp khó nhất với ResNet-18 (F1 0.30) và ResNet50 (F1 0.59). Với GoogLeNet lớp này cũng chỉ đạt 0.09.

## Ghi chú cài đặt

- GoogLeNet làm theo bảng tham số trong bài báo: conv đầu 7x7 stride 2 padding 3, tất cả MaxPool dùng `ceil_mode=True`, có dropout 0.4 trước lớp linear, không dùng auxiliary classifier.
- ResNet-18: theo đề, giữa các nhóm residual block có MaxPool (kernel 3, stride 2, padding 0). Lớp MaxPool này đã giảm kích thước nên các residual block dùng stride 1. Kích thước feature map: 56 → 27 → 13 → 6.
- ResNet50: thay lớp phân loại 1000 lớp bằng lớp 21 lớp, fine-tune toàn bộ mạng.
