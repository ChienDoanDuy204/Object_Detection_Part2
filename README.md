# 🚀 Faster R-CNN Object Detection Studio & Research Platform

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-Accelerated-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-16.4-black?style=for-the-badge&logo=next.js&logoColor=white)
![React](https://img.shields.io/badge/React-19-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![TailwindCSS](https://img.shields.io/badge/TailwindCSS-v4-06B6D4?style=for-the-badge&logo=tailwindcss&logoColor=white)
![Pascal VOC](https://img.shields.io/badge/Dataset-Pascal%20VOC%202007-blue?style=for-the-badge)

<p align="center">
  <b>Hệ thống nghiên cứu và triển khai mô hình Two-Stage Object Detection (Faster R-CNN) từ mức cốt lõi bằng PyTorch, kết hợp Engine suy luận tối ưu hóa FPS và ứng dụng Web tương tác thời gian thực (FastAPI + Next.js).</b>
</p>

[Tổng quan](#-tổng-quan-dự-án) •
[Kiến trúc Mô hình](#-kiến-trúc-mô-hình-faster-r-cnn) •
[Tiền xử lý & Dataset](#-tiền-xử-lý-dữ-liệu--data-augmentation) •
[Kết quả Huấn luyện](#-kết-quả-huấn-luyện-thực-nghiệm) •
[Tính năng Web Studio](#-tính-năng-hệ-thống-web-studio) •
[Cấu trúc Thư mục](#-cấu-trúc-thư-mục-dự-án) •
[Cài đặt & Khởi chạy](#-cài-đặt--khởi-chạy) •
[Tài liệu API](#-tài-liệu-api-endpoints) •
[Sử dụng Python SDK](#-sử-dụng-thư-viện-suy-luận-python)

</div>

---

## 📌 Tổng quan Dự án

Dự án **Object_Detection_Part2** là một nền tảng hoàn chỉnh từ nghiên cứu thuật toán đến ứng dụng thực tế về bài toán **Phát hiện đối tượng 2 giai đoạn (Two-Stage Object Detection)**:

1. **Nghiên cứu Tiền đề (Selective Search)**: Khảo sát trực quan thuật toán đề xuất vùng cổ điển dựa trên phân đoạn ảnh (`notebooks/selective_search.ipynb`) để làm rõ lý do ra đời và ưu thế vượt trội của mạng đề xuất vùng nơ-ron (Region Proposal Network - RPN).
2. **Hiện thực Faster R-CNN từ Cốt lõi (From Scratch)**: Xây dựng toàn bộ kiến trúc gồm Backbone **ResNet-18** (`resnet18.a1_in1k`), mạng **RPN** với 9 anchors đa tỉ lệ, tầng **RoI Align** chuẩn xác và **Fast R-CNN Detection Head** phân loại 20 lớp đối tượng.
3. **Data Pipeline nâng cao (Image Pyramid & Dynamic Batching)**: Tự xây dựng `VOCObjectDetectionDataset` hỗ trợ **Image Pyramid Data Augmentation** (phóng to/thu nhỏ đa tỷ lệ) và hàm `collate_fn` padding động kèm GT Box mask.
4. **Huấn luyện End-to-End với PyTorch AMP**: Tối ưu hóa đồng thời 4 hàm mất mát (Multi-task Loss) với kỹ thuật Mixed Precision (`torch.amp.GradScaler`), lưu checkpoint trọng số tối ưu tại `model_weight.pth`.
5. **Engine Suy luận Hiệu năng cao (Inference Engine)**: Hỗ trợ dự đoán ảnh tĩnh, video file và live stream camera với cơ chế **Target FPS Throttling**, Batched NMS trên GPU và bảng màu trực quan hóa 20 lớp chuyên nghiệp.
6. **Ứng dụng Full-stack Hiện đại**:
   - **Backend**: FastAPI (Python) cung cấp REST API và WebSocket 2 chiều thời gian thực.
   - **Frontend**: Next.js 16 + React 19 + Tailwind CSS v4 cung cấp 3 Studio trực quan: **Photo Studio**, **Video Pipeline Studio** và **Live Camera Studio**.

---

## 🧠 Kiến trúc Mô hình Faster R-CNN

Kiến trúc Faster R-CNN được hiện thực chi tiết tại [`model/FasterRCNN.py`](model/FasterRCNN.py) bao gồm 4 khối thành phần chính:

```text
               ┌───────────────────────────────┐
               │    Input Image (B, 3, H, W)   │
               └───────────────┬───────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. Feature Extractor Backbone (ResNet-18)                   │
│    - Pretrained resnet18.a1_in1k (timm)                     │
│    - Global Pooling & FC layers removed                     │
│    - Output Feature Map: C = 512, Stride = 32 (H/32, W/32)  │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Region Proposal Network (RPN)                            │
│    - Anchor Generation: 9 anchors/cell                      │
│      Scales: [64, 128, 256] px | Ratios: [0.5, 1.0, 2.0]    │
│    - Classifier Head: Conv2d(512 -> 2 * 9, k=3, p=1)        │
│    - Regression Head: Conv2d(512 -> 4 * 9, k=3, p=1)        │
│    - Proposal Generator: Delta Decode -> Boundary Clip      │
│      -> Top-2000 Pre-NMS -> Batched NMS (IoU=0.7) -> Top-K │
└──────────────────────────────┬──────────────────────────────┘
                               │ (Top candidate proposals)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. RoI Align Layer                                          │
│    - PyTorch torchvision.ops.roi_align                      │
│    - Spatial Scale = 1/32, Output Size = 7 x 7              │
│    - Output Tensor: (K, 512, 7, 7)                          │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Fast R-CNN Detection Head                                │
│    - Flatten & FC1 (512 * 7 * 7 -> 1024) + ReLU             │
│    - Classifier Head: FC2 (1024 -> 21 classes)              │
│    - Regressor Head: FC2 (1024 -> 4 coordinates)            │
│    - Post-processing: Softmax -> Batched NMS theo từng lớp  │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
      [ Final Detections: BBoxes + Labels + Confidences ]
```

### Thông số & Cơ chế toán học

#### 1. Anchor Delta Encoding & Decoding
Tọa độ bounding box thực $(x, y, w, h)$ được chuẩn hóa tương đối so với anchor box $(x_a, y_a, w_a, h_a)$ theo công thức:

$$
t_x = \frac{x - x_a}{w_a}, \quad t_y = \frac{y - y_a}{h_a}, \quad t_w = \ln\left(\frac{w}{w_a}\right), \quad t_h = \ln\left(\frac{h}{h_a}\right)
$$

Và giải mã ngược lại để khôi phục tọa độ hộp thực tế:

$$
x = x_a + t_x \cdot w_a, \quad y = y_a + t_y \cdot h_a, \quad w = w_a \cdot e^{t_w}, \quad h = h_a \cdot e^{t_h}
$$

#### 2. Chiến lược Gán nhãn Target (Target Matching)
- **RPN Target (`build_target`)**:
  - **Positive (1)**: Anchor có $\text{IoU} \ge 0.7$ với bất kỳ GT box nào, hoặc là anchor có IoU cao nhất với một GT box cụ thể.
  - **Negative (0)**: Anchor có $\text{IoU} < 0.3$ với tất cả GT boxes.
  - **Ignore (-1)**: Anchor nằm trong khoảng $0.3 \le \text{IoU} < 0.7$ (không tham gia tính gradient).
- **RoI Target (`build_roi_targets`)**:
  - **Positive**: RoI proposal có $\text{IoU} \ge 0.5$ với GT box.
  - **Negative**: RoI proposal có $0.1 \le \text{IoU} < 0.5$ (gán nhãn background = 0).
  - Lấy mẫu cân bằng mini-batch (128 RoIs/ảnh với 25% positive, 75% negative).

#### 3. Hàm mất mát Đa tác vụ (Multi-task Loss)
Mô hình tối ưu hóa đồng thời 4 thành phần loss:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{rpn}}^{\text{cls}} + \mathcal{L}_{\text{rpn}}^{\text{reg}} + \mathcal{L}_{\text{roi}}^{\text{cls}} + \mathcal{L}_{\text{roi}}^{\text{reg}}
$$

- $\mathcal{L}_{\text{rpn}}^{\text{cls}}$: Cross Entropy Loss phân loại foreground/background cho RPN (`ignore_index=-1`).
- $\mathcal{L}_{\text{rpn}}^{\text{reg}}$: Smooth L1 Loss hồi quy tọa độ cho positive anchors của RPN.
- $\mathcal{L}_{\text{roi}}^{\text{cls}}$: Cross Entropy Loss phân loại 21 lớp (20 đối tượng + 1 nền).
- $\mathcal{L}_{\text{roi}}^{\text{reg}}$: Smooth L1 Loss hồi quy tọa độ hộp bao cho positive RoIs.

#### 4. Danh sách 20 Lớp đối tượng Pascal VOC 2007:
> `aeroplane`, `bicycle`, `bird`, `boat`, `bottle`, `bus`, `car`, `cat`, `chair`, `cow`, `diningtable`, `dog`, `horse`, `motorbike`, `person`, `pottedplant`, `sheep`, `sofa`, `train`, `tvmonitor`.

---

## 📦 Tiền xử lý Dữ liệu & Data Augmentation

Hiện thực tại [`src/orgainize_data.py`](src/orgainize_data.py):

- **Image Pyramid Augmentation (`VOCObjectDetectionDataset`)**:
  - Hỗ trợ biến đổi đa kích thước: upscale (`up` - phóng to tối đa 2.5x để phát hiện vật thể nhỏ), downscale (`down` - thu nhỏ), hoặc kết hợp (`both`) với tỷ lệ mở rộng $1.5\times$.
  - Tự động co giãn tỷ lệ tọa độ hộp bao (Ground Truth BBoxes) tương ứng với kích thước ảnh mới và cắt viền (coordinate clamping).
- **Dynamic Batch Padding (`collate_fn`)**:
  - Tự động xác định $(H_{\max}, W_{\max})$ trong batch và đệm (padding) ảnh về cùng kích thước mà không làm méo tỉ lệ khung hình.
  - Đệm danh sách GT BBoxes và GT Labels theo số lượng hộp lớn nhất trong batch kèm mặt nạ logic `mask_GT_boxes`.

---

## 📊 Kết quả Huấn luyện Thực nghiệm

Quá trình huấn luyện End-to-End được thực hiện với lớp `TrainModel` (`model/FasterRCNN.py`) trên tập **Pascal VOC 2007 trainval** thông qua sổ tay [`notebooks/TrainFasterRCNN.ipynb`](notebooks/TrainFasterRCNN.ipynb):

### Cấu hình Huấn luyện
- **Phần cứng**: NVIDIA GPU hỗ trợ CUDA (8 GB VRAM).
- **Optimizer**: Adam với Differential Learning Rate:
  - Backbone ResNet-18: $\text{lr} = 0.0001$, weight decay = $0.0005$.
  - RPN & Fast R-CNN Heads: $\text{lr} = 0.001$, weight decay = $0.0005$.
- **Kỹ thuật tăng tốc**: PyTorch Automatic Mixed Precision (`torch.autocast` + `GradScaler`).
- **Batch size**: 16 | **Số epoch**: 10 | **Thời gian huấn luyện**: ~2 giờ 10 phút.

### Nhật ký (LOG) Giảm Loss & Độ chính xác qua 10 Epochs:

| Epoch | Total Loss | RPN Cls Loss | RPN Reg Loss | RPN Accuracy | Fast R-CNN Cls Loss | Fast R-CNN Reg Loss | Fast R-CNN Accuracy |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 0.8143 | 0.0149 | 0.2175 | 99.81% | 0.4151 | 0.1667 | 88.94% |
| **2** | 0.5053 | 0.0078 | 0.1343 | 99.88% | 0.3151 | 0.0482 | 89.70% |
| **3** | 0.4482 | 0.0070 | 0.1092 | 99.88% | 0.2865 | 0.0454 | 90.35% |
| **4** | 0.4138 | 0.0066 | 0.0963 | 99.88% | 0.2673 | 0.0435 | 90.82% |
| **5** | 0.3876 | 0.0063 | 0.0863 | 99.88% | 0.2530 | 0.0419 | 91.16% |
| **6** | 0.3637 | 0.0062 | 0.0783 | 99.88% | 0.2387 | 0.0406 | 91.56% |
| **7** | 0.3415 | 0.0060 | 0.0713 | 99.88% | 0.2247 | 0.0395 | 91.99% |
| **8** | 0.3244 | 0.0059 | 0.0663 | 99.88% | 0.2137 | 0.0386 | 92.27% |
| **9** | 0.3089 | 0.0058 | 0.0622 | 99.88% | 0.2034 | 0.0375 | 92.56% |
| **10** | **0.2939** | **0.0057** | **0.0580** | **99.88%** | **0.1935** | **0.0367** | **92.84%** |

Trọng số tối ưu đã được lưu trữ sẵn tại: [`model_weight.pth`](model_weight.pth) (~251 MB).

---

## 💻 Tính năng Hệ thống Web Studio

Ứng dụng web được xây dựng với giao diện Dark Mode hiện đại, hiệu ứng kính mờ (Glassmorphism) và 3 Studio chuyên biệt:

| Studio | Mô tả & Chức năng chi tiết |
| :--- | :--- |
| 🖼️ **Photo Studio** | • Tải ảnh lên (kéo thả hoặc chọn file) hoặc chọn nhanh **6 ảnh mẫu benchmark Pascal VOC**.<br>• Điều chỉnh thanh trượt **Confidence Threshold** ($0.01 - 1.0$) và **NMS IoU Threshold** ($0.01 - 1.0$).<br>• **Target Inspector**: Click trực tiếp vào bất kỳ bounding box nào trên ảnh để xem tên lớp, độ tin cậy %, tọa độ $[x_1, y_1, x_2, y_2]$ và kích thước $W \times H$.<br>• **Class Filter Pills**: Lọc và tô sáng riêng biệt từng lớp đối tượng kèm số lượng phát hiện.<br>• Chế độ chuyển đổi nhanh: Ảnh gốc vs Ảnh đã gắn nhãn (Split/Toggle). |
| 🎬 **Video Pipeline Studio** | • Tải lên video MP4/WebM hoặc chọn **Sample Video** mẫu (`sample_demo.mp4`).<br>• Tùy chỉnh **Target FPS** (1 đến 30 FPS) giúp kiểm soát lưu lượng xử lý.<br>• Trình phát video tích hợp chuyển đổi mượt mà giữa video gốc và video annotated (mã hóa chuẩn WebM VP80 tương thích trình duyệt).<br>• Bảng thống kê hiệu năng chi tiết: Source FPS, Effective Target FPS, số khung hình đã xử lý, tổng thời gian, processing FPS trung bình, biểu đồ phân bố lớp đối tượng và nút tải video kết quả. |
| 📹 **Live Camera Studio** | • Nhận diện trực tiếp qua Webcam/Camera ngoài với kênh **WebSocket 2 chiều (`/api/ws/webcam`)**.<br>• Tùy chọn thiết bị camera (đa webcam), lật gương (Mirror mode) và điều chỉnh Target FPS.<br>• HUD hiển thị thời gian thực: FPS thực tế (Actual FPS) và độ trễ suy luận (Latency ms).<br>• **Snapshot Capture**: Chụp nhanh khung hình hiện tại kèm bounding boxes và tự động chuyển sang Photo Studio để phân tích chuyên sâu. |

---

## 📂 Cấu trúc Thư mục Dự án

```text
Object_Detection_Part2/
├── backend/                       # Backend FastAPI
│   └── main.py                    # RESTful endpoints & WebSocket webcam streaming
├── backend_storage/               # Thư mục lưu trữ tạm thời cho backend
│   ├── processed/                 # Video WebM đã annotate phục vụ stream/tải về
│   └── uploads/                   # Video đầu vào được tải lên
├── data/                          # Dữ liệu phục vụ nghiên cứu & demo
│   ├── VOCdevkit/                 # Bộ dữ liệu Pascal VOC 2007 (Annotations, JPEGImages)
│   ├── sample_demo.mp4            # Video mẫu đầu vào
│   └── sample_demo_annotated.mp4  # Video mẫu sau khi được mô hình nhận diện
├── frontend/                      # Web Studio (Next.js 16 + React 19 + Tailwind CSS v4)
│   ├── src/app/
│   │   ├── globals.css            # Design system, glassmorphism & responsive styles
│   │   ├── layout.js              # Root layout & Metadata
│   │   └── page.js                # Điều hướng các Studio (Photo, Video, Camera)
│   ├── src/components/
│   │   ├── CameraStudio.js        # Giao diện Webcam thời gian thực qua WebSocket
│   │   ├── Navbar.js              # Thanh điều hướng & badge trạng thái phần cứng CUDA
│   │   ├── PhotoStudio.js         # Giao diện nhận diện ảnh tĩnh & Target Inspector
│   │   └── VideoStudio.js         # Giao diện xử lý video theo Target FPS
│   ├── src/utils/
│   │   └── api.js                 # Cấu hình API Base URL & WebSocket URL
│   └── package.json               # Dependencies Frontend
├── model/                         # Định nghĩa kiến trúc Faster R-CNN cốt lõi
│   └── FasterRCNN.py              # Feature_Extractor, FasterRCNN_model & TrainModel
├── notebooks/                     # Sổ tay Jupyter nghiên cứu
│   ├── selective_search.ipynb     # Khảo sát thuật toán đề xuất vùng Selective Search cổ điển
│   └── TrainFasterRCNN.ipynb      # Pipeline huấn luyện Faster R-CNN với PyTorch AMP
├── src/                           # Các module tiện ích
│   ├── download_data.py           # Script tự động tải bộ dữ liệu Pascal VOC 2007
│   ├── inference.py               # Engine FasterRCNNInferencer & hàm model_inference
│   └── orgainize_data.py          # VOCObjectDetectionDataset & collate_fn đệm động
├── model_weight.pth               # Checkpoint trọng số đã huấn luyện (~251 MB)
├── requirements.txt               # Thư viện Python phụ thuộc
├── run_backend.sh                 # Script khởi chạy riêng FastAPI Backend (:8000)
├── run_frontend.sh                # Script khởi chạy riêng Next.js Frontend (:3000)
├── start_all.sh                   # Script khởi chạy 1-Click đồng thời Backend & Frontend
└── README.md                      # Tài liệu dự án
```

---

## ⚙️ Cài đặt & Khởi chạy

### 1. Yêu cầu Tiên quyết
- **Hệ điều hành**: Linux (Ubuntu 20.04/22.04), macOS, hoặc Windows (WSL2).
- **Python**: `>= 3.10`
- **Node.js**: `>= 18.0.0` & `npm`
- **GPU (Khuyến nghị)**: NVIDIA GPU có CUDA để đạt hiệu năng xử lý video/webcam tối ưu (hệ thống tự động fallback về CPU nếu không có GPU).

### 2. Cài đặt Môi trường

```bash
# 1. Clone repository
git clone https://github.com/ChienDoanDuy204/Object_Detection_Part2.git
cd Object_Detection_Part2

# 2. Cài đặt Python dependencies
pip install -r requirements.txt

# 3. Cài đặt Frontend dependencies
cd frontend
npm install
cd ..
```
> File trọng số [`model_weight.pth`](model_weight.pth) (~251 MB) đã được huấn luyện sẵn và lưu ở trên huggingface. Nếu cần tải lại tập dữ liệu Pascal VOC 2007:
> ```bash
> python -m src.download_data
> ```

> Tải trọng số của model tại: https://huggingface.co/doanduychien204/Faster_RCNN
>```bash
> wget -O model_weight.pth https://huggingface.co/doanduychien204/Faster_RCNN/blob/main/model_weight.pth
>```
---

### 3. Khởi chạy Ứng dụng

#### Cách 1: Khởi chạy 1-Click tự động (Khuyên dùng) 🌟

```bash
chmod +x start_all.sh run_backend.sh run_frontend.sh
./start_all.sh
```

Script sẽ tự động kiểm tra và giải phóng các port `8000` và `3000`, sau đó khởi chạy song song Backend và Frontend. Nhấn `Ctrl + C` để dừng đồng thời cả 2 dịch vụ an toàn.

- **Frontend Studio**: [http://localhost:3000](http://localhost:3000)
- **Backend API Docs (Swagger UI)**: [http://localhost:8000/docs](http://localhost:8000/docs)

#### Cách 2: Khởi chạy Thủ công Từng Dịch vụ

```bash
# Terminal 1 - Backend FastAPI (:8000):
./run_backend.sh
# Hoặc: python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000

# Terminal 2 - Frontend Next.js (:3000):
./run_frontend.sh
# Hoặc: cd frontend && npm run dev -- -p 3000
```

---

## 📡 Tài liệu API Endpoints

Backend FastAPI hỗ trợ đầy đủ các chuẩn giao tiếp RESTful và WebSocket:

### REST API

| Phương thức | Đường dẫn | Chức năng | Tham số / Request Body | Response chính |
| :---: | :--- | :--- | :--- | :--- |
| `GET` | `/api/info` | Thông tin mô hình, thiết bị (CUDA/CPU), danh sách classes & mã màu HEX | Không | `{ status, model_name, num_classes, classes, colors, device, cuda_available }` |
| `GET` | `/api/samples` | Danh sách 6 ảnh mẫu benchmark Pascal VOC | Không | `{ samples: [{ id, title, category }, ...] }` |
| `GET` | `/api/sample/{filename}` | Lấy file ảnh mẫu benchmark từ tập dữ liệu | `filename: string` | File JPEG trực tiếp |
| `GET` | `/api/video/sample` | Lấy video demo mẫu | Không | File MP4 trực tiếp |
| `POST` | `/api/predict/image` | Nhận diện đối tượng trên ảnh tải lên | Multipart Form:<br>• `file`: File ảnh (JPG/PNG)<br>• `conf_threshold`: float (mặc định 0.35)<br>• `nms_threshold`: float (mặc định 0.35) | `{ filename, width, height, inference_time_ms, total_detections, counts, detections, annotated_image }` |
| `POST` | `/api/predict/video` | Xử lý video theo Target FPS | Multipart Form:<br>• `file`: File video (MP4/WebM)<br>• `target_fps`: float (1.0 - 30.0)<br>• `conf_threshold`: float<br>• `nms_threshold`: float | `{ video_id, source_fps, target_fps, total_source_frames, processed_frames, total_elapsed_s, avg_processing_fps, video_url, class_summary, frame_stats }` |
| `GET` | `/api/video/{filename}` | Stream hoặc tải video WebM đã gắn nhãn | `filename: string` | Stream `video/webm` |

### WebSocket API

| Giao thức | Đường dẫn | Chức năng | Chi tiết Payload |
| :---: | :--- | :--- | :--- |
| `WS` | `/api/ws/webcam` | Kênh stream webcam tương tác hai chiều thời gian thực | **Client gửi**: JSON `{ image: "<base64>", target_fps: 15, conf_threshold: 0.35, nms_threshold: 0.35, return_annotated: true }`<br>**Server trả**: JSON `{ detections: [...], inference_time_ms: 18.5, target_fps: 15, actual_fps: 14.8, width: 640, height: 480, annotated_image: "<base64>" }` |

---

## 🐍 Sử dụng Thư viện Suy luận Python

Bạn có thể tích hợp trực tiếp module suy luận [`src/inference.py`](src/inference.py) vào các dự án Python độc lập thông qua hàm `model_inference`:

```python
from src.inference import model_inference

# 1. Dự đoán trên ảnh tĩnh
image_result = model_inference(
    mode="picture",
    source="data/VOCdevkit/VOC2007/JPEGImages/000200.jpg",
    conf_threshold=0.4,
    nms_threshold=0.35,
    weights_path="model_weight.pth"
)
print("Số lượng đối tượng:", len(image_result["detections"]))
print("Độ trễ suy luận:", image_result["inference_time_ms"], "ms")

# 2. Xử lý video file với Target FPS
video_result = model_inference(
    mode="video",
    source="data/sample_demo.mp4",
    fps=15.0,  # Xử lý ở tốc độ 15 khung hình/giây
    output_path="backend_storage/processed/demo_result.mp4",
    conf_threshold=0.35,
    weights_path="model_weight.pth"
)
print(f"Đã xử lý {video_result['processed_frames']} frames với tốc độ trung bình {video_result['avg_processing_fps']} FPS")

# 3. Stream webcam trực tiếp
for annotated_frame, pred, fps in model_inference(mode="webcam", source=0, fps=15.0):
    # annotated_frame: Ảnh OpenCV BGR có sẵn bounding boxes và watermark
    # pred: Dictionary chứa danh sách detections và latency
    # fps: FPS đo lường thực tế
    print(f"Live FPS: {fps:.1f} | Detections: {len(pred['detections'])}")
    break
```

---

## 🔬 Sổ tay thử nghiệm (Jupyter Notebooks)

1. **Selective Search Exploration**:
   - File: [`notebooks/selective_search.ipynb`](notebooks/selective_search.ipynb)
   - Trực quan hóa thuật toán Selective Search cổ điển dựa trên phân đoạn ảnh bằng OpenCV (`cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()`) trên ảnh thử nghiệm, phân tích lý do thuật toán tiêu tốn từ 2 đến 5 giây cho mỗi ảnh và động lực phát triển RPN.
2. **Train Faster R-CNN From Scratch**:
   - File: [`notebooks/TrainFasterRCNN.ipynb`](notebooks/TrainFasterRCNN.ipynb)
   - Pipeline hoàn chỉnh: Phân tích cấu trúc thư mục VOC, tạo từ điển ánh xạ 20 nhãn, khởi tạo `VOCObjectDetectionDataset` với chuẩn hóa ImageNet và Image Pyramid, nạp dữ liệu bằng DataLoader kết hợp `collate_fn`, thiết lập mô hình `FasterRCNN_model` và thực thi vòng lặp huấn luyện `TrainModel` trong 10 epochs với Mixed Precision.

---

## 📜 Tác giả & Giấy phép

- **Tác giả**: Chien Doan ([@ChienDoanDuy204](https://github.com/ChienDoanDuy204))
- **Mục tiêu**: Nghiên cứu, học tập chuyên sâu và triển khai bài toán Computer Vision / Object Detection thực tế.
- Đóng góp và phản hồi luôn được hoan nghênh! ⭐ Hãy cho dự án 1 sao trên GitHub nếu bạn thấy hữu ích!
