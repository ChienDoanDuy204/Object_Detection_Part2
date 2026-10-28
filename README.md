# 🚀 Faster R-CNN Object Detection Studio & Research Platform

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-Accelerated-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-16-black?style=for-the-badge&logo=next.js&logoColor=white)
![Pascal VOC](https://img.shields.io/badge/Dataset-Pascal%20VOC%202007-blue?style=for-the-badge)

<p align="center">
  <b>Nghiên cứu & Triển khai toàn diện bài toán Object Detection: Từ phương pháp cổ điển (Selective Search), kiến trúc Mạng đề xuất vùng (Region Proposal Network - RPN) đến mô hình 2 giai đoạn Faster R-CNN hoàn chỉnh, kết hợp ứng dụng Web AI tương tác thời gian thực.</b>
</p>

[Tổng quan](#-tổng-quan-dự-án) •
[Kiến trúc mô hình](#-kiến-trúc-mô-hình-faster-r-cnn) •
[Tính năng](#-tính-năng-nổi-bật) •
[Cấu trúc thư mục](#-cấu-trúc-thư-mục-dự-án) •
[Cài đặt](#-hướng-dẫn-cài-đặt-chi-tiết) •
[Khởi chạy](#-hướng-dẫn-khởi-chạy-hệ-thống) •
[Tài liệu API](#-tài-liệu-api-endpoints) •
[Huấn luyện & Nghiên cứu](#-huấn-luyện-mô-hình--nghiên-cứu)

</div>

---

## 📌 Tổng quan Dự án

Dự án **Object_Detection_Part2** tập trung vào nghiên cứu chuyên sâu và xây dựng hệ thống phát hiện đối tượng 2 giai đoạn (Two-Stage Object Detection) từ mức cốt lõi (from scratch) bằng PyTorch, đồng thời đóng gói thành giải pháp Full-stack Production-ready phục vụ suy luận ảnh, video và webcam trực tiếp.

### Lộ trình tiến hóa công nghệ trong dự án:
1. **Classical Proposal (Selective Search)**: Sử dụng các thuật toán phân vùng ảnh dựa trên màu sắc, texture, kích thước để tạo ra hàng nghìn Region Proposals ứng viên (minh họa chi tiết tại notebook [`notebooks/selective_search.ipynb`](file:///home/chiendoan/Code_Module6/Object_Detection_Part2/notebooks/selective_search.ipynb)).
2. **Deep Region Proposal (RPN)**: Thay thế thuật toán Selective Search chậm chạp bằng mạng RPN học trực tiếp trên feature map của Backbone CNN, trượt các Anchor Boxes đa tỉ lệ và dự đoán đồng thời điểm *objectness* và tọa độ *bounding box regression*.
3. **Faster R-CNN End-to-End**: Kết hợp Backbone trích xuất đặc trưng (**ResNet-18**), mạng RPN, tầng **RoI Align** và các đầu phân loại / tinh chỉnh hộp giới hạn (**Fast R-CNN Head**), tối ưu hóa bằng hàm mất mát đa tác vụ (Multi-task Loss).
4. **Interactive Full-stack Application**: Cung cấp giao diện web hiện đại (**Next.js 16 + React 19 + Tailwind CSS**) giao tiếp với Backend hiệu năng cao (**FastAPI + WebSockets**) cho phép người dùng điều chỉnh ngưỡng tin cậy (Confidence), lọc trùng lặp (NMS), giới hạn tốc độ xử lý (Target FPS) và phân tích trực quan kết quả suy luận.

---

## 🧠 Kiến trúc Mô hình Faster R-CNN

Kiến trúc mô hình được hiện thực chi tiết tại [`model/FasterRCNN.py`](file:///home/chiendoan/Code_Module6/Object_Detection_Part2/model/FasterRCNN.py) bao gồm 4 khối thành phần chính:

```
[ Input Image (B, 3, H, W) ]
            │
            ▼
┌───────────────────────────────────────────────┐
│  1. Backbone: ResNet-18 Feature Extractor     │
│     (Output: Feature Map C=512, H/32, W/32)   │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────┐
│  2. Region Proposal Network (RPN)             │
│     - 3x3 Conv (512 -> 512)                   │
│     - 9 Anchors (3 scales x 3 aspect ratios)  │
│     - Classifier Head: 2 * 9 (Obj vs Bkg)     │
│     - Regression Head: 4 * 9 (dx, dy, dw, dh) │
│     - Proposal Filtering & Batched NMS        │
└───────────────────────┬───────────────────────┘
                        │ (Top candidate proposals)
                        ▼
┌───────────────────────────────────────────────┐
│  3. RoI Align Layer                           │
│     - Bilinear Interpolation pooling to 7x7   │
│     - Output Tensor: (K, 512, 7, 7)           │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────┐
│  4. Fast R-CNN Detection Head                 │
│     - Flatten & FC1 (512*7*7 -> 1024) + ReLU  │
│     - Cls Head: FC2 -> 21 classes             │
│     - Reg Head: FC2 -> 21 * 4 bounding boxes  │
│     - Batched Class NMS & Coordinate Clip     │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
   [ Final Detections: BBoxes + Labels + Scores ]
```

### Các thông số kiến trúc cốt lõi:
- **Backbone**: `resnet18.a1_in1k` trích xuất feature map cấp cao (stride = 32, channels = 512).
- **RPN Anchors**: 9 anchor boxes tại mỗi spatial point trên feature map:
  - **Scales**: `[64, 128, 256]` pixel.
  - **Aspect Ratios**: `[0.5, 1.0, 2.0]` (1:2, 1:1, 2:1).
- **RoI Align**: Trích xuất vùng đặc trưng đều đặn với kích thước $7 \times 7$.
- **Hàm mất mát (Multi-task Loss)**:
  $$\mathcal{L} = \mathcal{L}_{RPN\_cls} + \lambda_1 \mathcal{L}_{RPN\_reg} + \mathcal{L}_{ROI\_cls} + \lambda_2 \mathcal{L}_{ROI\_reg}$$
  - Phân loại: *Binary Cross Entropy* (RPN) & *Cross Entropy* (Fast R-CNN).
  - Hồi quy tọa độ: *Smooth L1 Loss* ($\beta = 1.0$).
- **Tập dữ liệu**: Huấn luyện trên **Pascal VOC 2007** gồm **20 lớp đối tượng** (và 1 lớp nền Background):
  > *aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train, tvmonitor.*

---

## ✨ Tính năng Nổi bật

| Tính năng | Mô tả chi tiết |
| :--- | :--- |
| 🖼️ **Image Studio** | Tải lên ảnh tùy ý hoặc chọn ảnh mẫu từ bộ dữ liệu Pascal VOC. Tự do tùy biến thanh trượt **Confidence Threshold** ($0.01 - 1.0$) và **NMS IoU Threshold** ($0.01 - 1.0$). Hiển thị bounding box với màu sắc riêng biệt cho từng lớp, bảng thống kê số lượng và độ trễ (latency ms). |
| 🎬 **Video Studio** | Phân tích video định dạng MP4/WebM với tùy chọn **Target Processing FPS** (5, 10, 15, 24, 30 FPS). Tự động nén và mã hóa sang chuẩn video WebM thân thiện với trình duyệt, cung cấp video player phát lại mượt mà kèm bảng phân bố đối tượng theo thời gian. |
| 📹 **Live Webcam Studio** | Truyền tải frame hình thời gian thực qua kênh **WebSocket 2 chiều (`/api/ws/webcam`)**. Cho phép điều chỉnh Target FPS nhằm kiểm soát băng thông mạng và tải GPU/CPU, đo lường chính xác FPS thực tế và thời gian suy luận từng frame. |
| 🧠 **Model Architecture Inspector** | Trực quan hóa cấu trúc từng tầng của Faster R-CNN ngay trên giao diện Web, giúp người dùng nắm bắt luồng truyền dữ liệu từ ảnh đầu vào, RPN, RoI Align đến đầu ra dự đoán. |
| ⚡ **Tối ưu hóa Suy luận** | Tích hợp **PyTorch AMP (Automatic Mixed Precision)**, cơ chế **Model Warmup** khi khởi động server, **Batched NMS** trên GPU giúp giảm thiểu tối đa độ trễ suy luận. |
| 🛠️ **1-Click Launch Scripts** | Cung cấp sẵn các file bash scripts (`start_all.sh`, `run_backend.sh`, `run_frontend.sh`) hỗ trợ khởi chạy đồng thời hoặc độc lập giữa Backend và Frontend, tự động giải phóng port bị chiếm dụng. |

---

## 📂 Cấu trúc Thư mục Dự án

```text
Object_Detection_Part2/
├── backend/                        # Mã nguồn Backend FastAPI
│   └── main.py                     # Khởi tạo API REST & WebSocket stream
├── backend_storage/                # Thư mục lưu trữ tạm thời cho file tải lên & video đã xử lý
│   ├── uploads/
│   └── processed/
├── data/                           # Thư mục chứa dữ liệu benchmark & mẫu thử nghiệm
│   ├── sample_demo.mp4             # Video mẫu thử nghiệm
│   └── VOCdevkit/                  # Thư mục dataset Pascal VOC 2007
├── frontend/                       # Ứng dụng Next.js 16 Web UI
│   ├── src/
│   │   ├── app/                    # Next.js App Router (page.js, layout.js, ...)
│   │   └── components/             # Các component giao diện (ImageStudio, VideoStudio, CameraStudio, ...)
│   ├── package.json                # Quản lý dependencies frontend (React 19, Lucide, Tailwind)
│   └── next.config.mjs             # Cấu hình Next.js
├── model/                          # Cấu trúc mạng nơ-ron học sâu
│   └── FasterRCNN.py               # Hiện thực Feature_Extractor, RPN, RoI Align, Fast R-CNN Head & Loss
├── notebooks/                      # Jupyter Notebooks nghiên cứu & huấn luyện
│   ├── selective_search.ipynb      # Thử nghiệm trích xuất Region Proposal cổ điển với Selective Search
│   └── TrainFasterRCNN.ipynb       # Pipeline huấn luyện Faster R-CNN với PyTorch & AMP
├── src/                            # Thư viện tiện ích xử lý dữ liệu & suy luận
│   ├── download_data.py            # Tự động tải bộ dữ liệu Pascal VOC qua torchvision
│   ├── inference.py                # Engine suy luận FasterRCNNInferencer (ảnh, video, webcam)
│   └── orgainize_data.py           # Custom PyTorch Dataset & Transforms cho Pascal VOC
├── model_weight.pth                # Trọng số mô hình đã huấn luyện (~251 MB)
├── requirements.txt                # Danh sách thư viện Python cần cài đặt
├── run_backend.sh                  # Script khởi chạy riêng FastAPI Backend (:8000)
├── run_frontend.sh                 # Script khởi chạy riêng Next.js Frontend (:3000)
├── start_all.sh                    # Script khởi chạy đồng thời Backend & Frontend
└── README.md                       # Tài liệu hướng dẫn sử dụng và đặc tả dự án
```

---

## 💻 Yêu cầu Hệ thống (Prerequisites)

Trước khi tiến hành cài đặt, hãy đảm bảo hệ thống của bạn đáp ứng các tiêu chuẩn sau:

- **Hệ điều hành**: Linux (Ubuntu 20.04/22.04/24.04 được khuyến nghị), macOS hoặc Windows (thông qua WSL2).
- **Python**: Phiên bản `3.10` trở lên (khuyên dùng Python 3.10 hoặc 3.11).
- **Node.js**: Phiên bản `v18.0.0` hoặc `v20.0.0+` kèm theo `npm`.
- **Card đồ họa (GPU)**: Khuyến nghị card NVIDIA có hỗ trợ CUDA (Compute Capability 7.0+, ví dụ: GTX 1660, RTX 2060, RTX 3060 trở lên) để đạt tốc độ xử lý video/webcam mượt mà (chế độ CPU vẫn hoạt động tự động nếu không có CUDA).
- **FFmpeg & Codecs**: Cần thiết cho việc giải mã/ghi video bằng OpenCV (`apt install ffmpeg libsm6 libxext6`).

---

## 🛠️ Hướng dẫn Cài đặt Chi tiết

### Bước 1: Clone kho lưu trữ

```bash
git clone https://github.com/ChienDoanDuy204/Object_Detection_Part2.git
cd Object_Detection_Part2
```

---

### Bước 2: Khởi tạo Môi trường Python

Bạn có thể sử dụng **Conda / Miniconda** (khuyến nghị) hoặc **Python venv**:

#### Lựa chọn A: Sử dụng Conda (Khuyên dùng)
```bash
# Tạo môi trường ảo với Python 3.10
conda create -n faster_rcnn_env python=3.10 -y

# Kích hoạt môi trường
conda activate faster_rcnn_env
```

#### Lựa chọn B: Sử dụng Python venv chuẩn
```bash
# Tạo virtual environment
python3 -m venv .venv

# Kích hoạt môi trường
source .venv/bin/activate      # Trên Linux/macOS
# Hoặc trên Windows: .venv\Scripts\activate
```

---

### Bước 3: Cài đặt các gói thư viện Python (`requirements.txt`)

1. **(Quan trọng) Cài đặt PyTorch với hỗ trợ CUDA tương thích**:
   Nếu máy của bạn có GPU NVIDIA, hãy truy cập [PyTorch Get Started](https://pytorch.org/get-started/locally/) để cài đặt bản PyTorch phù hợp với phiên bản CUDA của máy.
   *Ví dụ cho CUDA 12.4 / 12.6:*
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
   ```
   *Nếu chỉ dùng CPU:*
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   ```

2. **Cài đặt toàn bộ các thư viện phụ thuộc qua `requirements.txt`**:
   ```bash
   pip install -r requirements.txt
   ```

> [!NOTE]
> File [`requirements.txt`](file:///home/chiendoan/Code_Module6/Object_Detection_Part2/requirements.txt) đã được phân loại rõ ràng gồm:
> - **Deep Learning**: `torch`, `torchvision`, `timm` (cung cấp ResNet backbone pretrained)
> - **Computer Vision**: `opencv-python`, `pillow`, `numpy`
> - **Web API Backend**: `fastapi`, `uvicorn[standard]`, `python-multipart`, `websockets`, `pydantic`
> - **Training & Tracking**: `tqdm`, `matplotlib`, `requests`, `huggingface_hub`

3. **Kiểm tra cài đặt**:
   ```bash
   python -c "import torch, torchvision, timm, cv2, fastapi; print('PyTorch CUDA:', torch.cuda.is_available(), '| Thiết bị:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
   ```

---

### Bước 4: Cài đặt Dependencies cho Frontend

Di chuyển vào thư mục `frontend` và cài đặt các package Node.js:

```bash
cd frontend
npm install
cd ..
```

---

### Bước 5: Kiểm tra Trọng số Mô hình (`model_weight.pth`)

Đảm bảo file trọng số mô hình đã được đặt ngay tại thư mục gốc:
```bash
ls -lh model_weight.pth
```
File trọng số có dung lượng khoảng **251 MB**. Khi khởi động, Backend sẽ tự động load file này vào GPU/CPU.

---

### Bước 6: Tải dữ liệu mẫu Pascal VOC 2007 (Tùy chọn)

Nếu muốn phục vụ thư viện ảnh mẫu cho Image Studio hoặc phục vụ huấn luyện lại mô hình:
```bash
python -m src.download_data
```
Script sẽ tự động tải bộ dữ liệu Pascal VOC 2007 (`trainval` và ảnh JPEG) vào thư mục `data/VOCdevkit/`.

---

## 🚀 Hướng dẫn Khởi chạy Hệ thống

### Cách 1: Khởi chạy 1-Click tự động (Khuyến nghị) 🌟

Chúng tôi đã thiết lập sẵn script [`start_all.sh`](file:///home/chiendoan/Code_Module6/Object_Detection_Part2/start_all.sh) giúp khởi chạy cả Backend và Frontend cùng lúc chỉ bằng một lệnh duy nhất:

```bash
# Cấp quyền thực thi nếu cần
chmod +x start_all.sh run_backend.sh run_frontend.sh

# Khởi chạy hệ thống
./start_all.sh
```

**Đặc điểm nổi bật của script:**
- Tự động kiểm tra và giải phóng cổng `8000` và `3000` nếu bị tiến trình khác chiếm dụng.
- Khởi động song song Backend và Frontend.
- Bắt tín hiệu ngắt (`Ctrl + C`) để dọn dẹp sạch sẽ toàn bộ các tiến trình nền khi bạn dừng hệ thống.

---

### Cách 2: Khởi chạy Thủ công Từng Phần

Nếu bạn muốn debug hoặc theo dõi log riêng của từng thành phần:

#### 1. Khởi chạy Backend:
```bash
./run_backend.sh
# Hoặc khởi chạy thủ công bằng python:
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
- API sẽ khả dụng tại: `http://localhost:8000`
- Swagger UI tài liệu API: `http://localhost:8000/docs`

#### 2. Khởi chạy Frontend:
Mở một terminal mới:
```bash
./run_frontend.sh
# Hoặc khởi chạy thủ công bằng npm:
cd frontend
npm run dev -- -p 3000
```
- Mở trình duyệt và truy cập: **`http://localhost:3000`**

---

## 📡 Tài liệu API Endpoints

FastAPI Backend cung cấp đầy đủ các chuẩn RESTful và WebSocket phục vụ đa dạng nhu cầu:

### REST API

| Method | Endpoint | Mô tả | Tham số / Body |
| :--- | :--- | :--- | :--- |
| `GET` | `/api/info` | Trả về thông tin mô hình, phần cứng (CUDA/CPU), danh sách 20 classes và bảng mã màu | Không |
| `GET` | `/api/samples` | Lấy danh sách ảnh mẫu benchmark từ bộ dữ liệu Pascal VOC | Không |
| `GET` | `/api/sample/{filename}` | Lấy dữ liệu file ảnh mẫu | `filename: string` |
| `GET` | `/api/video/sample` | Lấy video demo mẫu phục vụ test nhanh Video Studio | Không |
| `POST` | `/api/predict/image` | Dự đoán vật thể trên ảnh tĩnh tải lên | `file: UploadFile`<br>`conf_threshold: float (0.01 - 1.0)`<br>`nms_threshold: float (0.01 - 1.0)` |
| `POST` | `/api/predict/video` | Xử lý nhận diện trên file video theo tốc độ target FPS | `file: UploadFile`<br>`target_fps: float (1 - 30)`<br>`conf_threshold: float`<br>`nms_threshold: float` |
| `GET` | `/api/video/{filename}` | Stream hoặc tải video WebM đã vẽ bounding box | `filename: string` |

### WebSocket API

| Protocol | Endpoint | Mô tả | Payload gửi lên / nhận về |
| :--- | :--- | :--- | :--- |
| `WS` | `/api/ws/webcam` | Kênh stream webcam tương tác 2 chiều thời gian thực | **Client gửi**: JSON `{ image: "<base64>", target_fps: 15, conf_threshold: 0.35, nms_threshold: 0.35, return_annotated: true }`<br>**Server trả**: JSON `{ detections: [...], inference_time_ms: 18.5, actual_fps: 14.8, annotated_image: "<base64>" }` |

---

## 🔬 Huấn luyện Mô hình & Nghiên cứu

Dự án cung cấp hai sổ tay Jupyter phục vụ mục đích nghiên cứu và tái huấn luyện:

### 1. Phân tích Region Proposals với Selective Search
- File: [`notebooks/selective_search.ipynb`](file:///home/chiendoan/Code_Module6/Object_Detection_Part2/notebooks/selective_search.ipynb)
- Mục đích: Minh họa trực quan thuật toán Selective Search cổ điển dựa trên OpenCV (`cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()`). Phân tích nhược điểm về thời gian tính toán (~2-5 giây/ảnh) là động lực để phát minh ra Region Proposal Network (RPN).

### 2. Huấn luyện Faster R-CNN từ đầu với PyTorch
- File: [`notebooks/TrainFasterRCNN.ipynb`](file:///home/chiendoan/Code_Module6/Object_Detection_Part2/notebooks/TrainFasterRCNN.ipynb)
- Nội dung:
  - Tải và tiền xử lý dữ liệu với XML Parser cho nhãn VOC Bounding Boxes.
  - Xây dựng tầng Anchor Generator sinh 9 anchors trên mỗi vị trí spatial.
  - Tính toán IoU giữa ground-truth và anchors để gán nhãn dương tính (positive $\ge 0.7$), âm tính (negative $< 0.3$) và bỏ qua (ignore).
  - Huấn luyện kết hợp RPN và Fast R-CNN Head bằng kỹ thuật Mixed Precision (`torch.amp.GradScaler`).
  - Đánh giá mAP (mean Average Precision) và lưu lại trọng số `model_weight.pth`.

---

## ❓ Xử lý Sự cố Thường gặp (Troubleshooting)

<details>
<summary><b>1. Trình duyệt báo lỗi "Requested device not found" ở tab Webcam Studio?</b></summary>

- **Nguyên nhân**: Máy tính của bạn chưa kết nối webcam vật lý, hoặc trình duyệt chưa được cấp quyền truy cập camera cho tên miền `http://localhost:3000`.
- **Cách xử lý**:
  1. Kiểm tra lại thiết bị webcam bằng lệnh `v4l2-ctl --list-devices` hoặc `ls /dev/video*`.
  2. Bấm vào biểu tượng ổ khóa/quyền trang web trên thanh địa chỉ trình duyệt và chuyển trạng thái Camera sang **Allow**.
  3. Nếu thử nghiệm trên máy chủ từ xa (Remote Server/SSH), tính năng webcam client sẽ sử dụng camera của máy khách (client laptop/PC) đang mở trình duyệt.
</details>

<details>
<summary><b>2. Cổng 8000 hoặc 3000 báo lỗi "Address already in use"?</b></summary>

- **Cách xử lý**: Chạy lệnh giải phóng cổng tức thì:
  ```bash
  fuser -k 8000/tcp 2>/dev/null || true
  fuser -k 3000/tcp 2>/dev/null || true
  ```
  Script `./start_all.sh` đã được tích hợp sẵn cơ chế này tự động.
</details>

<details>
<summary><b>3. Thông báo "OpenCV: FFMPEG: tag 0x30385056/'VP80' is not supported"?</b></summary>

- **Giải thích**: Đây là thông báo log của bộ nạp codec OpenCV FFMPEG khi khởi tạo `VideoWriter` mã hóa định dạng WebM VP80. Tiến trình vẫn fallback và ghi video bình thường. Video sau khi xử lý có thể phát trực tiếp trên trình duyệt Chrome/Firefox/Edge mà không gặp trở ngại.
</details>

<details>
<summary><b>4. Thông báo cảnh báo "Warning: You are sending unauthenticated requests to the HF Hub"?</b></summary>

- **Giải thích**: Thư viện `timm` tự động tải trọng số pretrained backbone `resnet18.a1_in1k` từ Hugging Face Hub ở lần chạy đầu tiên. Đây chỉ là thông báo khuyến nghị và không ảnh hưởng đến hoạt động của mô hình.
</details>

---

## 📜 Giấy phép & Tác giả

- **Tác giả**: Chien Doan ([@ChienDoanDuy204](https://github.com/ChienDoanDuy204))
- **Mục đích**: Nghiên cứu, học tập và triển khai mô hình học sâu Thị giác máy tính (Computer Vision / Object Detection).
- Dự án hoan nghênh mọi đóng góp, phản hồi và mở rộng! ⭐ Hãy cho dự án 1 sao nếu bạn thấy hữu ích!
