# Chạy trên Kaggle

Thư mục này chứa các notebook để chạy trên Kaggle. Mọi logic nằm trong `core/`, `cifar/`,
`experiments/`; notebook chỉ gọi lại các script đó, nên kết quả chạy trên Kaggle và trên
máy cá nhân là như nhau.

| Notebook              | Mục đích                                                                                   | Thời gian GPU (ước lượng) |
| --------------------- | --------------------------------------------------------------------------------------------- | ------------------------------ |
| `exp32_smoke.ipynb` | Chạy thử 3.2: đo giây/epoch (FP32 và AMP), test resume, ước lượng giờ cho 200 epoch | 10–20 phút                   |
| `exp32_full.ipynb` | Quản lý toàn bộ 3.2: chọn nhóm chạy, resume, kiểm tra tiến độ và tổng hợp kết quả | Tùy model và số run; chia nhiều phiên |

## Lần đầu: đưa code lên Kaggle

1. Nén thư mục `Source` thành `Source.zip`.
2. Kaggle → **Datasets → New Dataset** → kéo `Source.zip` vào → đặt tên, vd `mixup-source`
   → để **Private** → Create.
   Muốn chia sẻ cho cả nhóm: vào dataset → Settings → Sharing → thêm tài khoản các bạn.
3. Mỗi lần sửa code: vào dataset → **New Version** → tải `Source.zip` mới lên.

## Chạy notebook smoke

1. Kaggle → **Code → New Notebook** → File → **Import Notebook** → chọn `exp32_smoke.ipynb`.
2. Thanh bên phải:
   - **Session options → Accelerator: GPU P100** (bài báo dùng P100) hoặc GPU T4 x2.
     Code chỉ dùng 1 GPU nên T4 x2 không nhanh gấp đôi.
   - **Internet: On** (cần để tải CIFAR-10 lần đầu, ~170MB).
   - **Add Input** → tab Your Datasets → chọn `mixup-source`.
3. **Run All**. Xong thì xem bảng ở mục 3 của notebook và gửi lại cho nhóm con số
   giây/epoch + tên GPU.

## Chạy một cấu hình dài bằng notebook smoke

- Đặt `RUN_FULL = True`, chỉnh `FULL_ALPHA` (0 = ERM, 1 = mixup) và `FULL_SEED`.
  Phần này chỉ chạy một cấu hình CIFAR-10 + PreAct ResNet-18, chưa phải toàn bộ mục 3.2.
- Dùng **Save Version → Save & Run All (Commit)**: Kaggle chạy nền tới 12 giờ, có thể tắt trình duyệt.
- Kết quả nằm ở tab **Output** của version đó: `results/.../history.csv`, `summary.json`,
  `results_logs.zip` (không có file model để nhẹ).

### Nếu bị ngắt giữa chừng (hết 12 giờ, hết quota)

`last.pt` được lưu sau mỗi epoch. Để chạy tiếp ở phiên mới:

1. Add Input → chọn **output của notebook version cũ** (tab Notebook Output Files).
2. Trước khi chạy, chép thư mục `results` cũ vào `/kaggle/working/results`
   (cùng `--run-name`). Trainer tự đọc `last.pt` và tiếp tục đúng epoch đang dở.

## Quản lý toàn bộ mục 3.2 bằng `exp32_full.ipynb`

Import `exp32_full.ipynb`, bật GPU và Internet, rồi Add Input dataset chứa code như trên.
Notebook gọi lại `experiments/exp32_cifar.py`; không cần chép logic huấn luyện vào các cell.
Xem [protocol và kết quả tham chiếu](../experiments/README_exp32.md) để biết các lựa chọn tái lập.

Lưới cần hoàn thành gồm **2 dataset × 3 model × ERM/mixup × `TARGET_SEEDS`**.
Mặc định `TARGET_SEEDS = [0, 1, 2]` tương ứng **36 runs**, mỗi run 200 epoch,
batch 128, FP32, mixup α=1 hoặc ERM α=0. Ba seed là lựa chọn của đồ án;
paper không quy định số seed tại mục 3.2.

### Chọn công việc cho từng phiên

1. Chạy lần đầu với `RUN_TRAINING = False` để xem kế hoạch và bảng tiến độ.
2. Chỉnh các biến `SESSION_*` trong cell cấu hình. Mặc định chọn hai phương pháp
   ERM/mixup trên CIFAR-10, PreAct ResNet-18, seed 0:

   ```python
   SESSION_DATASETS = ["cifar10"]
   SESSION_MODELS = ["preact_resnet18"]
   SESSION_METHODS = None
   SESSION_SEEDS = [0]
   ```

   `None` chọn toàn bộ giá trị tương ứng trong lưới mục tiêu. Đặt cả bốn biến thành
   `None` để chọn toàn bộ 36 runs; nên chia theo model/dataset/seed để quản lý từng phiên.
3. Đặt `RUN_TRAINING = True` rồi chạy lại notebook. Các run đã hoàn thành được bỏ qua;
   các run còn dở tiếp tục từ checkpoint khi có.

`SOURCE_INPUT = None` tự tìm code trong `/kaggle/input`. Nếu có nhiều bản `Source`,
đặt đường dẫn chính xác để chọn bản cần chạy. `MAX_SESSION_HOURS = 8.0` giới hạn
thời gian bắt đầu run mới trong phiên; notebook vẫn để run đang chạy hoàn thành,
nên đây không phải giới hạn cứng cho tổng thời gian phiên.

### Tiếp tục hoặc gom kết quả từ nhiều phiên

Sau mỗi phiên, lưu notebook version kèm Output để giữ thư mục
`/kaggle/working/results`, bao gồm các file `.pt`. Ở phiên sau:

1. Add Input → chọn output của các notebook version trước.
2. Điền `RESUME_INPUTS` bằng danh sách đường dẫn **thư mục `results`** trong các input đó.
3. Chọn nhóm chạy tiếp qua `SESSION_*` và chạy notebook.

Notebook gom kết quả theo từng run, giữ snapshot có epoch cao nhất khi cấu hình
tương thích và từ chối các xung đột. Giữ nguyên cấu hình khoa học để resume;
`last.pt` là checkpoint cần cho việc tiếp tục huấn luyện. Resume đúng epoch không
bảo đảm chuỗi batch giống hệt một lần chạy liên tục; ưu tiên chạy liên tục từng run
khi có thể.

### Xem và tải kết quả

- Bảng tiến độ cho biết run/seed còn thiếu trong toàn bộ lưới mục tiêu.
- Báo cáo từ `exp32_report.py` tổng hợp các run hoàn thành và khớp protocol thành
  `comparison.csv`, `comparison.json`, cùng `training_curves.png`.
  `best_err` là lựa chọn báo cáo mặc định của repo; mục 3.2 không nói rõ quy tắc chọn checkpoint.
- `results_logs.zip` là bản log nhẹ để tải về và chia sẻ, không chứa `.pt`.
  Để resume ở phiên sau, dùng thư mục `results` được lưu trong Output, không dùng bản zip log.
