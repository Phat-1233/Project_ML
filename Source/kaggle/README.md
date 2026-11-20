# Chạy trên Kaggle

Thư mục này chứa các notebook để chạy trên Kaggle. Mọi logic nằm trong `core/`, `cifar/`,
`experiments/`; notebook chỉ gọi lại các script đó, nên kết quả chạy trên Kaggle và trên
máy cá nhân là như nhau.

| Notebook              | Mục đích                                                                                   | Thời gian GPU (ước lượng) |
| --------------------- | --------------------------------------------------------------------------------------------- | ------------------------------ |
| `exp32_smoke.ipynb` | Chạy thử 3.2: đo giây/epoch (FP32 và AMP), test resume, ước lượng giờ cho 200 epoch | 10–20 phút                   |
| `exp38_ablation.ipynb` | 3.8 ablation: chạy thử rồi chạy các biến thể của Bảng 5, gom bảng + biểu đồ so với bài báo | ~ số lần chạy × thời gian 1 lần |

## Lần đầu: đưa code lên Kaggle

1. Nén thư mục `Source` thành `Source.zip`.
2. Kaggle → **Datasets → New Dataset** → kéo `Source.zip` vào → đặt tên, vd `mixup-source`
   → để **Private** → Create.
   Muốn chia sẻ cho cả nhóm: vào dataset → Settings → Sharing → thêm tài khoản các bạn.
3. Mỗi lần sửa code: vào dataset → **New Version** → tải `Source.zip` mới lên.

## Chạy notebook

1. Kaggle → **Code → New Notebook** → File → **Import Notebook** → chọn `exp32_smoke.ipynb`.
2. Thanh bên phải:
   - **Session options → Accelerator: GPU P100** (bài báo dùng P100) hoặc GPU T4 x2.
     Code chỉ dùng 1 GPU nên T4 x2 không nhanh gấp đôi.
   - **Internet: On** (cần để tải CIFAR-10 lần đầu, ~170MB).
   - **Add Input** → tab Your Datasets → chọn `mixup-source`.
3. **Run All**. Xong thì xem bảng ở mục 3 của notebook và gửi lại cho nhóm con số
   giây/epoch + tên GPU.

## Chạy thật (dài)

- Đặt `RUN_FULL = True`, chỉnh `FULL_ALPHA` (0 = ERM, 1 = mixup) và `FULL_SEED`.
- Dùng **Save Version → Save & Run All (Commit)**: Kaggle chạy nền tới 12 giờ, có thể tắt trình duyệt.
- Kết quả nằm ở tab **Output** của version đó: `results/.../history.csv`, `summary.json`,
  `results_logs.zip` (không có file model để nhẹ).

### Nếu bị ngắt giữa chừng (hết 12 giờ, hết quota)

`last.pt` được lưu sau mỗi epoch. Để chạy tiếp ở phiên mới:

1. Add Input → chọn **output của notebook version cũ** (tab Notebook Output Files).
2. Trước khi chạy, chép thư mục `results` cũ vào `/kaggle/working/results`
   (cùng `--run-name`). Trainer tự đọc `last.pt` và tiếp tục đúng epoch đang dở.
