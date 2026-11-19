# Tái lập mixup: Beyond Empirical Risk Minimization (ICLR 2018)

Mã nguồn tái lập thực nghiệm cho đồ án CSC14005. Bài báo: Zhang, Cisse, Dauphin,
Lopez-Paz, *mixup: Beyond Empirical Risk Minimization*, ICLR 2018 (arXiv:1710.09412).

## Cài đặt

```bash
conda env create -f environment.yml
conda activate mixup
python -m pytest tests -q        # kiểm tra phần lõi, chạy trên CPU trong vài giây
```

Trên Kaggle/Colab đã có sẵn torch và torchvision, chỉ cần `pip install -r requirements.txt`
nếu thiếu gói nào.

## Cấu trúc

```
Source/
  core/              LÕI DÙNG CHUNG, không phụ thuộc kiểu dữ liệu
    mixup.py         lấy λ, chọn cặp, trộn batch, loss, trộn biểu diễn ẩn, mixup cho GAN
    trainer.py       vòng lặp train/eval chung cho mọi bài toán phân loại, có resume
    schedules.py     lịch learning rate (step, warm-up)
    attacks.py       FGSM, I-FGSM, đánh giá white-box/black-box (3.5)
    data_utils.py    làm nhiễu nhãn (3.4), dataset từ mảng numpy (3.6)
    config.py        cờ dòng lệnh chung cho mọi script
    utils.py         seed, thiết bị, log CSV/JSON/W&B, checkpoint, thông tin môi trường
  cifar/             dùng chung cho 3.2, 3.4, 3.5, 3.8
    models.py        PreAct ResNet-18 giống hệt mã gốc
    data.py          CIFAR-10/100, augmentation, nhãn nhiễu
  tests/test_core.py
```

Các thực nghiệm (thêm sau, mỗi người một file) chỉ gọi lại `core` và `cifar`:

| Mục | Dữ liệu | Dùng từ lõi |
|---|---|---|
| 3.2 CIFAR | ảnh | `Trainer`, `cifar.build_cifar`, `preact_resnet18` |
| 3.3 Speech | phổ âm thanh (B,1,F,T) | `Trainer` với `MixupConfig(warmup_epochs=5)`, `step_schedule([10,20])` |
| 3.4 Nhãn nhiễu | ảnh | `build_cifar(noise_rate=..., eval_train_subsets=True)`, `preact_resnet18(dropout=...)` |
| 3.5 Đối kháng | ảnh | `core.attacks` + `Normalized` trên model đã train |
| 3.6 UCI | bảng (B,D) | `TensorClassificationDataset` + `Trainer` |
| 3.7 GAN | ảnh/điểm 2D | `mixup_real_fake` trong vòng lặp GAN riêng |
| 3.8 Ablation | ảnh | cờ `--pairing`, `--label-mode`, `--smoothing`, `--input-noise`, `--latent-module`, `pair_index_fn` (KNN) |

## Cách dùng lõi trong một script thực nghiệm

```python
import argparse, torch
from core.config import add_common_args, mixup_config_from_args, default_run_name
from core.utils import set_seed, get_device, RunLogger
from core.schedules import step_schedule
from core.trainer import Trainer
from cifar import build_cifar, preact_resnet18

a = add_common_args(argparse.ArgumentParser()).parse_args()
rng = set_seed(a.seed, a.deterministic)
data = build_cifar("cifar10", a.data_dir, a.batch_size, workers=a.workers, seed=a.seed)
model = preact_resnet18(data.num_classes)
opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=a.momentum, weight_decay=a.wd)
sched = step_schedule(opt, a.milestones, a.gamma, official_offset=a.official_lr_offset)
name = a.run_name or default_run_name(a, "exp32_cifar10")
log = RunLogger(a.out_dir, name, vars(a), a.wandb, a.wandb_project, resume=not a.no_resume)
tr = Trainer(model, opt, sched, data.train, {"test": data.test, **data.extra_eval},
             data.num_classes, mixup_config_from_args(a), log, rng, get_device(a.device),
             a.epochs, amp=a.amp, limit_batches=a.limit_batches)
print(tr.fit(resume=not a.no_resume))
```

Mỗi lần chạy ghi vào `results/<run_name>/`: `config.json` (cấu hình, phiên bản thư viện, GPU),
`history.csv` (mỗi epoch một dòng), `summary.json` (lỗi best/last/median 10 epoch cuối),
`last.pt` (checkpoint để chạy tiếp), `best_model.pt`, `final_model.pt`.

**Chạy tiếp khi Kaggle/Colab bị ngắt:** chạy lại đúng lệnh cũ với cùng `--run-name` và
`--out-dir`; trainer tự đọc `last.pt` và tiếp tục từ epoch đang dở (gồm cả trạng thái
optimizer, scheduler và bộ sinh số ngẫu nhiên).

## Quy ước nhóm

1. `core/` và `cifar/` đã **đóng băng**. Muốn sửa thì mở pull request để cả nhóm xem.
2. Không đổi giá trị mặc định. Thực nghiệm nào cần khác thì truyền cờ, hoặc thêm cờ mới
   có mặc định giữ nguyên hành vi cũ.
3. Luôn ghi `--seed`. Mỗi cấu hình chạy với cùng danh sách seed (vd 0, 1, 2).

## Đối chiếu với mã chính thức (facebookresearch/mixup-cifar10)

Những chi tiết có trong mã gốc nhưng bài báo không nêu, và cách mã này xử lý:

| Chi tiết | Mã gốc | Mã này |
|---|---|---|
| Số λ mỗi batch | một λ cho cả batch | giống (có cờ `--per-sample-lambda` để thử khác) |
| Cách ghép cặp | hoán vị trong cùng batch (bài báo minh họa bằng 2 loader) | giống |
| Dạng loss | λ·CE(y_i) + (1-λ)·CE(y_j) | giống; tương đương CE với nhãn mềm (có test chứng minh) |
| Kiến trúc "ResNet18" | PreAct, stem conv-BN-ReLU, không có BN-ReLU cuối | giống, 11.171.274 tham số |
| Chuẩn hóa | std = (0.2023, 0.1994, 0.2010) | giống |
| Giảm lr | sau epoch 101 và 151 (lệch 1 epoch so với bài báo) | theo bài báo; `--official-lr-offset` để khớp mã gốc |
| Seed | chỉ seed torch; np.random (dùng lấy λ) không được seed; seed=0 nghĩa là không seed | seed toàn bộ, λ lấy từ Generator riêng, lưu được khi resume |
| Augmentation | RandomCrop(32, pad 4) + lật ngang | giống |
| Optimizer | SGD momentum 0.9, không Nesterov, wd 1e-4, batch 128 | giống (mặc định) |

## Nguồn tham khảo

- Mã chính thức: https://github.com/facebookresearch/mixup-cifar10 (CC-BY-NC 4.0). Kiến trúc
  PreAct ResNet-18 và hằng số chuẩn hóa được dựng lại theo mã này.
- kuangliu/pytorch-cifar: nguồn gốc của mã CIFAR trong repo chính thức.
- Zhang et al. (2017), *Understanding deep learning requires rethinking generalization*: cách tạo nhãn nhiễu.
