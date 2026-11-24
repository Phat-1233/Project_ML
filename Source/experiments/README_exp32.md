# Tái lập mục 3.2: CIFAR-10 và CIFAR-100

Thực nghiệm theo [mixup: Beyond Empirical Risk Minimization, bản v2, §3.2 và Figure 3](https://arxiv.org/html/1710.09412v2#S3.SS2).
Mã tham khảo: các file `train.py`, `models/resnet.py`, `models/densenet3.py` được cung cấp
từ [facebookresearch/mixup-cifar10](https://github.com/facebookresearch/mixup-cifar10),
và [WideResNet PyTorch của tác giả](https://github.com/szagoruyko/wide-residual-networks/tree/master/pytorch).
Các file tham khảo không cần nằm trên máy chạy thực nghiệm.

`exp32_cifar.py` gọi trực tiếp `Trainer`, `cifar.build_cifar`, `step_schedule`,
`RunLogger` và các hàm seed/config của repo. Cả ba kiến trúc nằm trong
`cifar/models.py`; `cifar.build_model` dùng lại `preact_resnet18` hiện có hoặc tạo
`WideResNet`/`DenseNetBC`. Phần code model có sẵn được giữ nguyên; hai kiến trúc
mới được thêm ở cuối file.

## Protocol và kết quả tham chiếu

Mỗi seed có **12 runs**: hai dataset × ba model × ERM/mixup. Danh sách seed 0, 1, 2
là lựa chọn cho đồ án; paper không quy định số seed.

| Tham số | Giá trị |
|---|---|
| Epoch / train batch / test batch | 200 / 128 / 100 |
| Optimizer | SGD, learning rate 0.1, momentum 0.9, không Nesterov |
| Weight decay / dropout | `1e-4` / không dùng |
| PreAct ResNet-18, DenseNet-BC-190 | LR × 0.1 sau 100 và 150 epoch |
| WideResNet-28-10 | LR × 0.1 sau 60, 120 và 180 epoch |
| ERM / mixup | `alpha=0` / `alpha=1`, lambda từ Beta(1,1) |
| Ghép cặp | Hoán vị trong minibatch; một lambda cho cả batch |
| Loss mixup | `lambda * CE(logits, y_a) + (1-lambda) * CE(logits, y_b)` |
| Augmentation train | RandomCrop(32, padding=4) + RandomHorizontalFlip |
| Đánh giá | Toàn bộ 10.000 ảnh test, không augmentation hoặc mixup |

Mean/std dùng từ `build_cifar`: `(0.4914, 0.4822, 0.4465)` /
`(0.2023, 0.1994, 0.2010)` cho cả hai dataset. File `train.py` tham khảo chỉ hỗ trợ
CIFAR-10; áp dụng cùng normalization cho CIFAR-100 là lựa chọn kế thừa từ repo,
không phải chi tiết được xác nhận bởi file đó hoặc §3.2.

DenseNet có depth 190, growth rate 40, bottleneck factor 4, compression 0.5:
ba dense block, mỗi block 31 bottleneck. Classifier nhận đúng 10 hoặc 100 lớp.
WRN dùng depth 28, width 10 và initialization của mã tác giả, gồm BN scale
khởi tạo Uniform(0,1). Không dùng dropout trong cả hai model.

Lỗi test (%) trong **Figure 3(a)**:

| Dataset | Model CLI | ERM | mixup |
|---|---|---:|---:|
| CIFAR-10 | `preact_resnet18` | 5.6 | 4.2 |
| CIFAR-10 | `wideresnet28_10` | 3.8 | 2.7 |
| CIFAR-10 | `densenet_bc190` | 3.7 | 2.7 |
| CIFAR-100 | `preact_resnet18` | 25.6 | 21.1 |
| CIFAR-100 | `wideresnet28_10` | 19.4 | 17.5 |
| CIFAR-100 | `densenet_bc190` | 19.0 | 16.8 |

Đây là số tham chiếu từ paper, không phải kết quả đã chạy của repo.
Paper không nói rõ quy tắc chọn checkpoint ở §3.2. Báo cáo mặc định dùng
`best_err` (lỗi test nhỏ nhất qua các epoch) như một **lựa chọn báo cáo**,
và cho phép đối chiếu `last_err` hoặc `median_last10_err`. Quy tắc median 10 epoch
cuối được paper nêu ở §3.8; không gán quy tắc đó cho §3.2.

## Chạy thực nghiệm

Chạy các lệnh dưới đây từ thư mục `Source`, với môi trường đã cài `requirements.txt`
hoặc `environment.yml`. Chọn bản PyTorch phù hợp GPU theo môi trường chạy;
môi trường `.venv` ở gốc repo do lần triển khai này tạo chỉ dùng PyTorch CPU để kiểm thử.

Xem trước 36 cấu hình, không tải dữ liệu hoặc tạo model/checkpoint:

```bash
python experiments/exp32_cifar.py --suite --seeds 0 1 2 --dry-run
```

Chạy đầy đủ 36 runs lần lượt, mỗi run ở một tiến trình riêng:

```bash
python experiments/exp32_cifar.py --suite --seeds 0 1 2 --device cuda --data-dir ./data --out-dir ./results
```

Chỉ chạy PreAct ResNet-18 cho cả hai bộ CIFAR, ERM và mixup:

```bash
python experiments/exp32_cifar.py --suite --models preact_resnet18 --seeds 0 1 2 --device cuda
```

Chia việc giữa các máy bằng `--datasets`, `--models`, `--methods`, `--seeds`:

```bash
python experiments/exp32_cifar.py --suite --datasets cifar100 --models densenet_bc190 --methods mixup --seeds 0 --device cuda
```

Chạy riêng một cấu hình:

```bash
python experiments/exp32_cifar.py --dataset cifar10 --model preact_resnet18 --alpha 0 --seed 0 --device cuda
python experiments/exp32_cifar.py --dataset cifar100 --model wideresnet28_10 --alpha 1 --seed 0 --device cuda
```

Nếu không truyền `--milestones`, script tự chọn theo model. Gamma mặc định là
**0.1 cho cả ba kiến trúc** theo §3.2, kể cả WRN; không dùng gamma 0.2 của WRN gốc.
`--official-lr-offset` giảm LR trễ một epoch để khớp vòng lặp `train.py` tham khảo;
mặc định theo mốc trong paper. Cờ này được ghi nhận là khác protocol mặc định.

Smoke test với dữ liệu thật (tải CIFAR khi chưa có):

```bash
python experiments/exp32_cifar.py --dataset cifar10 --epochs 2 --limit-batches 2 --batch-size 8 --workers 0 --device cpu --run-name exp32_smoke
```

`--limit-batches` giới hạn **cả train lẫn test** trong `Trainer` hiện có. Kết quả
này chỉ kiểm tra đường chạy. `--amp`, augmentation khác, epoch/batch/LR khác và
các tùy chọn ablation cũng được ghi `paper_protocol=false`, kèm lý do trong config.
`paper_protocol=true` chỉ xác nhận cấu hình đã chọn, không bảo đảm đạt đúng lỗi paper.

Paper dùng một Tesla P100. Hai model lớn cần bộ nhớ đáng kể ở batch 128;
máy được kiểm tra trong lần triển khai này chỉ có GTX 1650 4 GiB. Dùng GPU đủ bộ nhớ
trên Kaggle/Colab để chạy đầy đủ. Giảm batch hoặc bật AMP là biến thể cần ghi rõ.
Notebook `kaggle/exp32_smoke.ipynb` hiện có vẫn dùng được cho PreAct; chạy các lệnh
trên trong thư mục `Source` đã chép vào vùng working để dùng suite đầy đủ.

## Kết quả, resume và tổng hợp

Thư mục mặc định của run chuẩn, ví dụ:
`results/exp32_cifar100/wideresnet28_10/mixup_a1_seed0/`.
Mỗi run có `config.json` (cả môi trường, số tham số, protocol), `history.csv`,
`summary.json`, `last.pt`, `best_model.pt`, `final_model.pt`.
Cấu hình khác protocol nhận hậu tố `_custom_<hash>` để phân biệt khi không đặt tên.
`--run-name` đặt tên run riêng; trong `--suite`, nó là thư mục tiền tố của các run.

Chạy lại cùng lệnh để resume từ `last.pt`. Script kiểm tra cấu hình trước khi
ghi log hoặc tải dữ liệu; từ chối ghép checkpoint khi model, dataset, alpha, seed,
optimizer, lịch LR hoặc các cờ khoa học khác thay đổi. Có thể tăng `--epochs` khi
giữ cùng tên run. `--no-resume` yêu cầu thư mục run mới.

Resume khôi phục model, optimizer, scheduler và các RNG được `Trainer` lưu.
`Trainer` hiện chưa lưu generator riêng của DataLoader và RNG augmentation trong
worker, nên tiếp tục đúng epoch **không bảo đảm chuỗi batch và kết quả giống hệt**
một lần chạy liên tục. Tái lập so sánh nhiều seed nên ưu tiên các run liên tục.

Tổng hợp sau khi các run hoàn thành:

```bash
python experiments/exp32_report.py results --metric best_err --plot
```

Tạo `results/report/comparison.csv`, `comparison.json`, và `training_curves.png`
(khi có `--plot`, cần matplotlib). CSV/JSON ghi mean, sample std, số run, danh sách
seed, lỗi paper và `gap_to_paper` tính bằng điểm phần trăm. Với một seed, std để trống.
Biểu đồ có sáu ô dataset/model, trung bình test error theo epoch qua các seed.

Report mặc định chỉ nhận run đủ 200 epoch, không giới hạn batch và khớp protocol;
cảnh báo các ô hoặc seed còn thiếu. Không chèn kết quả paper vào ô chưa chạy.
Nếu không có run hợp lệ, lệnh báo lỗi. Khi cần xem smoke/custom, thêm
`--include-nonpaper`; nhãn và lý do khác protocol vẫn hiện trong báo cáo.
Hai run trùng dataset/model/method/seed, hoặc các seed có cấu hình khoa học
khác nhau trong cùng ô, sẽ gây lỗi để người chạy chọn lại thư mục kết quả.

## Kiểm tra mã

```bash
python -m pytest tests -q
```

Các test không tải dữ liệu: kiểm tra số tham số và shape cho cả ba kiến trúc,
gradient trên model nhỏ, lưới cấu hình, lịch LR, luồng train/eval ERM/mixup
cho 10/100 lớp, checkpoint/resume, và bộ tổng hợp với dữ liệu log giả lập.
Kiểm thử này không thay thế việc huấn luyện đủ 200 epoch trên CIFAR thật.
