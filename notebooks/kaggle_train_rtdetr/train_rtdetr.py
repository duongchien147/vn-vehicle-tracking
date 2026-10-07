"""Fine-tune RT-DETRv2 (họ Transformer) trên dataset xe Việt Nam — kernel dạng script chạy trên GPU Kaggle.

Thí nghiệm so sánh họ mô hình: YOLO26 (CNN) và RT-DETRv2 (Transformer) được train trên CÙNG dữ liệu (dataset v5),
CÙNG imgsz 960, mỗi họ dùng công thức train khuyến nghị của chính nó (chuẩn khi so sánh kiến trúc trong bài báo).
Công thức ở đây chép theo cấu hình chính thức của tác giả (github.com/lyuwenyu/RT-DETR, rtdetrv2_pytorch/configs/
rtdetrv2/rtdetrv2_r18vd_120e_coco.yml + include/optimizer.yml + include/dataloader.yml); chỗ nào khác đều ghi lý do.
Mô hình lấy từ Hugging Face (`PekingU/rtdetr_v2_r18vd`, Apache 2.0, pretrained COCO) qua thư viện transformers.

Đẩy lên:  python scripts/kaggle_train.py push --notebook-dir notebooks/kaggle_train_rtdetr
Chạy thử trên máy local (CPU, vài ảnh, imgsz nhỏ — chỉ để bắt lỗi code): python notebooks/kaggle_train_rtdetr/train_rtdetr.py

Kết quả trong /kaggle/working/:
  <RUN_NAME>_best/          trọng số EMA tốt nhất theo mAP50-95 val (save_pretrained: config + model.safetensors)
  <RUN_NAME>_epochs.csv     loss, lr, mAP val theo từng epoch
  <RUN_NAME>_test_metrics.csv  mAP trên tập test Roboflow theo lớp
  <RUN_NAME>_run_config.json   cấu hình + phiên bản thư viện + GPU
"""

import builtins
import contextlib
import csv
import faulthandler
import io
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

# Kernel dạng script ghi stdout có bộ đệm -> log trên Kaggle chỉ hiện khi kết thúc; flush ngay để theo dõi được tiến độ.
# Ghi thêm mọi dòng ra file log trong thư mục output: hai lần chạy trước chết giữa chừng mà log Kaggle trống trơn.
_LOG_FILE = None


def print(*args, **kwargs) -> None:  # noqa: A001
    builtins.print(*args, **kwargs, flush=True)
    if _LOG_FILE is not None:
        builtins.print(*args, file=_LOG_FILE, flush=True)


# ===================================== 1. Cấu hình thí nghiệm =====================================
HF_MODEL = "PekingU/rtdetr_v2_r18vd"  # bản nhỏ nhất (ResNet-18): gần cỡ YOLO26n/s nhất trong họ RT-DETRv2
# Ngân sách train và luật dừng GIỐNG HỆT YOLO26 (notebooks/kaggle_train): tối đa 100 epoch, dừng sớm khi mAP50-95
# val không cải thiện sau 20 epoch liên tiếp, giữ checkpoint EMA có mAP50-95 val cao nhất (= fitness của ultralytics)
EPOCHS = 100
PATIENCE = 20
# Tắt augmentation mạnh (PhotometricDistort, ZoomOut, IoUCrop) ở 10 epoch cuối để mô hình quen ảnh thật.
# ĐỒNG NHẤT với close_mosaic=10 của ultralytics (chính thức RT-DETRv2: 3 epoch cuối / 120). Chỉ xảy ra nếu chưa dừng sớm.
NO_AUG_EPOCHS = 10
IMGSZ = 960  # = imgsz của YOLO26 v5 -> hai họ nhìn ảnh cùng độ phân giải (chính thức: 640)
BATCH = 8
ACCUMULATE = 2  # batch hiệu dụng 8 x 2 = 16 = chính thức (4 GPU x 4 ảnh). Batch 8 đo được 7.4 GB -> batch 16 cần ~15 GB, sát giới hạn T4
LR = 1e-4
WEIGHT_DECAY = 1e-4  # lớp norm/bn không dùng weight decay (như chính thức)
BETAS = (0.9, 0.999)
# Tăng lr tuyến tính trong 3 epoch đầu — ĐỒNG NHẤT với warmup_epochs=3 của ultralytics. Chính thức RT-DETRv2 là
# 2000 bước, nhưng đó là ~0.3 epoch trên COCO (118k ảnh); trên 4.714 ảnh 2000 bước thành ~7 epoch. Sau warmup lr giữ nguyên.
WARMUP_EPOCHS = 3
# EMA: công thức và tham số y hệt ModelEMA của ultralytics (decay 0.9999, tau 2000 bước) — vốn trùng với chính thức
EMA_DECAY = 0.9999
EMA_WARMUPS = 2000
CLIP_MAX_NORM = 0.1
LOG_EVERY_STEPS = 50  # in tiến độ trong epoch để biết kernel đang chạy hay bị treo
TIME_LIMIT_HOURS = 11.0  # Kaggle cắt phiên ở 12 giờ: dừng sớm để còn thời gian chấm test và lưu kết quả
SEED = 42
SMOKE_TEST = False  # True: 2 epoch trên 5% dữ liệu để kiểm tra pipeline, đo thời gian 1 epoch và bộ nhớ GPU
TRANSFORMERS_VERSION = "5.18.0"  # cùng phiên bản với máy local để trọng số nạp lại được
RUN_TAG = "v5"

# Chạy ngoài Kaggle = kiểm tra code trên CPU: vài ảnh, ảnh nhỏ, 1 epoch
LOCAL_TEST = not Path("/kaggle").exists()
if SMOKE_TEST:
    EPOCHS, NO_AUG_EPOCHS = 2, 1
if LOCAL_TEST:
    EPOCHS, NO_AUG_EPOCHS, IMGSZ, BATCH, ACCUMULATE = 2, 1, 320, 2, 2
# Watchdog: quá hạn mà chưa xong thì in stack trace của mọi luồng (biết treo ở dòng nào) rồi thoát, không chiếm GPU tiếp
WATCHDOG_HOURS = 0.75 if SMOKE_TEST else TIME_LIMIT_HOURS + 0.75
faulthandler.dump_traceback_later(WATCHDOG_HOURS * 3600, exit=True)
RUN_NAME = f"rtdetrv2_r18_{RUN_TAG}_e{EPOCHS}_{IMGSZ}" + ("_smoke" if SMOKE_TEST else "") + ("_local" if LOCAL_TEST else "")
OUTPUT_DIR = Path("/kaggle/working") if not LOCAL_TEST else Path("outputs/rtdetr_local_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = open(OUTPUT_DIR / f"{RUN_NAME}_train.log", "a", encoding="utf-8")  # noqa: SIM115 (mở suốt quá trình chạy)
# Tiến trình chết vì lỗi tầng C (segfault, abort...) -> vẫn ghi được stack trace của mọi luồng vào file log
faulthandler.enable(file=_LOG_FILE, all_threads=True)


def _log_uncaught(exc_type, exc, tb) -> None:
    """Lỗi Python không bắt được -> ghi cả vào file log (ngoài stderr như bình thường)."""
    traceback.print_exception(exc_type, exc, tb, file=_LOG_FILE)
    if _LOG_FILE is not None:
        _LOG_FILE.flush()
    sys.__excepthook__(exc_type, exc, tb)


sys.excepthook = _log_uncaught
print("RUN_NAME =", RUN_NAME)

def available_ram_gb() -> float:
    """RAM còn trống của máy (Linux /proc/meminfo); để biết có sắp hết RAM không khi tiến trình bị giết."""
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return float("nan")
    for line in meminfo.read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1e6
    return float("nan")


# ===================================== 2. Thư viện và GPU =====================================
PIP_TIMEOUT_S = 900  # cài thư viện treo (mạng) thì báo lỗi rõ ràng thay vì chiếm GPU tới hết phiên
if not LOCAL_TEST:
    print("Cài thư viện...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", f"transformers=={TRANSFORMERS_VERSION}",
                    "pycocotools"], check=True, timeout=PIP_TIMEOUT_S)
    print("Cài xong")

import numpy as np  # noqa: E402  (import sau khi cài thư viện)
import torch  # noqa: E402
import transformers  # noqa: E402
from pycocotools.coco import COCO  # noqa: E402
from pycocotools.cocoeval import COCOeval  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from torchvision import tv_tensors  # noqa: E402
from torchvision.io import ImageReadMode, decode_image  # noqa: E402
from torchvision.ops import box_convert  # noqa: E402
from torchvision.transforms import v2 as T  # noqa: E402
from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not LOCAL_TEST and DEVICE.type != "cuda":
    raise RuntimeError("Không có GPU: kiểm tra kernel-metadata.json (enable_gpu: true).")
GPU_NAME = torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "cpu"
print(f"Thiết bị: {GPU_NAME} | torch {torch.__version__} | transformers {transformers.__version__}")

# Seed + chế độ tất định ĐỒNG NHẤT với ultralytics (deterministic=True, utils/torch_utils.init_seeds). warn_only:
# vài phép của deformable attention không có bản tất định trên GPU -> chỉ cảnh báo, như ultralytics.
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = str(SEED)

# ===================================== 3. Dataset =====================================
if LOCAL_TEST:
    DATASET = Path("data/processed/vn_vehicles")
else:
    data_yamls = sorted(Path("/kaggle/input").glob("**/data.yaml"))
    assert data_yamls, "Không thấy data.yaml trong /kaggle/input: kiểm tra dataset vn-vehicles-yolo đã được gắn."
    DATASET = data_yamls[0].parent
print("Dataset:", DATASET)
# Tên lớp theo id đọc từ chính data.yaml mà YOLO26 dùng -> hai họ mô hình học cùng một ánh xạ id -> lớp
import yaml  # noqa: E402

CLASS_NAMES = [name for _, name in sorted(yaml.safe_load((DATASET / "data.yaml").read_text())["names"].items())]
print("Lớp:", CLASS_NAMES)


def train_transforms(strong: bool) -> T.Compose:
    """Augmentation chính thức của RT-DETRv2. `strong=False` cho các epoch cuối (chỉ lật ngang + resize)."""
    ops = []
    if strong:
        ops += [
            T.RandomPhotometricDistort(p=0.5),
            T.RandomZoomOut(fill=0),
            T.RandomApply([T.RandomIoUCrop()], p=0.8),  # chính thức: RandomIoUCrop(p=0.8)
            T.SanitizeBoundingBoxes(min_size=1),
        ]
    ops += [
        T.RandomHorizontalFlip(),
        # Resize thẳng về hình vuông (không giữ tỉ lệ, không đệm viền) — đúng cách RT-DETR train/suy luận
        T.Resize((IMGSZ, IMGSZ)),
        T.SanitizeBoundingBoxes(min_size=1),
        T.ToDtype(torch.float32, scale=True),  # [0, 255] -> [0, 1]; RT-DETR không chuẩn hoá mean/std
    ]
    return T.Compose(ops)


EVAL_TRANSFORMS = T.Compose([T.Resize((IMGSZ, IMGSZ)), T.ToDtype(torch.float32, scale=True)])


class YoloDetectionDataset(Dataset):
    """Ảnh + nhãn YOLO (lớp cx cy w h, toạ độ chuẩn hoá) -> (ảnh float [0,1], nhãn theo định dạng transformers:
    `class_labels` và `boxes` cx cy w h chuẩn hoá theo ảnh đã resize)."""

    def __init__(self, split_dir: Path, transforms: T.Compose, fraction: float = 1.0, limit: int | None = None):
        self.images = sorted((split_dir / "images").glob("*.jpg"))
        if fraction < 1.0:
            self.images = random.Random(SEED).sample(self.images, max(1, int(len(self.images) * fraction)))
        if limit:
            self.images = self.images[:limit]
        self.labels_dir = split_dir / "labels"
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image_path = self.images[index]
        image = tv_tensors.Image(decode_image(str(image_path), mode=ImageReadMode.RGB))
        height, width = image.shape[-2:]
        label_path = self.labels_dir / f"{image_path.stem}.txt"
        rows = [line.split() for line in label_path.read_text().splitlines() if line.strip()] if label_path.exists() else []
        classes = torch.tensor([int(r[0]) for r in rows], dtype=torch.int64)
        cxcywh = torch.tensor([[float(v) for v in r[1:5]] for r in rows], dtype=torch.float32).reshape(-1, 4)
        cxcywh = cxcywh * torch.tensor([width, height, width, height])
        # pyright đọc nhầm chữ ký __init__ của torch.Tensor cho lớp con tv_tensors (lỗi stub, code đúng theo tài liệu)
        boxes = tv_tensors.BoundingBoxes(box_convert(cxcywh, "cxcywh", "xyxy"), format=tv_tensors.BoundingBoxFormat.XYXY,  # pyright: ignore[reportCallIssue]
                                         canvas_size=(int(height), int(width)))
        image, target = self.transforms(image, {"boxes": boxes, "labels": classes})
        out_h, out_w = image.shape[-2:]
        normalized = box_convert(target["boxes"].as_subclass(torch.Tensor), "xyxy", "cxcywh")
        normalized = normalized / torch.tensor([out_w, out_h, out_w, out_h], dtype=torch.float32)
        return image.as_subclass(torch.Tensor), {"class_labels": target["labels"], "boxes": normalized}

    def set_transforms(self, transforms: T.Compose) -> None:
        self.transforms = transforms


def collate(batch):
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)


fraction = 0.05 if SMOKE_TEST else 1.0
limit = 8 if LOCAL_TEST else None
train_set = YoloDetectionDataset(DATASET / "train", train_transforms(strong=True), fraction, limit)
val_set = YoloDetectionDataset(DATASET / "val", EVAL_TRANSFORMS, fraction, limit)
test_set = YoloDetectionDataset(DATASET / "test", EVAL_TRANSFORMS, fraction, limit)
WORKERS = 0 if LOCAL_TEST else 4
train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True, num_workers=WORKERS, collate_fn=collate,
                          drop_last=True, pin_memory=DEVICE.type == "cuda", persistent_workers=False)
print(f"Ảnh: train {len(train_set)}, val {len(val_set)}, test {len(test_set)}")

# ===================================== 4. Mô hình, tối ưu, EMA =====================================
ID2LABEL = dict(enumerate(CLASS_NAMES))
model = RTDetrV2ForObjectDetection.from_pretrained(
    HF_MODEL, id2label=ID2LABEL, label2id={v: k for k, v in ID2LABEL.items()},
    ignore_mismatched_sizes=True,  # đầu phân lớp 80 lớp COCO -> 4 lớp: khởi tạo lại, phần còn lại giữ pretrained
)
# Gọi thẳng nn.Module.to: kiểu khai báo của PreTrainedModel.to trong transformers 5 làm pyright báo lỗi oan
torch.nn.Module.to(model, DEVICE)
# Lưu kèm image processor đặt đúng imgsz: suy luận (local) đọc kích thước từ đây
processor = RTDetrImageProcessor.from_pretrained(HF_MODEL, size={"height": IMGSZ, "width": IMGSZ})

# Chính thức: tham số của lớp norm/bn không dùng weight decay
no_decay = [p for n, p in model.named_parameters() if p.requires_grad and ("norm" in n or "bn" in n)]
decay = [p for n, p in model.named_parameters() if p.requires_grad and not ("norm" in n or "bn" in n)]
optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": WEIGHT_DECAY}, {"params": no_decay, "weight_decay": 0.0}],
                              lr=LR, betas=BETAS)
# Đếm theo bước tối ưu (sau cộng dồn gradient); tối thiểu 100 bước như ultralytics
warmup_steps = max(round(WARMUP_EPOCHS * len(train_loader) / ACCUMULATE), 100)
warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, (step + 1) / warmup_steps))
use_amp = DEVICE.type == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)


class ModelEMA:
    """Trung bình trượt mũ của trọng số (như chính thức): hệ số tăng dần từ 0 tới `decay` trong `warmups` bước,
    để giai đoạn đầu EMA bám sát mô hình đang học thay vì giữ trọng số pretrained."""

    def __init__(self, source: RTDetrV2ForObjectDetection, decay: float, warmups: int):
        self.module = deepcopy(source).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay, self.warmups, self.updates = decay, warmups, 0

    @torch.no_grad()
    def update(self, source: torch.nn.Module) -> None:
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / self.warmups))
        current = source.state_dict()
        for key, value in self.module.state_dict().items():
            if value.dtype.is_floating_point:
                value.mul_(d).add_(current[key].detach(), alpha=1 - d)


ema = ModelEMA(model, EMA_DECAY, EMA_WARMUPS)


# ===================================== 5. Chấm mAP =====================================
# Tối đa 300 phát hiện/ảnh = số truy vấn của RT-DETR = max_det mặc định của ultralytics. Không dùng mức 100 mặc
# định của COCO: frame đông có ~110–140 xe, cắt ở 100 làm điểm thấp oan.
MAX_DETECTIONS = 300


def coco_map(truths: list[dict], predictions: list[dict]) -> dict:
    """mAP bằng pycocotools (bộ chấm chính thức của COCO), cùng cách với vtrack/coco_eval.py trên máy local.
    Box dạng xyxy pixel. Không dùng summarize()/stats: hàm đó viết cứng maxDets=100 và trả -1 khi đổi maxDets."""
    categories = [{"id": i + 1, "name": n} for i, n in enumerate(CLASS_NAMES)]
    images, annotations, detections = [], [], []
    for image_id, (truth, prediction) in enumerate(zip(truths, predictions), start=1):
        images.append({"id": image_id, "width": IMGSZ, "height": IMGSZ})
        for (x1, y1, x2, y2), label in zip(truth["boxes"].tolist(), truth["labels"].tolist()):
            annotations.append({"id": len(annotations) + 1, "image_id": image_id, "category_id": label + 1,
                                "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1), "iscrowd": 0})
        for (x1, y1, x2, y2), label, score in zip(prediction["boxes"].tolist(), prediction["labels"].tolist(),
                                                  prediction["scores"].tolist()):
            detections.append({"image_id": image_id, "category_id": label + 1, "bbox": [x1, y1, x2 - x1, y2 - y1],
                               "score": score})
    if not detections:
        return {"mAP50": 0.0, "mAP50-95": 0.0}
    with contextlib.redirect_stdout(io.StringIO()):  # pycocotools in rất nhiều dòng
        coco_truth = COCO()
        # stub kiểu của pycocotools quá hẹp (dataset, loadRes nhận cả list kết quả) -> cast như vtrack/coco_eval.py
        coco_truth.dataset = cast(Any, {"images": images, "annotations": annotations, "categories": categories})
        coco_truth.createIndex()
        evaluator = COCOeval(coco_truth, coco_truth.loadRes(cast(Any, detections)), "bbox")
        evaluator.params.maxDets = [1, 10, MAX_DETECTIONS]
        evaluator.evaluate()
        evaluator.accumulate()
    # precision: [IoU 0.50:0.05:0.95, mức recall, lớp, vùng diện tích (0 = tất cả), maxDets]; -1 = lớp không có nhãn
    precision = evaluator.eval["precision"][:, :, :, 0, -1]

    def mean_valid(values: np.ndarray) -> float:
        valid = values[values > -1]
        return float(valid.mean()) if valid.size else float("nan")

    return {"mAP50": mean_valid(precision[0]), "mAP50-95": mean_valid(precision),
            **{f"mAP50_{n}": mean_valid(precision[0, :, i]) for i, n in enumerate(CLASS_NAMES)},
            **{f"mAP50-95_{n}": mean_valid(precision[:, :, i]) for i, n in enumerate(CLASS_NAMES)}}


@torch.no_grad()
def evaluate(eval_model: torch.nn.Module, dataset: Dataset) -> dict:
    """Dự đoán trên cả tập (toạ độ pixel của ảnh đã resize IMGSZ x IMGSZ) rồi chấm mAP kiểu COCO."""
    loader = DataLoader(dataset, batch_size=BATCH * 2, shuffle=False, num_workers=WORKERS, collate_fn=collate)
    scale = torch.tensor([IMGSZ, IMGSZ, IMGSZ, IMGSZ], dtype=torch.float32)
    eval_model.eval()
    truths, predictions = [], []
    for images, targets in loader:
        with torch.autocast(DEVICE.type, dtype=torch.float16, enabled=use_amp):
            outputs = eval_model(pixel_values=images.to(DEVICE))
        # Hậu xử lý chính thức: sigmoid, lấy top-300 cặp (truy vấn, lớp) -> xyxy chuẩn hoá (không có NMS)
        for result in processor.post_process_object_detection(outputs, threshold=0.0):
            predictions.append({"boxes": result["boxes"].float().cpu() * scale, "scores": result["scores"].float().cpu(),
                                "labels": result["labels"].cpu()})
        truths += [{"boxes": box_convert(t["boxes"], "cxcywh", "xyxy") * scale, "labels": t["class_labels"]} for t in targets]
    return coco_map(truths, predictions)


# ===================================== 6. Huấn luyện (train tiếp được qua nhiều phiên) =====================================
# Một phiên Kaggle tối đa 12 giờ, chỉ đủ ~70 epoch -> mỗi epoch lưu đủ trạng thái vào <RUN_NAME>_last.pt. Phiên sau gắn
# output phiên trước làm input (dataset) thì chạy tiếp đúng chỗ dừng, như một lần train liền mạch (cùng ngân sách 100
# epoch như YOLO26).
started = time.time()
best_dir = OUTPUT_DIR / f"{RUN_NAME}_best"
epoch_log = OUTPUT_DIR / f"{RUN_NAME}_epochs.csv"
last_checkpoint = OUTPUT_DIR / f"{RUN_NAME}_last.pt"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
best_fitness, best_epoch, step, start_epoch, previous_hours, stopped_early = -1.0, -1, 0, 0, 0.0, False
epoch_rows: list[dict] = []

RESUME_SEARCH_DIR = Path(os.environ.get("RTDETR_RESUME_DIR", "/kaggle/input"))
resume_files = sorted(RESUME_SEARCH_DIR.glob(f"**/{RUN_NAME}_last.pt")) if RESUME_SEARCH_DIR.exists() else []
if resume_files:
    state = torch.load(resume_files[0], map_location="cpu", weights_only=False)  # file do chính script này ghi
    model.load_state_dict(state["model"])
    ema.module.load_state_dict(state["ema"])
    ema.updates = state["ema_updates"]
    optimizer.load_state_dict(state["optimizer"])
    scaler.load_state_dict(state["scaler"])
    warmup.load_state_dict(state["warmup"])
    best_fitness, best_epoch, step = state["best_fitness"], state["best_epoch"], state["step"]
    epoch_rows, previous_hours, start_epoch = state["epoch_rows"], state["train_hours"], state["epoch"] + 1
    random.setstate(state["rng_python"])
    np.random.set_state(state["rng_numpy"])
    torch.set_rng_state(state["rng_torch"])
    if DEVICE.type == "cuda" and state["rng_cuda"] is not None:
        torch.cuda.set_rng_state_all(state["rng_cuda"])
    shutil.copytree(resume_files[0].parent / best_dir.name, best_dir, dirs_exist_ok=True)
    print(f"Train tiếp từ epoch {start_epoch} ({resume_files[0]}); tốt nhất đến giờ: epoch {best_epoch}, "
          f"mAP50-95 val {best_fitness:.4f}")

for epoch in range(start_epoch, EPOCHS):
    strong = epoch < EPOCHS - NO_AUG_EPOCHS
    train_set.set_transforms(train_transforms(strong=strong))
    if epoch == EPOCHS - NO_AUG_EPOCHS:
        print(f"Epoch {epoch}: tắt augmentation mạnh")
    epoch_started = time.time()
    model.train()
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    for batch_index, (images, targets) in enumerate(train_loader):
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        with torch.autocast(DEVICE.type, dtype=torch.float16, enabled=use_amp):
            loss = model(pixel_values=images.to(DEVICE), labels=targets).loss
        scaler.scale(loss / ACCUMULATE).backward()
        total_loss += loss.item()
        if (batch_index + 1) % ACCUMULATE == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_MAX_NORM)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            warmup.step()
            ema.update(model)
            step += 1
            if step % LOG_EVERY_STEPS == 0 or step == 1:
                gpu_gb = torch.cuda.memory_allocated() / 1e9 if DEVICE.type == "cuda" else 0.0
                print(f"  bước {step} | loss {loss.item():.3f} | {time.time() - epoch_started:.0f} s từ đầu epoch"
                      f" | RAM trống {available_ram_gb():.1f} GB | GPU {gpu_gb:.1f} GB")

    val = evaluate(ema.module, val_set)
    fitness = val["mAP50-95"]  # = tiêu chí chọn best.pt của ultralytics 8.4 (fitness chỉ tính mAP50-95)
    if fitness > best_fitness:
        best_fitness, best_epoch = fitness, epoch
        ema.module.save_pretrained(best_dir)
        processor.save_pretrained(best_dir)
    epoch_seconds = time.time() - epoch_started
    row = {"epoch": epoch, "train_loss": total_loss / max(len(train_loader), 1), "lr": optimizer.param_groups[0]["lr"],
           "val_mAP50": val["mAP50"], "val_mAP50-95": val["mAP50-95"], "seconds": epoch_seconds,
           "gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if DEVICE.type == "cuda" else 0.0}
    epoch_rows.append(row)
    with epoch_log.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerows(epoch_rows)
    print(f"epoch {epoch:3d} | loss {row['train_loss']:.3f} | val mAP50 {val['mAP50']:.3f} mAP50-95 {val['mAP50-95']:.3f}"
          f" | {epoch_seconds / 60:.1f} phút | GPU {row['gpu_mem_gb']:.1f} GB")
    torch.save({
        "epoch": epoch, "model": model.state_dict(), "ema": ema.module.state_dict(), "ema_updates": ema.updates,
        "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "warmup": warmup.state_dict(),
        "best_fitness": best_fitness, "best_epoch": best_epoch, "step": step, "epoch_rows": epoch_rows,
        "train_hours": previous_hours + (time.time() - started) / 3600,
        "rng_python": random.getstate(), "rng_numpy": np.random.get_state(), "rng_torch": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if DEVICE.type == "cuda" else None,
    }, last_checkpoint)

    if epoch - best_epoch >= PATIENCE:
        print(f"Dừng sớm ở epoch {epoch}: mAP50-95 val không cải thiện sau {PATIENCE} epoch (tốt nhất: epoch {best_epoch})")
        break
    elapsed_hours = (time.time() - started) / 3600
    if elapsed_hours + 1.5 * epoch_seconds / 3600 > TIME_LIMIT_HOURS:
        print(f"Dừng ở epoch {epoch}: sắp hết giới hạn {TIME_LIMIT_HOURS} giờ của phiên Kaggle")
        stopped_early = True
        break

# ===================================== 7. Chấm test và lưu =====================================
best_model = RTDetrV2ForObjectDetection.from_pretrained(str(best_dir.resolve()))
torch.nn.Module.to(best_model, DEVICE)
test = evaluate(best_model, test_set)
with (OUTPUT_DIR / f"{RUN_NAME}_test_metrics.csv").open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["metric", "value"])
    writer.writerows((k, round(v, 4)) for k, v in test.items())
print("Test:", {k: round(v, 4) for k, v in test.items()})

run_config = {
    "run_name": RUN_NAME, "hf_model": HF_MODEL, "epochs": EPOCHS, "patience": PATIENCE, "epochs_done": len(epoch_rows),
    "classes": CLASS_NAMES,
    "stopped_by_time_limit": stopped_early, "best_epoch": best_epoch, "best_val_mAP50-95": best_fitness,
    "no_aug_epochs": NO_AUG_EPOCHS, "imgsz": IMGSZ, "batch": BATCH, "accumulate": ACCUMULATE, "lr": LR,
    "weight_decay": WEIGHT_DECAY, "warmup_epochs": WARMUP_EPOCHS, "warmup_steps": warmup_steps, "ema_decay": EMA_DECAY, "clip_max_norm": CLIP_MAX_NORM,
    "seed": SEED, "smoke_test": SMOKE_TEST, "dataset": str(DATASET), "gpu": GPU_NAME,
    "torch": torch.__version__, "transformers": transformers.__version__,
    "resumed_from_epoch": start_epoch if resume_files else None,
    "train_hours": round(previous_hours + (time.time() - started) / 3600, 2),
}
(OUTPUT_DIR / f"{RUN_NAME}_run_config.json").write_text(json.dumps(run_config, indent=2, ensure_ascii=False))
for path in sorted(OUTPUT_DIR.glob(f"{RUN_NAME}*")):
    print(path.name)
shutil.rmtree(OUTPUT_DIR / "__pycache__", ignore_errors=True)
