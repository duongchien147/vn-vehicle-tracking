"""Export mô hình YOLO sang OpenVINO (FP32, hoặc INT8 lượng tử hoá sau huấn luyện bằng NNCF) để chạy nhanh trên CPU.

imgsz = đúng imgsz lúc train (đọc từ file trọng số). INT8 hiệu chỉnh bằng 300 ảnh ngẫu nhiên của tập TRAIN
(vtrack.quantization: không lấy val vì val Hà Nội là tập dev dùng để quyết định có nhận bản INT8 không).
Kết quả: thư mục <weights>_openvino_model hoặc <weights>_int8_openvino_model cạnh file trọng số;
danh sách ảnh hiệu chỉnh + cấu hình đã dùng ở outputs/openvino_export/<tên>/.

Ví dụ:
    python scripts/export_openvino.py --weights weights/yolo26n_vn_v5_best.pt --int8
    python scripts/export_openvino.py --weights weights/yolo26s_vn_v5_best.pt
"""

import argparse
import json
from pathlib import Path

import yaml

from vtrack.models import load_model, training_imgsz
from vtrack.quantization import CALIBRATION_IMAGES, sample_calibration_images, write_calibration_dataset

DATASET_DIR = Path("data/processed/vn_vehicles")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True, help="File .pt cần export")
    parser.add_argument("--int8", action="store_true", help="Lượng tử hoá INT8 (cần ảnh hiệu chỉnh), mặc định FP32")
    parser.add_argument("--calibration-images", type=int, default=CALIBRATION_IMAGES)
    parser.add_argument("--seed", type=int, default=0, help="Seed chọn ảnh hiệu chỉnh")
    parser.add_argument("--out-dir", default="outputs/openvino_export")
    args = parser.parse_args()

    weights = Path(args.weights)
    model = load_model(str(weights))
    imgsz = training_imgsz(model)
    if imgsz is None:
        raise SystemExit(f"Không đọc được imgsz lúc train từ {weights}")

    out_dir = Path(args.out_dir) / (weights.stem + ("_int8" if args.int8 else "_fp32"))
    out_dir.mkdir(parents=True, exist_ok=True)
    export_args: dict = {"format": "openvino", "imgsz": imgsz}
    config: dict = {"weights": str(weights), "imgsz": imgsz, "precision": "int8" if args.int8 else "fp32"}
    if args.int8:
        names = yaml.safe_load((DATASET_DIR / "data.yaml").read_text(encoding="utf-8"))["names"]
        images = sample_calibration_images(DATASET_DIR / "train" / "images", args.calibration_images, args.seed)
        export_args.update(quantize=8, data=str(write_calibration_dataset(images, names, out_dir)))
        config.update(calibration_split="train", calibration_images=len(images), seed=args.seed)
        print(f"Hiệu chỉnh INT8 bằng {len(images)} ảnh train (seed {args.seed})")

    exported = model.export(**export_args)
    config["exported"] = str(exported)
    (out_dir / "export_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Đã export: {exported}")


if __name__ == "__main__":
    main()
