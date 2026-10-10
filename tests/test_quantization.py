from pathlib import Path

import yaml

from vtrack.quantization import sample_calibration_images, write_calibration_dataset


def make_images(directory: Path, count: int) -> None:
    for i in range(count):
        (directory / f"img_{i:03d}.jpg").write_bytes(b"")
    (directory / "notes.txt").write_text("không phải ảnh", encoding="utf-8")


def test_sample_is_reproducible_and_only_images(tmp_path):
    make_images(tmp_path, 20)
    first = sample_calibration_images(tmp_path, count=5, seed=0)
    assert first == sample_calibration_images(tmp_path, count=5, seed=0)
    assert len(first) == 5 and all(p.suffix == ".jpg" for p in first)
    assert first != sample_calibration_images(tmp_path, count=5, seed=1)


def test_sample_takes_all_when_too_few(tmp_path):
    make_images(tmp_path, 3)
    assert len(sample_calibration_images(tmp_path, count=300)) == 3


def test_calibration_yaml_points_val_to_list(tmp_path):
    make_images(tmp_path, 4)
    images = sample_calibration_images(tmp_path, count=2)
    data = yaml.safe_load(write_calibration_dataset(images, {0: "motorbike"}, tmp_path / "out").read_text("utf-8"))
    listed = Path(data["val"]).read_text("utf-8").split()
    assert listed == [p.resolve().as_posix() for p in images]
    assert data["names"] == {0: "motorbike"}
