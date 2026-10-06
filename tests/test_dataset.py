import numpy as np

from vtrack.dataset import (
    dhash,
    dhash_bits,
    group_near_duplicates,
    has_min_boxes,
    merge_riders_into_motorbikes,
    nearest_hamming_distances,
    remap_label_lines,
    sample_items,
    split_by_group,
    visdrone_to_yolo_lines,
)


def test_visdrone_conversion_normalizes_and_skips_ignored():
    rows = [
        "100,50,20,40,1,10,0,0",  # motor (id gốc 10 -> YOLO 9), tâm (110, 70)
        "0,0,50,50,0,4,0,0",  # score 0 = vùng bỏ qua
        "10,10,5,5,1,11,0,0",  # id 11 = "others" -> bỏ
        "10,10,0,5,1,4,0,0",  # box rỗng -> bỏ
    ]
    assert visdrone_to_yolo_lines(rows, image_width=200, image_height=100) == [
        "9 0.550000 0.700000 0.100000 0.400000"
    ]


def test_rider_is_merged_into_the_motorbike_box():
    motor = (9, 10.0, 50.0, 30.0, 80.0)  # xe: x 10–30, y 50–80
    rider = (1, 12.0, 30.0, 28.0, 65.0)  # người lái: nửa dưới chồng lên xe
    bystander = (1, 100.0, 30.0, 110.0, 65.0)  # người ở xa, không chồng lên xe nào
    car = (3, 200.0, 200.0, 260.0, 240.0)
    merged = merge_riders_into_motorbikes([motor, rider, bystander, car])
    assert merged[0] == (9, 10.0, 30.0, 30.0, 80.0)  # box xe mở rộng lên trên để bao người lái
    assert merged[1:] == [rider, bystander, car]  # các box khác giữ nguyên (lớp people sẽ bị bỏ khi đổi lớp)


def test_has_min_boxes_counts_project_classes():
    labels = ["0 0.1 0.1 0.1 0.1"] * 3 + ["1 0.5 0.5 0.2 0.2"]  # 3 xe máy, 1 ô tô
    assert has_min_boxes(labels, {"motorbike": 3})
    assert not has_min_boxes(labels, {"motorbike": 4})
    assert not has_min_boxes(labels, {"motorbike": 1, "bus": 1})
    assert has_min_boxes([], {})


def test_sample_items_is_reproducible_and_keeps_order():
    items = list(range(100))
    first = sample_items(items, 10, seed=1)
    assert first == sample_items(items, 10, seed=1)
    assert first == sorted(first) and len(first) == 10
    assert sample_items(items, None, seed=1) == items
    assert sample_items(items, 500, seed=1) == items


def test_remap_converts_source_ids_and_drops_unknown_classes():
    source_names = ["0", "1", "2", "3", "person"]
    class_map = {"0": "car", "1": "motorbike", "2": "truck", "3": "bus"}
    lines = ["0 0.5 0.5 0.1 0.1", "1 0.2 0.2 0.05 0.05", "4 0.3 0.3 0.1 0.2", ""]
    # dự án: motorbike=0, car=1, bus=2, truck=3
    assert remap_label_lines(lines, source_names, class_map) == [
        "1 0.5 0.5 0.1 0.1",
        "0 0.2 0.2 0.05 0.05",
    ]


def test_dhash_is_stable_under_small_brightness_change():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 200, size=(120, 160, 3), dtype=np.uint8)
    brighter = (image.astype(int) + 10).clip(0, 255).astype(np.uint8)
    other = rng.integers(0, 200, size=(120, 160, 3), dtype=np.uint8)
    distance = lambda a, b: bin(dhash(a) ^ dhash(b)).count("1")  # noqa: E731
    assert distance(image, brighter) <= 2
    assert distance(image, other) > 10


def test_group_near_duplicates_is_transitive():
    a = 0b0000
    b = 0b0001  # cách a 1 bit
    c = 0b0011  # cách b 1 bit, cách a 2 bit
    far = (1 << 64) - 1  # cách tất cả > 60 bit
    groups = group_near_duplicates([a, b, c, far], max_distance=1)
    assert groups[0] == groups[1] == groups[2]
    assert groups[3] != groups[0]


def test_split_keeps_groups_together_and_respects_ratios():
    group_ids = [i // 5 for i in range(100)]  # 20 nhóm, mỗi nhóm 5 ảnh
    splits = split_by_group(group_ids, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=0)
    for group in set(group_ids):
        assert len({splits[i] for i, g in enumerate(group_ids) if g == group}) == 1
    assert splits.count("train") == 70
    assert splits.count("val") == 15
    assert splits.count("test") == 15


def test_nearest_hamming_distances_finds_closest_reference():
    query = np.array([[1, 1, 0, 0], [0, 0, 0, 0]], dtype=bool)
    reference = np.array([[1, 1, 0, 1], [1, 1, 1, 1]], dtype=bool)

    assert nearest_hamming_distances(query, reference).tolist() == [1, 3]


def test_dhash_bits_is_stable_under_small_noise_and_differs_for_another_scene():
    rng = np.random.default_rng(0)
    scene = rng.integers(0, 255, (120, 160), dtype=np.uint8)
    noisy = np.clip(scene.astype(int) + rng.integers(-3, 4, scene.shape), 0, 255).astype(np.uint8)
    other = rng.integers(0, 255, (120, 160), dtype=np.uint8)

    same = (dhash_bits(scene) != dhash_bits(noisy)).sum()
    different = (dhash_bits(scene) != dhash_bits(other)).sum()
    assert dhash_bits(scene).shape == (256,)
    assert same < different
