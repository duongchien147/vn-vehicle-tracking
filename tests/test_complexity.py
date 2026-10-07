import torch

from vtrack.complexity import count_gflops, count_params_m, yolo_input_hw


def test_yolo_input_is_minimal_letterbox_padded_to_stride():
    assert yolo_input_hw(960, (1080, 1920)) == (544, 960)  # 540 đệm lên bội số 32
    assert yolo_input_hw(640, (1080, 1920)) == (384, 640)  # 360 -> 384


def test_count_gflops_uses_two_flops_per_multiply_add():
    conv = torch.nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False)
    # 8 x 8 vị trí x 8 kênh ra x (3 x 3 x 3) phép nhân-cộng, mỗi phép = 2 FLOPs
    assert count_gflops(conv, torch.zeros(1, 3, 8, 8)) * 1e9 == 2 * 8 * 8 * 8 * 27
    assert count_params_m(conv) * 1e6 == 8 * 27
