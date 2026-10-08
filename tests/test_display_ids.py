from vtrack.display_ids import DisplayIds


def test_display_ids_are_sequential_in_order_of_first_appearance_and_stable():
    ids = DisplayIds()
    assert [ids.get(t) for t in (2719, 35, 2719, 4000, 35)] == [1, 2, 1, 3, 2]


def test_label_mask_skips_small_far_boxes():
    import numpy as np

    from vtrack.display_ids import label_mask

    xyxy = np.array([[0, 0, 10, 20], [0, 0, 30, 50]], dtype=float)  # cao 20 px và 50 px
    assert label_mask(xyxy, frame_height=1000, min_fraction=0.04).tolist() == [False, True]
