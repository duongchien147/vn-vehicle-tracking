import numpy as np

from vtrack.display_ids import CountedClasses, DisplayIds


def test_display_ids_are_sequential_in_order_of_first_appearance_and_stable():
    ids = DisplayIds()
    assert [ids.get(t) for t in (2719, 35, 2719, 4000, 35)] == [1, 2, 1, 3, 2]


def test_label_mask_skips_small_far_boxes():
    import numpy as np

    from vtrack.display_ids import label_mask

    xyxy = np.array([[0, 0, 10, 20], [0, 0, 30, 50]], dtype=float)  # cao 20 px và 50 px
    assert label_mask(xyxy, frame_height=1000, min_fraction=0.04).tolist() == [False, True]


def test_counted_classes_keep_class_at_first_crossing():
    counted = CountedClasses()
    counted.lock(7, 2)  # xe 7 được đếm là bus (2)
    counted.lock(7, 3)  # lần cắt vạch sau không đổi lớp đã khoá

    shown = counted.shown(np.array([7, 9]), np.array([3, 1]))

    assert shown.tolist() == [2, 1]  # xe 7 giữ bus dù bỏ phiếu đã thành truck; xe 9 chưa đếm giữ lớp hiện tại
