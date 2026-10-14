"""Ghi kết quả tracking thành nhãn sơ bộ định dạng "CVAT for video 1.1" (XML) để người gán nhãn sửa trong CVAT.

Chỉ ghi keyframe thưa (mỗi `keyframe_every` frame) — CVAT tự nội suy tuyến tính giữa hai keyframe, nên người
gán chỉ phải chỉnh ở keyframe. Không dùng định dạng MOT để nhập: bộ nhập MOT của CVAT coi frame bị thiếu giữa hai
dòng là xe "biến mất" (outside), nên keyframe thưa sẽ thành track đứt đoạn.
"""

from collections import defaultdict
from xml.sax.saxutils import quoteattr

from vtrack.mot_format import MotBox


def tracks_to_cvat_xml(
    boxes: list[MotBox], track_labels: dict[int, str], keyframe_every: int = 10, num_frames: int | None = None,
) -> str:
    """`boxes` có frame đánh từ 1 (MOTChallenge) -> XML CVAT đánh từ 0. `track_labels`: track_id -> tên nhãn CVAT.

    Mỗi track: keyframe ở frame đầu, mỗi `keyframe_every` frame, frame cuối của mỗi đoạn liên tục; nếu track
    vắng mặt giữa chừng hoặc kết thúc thì thêm box `outside="1"` ngay sau đó (đúng cách CVAT đánh dấu xe rời cảnh).
    `num_frames` (số frame của clip): không ghi box `outside` sau frame cuối, vì CVAT từ chối frame không tồn tại.
    """
    by_track: dict[int, list[MotBox]] = defaultdict(list)
    for box in boxes:
        by_track[box.track_id].append(box)

    lines = ['<?xml version="1.0" encoding="utf-8"?>', "<annotations>", "  <version>1.1</version>"]
    for cvat_id, track_id in enumerate(sorted(by_track)):
        track = sorted(by_track[track_id], key=lambda b: b.frame)
        lines.append(f'  <track id="{cvat_id}" label={quoteattr(track_labels[track_id])} source="auto">')
        for box, outside in _keyframes(track, keyframe_every):
            if num_frames is not None and box.frame > num_frames:
                continue
            lines.append("    " + _box_xml(box, outside))
        lines.append("  </track>")
    lines.append("</annotations>")
    return "\n".join(lines) + "\n"


def _keyframes(track: list[MotBox], keyframe_every: int) -> list[tuple[MotBox, bool]]:
    """(box, outside) cần ghi cho một track đã sắp theo frame."""
    keyframes: list[tuple[MotBox, bool]] = []
    segment_start = track[0].frame
    for i, box in enumerate(track):
        next_box = track[i + 1] if i + 1 < len(track) else None
        segment_ends = next_box is None or next_box.frame != box.frame + 1
        if box.frame == segment_start or (box.frame - segment_start) % keyframe_every == 0 or segment_ends:
            keyframes.append((box, False))
        if segment_ends:
            # xe vắng mặt từ frame kế tiếp (đến khi đoạn sau bắt đầu, hoặc hết track)
            keyframes.append((MotBox(box.frame + 1, box.track_id, box.left, box.top, box.width, box.height), True))
            if next_box is not None:
                segment_start = next_box.frame
    return keyframes


def _box_xml(box: MotBox, outside: bool) -> str:
    return (
        f'<box frame="{box.frame - 1}" outside="{int(outside)}" occluded="0" keyframe="1" '
        f'xtl="{box.left:.2f}" ytl="{box.top:.2f}" xbr="{box.left + box.width:.2f}" '
        f'ybr="{box.top + box.height:.2f}" z_order="0"></box>'
    )
