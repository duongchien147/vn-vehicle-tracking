import xml.etree.ElementTree as ET

from vtrack.cvat_format import tracks_to_cvat_xml
from vtrack.mot_format import MotBox


def parse_boxes(xml: str) -> list[tuple[int, int, int]]:
    """(id track CVAT, frame, outside) của mọi box trong XML."""
    root = ET.fromstring(xml)
    return [
        (int(track.get("id", "")), int(box.get("frame", "")), int(box.get("outside", "")))
        for track in root.iter("track") for box in track.iter("box")
    ]


def test_continuous_track_gets_sparse_keyframes_and_closing_outside_box():
    track = [MotBox(f, 7, 10.0 * f, 0, 20, 20) for f in range(1, 26)]  # frame 1..25 -> CVAT 0..24

    boxes = parse_boxes(tracks_to_cvat_xml(track, {7: "car"}, keyframe_every=10))

    assert boxes == [(0, 0, 0), (0, 10, 0), (0, 20, 0), (0, 24, 0), (0, 25, 1)]


def test_gap_in_track_is_marked_outside_until_it_reappears():
    track = [MotBox(f, 3, 0, 0, 20, 20) for f in (1, 2, 3, 8, 9)]

    boxes = parse_boxes(tracks_to_cvat_xml(track, {3: "motorbike"}, keyframe_every=10))

    assert boxes == [(0, 0, 0), (0, 2, 0), (0, 3, 1), (0, 7, 0), (0, 8, 0), (0, 9, 1)]


def test_label_and_coordinates_are_written_in_corner_format():
    xml = tracks_to_cvat_xml([MotBox(1, 1, 10, 20, 30, 40)], {1: "bus"})
    track = ET.fromstring(xml).find("track")

    assert track is not None and track.get("label") == "bus"
    box = track.find("box")
    assert box is not None
    assert (box.get("xtl"), box.get("ytl"), box.get("xbr"), box.get("ybr")) == ("10.00", "20.00", "40.00", "60.00")
