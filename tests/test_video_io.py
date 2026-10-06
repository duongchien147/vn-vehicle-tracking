import pytest

from vtrack.video_io import safe_upload_name

ALLOWED = ("mp4", "avi", "mov")


@pytest.mark.parametrize("name, expected", [
    ("clip.mp4", "clip.mp4"),
    ("Clip.MOV", "Clip.MOV"),
    ("../../secret.mp4", "secret.mp4"),
    (r"..\..\Windows\evil.mp4", "evil.mp4"),
    ("C:/Users/a/x.avi", "x.avi"),
    ("script.py", None),
    (".hidden.mp4", None),
    ("..", None),
    ("", None),
])
def test_safe_upload_name_strips_directories_and_rejects_bad_names(name, expected):
    assert safe_upload_name(name, ALLOWED) == expected
