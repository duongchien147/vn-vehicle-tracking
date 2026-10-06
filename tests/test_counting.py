import numpy as np

from vtrack.counting import TrackClassVoter


def test_voter_keeps_majority_class_when_label_flickers():
    voter = TrackClassVoter()
    # Track 7 là car (1) ở hầu hết các frame, thỉnh thoảng bị nhận nhầm thành truck (3).
    for cls in [1, 1, 3, 1, 3, 1]:
        voted = voter.update(np.array([7]), np.array([cls]), np.array([0.8]))
    assert voted[0] == 1
    assert voter.class_of(7) == 1


def test_voter_weights_by_confidence():
    voter = TrackClassVoter()
    voter.update(np.array([1]), np.array([2]), np.array([0.9]))
    voted = voter.update(np.array([1]), np.array([3]), np.array([0.3]))
    assert voted[0] == 2


def test_voter_tracks_are_independent():
    voter = TrackClassVoter()
    voted = voter.update(np.array([1, 2]), np.array([0, 1]), np.array([0.5, 0.5]))
    assert voted.tolist() == [0, 1]
    assert voter.class_of(3) is None
