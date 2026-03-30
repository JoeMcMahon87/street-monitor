from street_monitor.config import TrackingConfig
from street_monitor.detector import Detection
from street_monitor.tracker import CentroidTracker


def _det(x1, y1, x2, y2, cls="car"):
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    return Detection(bbox=(x1, y1, x2, y2), class_id=2,
                     class_name=cls, confidence=0.9, centroid=(cx, cy))


def _cfg():
    return TrackingConfig(max_disappeared=5, max_distance=80)


def test_register_new_track():
    tracker = CentroidTracker(_cfg())
    det = _det(100, 100, 200, 200)
    tracks = tracker.update([det], frame_number=0)
    assert len(tracks) == 1
    track = list(tracks.values())[0]
    assert track.class_name == "car"
    assert track.frames_seen == 1
    assert not track.confirmed


def test_track_becomes_confirmed_after_3_frames():
    tracker = CentroidTracker(_cfg())
    for i in range(3):
        tracks = tracker.update([_det(100 + i, 100, 200 + i, 200)], frame_number=i)
    track = list(tracks.values())[0]
    assert track.confirmed


def test_track_disappears_and_is_removed():
    tracker = CentroidTracker(_cfg())
    tracker.update([_det(100, 100, 200, 200)], frame_number=0)
    # No detections for max_disappeared + 1 frames
    for i in range(1, 8):
        tracks = tracker.update([], frame_number=i)
    assert len(tracks) == 0


def test_two_vehicles_tracked_separately():
    tracker = CentroidTracker(_cfg())
    d1 = _det(100, 100, 200, 200)
    d2 = _det(500, 100, 600, 200)
    tracks = tracker.update([d1, d2], frame_number=0)
    assert len(tracks) == 2
    ids = {t.track_id for t in tracks.values()}
    assert len(ids) == 2


def test_vehicle_moves_and_keeps_same_id():
    tracker = CentroidTracker(_cfg())
    tracks0 = tracker.update([_det(100, 100, 200, 200)], frame_number=0)
    orig_id = list(tracks0.keys())[0]
    # Move 30 pixels right
    tracks1 = tracker.update([_det(130, 100, 230, 200)], frame_number=1)
    assert list(tracks1.keys())[0] == orig_id
