"""Known 90 mm stereo rig and planar checkerboards, generated only for tests."""

from pathlib import Path
import tempfile

import cv2
import numpy as np

from app.stereo.calibration import BoardSettings, Observation


BOARD = BoardSettings(9, 6, 24, "mm")
SIZE = (960, 720)
CAMERA = np.array([[800., 0, 480], [0, 805, 360], [0, 0, 1]])
STEREO_R = cv2.Rodrigues(np.array([0.002, 0.006, -0.001]))[0]
STEREO_T = np.array([[-90.], [0.], [0.]])


def workspace():
    # Bound temporary test creation and recursive cleanup to the project directory.
    root = (Path(__file__).resolve().parents[1] / "results").resolve()
    root.mkdir(exist_ok=True)
    return tempfile.TemporaryDirectory(prefix="test_", dir=root)


def poses(count=12):
    rng = np.random.default_rng(818)
    accepted = 0
    while accepted < count:
        rotation = rng.uniform([-0.35, -0.35, -0.12], [0.35, 0.35, 0.12])
        translation = rng.uniform([-260, -170, 500], [120, 100, 900]).reshape(3, 1)
        rr, tr = right_pose(rotation, translation)
        outer = np.array([[-1, -1, 0], [BOARD.columns, -1, 0],
                          [BOARD.columns, BOARD.rows, 0], [-1, BOARD.rows, 0]], np.float32) * BOARD.square_size
        projections = [cv2.projectPoints(outer, r, t, CAMERA, None)[0].reshape(-1, 2)
                       for r, t in ((rotation, translation), (rr, tr))]
        if any(np.any(p < [30, 30]) or np.any(p > [SIZE[0] - 30, SIZE[1] - 30]) for p in projections):
            continue
        accepted += 1
        yield rotation, translation


def right_pose(rotation, translation):
    return cv2.Rodrigues(STEREO_R @ cv2.Rodrigues(rotation)[0])[0], STEREO_R @ translation + STEREO_T


def projected_observations(count=12, noise=0.04):
    rng = np.random.default_rng(712)
    observations = []
    for index, (rotation, translation) in enumerate(poses(count)):
        left = cv2.projectPoints(BOARD.object_points(), rotation, translation, CAMERA, None)[0]
        right_rotation, right_translation = right_pose(rotation, translation)
        right = cv2.projectPoints(BOARD.object_points(), right_rotation, right_translation, CAMERA, None)[0]
        left += rng.normal(0, noise, left.shape).astype(np.float32)
        right += rng.normal(0, noise, right.shape).astype(np.float32)
        observations.append(Observation(str(index), left, right, SIZE))
    return observations


def render_checkerboard(rotation, translation):
    scale = 4
    width, height = SIZE
    image = np.full((height * scale, width * scale, 3), 170, np.uint8)
    for y in range(-1, BOARD.rows):
        for x in range(-1, BOARD.columns):
            square = np.array([[x, y, 0], [x + 1, y, 0],
                               [x + 1, y + 1, 0], [x, y + 1, 0]], np.float32) * BOARD.square_size
            projected = cv2.projectPoints(square, rotation, translation, CAMERA, None)[0]
            polygon = np.round(projected.reshape(-1, 2) * scale).astype(np.int32)
            value = 25 if (x + y) % 2 == 0 else 245
            cv2.fillConvexPoly(image, polygon, (value, value, value), lineType=cv2.LINE_AA)
    return cv2.resize(image, SIZE, interpolation=cv2.INTER_AREA)


def write_jpeg(path, image):
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok
    Path(path).write_bytes(encoded.tobytes())
