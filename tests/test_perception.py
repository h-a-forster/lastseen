"""Gate 1 scoring without the network: tiles map people to the right pixels, and an alarm counts as a find or a false
alarm by the rules in `gate.py`."""

from pathlib import Path

import numpy as np

from app.perception.gate import Tile, score, tiles_at
from app.perception.heridal import Photo

PHOTO = Photo("p", Path("p.jpg"), ((2100, 1100, 2160, 1180), (990, 10, 1030, 60)))


def test_tiles_cover_the_photo_and_shrink_people_with_scale():
    assert len(tiles_at(PHOTO, 1)) == 12 and len(tiles_at(PHOTO, 3)) == 2
    tile = Tile(PHOTO, 2, 2000, 1000)
    assert tile.people(fully=True) == [PHOTO.people[0]]
    assert tile.to_tile(PHOTO.people[0]) == [50, 50, 80, 90]  # 60x80 px in the photo is 30x40 px in the tile


def test_a_find_needs_the_alarm_on_the_person_and_cut_off_people_are_not_false_alarms():
    record = {"people": [[100, 100, 120, 130]], "touching": [[100, 100, 120, 130], [990, 0, 1000, 20]],
              "candidates": [{"box": [105, 110, 118, 128], "c": 0.6, "spark": {"person": True}},
                             {"box": [992, 2, 999, 15], "c": 0.6, "spark": {"person": False}},  # a cut-off person
                             {"box": [500, 500, 520, 520], "c": 0.6, "spark": {"person": False}}]}  # a rock
    sam = score([record], 0.5, with_spark=False)
    assert sam["recall"] == [1, 1] and sam["false_alarms_per_tile"] == 1
    checked = score([record], 0.5, with_spark=True)
    assert checked["recall"] == [1, 1] and checked["false_alarms_per_tile"] == 0
    assert score([record], 0.7, with_spark=False)["recall"] == [0, 1]


def test_measured_table_loads_and_detection_falls_with_altitude():
    from app.sar.detection import DetectionModel, TABLE
    from app.sar.grid import Terrain
    import numpy as np

    model = DetectionModel.load(TABLE)
    assert model.calibrated
    open_ground = np.full((1, 2), Terrain.OPEN)
    low, high = model.pod(1.5, open_ground)[0, 0], model.pod(5.0, open_ground)[0, 0]
    assert 0.8 < low <= 1.0 and high < low
    assert model.pod(1.5, np.full((1, 1), Terrain.FOREST))[0, 0] < low


def test_photo_pool_draws_a_person_the_size_the_drone_would_see():
    from app.perception.pool import PhotoPool, person_px

    def record(i, side, found):
        person = [100, 100, 100 + side / 2, 100 + side]
        candidates = [{"box": person, "c": 0.9, "spark": {"person": True, "looks_like": "person"}}] if found else []
        return {"key": f"p{i}@1:0,0", "people": [person], "touching": [person], "candidates": candidates}

    pool = PhotoPool([record(i, 20, False) for i in range(5)] + [record(i + 5, 70, True) for i in range(5)])
    rng = np.random.default_rng(0)
    assert all(pool.shot(120 / 70, rng).detected for _ in range(20))
    assert not any(pool.shot(120 / 20, rng).detected for _ in range(20))
    assert round(person_px(1.65)) == 73
