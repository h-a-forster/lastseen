"""HERIDAL: 4000x3000 drone photos of staged search scenes in Croatian mountains, people labelled with boxes.

Source: the HERIDAL database (University of Split, ipsar.fesb.unist.hr), as repacked by Pasi Pyrrö (Aalto University),
"HERIDAL dataset in keras-retinanet PASCAL VOC format", Zenodo, doi:10.5281/zenodo.5662350, licence CC BY 3.0.
Photos are fetched by this script, never committed. Only the official test split is used here (101 photos, its own
scenes), read member by member from the 8.3 GB zip with HTTP range requests, so about 0.5 GB is downloaded.

    uv run python -m app.perception.heridal
"""

import io
import time
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path

import httpx

URL = "https://zenodo.org/api/records/5662351/files/heridal_keras_retinanet_voc.zip/content"
ROOT = Path(__file__).resolve().parents[2] / "data" / "datasets" / "heridal"
PREFIX = "heridal_keras_retinanet_voc/"
CREDIT = ("HERIDAL database (University of Split), repacked by P. Pyrrö, Aalto University "
          "(doi:10.5281/zenodo.5662350), CC BY 3.0")

Box = tuple[int, int, int, int]  # x1, y1, x2, y2 in pixels


@dataclass(frozen=True)
class Photo:
    id: str
    path: Path
    people: tuple[Box, ...]


class _RemoteFile(io.RawIOBase):
    """A read-only, seekable view of a remote file; each read is one HTTP range request."""

    def __init__(self, url: str) -> None:
        self.url, self.pos = url, 0
        self.client = httpx.Client(follow_redirects=True, timeout=120)
        self.size = int(self._get(0, 0).headers["content-range"].split("/")[1])

    def _get(self, start: int, end: int, attempts: int = 4) -> httpx.Response:
        for attempt in range(1, attempts + 1):
            r = self.client.get(self.url, headers={"Range": f"bytes={start}-{end}"})
            if r.status_code == 206:
                return r
            if r.status_code not in (429, 500, 502, 503) or attempt == attempts:
                raise IOError(f"range request refused: HTTP {r.status_code}")
            time.sleep(float(r.headers.get("retry-after", 2 ** attempt)))  # Zenodo rate-limits bursts with 429
        raise AssertionError("unreachable")

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        self.pos = {io.SEEK_SET: offset, io.SEEK_CUR: self.pos + offset, io.SEEK_END: self.size + offset}[whence]
        return self.pos

    def readinto(self, buffer) -> int:
        if self.pos >= self.size or not len(buffer):
            return 0
        data = self._get(self.pos, min(self.pos + len(buffer), self.size) - 1).content
        buffer[:len(data)] = data
        self.pos += len(data)
        return len(data)


def fetch(split: str = "test") -> int:
    """Download the split's photos and labels into ROOT, skipping files already there. Returns how many it fetched."""
    archive = zipfile.ZipFile(io.BufferedReader(_RemoteFile(URL), buffer_size=1 << 20))
    wanted = [m for m in archive.infolist() if m.filename.startswith(PREFIX) and m.file_size
              and Path(m.filename).name.startswith(f"{split}_")]
    fetched = 0
    for i, member in enumerate(wanted, 1):
        target = ROOT / Path(member.filename).relative_to(PREFIX)
        if target.exists() and target.stat().st_size == member.file_size:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(target.suffix + ".part")
        partial.write_bytes(archive.read(member))  # zipfile checks the CRC
        partial.replace(target)
        fetched += 1
        print(f"{i}/{len(wanted)} {target.name}", flush=True)
    (ROOT / "CREDIT.txt").write_text(CREDIT + "\n", encoding="utf-8")
    return fetched


def photos(split: str = "test") -> list[Photo]:
    """The split's photos with their labelled people, in a fixed order. Two test labels have no photo in the repack
    (GRO_0005, SB_0004); labels without a photo are skipped."""
    out = []
    for xml in sorted((ROOT / "Annotations").glob(f"{split}_*.xml")):
        if not (ROOT / "JPEGImages" / f"{xml.stem}.jpg").exists():
            continue
        text = xml.read_text(encoding="utf-8")
        if "<!ENTITY" in text:  # a label file has no reason to define entities
            raise ValueError(f"{xml.name}: unexpected XML entity definition")
        people = tuple(tuple(int(float(o.findtext(f"bndbox/{k}"))) for k in ("xmin", "ymin", "xmax", "ymax"))
                       for o in ET.fromstring(text).iter("object") if o.findtext("name") == "person")
        out.append(Photo(xml.stem, ROOT / "JPEGImages" / f"{xml.stem}.jpg", people))
    return out


if __name__ == "__main__":
    print(f"fetched {fetch()} files into {ROOT}")
    ps = photos()
    print(f"{len(ps)} photos, {sum(len(p.people) for p in ps)} people; credit: {CREDIT}")
