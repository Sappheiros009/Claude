"""8시점 45° 링 카메라 정의.

모든 시점은 정사영(orthographic)이다. 생성 이미지의 초점거리를 알 수 없기 때문에
원근 효과를 배제한다.

좌표계는 Blender 기준이다. Z가 위쪽이고, 정면(view 0) 카메라는 -Y 쪽에서 +Y를 바라본다.
view i의 카메라 방위각은 i * step 이고, direction="ccw"이면 위에서 볼 때 반시계 방향으로 돈다.
즉 view 2(90°) 카메라는 +X 쪽에 있다. 이는 물체가 위에서 볼 때 시계 방향으로 도는 턴테이블과 같다.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class RingConfig:
    n_views: int = 8
    step_deg: float = 45.0
    elevation_deg: float = 0.0
    direction: str = "ccw"  # "ccw" 또는 "cw"
    # 정규화된 캔버스에서 물체 높이가 차지하는 비율.
    # elevation 0의 정사영에서는 물체의 화면 높이가 방위각과 무관하게 일정하므로,
    # 높이 기준으로 정규화하면 생성 이미지의 프레이밍 차이를 흡수할 수 있다.
    frame_frac: float = 0.6

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def from_json(cls, path: str | Path) -> "RingConfig":
        return cls(**json.loads(Path(path).read_text()))


@dataclass
class OrthoCamera:
    index: int
    azimuth_deg: float
    position_dir: np.ndarray  # 원점에서 카메라 쪽을 향하는 단위 벡터
    forward: np.ndarray  # 카메라가 바라보는 방향 (= -position_dir)
    right: np.ndarray
    up: np.ndarray


def ring_cameras(cfg: RingConfig) -> list[OrthoCamera]:
    if abs(cfg.elevation_deg) > 1e-6:
        # 높이 기준 정규화는 elevation 0에서만 시점 간 불변이다.
        import warnings

        warnings.warn(
            "elevation_deg != 0: 높이 기준 정규화가 시점마다 달라질 수 있습니다.",
            stacklevel=2,
        )
    sign = 1.0 if cfg.direction == "ccw" else -1.0
    e = math.radians(cfg.elevation_deg)
    cams = []
    for i in range(cfg.n_views):
        az = sign * i * cfg.step_deg
        t = math.radians(az)
        c = np.array([math.sin(t) * math.cos(e), -math.cos(t) * math.cos(e), math.sin(e)])
        f = -c
        z = np.array([0.0, 0.0, 1.0])
        r = np.cross(f, z)
        r /= np.linalg.norm(r)
        u = np.cross(r, f)
        cams.append(OrthoCamera(i, az, c, f, r, u))
    return cams
