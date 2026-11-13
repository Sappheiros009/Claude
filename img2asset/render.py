"""Blender(mathutils BVHTree) 레이캐스팅으로 시점별 깊이·마스크를 계산한다.

피팅 루프 안에서 쓰는 렌더러다. Cycles 렌더 대신 BVH 레이캐스팅을 써서
EXR 입출력과 Blender 프로세스 재시작 비용 없이 같은 Python 프로세스 안에서 돈다.
(`pip install bpy` 로 Blender를 Python 모듈로 사용)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cameras import OrthoCamera


@dataclass
class ViewRender:
    mask: np.ndarray  # (R, R) bool
    height: np.ndarray  # (R, R) float, 카메라 쪽으로 튀어나올수록 큼 (= -깊이), 월드 단위
    pixel_size: float  # 월드 단위 / 픽셀
    # 메시 꼭짓점을 카메라 축에 투영해 얻은 정확한 연속 bbox (행 중심, 열 중심, 높이[px]).
    # 이진 마스크의 정수 bbox는 ±1px 양자화 오차가 있어 정규화 스케일이 흔들린다.
    frame: tuple[float, float, float] | None = None


def mesh_bounds(vertices: np.ndarray) -> tuple[np.ndarray, float]:
    center = 0.5 * (vertices.min(0) + vertices.max(0))
    radius = float(np.linalg.norm(vertices - center, axis=1).max())
    return center, radius


def render_views(
    vertices: np.ndarray,
    triangles: np.ndarray,
    cameras: list[OrthoCamera],
    resolution: int,
) -> list[ViewRender]:
    import bpy  # noqa: F401  (bpy를 먼저 import해야 mathutils가 로드됨)
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree

    tree = BVHTree.FromPolygons(
        [tuple(v) for v in vertices.tolist()],
        [tuple(t) for t in triangles.tolist()],
        all_triangles=True,
    )
    center, radius = mesh_bounds(vertices)
    extent = radius * 1.05
    dist = radius * 3.0
    px = 2.0 * extent / resolution
    coords = (np.arange(resolution) + 0.5) * px - extent  # 픽셀 중심

    out = []
    for cam in cameras:
        mask = np.zeros((resolution, resolution), bool)
        height = np.zeros((resolution, resolution), np.float64)
        direction = Vector(cam.forward.tolist())
        base = center + cam.position_dir * dist
        r, u = cam.right, cam.up
        ray_cast = tree.ray_cast
        for row in range(resolution):
            y = -coords[row]  # 0번 행이 화면 위쪽
            row_base = base + u * y
            for col in range(resolution):
                o = row_base + r * coords[col]
                hit = ray_cast(Vector((o[0], o[1], o[2])), direction, dist * 2.0)
                if hit[0] is not None:
                    mask[row, col] = True
                    height[row, col] = -hit[3]
        pu = (vertices - center) @ u
        pr = (vertices - center) @ r
        frame = (
            (extent - 0.5 * (pu.max() + pu.min())) / px - 0.5,
            (0.5 * (pr.max() + pr.min()) + extent) / px - 0.5,
            (pu.max() - pu.min()) / px,
        )
        out.append(ViewRender(mask, height, px, frame))
    return out
