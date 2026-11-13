"""gpt-image용 턴어라운드 프롬프트.

M3: 형상 추정 전용 "클레이 턴어라운드"를 질감 버전과 따로 생성한다.
참조 이미지 체이닝: 히어로(질감) 이미지를 참조로 함께 넣어 같은 제품을 유지시킨다.
"""

from __future__ import annotations

from .cameras import RingConfig

ANGLE_WORDS = {
    0: "front",
    45: "front-right three-quarter",
    90: "right side",
    135: "back-right three-quarter",
    180: "back",
    225: "back-left three-quarter",
    270: "left side",
    315: "front-left three-quarter",
}


def _layout(cfg: RingConfig, rows: int, cols: int) -> str:
    cells = []
    for i in range(cfg.n_views):
        az = int(round(i * cfg.step_deg)) % 360
        r, c = divmod(i, cols)
        cells.append(f"row {r + 1} col {c + 1}: {az} deg ({ANGLE_WORDS.get(az, 'view')})")
    return "; ".join(cells)


def clay_turnaround_prompt(product: str, cfg: RingConfig | None = None, rows: int = 2, cols: int = 4) -> str:
    cfg = cfg or RingConfig()
    return (
        f"A technical turnaround sheet of {product}, shown as an untextured matte mid-gray clay model. "
        f"{rows} rows x {cols} columns grid of equal cells, one view per cell, the object rotating "
        f"{cfg.step_deg:g} degrees per cell around its vertical axis: {_layout(cfg, rows, cols)}. "
        "Strict orthographic projection, no perspective distortion, camera at the object's mid-height, "
        "level horizon. Same object scale and same vertical position in every cell, object centered, "
        "the whole object visible with margin. Uniform flat diffuse lighting, no cast shadows, no "
        "reflections, no specular highlights, no ambient-occlusion darkening. Pure white seamless "
        "background. Keep every recess, groove, button, hole and edge fillet clearly readable by shading. "
        "No text, no labels, no grid lines, no annotations."
    )


def chaining_note() -> str:
    return (
        "참조 이미지 체이닝: 히어로(질감) 이미지를 참조로 첨부하고 "
        "'Use the attached image as the exact product design reference; keep all proportions and features "
        "identical.' 를 프롬프트 앞에 붙인다. 클레이 시트와 이후 질감 시트 모두 같은 히어로를 참조한다."
    )
