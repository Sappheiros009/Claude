"""예시 하드서피스 제품: 둥근 모서리 본체 + 전면 화면 홈(오목) + 상단 버튼(돌출).

실제 사용 시 Claude Code가 클레이 턴어라운드를 보고 이 형식으로 새 모델 파일을 작성한다.
- 구조(무엇을 더하고 빼는지)는 코드로 정한다.
- 치수는 모두 PARAMS로 노출한다: (초깃값, 최솟값, 최댓값), 단위 mm.
- 필렛은 실제 B-rep 필렛이라 오목한 모서리(홈 안쪽)도 곡면으로 처리된다.
"""

import build123d as bd

PARAMS = {
    "width": (70.0, 40.0, 110.0),
    "depth": (28.0, 15.0, 45.0),
    "height": (120.0, 80.0, 160.0),
    "corner_r": (8.0, 3.0, 12.0),  # 본체 세로 모서리 필렛
    "top_r": (2.5, 0.5, 4.5),  # 본체 위아래 모서리 필렛
    "pocket_w": (48.0, 20.0, 60.0),  # 전면 화면 홈
    "pocket_h": (60.0, 20.0, 75.0),
    "pocket_z": (12.0, -10.0, 25.0),
    "pocket_depth": (3.0, 0.5, 8.0),
    "knob_r": (7.0, 3.0, 11.0),  # 상단 버튼
    "knob_h": (8.0, 2.0, 16.0),
}


def build(p: dict) -> bd.Part:
    body = bd.Box(p["width"], p["depth"], p["height"])
    body = bd.fillet(body.edges().filter_by(bd.Axis.Z), p["corner_r"])
    body = bd.fillet(body.edges().group_by(bd.Axis.Z)[-1] + body.edges().group_by(bd.Axis.Z)[0], p["top_r"])

    pocket = bd.Pos(0, -p["depth"] / 2, p["pocket_z"]) * bd.Box(p["pocket_w"], 2 * p["pocket_depth"], p["pocket_h"])
    pocket = bd.fillet(pocket.edges().filter_by(bd.Axis.Y), min(4.0, 0.2 * min(p["pocket_w"], p["pocket_h"])))
    body = body - pocket

    knob = bd.Pos(0, 0, p["height"] / 2 + p["knob_h"] / 2 - 0.5) * bd.Cylinder(p["knob_r"], p["knob_h"])
    return body + knob
