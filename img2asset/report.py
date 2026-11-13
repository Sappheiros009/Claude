"""M6: 시점별 잔차 히트맵과 구조 검토 체크포인트 문서를 만든다.

치수 최적화는 "없는 형상"을 만들지 못한다. 그래서 피팅이 끝나면 남은 오차가 어디에 몰려 있는지를
그림(report.png)과 글(checkpoint.md)로 남긴다. Claude Code는 이것을 읽고 모델 구조(홈 추가,
돌출부 추가 등)를 수정하고, 사용자는 다음 단계로 넘어가기 전에 확인한다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage

from .cameras import RingConfig, ring_cameras

DEPTH_CLIP = 0.5  # t_hat 단위 (표준편차 1 기준)
NORMAL_CLIP = 60.0  # 도
BLOB_THRESHOLD = 0.3
MIN_BLOB_FRAC = 0.004


def _gray(a, mask):
    v = a[mask]
    out = np.full(a.shape + (3,), 30, np.uint8)
    if v.size:
        lo, hi = np.percentile(v, [2, 98])
        g = np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1)
        out[mask] = (40 + 200 * g[mask])[:, None].astype(np.uint8)
    return out


def _diverging(a, mask, clip):
    x = np.clip(a / clip, -1, 1)
    out = np.full(a.shape + (3,), 30, np.uint8)
    pos, neg = np.clip(x, 0, 1), np.clip(-x, 0, 1)
    rgb = np.stack([255 - 200 * neg, 255 - 200 * (pos + neg) / 1.0, 255 - 200 * pos], -1)
    rgb = np.clip(rgb, 0, 255)
    out[mask] = rgb[mask].astype(np.uint8)
    return out


def _sequential(a, mask, clip):
    x = np.clip(a / clip, 0, 1)
    rgb = np.stack([255 * np.ones_like(x), 255 * (1 - x), 200 * (1 - x)], -1)
    out = np.full(a.shape + (3,), 30, np.uint8)
    out[mask] = rgb[mask].astype(np.uint8)
    return out


def _silhouette(only_t, only_m, inter):
    out = np.full(only_t.shape + (3,), 30, np.uint8)
    out[inter] = (90, 90, 90)
    out[only_t] = (230, 60, 200)  # 생성 이미지에만 있음 → 모델 부족
    out[only_m] = (60, 200, 230)  # 모델에만 있음 → 모델 넘침
    return out


def _where(cy, cx, R):
    v = ["위", "가운데", "아래"][min(2, int(3 * cy / R))]
    h = ["왼쪽", "가운데", "오른쪽"][min(2, int(3 * cx / R))]
    return f"{v}-{h}"


def _blobs(mask, R):
    lab, n = ndimage.label(mask)
    out = []
    for i in range(1, n + 1):
        comp = lab == i
        area = comp.sum() / (R * R)
        if area < MIN_BLOB_FRAC:
            continue
        cy, cx = ndimage.center_of_mass(comp)
        out.append((area, _where(cy, cx, R)))
    return sorted(out, reverse=True)


def write_report(result, targets, cfg: RingConfig, out: Path, params: dict) -> None:
    from PIL import Image, ImageDraw

    cams = ring_cameras(cfg)
    R = targets[0].mask.shape[0]
    rows = []
    md = [
        "# 구조 검토 체크포인트 (M6)",
        "",
        f"- 전체 손실: **{result.total:.4f}**",
        "- 그림: `report.png` (행 = 시점, 열 = 추정 깊이 | 모델 깊이 | 깊이 잔차 | 노멀 오차 | 실루엣 차이)",
        "- 깊이 잔차 색: 빨강 = 모델이 더 튀어나옴(생성 이미지 쪽이 오목), 파랑 = 모델이 더 들어감(생성 이미지 쪽이 돌출)",
        "- 실루엣 색: 자홍 = 생성 이미지에만 있음(모델 부족), 하늘 = 모델에만 있음(모델 넘침)",
        "",
        "| 시점 | 방위각 | IoU | 깊이 Huber | 노멀 오차(°) |",
        "|---|---|---|---|---|",
    ]
    findings = []
    for cam, v, tg in zip(cams, result.views, targets):
        inter = tg.mask & ~v.only_target  # = 생성 이미지 ∩ 모델
        model_aligned = np.where(inter, tg.that + v.depth_residual, 0.0)
        tiles = [
            _gray(tg.that, tg.mask),
            _gray(model_aligned, inter),
            _diverging(v.depth_residual, inter, DEPTH_CLIP),
            _sequential(v.normal_err_deg, v.normal_err_deg > 0, NORMAL_CLIP),
            _silhouette(v.only_target, v.only_model, inter),
        ]
        rows.append(np.concatenate(tiles, axis=1))
        md.append(f"| {cam.index} | {cam.azimuth_deg:.0f}° | {v.iou:.3f} | {v.depth:.4f} | {v.normal_deg:.1f} |")
        for area, where in _blobs(v.depth_residual > BLOB_THRESHOLD, R):
            findings.append((area, cam, where, "모델이 더 튀어나옴 → 생성 이미지에는 홈/함몰이 있을 수 있음"))
        for area, where in _blobs(v.depth_residual < -BLOB_THRESHOLD, R):
            findings.append((area, cam, where, "모델이 더 들어감 → 생성 이미지에는 돌출부가 있을 수 있음"))
        for area, where in _blobs(v.only_target, R):
            findings.append((area, cam, where, "실루엣 부족 → 모델에 없는 부품/돌출"))
        for area, where in _blobs(v.only_model, R):
            findings.append((area, cam, where, "실루엣 넘침 → 모델에 불필요한 부피, 또는 생성 이미지 쪽 구멍"))

    sheet = np.concatenate(rows, axis=0)
    k = max(1, 192 // R)  # 작은 해상도로 피팅해도 리포트는 읽을 수 있게 확대
    img = Image.fromarray(sheet).resize((sheet.shape[1] * k, sheet.shape[0] * k), Image.NEAREST)
    draw = ImageDraw.Draw(img)
    for i, cam in enumerate(cams):
        draw.text((3, i * R * k + 3), f"v{cam.index} {cam.azimuth_deg:.0f}deg", fill=(255, 255, 0))
    img.save(out / "report.png")

    md += ["", "## 남은 오차 영역 (면적 큰 순)", ""]
    if findings:
        md.append("| 시점 | 위치 | 면적(%) | 해석 |")
        md.append("|---|---|---|---|")
        for area, cam, where, meaning in sorted(findings, key=lambda f: -f[0])[:20]:
            md.append(f"| {cam.index} ({cam.azimuth_deg:.0f}°) | {where} | {100 * area:.1f} | {meaning} |")
    else:
        md.append("임계값을 넘는 오차 영역이 없습니다.")
    md += [
        "",
        "## 다음 행동",
        "",
        "1. 여러 시점에서 같은 부위에 같은 종류의 오차가 반복되면 **구조 누락**입니다. 모델 파일에 형상(홈, 돌출, 구멍)을 추가하고 해당 치수를 PARAMS로 노출한 뒤 다시 피팅합니다.",
        "2. 한 시점에만 나타나는 오차는 생성 이미지의 시점 간 불일치일 가능성이 큽니다. 구조를 바꾸기 전에 해당 시점 이미지를 확인합니다.",
        "3. **사용자 확인 후** 다음 단계(Phase 2)로 넘어갑니다.",
        "",
        "## 피팅된 치수",
        "",
        "```json",
        __import__("json").dumps(params, indent=2, ensure_ascii=False),
        "```",
    ]
    (out / "checkpoint.md").write_text("\n".join(md), encoding="utf-8")
