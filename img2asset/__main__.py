"""Phase 1 CLI.

    python -m img2asset prompt   --product "a compact bluetooth speaker"
    python -m img2asset slice    --sheet clay_sheet.png --out work/clay
    python -m img2asset estimate --views work/clay --out work/targets
    python -m img2asset fit      --model models/my_product.py --targets work/targets --out work/fit1
    python -m img2asset synth    --model models/example_device.py --params gt.json --out work/synth
"""

from __future__ import annotations

import argparse
import json
import sys

from .cameras import RingConfig


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="img2asset")
    ap.add_argument("--cameras", help="RingConfig JSON (기본: 8시점 45°, elevation 0)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prompt", help="M3 클레이 턴어라운드 프롬프트 출력")
    p.add_argument("--product", required=True, help="제품 설명 (영어 권장)")
    p.add_argument("--rows", type=int, default=2)
    p.add_argument("--cols", type=int, default=4)

    s = sub.add_parser("slice", help="턴어라운드 시트를 시점별 이미지와 마스크로 분할")
    s.add_argument("--sheet", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--rows", type=int, default=2)
    s.add_argument("--cols", type=int, default=4)
    s.add_argument("--threshold", type=float, default=18.0, help="배경 색과의 RGB 거리 임계값")

    e = sub.add_parser("estimate", help="CPU 깊이 추정 (transformers, torch 필요)")
    e.add_argument("--views", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--model-id", default=None)

    f = sub.add_parser("fit", help="CAD 치수 피팅 + 잔차 리포트 + STEP/STL 내보내기")
    f.add_argument("--model", required=True)
    f.add_argument("--targets", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--res", type=int, default=160, help="정규화 캔버스 해상도 (얇은 시점이 많으면 높일 것)")
    f.add_argument("--oversample", type=float, default=1.75, help="렌더 해상도 / 캔버스 해상도")
    f.add_argument("--maxfev", type=int, default=400)
    f.add_argument("--init", help="초깃값 JSON (이전 피팅의 params.json 도 가능)")
    f.add_argument("--fix", action="append", default=[], metavar="NAME=VALUE",
                   help="고정할 실제 치수 (예: --fix height=132). 절대 스케일 기준으로 하나 이상 필요")

    y = sub.add_parser("synth", help="정답 치수로 합성 타깃 생성 (검증용)")
    y.add_argument("--model", required=True)
    y.add_argument("--params", help="정답 치수 JSON (없으면 PARAMS 초깃값)")
    y.add_argument("--out", required=True)
    y.add_argument("--res", type=int, default=256)
    y.add_argument("--noise", type=float, default=0.0)

    a = ap.parse_args(argv)
    cfg = RingConfig.from_json(a.cameras) if a.cameras else RingConfig()

    if a.cmd == "prompt":
        from .prompts import chaining_note, clay_turnaround_prompt

        print(clay_turnaround_prompt(a.product, cfg, a.rows, a.cols))
        print("\n# " + chaining_note())
    elif a.cmd == "slice":
        from .views import slice_sheet

        for pth in slice_sheet(a.sheet, a.rows, a.cols, a.out, a.threshold):
            print(pth)
    elif a.cmd == "estimate":
        from .estimate import DEFAULT_MODEL, estimate_depth

        estimate_depth(a.views, a.out, cfg.n_views, a.model_id or DEFAULT_MODEL)
    elif a.cmd == "fit":
        from .fit import fit

        init = None
        if a.init:
            raw = json.loads(open(a.init).read())
            init = raw.get("params", raw)
        fixed = {k: float(v) for k, v in (item.split("=", 1) for item in a.fix)}
        summary = fit(a.model, a.targets, a.out, cfg, a.res, a.maxfev, init_params=init, fixed=fixed,
                      oversample=a.oversample)
        print(json.dumps({k: summary[k] for k in ("loss", "function_evals", "params")}, indent=2, ensure_ascii=False))
        print(f"\nCHECKPOINT: {a.out}/checkpoint.md 와 report.png 를 확인한 뒤 다음 단계로 진행하세요.")
    elif a.cmd == "synth":
        from .synth import make_synthetic_targets

        params = json.loads(open(a.params).read()) if a.params else {}
        make_synthetic_targets(a.model, params, a.out, cfg, a.res, noise=a.noise)
    return 0


if __name__ == "__main__":
    sys.exit(main())
