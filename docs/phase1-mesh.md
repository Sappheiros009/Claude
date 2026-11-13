# Phase 1: 하드서피스 메시 파이프라인

AI 생성 이미지(gpt-image)에서 360° 재조명 가능한 하드서피스 에셋을 만드는 파이프라인의 첫 단계다.
여기서는 **CAD 코드 모델 + 8시점 피팅**으로 기본 메시를 만든다. 이후 단계(텍스처 생성, 역투영, 재질)는 이 메시를 기준으로 삼는다.

## 확정된 설계 결정

| 항목 | 결정 |
|---|---|
| 대상 | 하드서피스 제품 |
| 시점 | 8개, 45° 링, elevation 0, 정사영 |
| 모델링 커널 | build123d (OpenCascade B-rep). 필렛·불리언·로프트로 곡면과 오목부를 정확히 표현 |
| 모델 작성 | Claude Code가 클레이 턴어라운드를 보고 `models/<제품>.py` 작성 (구조 = 코드, 치수 = PARAMS) |
| 피팅 오차 | 소프트 실루엣 IoU + 상대 깊이(Huber, IRLS 정렬) + 노멀 각도 |
| 사전 부검에서 막기로 한 항목 | **M3** 클레이 참조 이미지로 깊이 추정, **M6** 잔차 히트맵 + 사용자 확인 체크포인트 |
| 렌더러 | Blender(bpy) mathutils BVH 레이캐스팅, 같은 Python 프로세스 안에서 실행 |

막지 않기로 한 항목(M1 계층 피팅, M2 별도 렌더러, M4 카메라 공동 보정, M5 필렛 폴백)은 구현하지 않았다.
CAD 빌드가 실패한 평가는 피팅이 멈추지 않도록 큰 벌점(10.0)만 준다(M5의 반경 축소 재시도는 없음).

## 작업 흐름

```
1. prompt    : 클레이 턴어라운드 프롬프트 생성 (히어로 이미지를 참조로 첨부)
2. (gpt-image): 2x4 클레이 턴어라운드 시트 생성
3. slice     : 시트 → view_i.png + mask_i.png
4. estimate  : CPU 깊이 추정 → depth_i.npy
5. (Claude)  : 턴어라운드를 보고 models/<제품>.py 작성
6. fit       : 치수 피팅 → params.json, model.step, model.stl, report.png, checkpoint.md
7. CHECKPOINT: checkpoint.md 의 오차 영역을 보고 구조 수정 → 6 반복 → 사용자 확인 후 Phase 2
```

```bash
python -m img2asset prompt   --product "a compact handheld bluetooth speaker"
python -m img2asset slice    --sheet clay_sheet.png --out work/clay
python -m img2asset estimate --views work/clay --out work/clay
python -m img2asset fit      --model models/my_speaker.py --targets work/clay --out work/fit1 --fix height=132
python -m img2asset fit      --model models/my_speaker.py --targets work/clay --out work/fit2 --fix height=132 --init work/fit1/params.json
```

## 반드시 알아야 할 조건

1. **`--fix` 로 실제 치수 하나를 고정해야 한다.** 이미지에는 절대 크기 정보가 없다. 고정하지 않으면 모양 비율은 맞아도 전체 크기가 임의로 나온다(합성 테스트에서 모든 치수가 정답의 0.88~0.93배로 수렴한 사례가 있음).
2. **클레이 시트의 칸 순서 = 시점 순서.** 왼쪽 위부터 행 우선으로 0°, 45°, …, 315°. view 2(90°)는 정면 기준 오른쪽(+X)에서 본 모습이다.
3. **물체 높이 기준 정규화.** elevation 0 정사영에서는 물체의 화면 높이가 방위각과 무관하게 같으므로, 생성 이미지와 렌더를 같은 규칙으로 정규화해 프레이밍 차이를 없앤다. 시트의 칸마다 물체 크기가 다르게 생성되어도 흡수되지만, 칸마다 **비율**이 다르면 흡수되지 않는다.
4. **모델 파일 규약**: `PARAMS = {"이름": (초깃값, 최솟값, 최댓값)}`, `def build(p) -> build123d.Part`. 예시는 `models/example_device.py`.

## 해상도와 속도 (이 컨테이너 4코어 CPU 기준 측정값)

정답 치수(GT)와 홈을 지운 모델(flat)의 손실 비율이 클수록 오목부 신호가 앨리어싱 노이즈보다 강하다.

| 렌더 배율 | 캔버스 | GT 손실 | flat 손실 | 비율 | 최소 IoU | 평가 1회 |
|---|---|---|---|---|---|---|
| 1.25 | 96 | 0.0502 | 0.0661 | 1.3 | 0.891 | 0.41s |
| 1.75 | 96 | 0.0231 | 0.0385 | 1.7 | 0.972 | 0.60s |
| 1.75 | 128 | 0.0189 | 0.0355 | 1.9 | 0.970 | 0.93s |
| 1.75 | 160 | 0.0080 | 0.0237 | 3.0 | 0.989 | 1.36s |

기본값은 캔버스 160, 배율 1.75다. 측면이 아주 얇은 제품은 `--res` 를 더 올린다.
파라미터 10개 기준 Powell 최적화는 평가 400~500회가 필요하다(기본 설정에서 약 10분).

## 정밀도 한계

합성 타깃, 기본 설정(160px, 평가 500회, `--fix height=132`)에서 정답 대비 오차:

| width | depth | corner_r | top_r | pocket_w | pocket_h | pocket_z | pocket_depth | knob_r | knob_h |
|---|---|---|---|---|---|---|---|---|---|
| +0.85 | +0.33 | +0.62 | −0.52 | −0.01 | −2.79 | −1.02 | −0.21 | +0.23 | −0.15 |

- **외곽 치수**는 소프트 IoU 덕분에 1mm 미만으로 맞는다.
- **실루엣 안쪽 특징(홈의 경계 위치)**은 깊이 잔차로만 보이기 때문에 캔버스 픽셀 1~2개가 한계다. 160px 캔버스에서 높이 143mm 물체의 1픽셀은 약 1.5mm다. 내부 특징을 더 정밀하게 맞추려면 `--res` 를 올린다(평가 시간은 대략 해상도 제곱에 비례).
- 오목부의 **깊이**(pocket_depth)는 잘 맞는다(−0.21mm).

## 검증 상태

- 합성 타깃(정답 치수로 렌더 + 임의 scale/offset + 노이즈)으로 검증: `pytest -q tests/test_phase1.py`
- 실제 gpt-image 시트와 Depth Anything 추정값으로는 **아직 검증하지 않았다**(이 개발 환경에 GPU·이미지 생성 접근이 없음). 실제 데이터에서는 시점 간 불일치와 추정 노이즈 때문에 손실 바닥값이 합성 테스트보다 높게 나올 것이다.
