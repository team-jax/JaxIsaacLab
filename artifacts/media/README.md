# 영상·사진 모음

프로젝트에서 만든 영상 1168개와 사진 104개를 이름 형식 하나로 맞춰 모은 폴더입니다(총 707 MB). 원본은 원래 자리(`logs/rsl_rl/…`, `artifacts/…`)에 그대로 있습니다. `INDEX.csv`에 새 이름과 원본 경로가 한 줄씩 짝지어져 있습니다.

## 이름 형식

```
{날짜}_{시각}_{단계}_{실행}_{모델}_{화면}_{설명}.{확장자}
```

- 칸 사이는 `_`, 칸 안의 낱말 사이는 `-`입니다. 비어 있는 칸은 `none`입니다.
- 이름이 날짜·시각으로 시작해서, 이름순 정렬이 곧 만든 순서입니다.

| 칸 | 뜻 | 예 |
|---|---|---|
| 날짜, 시각 | 파일을 만든 때 (한국 시간) | `2026-11-05_1703` |
| 단계 | 학습 태스크. `v2-`는 JAX V2(URDF_1006) 태스크, `oaj-`는 그 전 one_arm_jax 태스크, `artifacts`는 `artifacts/`에 있던 파일, `analysis`는 결과를 확인하며 뽑은 장면 | `v2-front-grasp-lift-v16` |
| 실행 | 학습 실행 이름 (시각 부분은 뺌) | `v50-r3` |
| 모델 | 체크포인트 번호 | `m13075` |
| 화면 | 카메라 `front`·`side`·`close-front`·`close`, 3×3 체인 모음 `montage-front`, 그 밖의 영상 `clip`·`single`, 스크립트 기준 동작 `ref`, 사진 `image`·`frame` | `montage-front` |
| 설명 | 원래 이름에 있던 나머지 정보. `chain-x-0.03-y0.0`은 체인 영상의 큐브 위치(기준에서 x −3 cm, y 0 cm) | `v50-deploy` |

예: `video/2026-11-05/2026-11-05_1703_v2-front-grasp-lift-v16_v50-r3_m13075_montage-front_v50-deploy.mp4`

## 폴더

| 폴더 | 내용 |
|---|---|
| `video/날짜/` | mp4 1167개, gif 1개 |
| `image/날짜/` | png·jpg 104개 |
| `INDEX.csv` | 새 경로, 종류, 만든 때, 각 칸 값, 크기, 원본 경로 |

화면 종류별 개수: front 497, side 263, single 173, close-front 92, image 60, clip 59, montage-front 44, frame 42, ref 35, close 7

빠진 것: `logs/rsl_rl/h1_rough`(2026-09 Isaac Lab 예제 H1 영상, 이 프로젝트와 무관), `artifacts/deploy_2026-11-05/video/`(아래 몽타주의 복사본).

## 체인 모음 영상 (3×3, 큐브 위치 9곳 × 16번)

늘어뜨린 팔에서 넘겨주기까지 이어지는 전체 체인을 큐브 위치 9곳으로 찍어 한 화면에 모은 영상입니다. 프로젝트 후반의 진행을 한눈에 볼 수 있습니다.

| 만든 때 | 실행 | 모델 | 내용 | 파일 |
|---|---|---|---|---|
| 2026-11-02 00:21 | `v27b` | `m3075` | 넓은 큐브 범위 체인 첫 시도 | `video/2026-11-02/2026-11-02_0021_v2-front-grasp-lift-v16_v27b_m3075_montage-front_none.mp4` |
| 2026-11-02 01:31 | `v27d` | `m3550` | 넓은 큐브 범위 체인 (v27 회차) | `video/2026-11-02/2026-11-02_0131_v2-front-grasp-lift-v16_v27d_m3550_montage-front_none.mp4` |
| 2026-11-02 01:41 | `v27c` | `m3425` | 넓은 큐브 범위 체인 (v27 회차) | `video/2026-11-02/2026-11-02_0141_v2-front-grasp-lift-v16_v27c_m3425_montage-front_none.mp4` |
| 2026-11-02 02:07 | `v27e` | `m3625` | 넓은 큐브 범위 체인 (v27 회차) | `video/2026-11-02/2026-11-02_0207_v2-front-grasp-lift-v16_v27e_m3625_montage-front_none.mp4` |
| 2026-11-02 02:37 | `v27f` | `m3900` | 넓은 큐브 범위 체인 (v27 회차) | `video/2026-11-02/2026-11-02_0237_v2-front-grasp-lift-v16_v27f_m3900_montage-front_none.mp4` |
| 2026-11-02 03:08 | `v27g` | `m4124` | 넓은 큐브 범위 체인 (v27 회차) | `video/2026-11-02/2026-11-02_0308_v2-front-grasp-lift-v16_v27g_m4124_montage-front_none.mp4` |
| 2026-11-02 03:43 | `v27h` | `m3900` | 넓은 큐브 범위 체인 (v27 회차) | `video/2026-11-02/2026-11-02_0343_v2-front-grasp-lift-v16_v27h_m3900_montage-front_none.mp4` |
| 2026-11-02 04:17 | `v27i` | `m4550` | 넓은 큐브 범위 체인 (v27 회차) | `video/2026-11-02/2026-11-02_0417_v2-front-grasp-lift-v16_v27i_m4550_montage-front_none.mp4` |
| 2026-11-02 05:26 | `v27k` | `m5175` | 넓은 큐브 범위 체인 (v27 회차) | `video/2026-11-02/2026-11-02_0526_v2-front-grasp-lift-v16_v27k_m5175_montage-front_none.mp4` |
| 2026-11-02 06:01 | `v27l` | `m4950` | 넓은 큐브 범위 체인 (v27 회차) | `video/2026-11-02/2026-11-02_0601_v2-front-grasp-lift-v16_v27l_m4950_montage-front_none.mp4` |
| 2026-11-02 16:32 | `v27m` | `m5525` | 넓은 큐브 범위 체인 (v27 회차) | `video/2026-11-02/2026-11-02_1632_v2-front-grasp-lift-v16_v27m_m5525_montage-front_none.mp4` |
| 2026-11-02 16:33 | `v27j` | `m4675` | 넓은 큐브 범위 체인 (v27 회차) | `video/2026-11-02/2026-11-02_1633_v2-front-grasp-lift-v16_v27j_m4675_montage-front_none.mp4` |
| 2026-11-02 17:12 | `v27o` | `m4875` | 넓은 큐브 범위 체인 (v27 회차) | `video/2026-11-02/2026-11-02_1712_v2-front-grasp-lift-v16_v27o_m4875_montage-front_none.mp4` |
| 2026-11-02 23:46 | `v28e-ik` | `m6725` | IK 보조 들기 | `video/2026-11-02/2026-11-02_2346_v2-front-grasp-lift-v16_v28e-ik_m6725_montage-front_none.mp4` |
| 2026-11-03 00:21 | `v28f-ik` | `m7075` | IK 보조 들기 (이후 기준 모델) | `video/2026-11-03/2026-11-03_0021_v2-front-grasp-lift-v16_v28f-ik_m7075_montage-front_none.mp4` |
| 2026-11-03 00:56 | `v28f-ik` | `m7075` | IK 보조 들기 (이후 기준 모델) — ㄴ자 복귀 | `video/2026-11-03/2026-11-03_0056_v2-front-grasp-lift-v16_v28f-ik_m7075_montage-front_retl.mp4` |
| 2026-11-03 01:59 | `v28h-ik` | `m7275` | IK 보조 들기 | `video/2026-11-03/2026-11-03_0159_v2-front-grasp-lift-v16_v28h-ik_m7275_montage-front_none.mp4` |
| 2026-11-03 02:11 | `v28h-ik` | `m7275` | IK 보조 들기 — ㄴ자 복귀 | `video/2026-11-03/2026-11-03_0211_v2-front-grasp-lift-v16_v28h-ik_m7275_montage-front_retl.mp4` |
| 2026-11-04 07:51 | `v28f-ik` | `m7075` | IK 보조 들기 (이후 기준 모델) — 완료 판정 | `video/2026-11-04/2026-11-04_0751_v2-front-grasp-lift-v16_v28f-ik_m7075_montage-front_done.mp4` |
| 2026-11-04 08:14 | `v28f-ik` | `m7075` | IK 보조 들기 (이후 기준 모델) — 손 편 ㄴ자 | `video/2026-11-04/2026-11-04_0814_v2-front-grasp-lift-v16_v28f-ik_m7075_montage-front_straight.mp4` |
| 2026-11-04 12:15 | `v34-rel` | `m7425` | 책상 높이 −3 / 0 / +3 cm | `video/2026-11-04/2026-11-04_1215_v2-front-grasp-lift-v16_v34-rel_m7425_montage-front_tdz-0.03.mp4` |
| 2026-11-04 13:28 | `v34-rel` | `m7425` | 책상 높이 −3 / 0 / +3 cm | `video/2026-11-04/2026-11-04_1328_v2-front-grasp-lift-v16_v34-rel_m7425_montage-front_tdz0.0.mp4` |
| 2026-11-04 13:38 | `v34-rel` | `m7425` | 책상 높이 −3 / 0 / +3 cm | `video/2026-11-04/2026-11-04_1338_v2-front-grasp-lift-v16_v34-rel_m7425_montage-front_tdz0.03.mp4` |
| 2026-11-04 14:54 | `v34-relret` | `m7550` | 책상 높이 바꿈 + ㄴ자 복귀 | `video/2026-11-04/2026-11-04_1454_v2-front-grasp-lift-v16_v34-relret_m7550_montage-front_tdz-0.03.mp4` |
| 2026-11-04 15:04 | `v34-relret` | `m7550` | 책상 높이 바꿈 + ㄴ자 복귀 | `video/2026-11-04/2026-11-04_1504_v2-front-grasp-lift-v16_v34-relret_m7550_montage-front_tdz0.0.mp4` |
| 2026-11-04 15:10 | `v34-relret` | `m7550` | 책상 높이 바꿈 + ㄴ자 복귀 | `video/2026-11-04/2026-11-04_1510_v2-front-grasp-lift-v16_v34-relret_m7550_montage-front_tdz0.03.mp4` |
| 2026-11-04 16:53 | `v34-relret` | `m7550` | 책상 높이 바꿈 + ㄴ자 복귀 | `video/2026-11-04/2026-11-04_1653_v2-front-grasp-lift-v16_v34-relret_m7550_montage-front_tdz-0.01.mp4` |
| 2026-11-04 17:03 | `v34-relret` | `m7550` | 책상 높이 바꿈 + ㄴ자 복귀 | `video/2026-11-04/2026-11-04_1703_v2-front-grasp-lift-v16_v34-relret_m7550_montage-front_tdz0.01.mp4` |
| 2026-11-04 17:14 | `v34-relret` | `m7550` | 책상 높이 바꿈 + ㄴ자 복귀 | `video/2026-11-04/2026-11-04_1714_v2-front-grasp-lift-v16_v34-relret_m7550_montage-front_table-pm1cm.mp4` |
| 2026-11-04 17:15 | `v34-relret` | `m7550` | 책상 높이 바꿈 + ㄴ자 복귀 | `video/2026-11-04/2026-11-04_1715_v2-front-grasp-lift-v16_v34-relret_m7550_montage-front_table-pm1cm-seq.mp4` |
| 2026-11-04 19:09 | `v38f-r1` | `m7825` | 마찰 0.4 학습 — 마찰 0.4 | `video/2026-11-04/2026-11-04_1909_v2-front-grasp-lift-v16_v38f-r1_m7825_montage-front_mu04.mp4` |
| 2026-11-04 22:49 | `v44-r1` | `m8725` | 사전 파지 손 굴림 15° | `video/2026-11-04/2026-11-04_2249_v2-front-grasp-lift-v16_v44-r1_m8725_montage-front_v44-roll15.mp4` |
| 2026-11-04 23:04 | `v44-r1` | `m8725` | 사전 파지 손 굴림 15° | `video/2026-11-04/2026-11-04_2304_v2-front-grasp-lift-v16_v44-r1_m8725_montage-front_v44-roll15-early.mp4` |
| 2026-11-05 01:12 | `v45-r3` | `m9200` | 쥐는 힘·끝마디 굽힘 학습 | `video/2026-11-05/2026-11-05_0112_v2-front-grasp-lift-v16_v45-r3_m9200_montage-front_v45-learnedgrip.mp4` |
| 2026-11-05 03:12 | `v46-r2` | `m9500` | ㄴ자 복귀 4 cm 트리거 | `video/2026-11-05/2026-11-05_0312_v2-front-grasp-lift-v16_v46-r2_m9500_montage-front_v46-ret4cm.mp4` |
| 2026-11-05 04:54 | `v47b-r3` | `m10675` | 손 1.5 cm 깊게 | `video/2026-11-05/2026-11-05_0454_v2-front-grasp-lift-v16_v47b-r3_m10675_montage-front_v47b-deep15.mp4` |
| 2026-11-05 07:58 | `v48b2-r3` | `m11625` | 손 2 cm 깊게 (144/144) | `video/2026-11-05/2026-11-05_0758_v2-front-grasp-lift-v16_v48b2-r3_m11625_montage-front_v48b2-deep20.mp4` |
| 2026-11-05 10:34 | `v48e-r4` | `m11950` | 2 cm 깊게, 굴림 12° (143/144) | `video/2026-11-05/2026-11-05_1034_v2-front-grasp-lift-v16_v48e-r4_m11950_montage-front_v48e-final.mp4` |
| 2026-11-05 13:36 | `v48i2-r1` | `m12750` | 2.5 cm 깊게, 첫 마디까지 쥠 (144/144) | `video/2026-11-05/2026-11-05_1336_v2-front-grasp-lift-v16_v48i2-r1_m12750_montage-front_v48i2-final.mp4` |
| 2026-11-05 15:27 | `v49c-r1` | `m13100` | 넓은 범위 (일부 위치 실패) — (+2, −10 cm) 0/16 | `video/2026-11-05/2026-11-05_1527_v2-front-grasp-lift-v16_v49c-r1_m13100_montage-front_v49c-wide.mp4` |
| 2026-11-05 15:30 | `v49c-r1` | `m13100` | 넓은 범위 (일부 위치 실패) — 좌우 +10 cm 줄 0/16 | `video/2026-11-05/2026-11-05_1530_v2-front-grasp-lift-v16_v49c-r1_m13100_montage-front_v49c-y10.mp4` |
| 2026-11-05 17:03 | `v50-r3` | `m13075` | 실물용 13075 — 기본 조건 144/144 | `video/2026-11-05/2026-11-05_1703_v2-front-grasp-lift-v16_v50-r3_m13075_montage-front_v50-deploy.mp4` |
| 2026-11-05 17:21 | `v50-r3` | `m13075` | 실물용 13075 — 지연 67 ms 142/144 | `video/2026-11-05/2026-11-05_1721_v2-front-grasp-lift-v16_v50-r3_m13075_montage-front_v50-delay67ms.mp4` |
| 2026-11-05 17:33 | `v50-r3` | `m13075` | 실물용 13075 — 지연 33 ms 144/144 | `video/2026-11-05/2026-11-05_1733_v2-front-grasp-lift-v16_v50-r3_m13075_montage-front_v50-delay33ms.mp4` |
