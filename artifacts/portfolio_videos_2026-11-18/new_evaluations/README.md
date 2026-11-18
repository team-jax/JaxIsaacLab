# 새 포트폴리오 영상의 출처 (2026-11-18)

이 폴더의 영상은 과거 촬영본이 아니다. 보존된 모델 또는 원본 학습 로그를 현재 코드로 다시 평가·시각화했다. 원본 체크포인트와 로그는 변경하지 않았다.

| 사례 | 새 자료 | 입력 | 조건과 한계 |
|---|---|---|---|
| 5 | `case05/videos/evaluation_seed_12345/rl-video-step-0.mp4` | `logs/rsl_rl/one_arm_jax_minimal_stage1_cube5cm_v3_5k/2026-10-13_19-15-17/model_4999.pt` | 현재 `evaluate.py`; seed 12345, 16환경, 480스텝. 결과 `case05/evaluation_minimal_seed_12345.json`. 과거 환경 코드와 동일함을 보장하지 않음. |
| 12 | `case12/case12_tensorboard_trends_new_20261118.mp4` | `jax_v2_1006_raise_table_v22/2026-10-29_20-01-00_v22_table` 및 `2026-11-01_19-48-38_v22_rs`의 TensorBoard event 파일 | `case12/render_metrics.py`로 원본 스칼라를 애니메이션화. 로봇 평가 영상이 아니며 보상 배율이 다른 실행의 절대값을 직접 비교하지 않음. |
| 19 | `case19/videos/front_grasp_lift_team_model_25550_portfolio_20261118/rl-video-step-0.mp4` | `logs/rsl_rl/jax_v2_1006_front_grasp_lift_v16/2026-11-11_21-12-12_v81b_r3/model_25550.pt` | 현재 `render_front_grasp.py`에 `v81b_full.sh`의 `COMMON` 설정, 팀 배치, 큐브 위치 `(0.14, 0.25)` 적용. 1환경, seed 미지정. 결과 `chain.json`. **Isaac Lab 영상이며 MuJoCo 실행 증거가 아님.** |

포트폴리오용 최상위 MP4 사본과 각 원본 파일의 관계는 `../manifest.json`에 기록했다.
