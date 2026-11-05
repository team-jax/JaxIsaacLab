# JAX 로봇 팔 프로젝트: Orca 인계 메모

## 현재 목표

Isaac Lab + RSL-RL PPO로 JAX V2 단일 팔이 5 cm 정육면체를 집기 전 자세를 학습한다. 순서는 책상을 피해서 팔 들기 → 손목 들어 올리기 → `RotationBody`와 손가락을 이용해 큐브 한쪽 면에 손가락 1개, 반대 면에 2개 배치 → 열린 사전 파지 자세를 0.5초 유지하기다. 이 단계에서는 손가락을 닫아 실제로 파지하거나 물체를 들어 올리지 않는다.

## 실행 환경과 파일

- 프로젝트: `/home/jeongwoo/isaacsim/JaxIsaacLab`
- 최신 태스크: `Isaac-JaxV2-LiftThenPregrasp-Direct-v6`
- 환경: `source/isaaclab_tasks/isaaclab_tasks/direct/one_arm_jax_minimal/v2_pregrasp_pose.py`
- 앞 단계 팔 들기: 같은 폴더의 `v2_lift_only.py`
- 등록: 같은 폴더의 `__init__.py`
- PPO 설정: 같은 폴더의 `agents/rsl_rl_ppo_cfg.py`
- 평가: 같은 폴더의 `evaluate_v2_lift_only.py --task pregrasp --camera wide_front`
- URDF: `source/isaaclab_assets/data/Robots/JaxV2Assembly1006/urdf/JAXV2_Assembly.urdf`
- 관측 40차원, 행동 11차원. 8개 팔 관절과 3개 손가락 행동. 손가락은 팔을 들어 올린 후에만 행동이 적용된다.
- 학습은 사용자가 자신의 터미널에서 수행한다. 보통 7056개 환경, 새 정책으로 1000 iteration씩 실행하며, 기존 체크포인트를 지우지 않는다.

## 최신 학습과 평가

- 최신 완료 학습: `logs/rsl_rl/jax_v2_1006_pregrasp_pose_v6/2026-10-15_20-11-38_pregrasp_table_clearance_fresh_1k/model_999.pt`
- 해당 정책을 학습 당시 코드로 평가한 결과: 중간 팔 자세 0/16, 팔 완전 들기 0/16, 사전 파지 0/16.
- 평가 영상: 같은 실행 폴더의 `videos/evaluation_pregrasp_wide_front_seed_12345_model_999_table_clearance_final_1k/rl-video-step-0.mp4`
- 영상에서는 팔 방향이 큐브에 비해 틀어져 있고 손가락은 중립이다.
- 관절 궤적 약 5초 이후: 어깨 pitch −0.447 rad, 어깨 yaw −0.796 rad, 팔꿈치 1.410 rad, 손목 pitch −0.024 rad, `RotationBody` 0 rad. 손가락도 거의 0 rad.

## 마지막으로 확인한 직접 원인과 코드 수정

- 기존 첫 단계는 `(-0.45, 0, -0.50, 1.45, 0, 0, 0, 0)` 관절 목표의 각 관절 오차가 모두 0.18 rad 미만이어야 했다. 최신 정책의 yaw 오차가 약 0.296 rad여서 영원히 첫 단계를 통과하지 못했다. 그 결과 손목·손가락 단계의 보상이 켜지지 않고 손가락 행동도 잠겼다.
- `v2_lift_only.py`에 v6 전용 안전 범위 전환을 추가했다. 팔 pitch, yaw, 팔꿈치 및 손목 중립 범위를 확인하되 정확한 yaw 목표 일치는 요구하지 않는다. v5에는 적용되지 않는다.
- `v2_pregrasp_pose.py`에서 손 높이 *진행* 보상을 첫 단계부터 적용한다.
- 기존 `model_999.pt`를 수정된 코드로 재평가한 진단 결과 첫 단계 통과는 16/16이 되었지만 팔 완전 들기·사전 파지는 여전히 0/16이다. 이것은 **재학습 결과가 아니라 코드 진단**이다.
- 수정 후 syntax 및 2환경 reset/step 검증은 통과했다. 수정 후 새 학습은 아직 없다.

## 책상 관련 주의

- 책상 상판은 world Z=0.74 m이다. 기존 책상 안전 벌점은 가장자리 6 cm 영역에서만 활성화되어, 손가락이 상판 높이 아래에 있어도 0으로 기록되었다. v6 안전 영역을 18 cm, 벌점 scale을 0.10으로 변경하고 손 높이 진행 보상을 추가했다.
- 이전 오프라인 URDF 충돌 메시 꼭짓점 거리 검사에서는 책상과 최소 약 14.8 cm가 나왔으나, 화면에서는 손이 책상 아래로 내려가는 모습이 있다. 해당 검사는 완전한 물리 접촉 증거가 아니다. GPU의 필터링된 책상 접촉 센서는 `GPU contact filter ... not supported` 경고가 있어 0 N을 신뢰할 수 없다. 실제 아랫면 충돌 여부를 단정하지 말고 추가 검증해야 한다.

## 다음 판단

수정된 v6를 새 정책으로 학습한 후, 반드시 정면 전체 영상을 직접 보고 첫 단계 통과, 손목 상승, 손가락 개방, `RotationBody` 회전, 큐브의 양쪽 면 배치, 책상 근접을 분리해 평가한다. 보상 수치만으로 성공 판단하지 않는다. 사용자에게는 사실과 추측을 구분해 간결하게 보고한다.

전체 대화의 텍스트 기록은 `artifacts/orca_codex_conversation_2026-10-18.md`에 있다.
