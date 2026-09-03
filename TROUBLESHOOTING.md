# Team JAX — Isaac Lab 트러블슈팅 가이드

**프로젝트:** 휴머노이드 로봇 Bimanual Box Grasp  
**마지막 업데이트:** 2026-07-09  
**환경:** Isaac Lab (JaxIsaacLab), Isaac Sim 5.1.0, RTX 4060 Laptop 8GB  
**로봇:** Unitree G1 + Inspire 5-finger Hand (G1_INSPIRE_FTP_CFG)  
**태스크:** 6 DOF 양팔로 박스를 잡고 들어올리기

---

## 목차 (TOC)

0. [현재 상태 (2026-07-09)](#0-현재-상태-2026-07-09)
1. [환경 설정 / 실행](#1-환경-설정--실행)
2. [ArticulationCfg / 로봇 설정](#2-articulationcfg--로봇-설정)
3. [보상 함수 설계](#3-보상-함수-설계)
4. [학습 발산 / 수렴 문제](#4-학습-발산--수렴-문제)
5. [관측 / 센서 설정](#5-관측--센서-설정)
6. [체크포인트 / 학습 재개](#6-체크포인트--학습-재개)
7. [병렬 환경 수 / GPU 설정](#7-병렬-환경-수--gpu-설정)
8. [미해결 이슈 / 향후 계획](#8-미해결-이슈--향후-계획)

---

## 0. 현재 상태 (2026-07-09)

### 보상 구조 (최종)

| 항목 | Weight | 조건 |
|------|--------|------|
| left_hand_contact | +1.0 | 왼손 접촉 (측면=1.0, 기타=0.2) |
| right_hand_contact | +1.0 | 오른손 접촉 (측면=1.0, 기타=0.2) |
| bimanual_grasp | +3.0 | 양손 동시 접촉 (측면=1.0, 기타=0.2) |
| contact_gated_lift | +6.0 | 양손 측면 접촉 AND 박스 높이 상승 |
| hold_at_height | +5.0 | 양손 측면 접촉 AND 1.5m 유지 |
| sustained_hold | +2.0 | 10스텝 내내 양손 접촉 AND 목표 높이 |
| action_rate | −0.01 | 연속 액션 변화량 L2 |
| action_l2 | −0.001 | 액션 크기 L2 |
| joint_torques | −0.0001 | 관절 토크 L2 |
| joint_vel | −0.005 | 관절 속도 L2 |
| joint_limits | −0.5 | 관절 한계 초과 |

### 액추에이터 설정 (최종)

```python
"active_arms":  stiffness=40.0,    damping=10.0,  effort_limit_sim=300  # 6 DOF 학습
"locked_upper": stiffness=10000.0, damping=100.0, effort_limit_sim=300  # shoulder_yaw, wrist
"locked_body":  stiffness=10000.0, damping=100.0, effort_limit_sim=500  # hip/knee/ankle/waist
"locked_hands": stiffness=2000.0,  damping=50.0,  effort_limit_sim=200  # 손가락 전체
```

### 학습 결과 요약

| Run | Iterations | Mean Reward | Ep. Length | 비고 |
|-----|-----------|-------------|------------|------|
| 2026-07-07_08-01-57 | 2249 (완료) | **145.59** | 600/600 | 최고 성능. LR 소멸 |
| 2026-07-07_19-48-57 | 763 (중단) | 138.88 | 600/600 | LR 소멸 직전 |
| 2026-07-06_19-26-20 | 2000 (완료) | −0.81 | — | 측면 필터 너무 엄격 |

- `mean_episode_length = 600` = 에피소드 타임아웃. 박스가 안 떨어지고 끝까지 버팀 (긍정적)
- 학습률이 매번 0으로 소멸하는 문제는 미해결 (아래 섹션 8 참고)

---

## 1. 환경 설정 / 실행

### [환경 설정] `--video` 사용 시 headless가 아닌데 렌더링 에러 발생

- **증상**: `play.py --video` 실행 시 화면 없이 죽음
- **원인**: `--video`는 오프스크린 렌더링 모드를 요구하는데 일반 kit 파일로 실행됨
- **해결**: `--headless` 플래그 같이 추가 → `isaaclab.python.headless.rendering.kit` 로드됨
- **핵심 교훈**: `--video`는 항상 `--headless`와 함께 써야 함

---

### [환경 설정] CPU powersave 경고

- **증상**: `CPU performance profile is set to powersave` 경고 반복 출력
- **원인**: 노트북 전원 관리 모드가 절전으로 설정됨
- **해결**: 무시해도 학습에 영향 없음. 성능 개선 원하면 `performance` 모드로 변경
- **핵심 교훈**: 경고일 뿐 에러 아님

---

### [환경 설정] 학습 중 headless ↔ 렌더링 전환 불가

- **증상**: 학습 중에 `--headless` 전환하고 싶은데 방법이 없음
- **원인**: headless 여부는 Isaac Sim 시작 시 렌더링 시스템 초기화를 결정하므로 런타임 변경 불가
- **해결**: Isaac Sim 창 우측 하단 **Isaac Lab 패널 → Render Mode → No Rendering** 선택으로 부하 최소화
- **핵심 교훈**: 렌더링 모드는 런타임에 UI에서만 변경 가능. headless 전환은 재시작 필요

---

## 2. ArticulationCfg / 로봇 설정

### [로봇 설정] H1과 G1 관절명 표기 차이

- **증상**: H1 기반 설정을 G1으로 교체 후 관절을 못 찾는 에러
- **원인**: H1 관절명: `left_shoulder_pitch` (접미사 없음) / G1 관절명: `left_shoulder_pitch_joint` (`_joint` 접미사 있음)
- **해결**: G1 전환 시 모든 관절명에 `_joint` 접미사 추가
- **핵심 교훈**: H1 → G1 마이그레이션 시 관절명 접미사 확인 필수

---

### [로봇 설정] fix_root_link 설정 위치

- **증상**: `ArticulationCfg`에서 `fix_root_link` 위치를 못 찾음
- **원인**: `spawn.articulation_props` 안에 있음 (최상위가 아님)
- **해결**:
  ```python
  cfg.spawn.articulation_props = sim_utils.ArticulationRootPropertiesCfg(fix_root_link=True)
  ```
  G1_INSPIRE_FTP_CFG는 이미 `fix_root_link=True`가 기본값
- **핵심 교훈**: `fix_root_link`는 `spawn.articulation_props` 안에 있음

---

### [로봇 설정] stiffness=10000으로 관절 고정 (pseudo fixed joint)

- **증상**: 특정 관절을 소프트웨어로 고정하고 싶은데 Isaac Sim에서 fixed joint 직접 설정 불가
- **원인**: Isaac Sim에서 런타임 fixed joint 생성을 지원하지 않음
- **해결**:
  ```python
  "locked_joints": ImplicitActuatorCfg(
      joint_names_expr=[".*_shoulder_yaw_joint", ...],
      stiffness=10000.0,
      damping=100.0,
  )
  ```
- **핵심 교훈**: fixed joint 효과는 stiffness=10000, damping=100으로 대체 가능

---

### [로봇 설정] 손가락 stiffness 너무 낮으면 물리적으로 움직임

- **증상**: 손가락을 잠갔는데 시뮬레이션에서 손가락이 흔들림
- **원인**: `locked_hands` actuator에 stiffness=100, damping=5 로 설정 → 접촉력/관성에 의해 물리적으로 움직임
  다른 잠금 관절은 stiffness=10000인데 손가락만 낮게 설정한 실수
- **해결**:
  ```python
  "locked_hands": ImplicitActuatorCfg(
      joint_names_expr=[".*_index_.*", ".*_middle_.*", ".*_thumb_.*", ".*_ring_.*", ".*_pinky_.*"],
      effort_limit_sim=100,
      stiffness=10000.0,  # 100 → 10000
      damping=100.0,      # 5 → 100
  )
  ```
- **핵심 교훈**: 잠금 관절은 **전부** stiffness=10000, damping=100. 예외 없음

---

### [로봇 설정] 손가락 stiffness — 재발 및 최종 해결 (2026-07-09)

- **증상**: 이전에 해결된 줄 알았으나 새 학습 시작 후에도 손가락이 꺾임. `stiffness=100, effort_limit=30`으로 복귀돼 있었음
- **원인 추가**: USD 기본 관절 위치(init_state)도 함께 확인해야 한다고 판단해 `init_state.joint_pos.update({".*_index_.*": 0.0, ...})`를 추가했는데, G1_INSPIRE_FTP_CFG에 이미 `".*": 0.0` (catch-all)이 있어 중복 매칭 에러 발생
  ```
  ValueError: Multiple matches for 'L_index_proximal_joint': '.*' and '.*_index_.*'!
  ```
- **최종 해결**:
  1. `init_state.joint_pos.update(...)` 제거 — G1_INSPIRE_FTP_CFG가 이미 전체를 0으로 초기화함
  2. `locked_hands` actuator 값 조정: `effort_limit_sim=200, stiffness=2000, damping=50`
  (stiffness=10000은 물리 불안정 유발, 2000은 안전하고 접촉력 충분히 버팀)
- **핵심 교훈**: `G1_INSPIRE_FTP_CFG.init_state.joint_pos`에 `".*": 0.0`이 있음. 추가 패턴을 update()하면 중복 매칭 에러. `init_state`는 건드리지 말 것

---

### [로봇 설정] 22 DOF(손가락 포함) 학습 — 발산

- **증상**: 손가락 16 DOF + 팔 6 DOF = 22 DOF로 학습했더니 정책이 완전히 붕괴 ("병신됐는데?")
- **원인**: 22 DOF는 6 DOF보다 탐색 공간이 극적으로 넓음. 2850 iteration으로는 수렴 불가능.
  손가락 관절이 팔 동작과 coupling되어 학습 신호가 희석됨
- **해결**: 6 DOF (shoulder_pitch/roll + elbow × 양팔)로 복귀. 손가락은 전부 잠금
  22 DOF를 학습하려면 최소 5000~10000 iteration 필요
- **핵심 교훈**: DOF 줄일수록 학습 빠름. 검증된 단순한 구조로 먼저 학습한 뒤 DOF 추가할 것

---

### [로봇 설정] G1+Inspire Hand — 손바닥 링크 이름

- **증상**: 손바닥에 ContactSensor 달려고 `L_palm_link` 사용 → `RuntimeError: Sensor at path '.../L_palm_link' could not find any bodies`
- **원인**: G1+Inspire Hand에는 `L_palm_link`라는 링크가 존재하지 않음.
  Nucleus 원격 서버에 USD 있어서 로컬에서 직접 확인 불가
- **해결**: `left_wrist_yaw_link` / `right_wrist_yaw_link` 사용. 공식 pick_place 예제도 이 링크 사용
  ```python
  ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/left_wrist_yaw_link", ...)
  ```
- **핵심 교훈**: G1+Inspire Hand EE = `{left/right}_wrist_yaw_link`. `L_palm_link`는 없음

---

### [로봇 설정] H1 팔 길이와 박스 위치 계산

- **증상**: 박스가 x=0.40m에 있을 때 팔이 닿지 않음
- **원인**: G1 어깨(0, ±0.2, 1.3) → 박스 측면(0.40, ±0.14, 0.85) 거리 = 0.587m > G1 팔 길이 0.58m
- **해결**: 박스를 x=0.30m, z=1.00m으로 조정 → 거리 0.428m (여유 있음)
- **핵심 교훈**: 박스 위치는 반드시 팔 길이 이내로 계산 후 배치

---

### [로봇 설정] `effort_limit` vs `effort_limit_sim` Deprecation 경고

- **증상**: `ImplicitActuatorCfg` 사용 시 `effort_limit` deprecated 경고
- **원인**: 최신 Isaac Lab에서 `effort_limit_sim`으로 파라미터명 변경됨
- **해결**: `effort_limit` → `effort_limit_sim` 으로 교체
- **핵심 교훈**: `effort_limit_sim` 을 사용할 것

---

## 3. 보상 함수 설계

### [보상] 접촉 없이 점수 주는 보상 — 가장 흔한 설계 실수

- **증상**: 로봇이 박스 근처에서 맴돌기만 하고 실제로 잡지 않음. 팔이 박스 없이도 위아래로 움직임
- **원인**: 아래 보상들은 접촉 없어도 점수를 줌:
  - `palm_approach_box_side` — 박스 옆에 가까이만 가도 보상 (접촉 불필요)
  - `palm_facing_box` — 손바닥 방향만 맞춰도 보상 (접촉 불필요)
  - `box_upright_bonus` — 박스가 안 쓰러지면 항상 보상 (아무것도 안 해도 됨)
  - `box_alive` — 박스가 테이블 위에 있으면 항상 보상 (아무것도 안 해도 됨)
  - `box_placed_on_table` — 박스가 테이블 높이 + 손 안 닿으면 보상 → **에피소드 초반 내내 점수**
  
  이 보상들이 실제 태스크 보상(bimanual_grasp, box_lift)보다 더 크거나 쉬워서 local optimum 형성
- **해결**: **접촉 없이 주는 보상을 전부 제거**. 오직 ContactSensor 기반 보상만 남김
  ```python
  # ✅ 접촉 기반 보상만 사용
  left_hand_contact    # 한 손 접촉 (탐색용)
  right_hand_contact   # 한 손 접촉 (탐색용)
  bimanual_grasp       # 양손 동시 접촉
  contact_gated_lift   # 양손 접촉 + 들기
  hold_at_height       # 양손 접촉 + 높이 유지
  sustained_hold       # 지속 접촉 + 높이
  ```
- **핵심 교훈**: 보상 설계 원칙 — **태스크를 수행하지 않았을 때 보상 > 0 이면 잘못된 설계**

---

### [보상] palm_facing 보상이 팔을 위로 올리게 만듦

- **증상**: 박스를 안 잡아도 팔이 위로 올라감
- **원인**: `palm_facing_box` weight=2.0 (너무 강함). 팔을 박스 위로 올리면 wrist Y축이 아래쪽(=박스 방향)을 향하게 됨
  → 위로 올리는 것만으로도 palm_facing 보상 획득 가능 (reward hacking)
- **해결**: weight 2.0 → 0.5 → 결국 완전 제거. 접촉 기반 보상으로 대체
- **핵심 교훈**: orientation 정렬 보상은 위치 보상과 독립적이면 reward hacking 취약. 접촉 조건과 묶거나 weight 매우 낮게

---

### [보상] no_contact_penalty가 박스를 못 놓게 만듦

- **증상**: 로봇이 박스를 잡았다가 절대 안 놓음. 에피소드 끝까지 들고 있음
- **원인**: `no_contact_penalty` weight=-3.0 → 손이 박스에서 떨어지면 매 스텝 -3점
  → "항상 잡고 있는 것"이 optimal이 되어버림. 내려놓기(place) 학습 불가능
- **해결**: `no_contact_penalty` 제거. 내려놓기는 `box_placed_on_table`로 양성 보상으로 유도
- **핵심 교훈**: 접촉 해제에 패널티 주면 내려놓기 학습 불가. 내려놓기는 양성 보상으로만 유도할 것

---

### [보상] box_placed_on_table이 에피소드 초반에 항상 만점

- **증상**: 텐서보드에서 `box_placed = 4.98` (weight=5.0) 이 처음부터 항상 나옴
- **원인**: 조건: "박스가 테이블 높이(z: 0.88~1.05)에 있고 손이 안 닿으면 점수"
  → 에피소드 시작부터 박스는 테이블 위(z=1.00)에 있고 손도 안 닿음 → 초반 내내 만점
- **해결**: `box_placed_on_table` 보상 제거. "잡고 들기"까지만 학습하기로 변경
- **핵심 교훈**: 보상 조건이 초기 상태에서 이미 만족되면 그 보상은 무의미하거나 역효과

---

### [보상] action_rate 패널티가 너무 강하면 팔이 안 움직임

- **증상**: 1시간 학습 후에도 팔이 가만히 있거나 아주 살짝만 흔들림
- **원인**: `action_rate weight=-0.05`가 초기 탐색 단계에서 너무 강하게 작동
  → "아무것도 안 하는 것"이 local minimum이 됨 (접촉 보상보다 action_rate 패널티 회피가 더 이득)
- **해결**: 커리큘럼 적용 — 초반(10000 step)은 -0.001로 시작, 이후 -0.05로 증가
  ```python
  curriculum:
    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -0.001, "num_steps": 10000},
    )
  ```
- **핵심 교훈**: 정규화 패널티(action_rate, joint_vel 등)는 초반 탐색을 방해할 수 있음. 커리큘럼으로 점진 적용

---

### [보상] 거리 기반 접촉 판정의 한계 — 손목 vs 손바닥 문제

- **증상**: 박스를 잡긴 잡는데 손바닥이 아닌 손목 뒤쪽으로 잡음 (영상 확인)
- **원인**: `wrist_yaw_link`를 EE로 사용하면서 접근 목표를 박스 면(±0.14m)으로 설정
  → 손목이 박스에 먼저 닿고 손바닥은 그 뒤에 있어서 닿지 않음
- **해결**: `palm_offset=0.10` 파라미터로 접근 목표 보정
  ```python
  target = box_pos + [0, ±(box_half_y + palm_offset), 0]  # ±0.24m
  ```
  + ContactSensor로 실제 물리 접촉 검증 추가
- **핵심 교훈**: EE가 손목일 때 손바닥 면까지 오프셋 추가 필요 (~10cm)

---

### [보상] 밀어서 날리기 꼼수 — 접촉 없이 들기 보상 발생

- **증상**: 로봇이 박스를 잡지 않고 밀어서 날려도 `box_lift` 보상을 받음
- **원인**: 기존 `box_lift_reward`가 박스 z 높이만 보고 접촉 여부를 확인하지 않음
- **해결**: `contact_gated_lift` — 양손 접촉 AND 높이 조건 동시 충족 시에만 보상
  ```python
  return lift_reward * both_contact  # 접촉 없으면 0
  ```
- **핵심 교훈**: 들기/이동 보상은 반드시 접촉 조건과 AND로 묶어야 꼼수 차단됨

---

### [보상] bimanual 조건 — 한 손만 쓰는 지름길 방지

- **증상**: 한 손만 박스에 대고 밀어서 처리하는 전략 학습
- **원인**: 왼/오른 손 보상을 각각 독립적으로 주면 한 손만 써도 절반 보상 획득 가능
- **해결**: `bimanual_contact_grasp`에서 AND 조건 사용
  ```python
  return (left_contact & right_contact).float()  # 둘 다 닿아야만 1.0
  ```
- **핵심 교훈**: 양팔 협조를 강제하려면 보상 조건을 AND로 묶어야 함

---

### [보상] 보상 스케일 과대 — NaN 발생

- **증상**: iter ~800에서 `mean_reward = -10^25`로 붕괴
- **원인**: `box_lift` weight=5.0이 너무 커서 total reward scale이 100+ → 그래디언트 폭발 → NaN
- **해결**: `box_lift` weight=5.0 → 2.0으로 낮춤. `obs_normalization=True` 추가
- **핵심 교훈**: 보상 합산 최대값이 10~15 이내가 안정적. 개별 weight는 1~5 범위 권장

---

### [보상] action_rate 패널티 부족 — 떨림(jitter) 현상

- **증상**: 학습된 정책이 박스를 잡은 채 심하게 떨림
- **원인**: `action_rate weight=-0.0001`이 너무 작아서 급격한 관절 명령 변화를 억제하지 못함
- **해결**: `action_rate weight=-0.01` (100배), `action scale=0.5 → 0.25`, `joint_vel weight` 강화, `joint_acc_l2` 추가
- **핵심 교훈**: 떨림이 있으면 action_rate 패널티부터 5~10배 올릴 것

---

### [보상] 측면 접촉 필터 너무 엄격 — 학습 신호 희소 (2026-07-07)

- **증상**: `_is_side_contact()` 추가 후 2000 iteration 동안 `bimanual_grasp=0.01`, `mean_reward=-0.81`. 정책이 전혀 학습하지 못함
- **원인**: 탐색 초반에는 손이 박스 측면에 정확히 닿을 확률이 매우 낮음. 측면에만 보상을 주면 거의 항상 보상=0 → gradient 신호 없음
- **해결**: 완전 차단 대신 **등급제** 도입
  - 측면 접촉: `side_reward=1.0` × weight (정상 보상)
  - 기타 접촉: `other_reward=0.2` × weight (낮은 보상, 탐색 허용)
  ```python
  is_side = _is_side_contact(wrist_pos, box_pos, side)
  reward = torch.where(is_side, side_reward, other_reward)
  return reward * has_contact.float()
  ```
- **핵심 교훈**: 탐색 초반에 보상 신호가 너무 희소하면 학습 불가. 완전 차단보다 등급제가 효과적

---

### [보상] 잡고 나서 망설임 — box_lift weight 부족 / action_rate 과도 (2026-07-09)

- **증상**: 로봇이 박스를 잡은 후 들어올리지 않고 그 자리에서 망설임. 충분히 학습됐음에도 움직임이 소극적
- **원인 1**: `action_rate weight=-0.05`가 동작 전환을 억제해 "가만히 있기"가 안전한 전략이 됨
- **원인 2**: 접촉 유지 보상 합계(1+1+3=5 per step)보다 들기 보상(`contact_gated_lift` weight=3)이 상대적으로 작음. 잡고 있는 것이 더 안전한 local optimum
- **해결**:
  - `contact_gated_lift` weight: 3.0 → **6.0**
  - `contact_gated_lift` std: 0.08 → **0.05** (조금만 들어도 보상이 급격히 증가)
  - `action_rate` weight: −0.05 → **−0.01** (동작 전환 억제 완화)
- **핵심 교훈**: 다음 단계 보상(들기)이 현재 단계 보상(접촉 유지)보다 명확히 커야 정책이 전진함

---

## 4. 학습 발산 / 수렴 문제

### [학습] Policy/mean_std 폭발 후 학습 붕괴

- **증상**: iter 800까지 학습 잘 되다가 `Policy/mean_std`가 1.98→2.16으로 치솟고 `mean_reward`가 NaN으로 붕괴
- **원인**: `obs_normalization=False` + `entropy_coef=0.01` 조합이 관측 스케일 불일치와 과도한 탐색을 유발
- **해결**:
  ```python
  actor_obs_normalization=True   # 실시간 관측 정규화
  entropy_coef=0.001             # 탐색 억제
  init_noise_std=0.5             # 초기 std 낮춤
  ```
- **핵심 교훈**: `Policy/mean_std > 1.5` 이면 붕괴 전조. obs_normalization 반드시 켤 것

---

### [학습] Learning rate가 0으로 수렴 — 학습 완전 멈춤

- **증상**: 텐서보드에서 `Loss/learning_rate = 0.0000`. `bimanual_grasp = 0.0000`으로 수백 iteration 고정
- **원인**: RSL-RL의 `schedule="adaptive"` + `desired_kl=0.01` 조합
  → 정책이 변하지 않으면 KL divergence ≈ 0 → lr 계속 반토막 → 결국 0에 수렴
  보상 하킹(reward hacking)으로 flat reward에 수렴하면 이 현상 발생
- **해결**: 보상 설계를 수정해서 gradient 신호가 살아있게 만들기. 접촉 기반 보상으로 전환 후 해결
- **핵심 교훈**: lr=0 이면 보상 설계 문제. 네트워크 문제 아님

---

### [학습] "표준편차가 음수" 에러

- **증상**: 학습 초반에 `std is negative` 또는 관련 에러 발생
- **원인**: learning_rate=1e-3이 너무 커서 초기에 그래디언트 폭발
- **해결**: `learning_rate=3e-4`로 낮춤
- **핵심 교훈**: manipulation 태스크는 learning_rate=3e-4 이하가 안정적

---

### [학습] 학습 붕괴 후 최적 모델 구출

- **증상**: 학습이 붕괴했지만 중간에 좋은 성능을 보인 체크포인트가 있음
- **해결**: `--load_run {폴더명} --checkpoint model_850.pt`로 특정 체크포인트 지정해서 플레이
- **핵심 교훈**: 체크포인트는 50 iter마다 자동 저장됨. 붕괴 직전 모델 구출 가능

---

### [학습] 환경 수 늘려도 학습 효과 없는 경우

- **증상**: num_envs 증가시켜도 학습 속도 안 빨라짐
- **원인**: 서멀 스로틀링 — 노트북 온도 상승으로 GPU 클럭 자동 감소
- **해결**: num_envs를 줄이거나 (4096 → 1024), 노트북 받침대로 통풍 확보
- **핵심 교훈**: GPU 90% 이상 + 온도 상승 = 스로틀링. 더 낮은 환경 수가 실제론 빠를 수 있음

---

## 5. 관측 / 센서 설정

### [센서] ContactSensor와 RigidObject 간 접촉 감지

- **증상**: ContactSensor를 추가했는데 접촉이 감지되지 않음
- **원인**: `filter_prim_paths_expr` 미설정 시 박스와의 접촉을 특정할 수 없음
- **해결**:
  ```python
  ContactSensorCfg(
      prim_path="{ENV_REGEX_NS}/Robot/left_wrist_yaw_link",
      filter_prim_paths_expr=["{ENV_REGEX_NS}/Box"],  # 박스 접촉만 감지
      history_length=10,
  )
  ```
- **핵심 교훈**: `filter_prim_paths_expr` 없으면 모든 접촉이 합산됨. 특정 오브젝트 감지 시 필수

---

### [센서] 접지센서(ContactSensor) 를 관측(observation)으로 활용

- **증상**: 로봇이 박스를 잡고 있는지 스스로 인식 못 함 (시각 정보 없음)
- **원인**: 기존 obs는 joint pos/vel + palm pose + box pos/quat 뿐 → 접촉 여부를 모름
- **해결**: ContactSensor 힘 크기를 정규화해서 obs에 추가
  ```python
  def contact_force_obs(env, contact_cfg, max_force=50.0):
      sensor = env.scene.sensors[contact_cfg.name]
      force = sensor.data.net_forces_w_history[:, 0, 0, :]
      return (torch.norm(force, dim=-1, keepdim=True) / max_force).clamp(0.0, 1.0)
  ```
  → obs 차원 39 → 41 (왼손, 오른손 접촉력 각 1개 추가)
- **핵심 교훈**: 접촉 태스크에서 policy가 접촉 여부를 모르면 학습이 매우 어려움. 접지센서 obs 추가 필수

---

### [센서] ContactSensor history_length — 지속 접촉 감지

- **증상**: 순간 접촉과 지속 접촉을 구분하고 싶음 (순간 툭 치는 것 vs 실제 잡기)
- **해결**: `history_length=10`으로 설정 → `net_forces_w_history: (N, 10, num_bodies, 3)`
  모든 히스토리 스텝에서 접촉이 있어야 "지속 접촉"으로 인정
  ```python
  left_all = (torch.norm(left_hist, dim=-1).squeeze(-1) > threshold).all(dim=-1)
  ```
- **핵심 교훈**: history_length로 순간 접촉 vs 지속 접촉 구분 가능

---

### [센서] SceneEntityCfg joint_names 필터로 관측/행동 차원 제어

- **증상**: 전체 관절(44+개)이 관측에 포함되어 불필요한 데이터가 섞임
- **해결**:
  ```python
  ACTIVE_ARM_CFG = SceneEntityCfg("robot", joint_names=ACTIVE_ARM_JOINTS)
  arm_joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": ACTIVE_ARM_CFG})
  ```
- **핵심 교훈**: SceneEntityCfg에 joint_names 지정하면 obs/action 차원이 자동으로 해당 관절만으로 제한됨

---

## 6. 체크포인트 / 학습 재개

### [체크포인트] `--resume` 없이 `--checkpoint` 만 주면 무시됨

- **증상**: `--load_run`과 `--checkpoint`를 줬는데 처음부터 학습 시작
- **원인**: `train.py` 내부에서 `if agent_cfg.resume:` 조건 체크 후에만 체크포인트 로드
- **해결**: 항상 `--resume` + `--load_run` 세트로 사용
  ```bash
  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-H1-BimanualBoxGrasp-v0 \
    --num_envs 7056 \
    --headless \
    --resume \
    --load_run 2026-07-02_19-53-55
  ```
- **핵심 교훈**: `--resume`은 체크포인트 로드 활성화 스위치. 없으면 나머지는 무시됨

---

### [체크포인트] `--load_checkpoint` 는 존재하지 않음

- **증상**: `train.py: error: unrecognized arguments: --load_checkpoint`
- **해결**: `--checkpoint model_2100.pt` (load_ 접두사 없음)
- **핵심 교훈**: 올바른 플래그: `--checkpoint`

---

### [체크포인트] RSL-RL resume 시 이터레이션 카운터 0부터 재시작

- **증상**: model_2850.pt에서 resume했는데 ETA가 7시간 (처음부터 시작하는 것과 동일)
- **원인**: RSL-RL은 resume 시 이터레이션 카운터를 0부터 재시작. max_iterations=2000을 다시 돌림
  → 실제로는 2850 + 2000 = 4850 iteration 학습하는 것과 같음
- **해결**: resume 전에 `max_iterations`를 "추가로 학습할 iteration 수"로 조정
  ```python
  max_iterations = 150  # 이미 2850 했으니 3000까지만
  ```
- **핵심 교훈**: RSL-RL resume = 카운터 리셋. max_iterations 조정 필요

---

### [체크포인트] `--resume`이 빈 폴더를 가리켜 에러

- **증상**: `ValueError: No checkpoints in the directory match model_.*.pt`
- **원인**: Isaac Lab은 train.py 실행과 동시에 타임스탬프 폴더를 생성하므로 실패해도 빈 폴더가 남음
  `--resume`은 가장 최신 폴더를 자동 선택 → 빈 폴더 선택됨
- **해결**: `--load_run 2026-06-11_10-03-43` 으로 체크포인트가 있는 폴더 명시
- **핵심 교훈**: 빈 폴더 있으면 `--load_run`으로 폴더 명시 필요

---

### [체크포인트] play.py에서 체크포인트 경로 형식 차이

- **증상**: play.py에서 `FileNotFoundError: Unable to find the file: model_2100.pt`
- **원인**: play.py의 `--checkpoint`는 **전체 절대 경로** 요구. train.py는 파일명만 요구
- **해결**:
  ```bash
  # train.py: 폴더명 + 파일명 분리
  --load_run 2026-06-11_10-23-53 --checkpoint model_2100.pt
  # play.py: 전체 절대 경로
  --checkpoint /home/jeongwoo/isaacsim/JaxIsaacLab/logs/rsl_rl/h1_bimanual_box_grasp/2026-06-11_10-23-53/model_2100.pt
  ```
- **핵심 교훈**: play.py는 체크포인트 **전체 경로**, train.py는 **폴더명 + 파일명** 분리

---

## 7. 병렬 환경 수 / GPU 설정

### [GPU] num_envs=7000 에러 — 격자 배치 제약

- **증상**: `--num_envs 7000` 실행 시 환경에서 차단
- **원인**: Isaac Lab이 환경을 √n × √n 격자로 배치하는데 7000은 정수 제곱근이 없음
- **해결**: 완전제곱수 사용: `4096(64×64)`, `6400(80×80)`, `7056(84×84)`
- **핵심 교훈**: num_envs는 완전제곱수여야 함

---

### [GPU] VRAM 사용량이 선형 스케일이 아님

- **증상**: 1024 envs=4000MB, 2048 envs=4042MB로 거의 차이 없음 → 4096=5578MB로 점프
- **원인**: 고정 오버헤드(모델 가중치, CUDA 컨텍스트)가 대부분. 환경당 비용은 작음
- **해결**: 실측 후 결정. RTX 4060 8GB 기준 → **7056 envs headless 사용 중** (약 7~7.5GB 추정)
- **핵심 교훈**: VRAM 비선형 스케일. 실측 후 결정할 것

---

### [GPU] 렌더링 with 학습 시 적정 환경 수

- **증상**: 2048 envs headless=4042MB이지만 렌더링 켜면 GPU 92% + 노트북 과열
- **해결**: 렌더링 켤 때는 num_envs=100~200 권장
- **핵심 교훈**: 렌더링 with 학습은 100~200 envs. 연구/확인용으로만 사용

---

## 8. 미해결 이슈 / 향후 계획

### [단기] 손가락 고정 시각 검증 필요

- **상태**: `stiffness=2000, effort_limit=200`으로 수정 완료. 아직 새 run에서 육안 미검증
- **검증 방법**: `--num_envs 4`로 실행해 Isaac Sim 뷰어에서 손가락 움직임 확인
- **예상**: 접촉 상황에서도 손가락이 펴진 상태를 유지해야 함

---

### [단기] 학습률 소멸 방지

- **상태**: 모든 성공 run에서 `Loss/learning_rate`가 결국 0.0000으로 소멸. 학습이 중간에 멈추는 주요 원인
- **현상**: `schedule="adaptive"` + `desired_kl=0.01` 조합에서 flat reward 구간이 생기면 KL≈0 → lr 반복 반감
- **시도할 해결책**:
  - `min_lr` 파라미터 추가 (RSL-RL 버전 확인 필요)
  - `schedule="fixed"`로 전환 (adaptive 포기)
  - `desired_kl` 값 올리기 (0.01 → 0.02)

---

### [단기] 망설임 해소 검증

- **상태**: `box_lift` weight=6.0, std=0.05 / `action_rate` weight=-0.01로 수정 완료. 다음 run에서 결과 확인 필요
- **기대**: 잡은 직후 빠르게 들어올리는 동작 학습

---

### [중기] 실제 손바닥 링크 이름 확인

- **상태**: `wrist_yaw_link` 사용 중이나 실제 손바닥 면의 링크 이름 미확인
- **내용**: G1+Inspire Hand USD가 Nucleus 서버에 있어서 로컬에서 직접 확인 불가
  런타임에 `print(env.scene["robot"].data.body_names)` 로 전체 바디 이름 출력 후 확인 필요
- **영향**: ContactSensor가 손목 관절에 달려 있어 손바닥 접촉 판정이 부정확할 수 있음

---

### [중기] 손가락 파지 학습 추가

- **상태**: Inspire Hand 5손가락이 모델에 있지만 모두 `stiffness=2000`으로 잠금
- **내용**: 현재 6 DOF (shoulder_pitch/roll + elbow × 양팔)만 학습. 손가락 파지 미구현
- **계획**: 6 DOF 정책이 안정적으로 수렴한 뒤 22 DOF로 확장. 최소 5000+ iteration 필요
- **주의**: 22 DOF 학습 시 init_noise_std, entropy_coef 재조정 필요

---

### [중기] 실제 H1 + 커스텀 손 USD 통합

- **상태**: 실제 하드웨어는 H1 기반 + 커스텀 손가락이지만 시뮬레이션에서는 G1+Inspire로 대체 중
- **내용**: H1 URDF + Inspire Hand URDF merge 작업 필요
- **예상 작업량**: URDF merge → USD 변환 → Isaac Lab 통합 (반나절~하루 예상)

---

### [장기] Sim-to-Real 전송

- **상태**: 시뮬레이션 전용. 실제 로봇 실험 미진행
- **필요 사항**: Domain randomization (마찰, 질량, 관절 노이즈), 실제 로봇 제어 인터페이스 연동
- **참고**: 현재 `fix_root_link=True`, `disable_gravity=True` 설정으로 실제 환경과 괴리 있음. Sim-to-Real 전 해제 필요
