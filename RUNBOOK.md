# Isaac-PnP 런북

---

## 보상 & 벌점 요약표

### 보상 (Reward)

| 항목 | 가중치 | 활성 조건 | 스텝당 최대값 | 역할 |
|------|--------|----------|-------------|------|
| `reach_object` | **+1.0** | 항상 | +1.0 | EE↔물체 거리 감소 유도 (tanh, std=0.10m) |
| `grasp_effort` | **+12.0** | 물체 15cm 이내 + 그리퍼 닫힘 | +12.0 | 파지 동작 발견 유도 |
| `lift_object` | **+25.0** | 물체 Z > 4cm | +25.0 | 핵심 보상. 들어올리는 동안 매 스텝 |
| `move_to_goal` | **+10.0** | 물체 Z > 4cm (들어올린 상태) | +10.0 | 들어올린 채 목표(0.5, 0.35, 0.25)로 이동 |
| `place_success` | **+50.0** | 물체↔목표 거리 < 6cm | +50.0 | 최종 목표. 매 스텝 지속 보상 |

### 벌점 (Penalty)

| 항목 | 가중치 | 활성 조건 | 스텝당 최대값 | 역할 |
|------|--------|----------|-------------|------|
| `ee_floor_penalty` | **−15.0** | EE Z < 5cm | −5.0 | 테이블에 쳐박히는 동작 억제 (tanh, std=0.03m) |
| `obstacle_penalty` | **0 → −8.0** | 장애물 12cm 이내 (커리큘럼) | −3.0 | 장애물 근접 억제. 50만 스텝에 걸쳐 강화 |
| `action_rate` | **−0.1** | 항상 | 제한 없음 | 액션 급변 억제 → 진동/꾸물거림 방지 |
| `joint_vel` | **−0.01** | 항상 | 제한 없음 | 관절 속도 억제 → 부드러운 동작 유도 |

### 종료 조건 (암묵적 벌점)

| 항목 | 트리거 | 종류 | 효과 |
|------|--------|------|------|
| `time_out` | 20초(1000스텝) 경과 | 정상 종료 | 패널티 없음 (bootstrap) |
| `object_dropped` | 물체 Z < −10cm | **실패 종료** | 미래 보상 전부 소멸 |
| `obstacle_contact` | 접촉력 > 1N + 기하학 확인 | **실패 종료** | 미래 보상 전부 소멸 |

> **가중치 계층**: 접근(1) < 파지시도(5) < 이동(10) < 바닥벌점(5) < 들어올리기(25) < 놓기(50)

---

## 파일 구조

```
pick_place_obstacle/
├── __init__.py                          # 태스크 등록 + 명령어 안내
├── pick_place_obstacle_env_cfg.py       # 추상 베이스 환경 설정
├── mdp/
│   ├── __init__.py
│   ├── rewards.py                       # 접근·파지·리프트·골·장애물 보상
│   ├── observations.py                  # 관측 함수 (66차원)
│   ├── events.py                        # 장애물 리셋 (커리큘럼 스테이지 체크)
│   └── curriculum.py                    # 스텝 기반 장애물 도입 커리큘럼
└── config/franka/
    ├── __init__.py                      # gym.register — Train + Play 태스크
    ├── franka_pnp_env_cfg.py            # Franka 구체 설정 (SWAP 포인트 5곳)
    └── agents/rsl_rl_ppo_cfg.py         # PPO 설정 (entropy_coef=0.01)

scripts/reinforcement_learning/rsl_rl/
├── train.py                             # 학습 진입점
└── play_debug_ee.py                     # 평가 + 대시보드 JSON 출력

scripts/
├── dashboard_server.py                  # 실시간 대시보드 HTTP 서버
└── watch_debug.sh                       # 디버그 로그 실시간 모니터링
```

---

## 주요 설계 포인트

| 항목 | 내용 |
|------|------|
| **장애물 비활성화** | 비활성 장애물은 `(200, 200, 0)` 위치로 주차 → proximity penalty ≈ 0 자동 처리 |
| **장애물 variant** | 슬롯당 5종, 에피소드마다 랜덤 선택. `_obstacle_active_variants` 텐서로 추적 |
| **오브젝트 스폰 커리큘럼** | 좁은 범위(x: 0.38~0.46) → 20만 스텝 후 선형 확장 → 60만 스텝에 전체 범위 도달 |
| **URDF 교체** | `franka_pnp_env_cfg.py`의 `# [SWAP]` 주석 5곳만 수정하면 교체 가능 |
| **ContactSensor** | `FRANKA_PANDA_CFG.copy()` 후 `activate_contact_sensors=True` 필수 |
| **ObservationsCfg** | 모든 필드는 `ObservationGroupCfg` 타입이어야 함. bool 필드 직접 선언 시 TypeError |

---

## 보상 함수

> 가중치 계층: 접근(1) < 파지시도(5) < 이동(10) < 들어올리기(25) < 놓기(50)
>
> **설계 원칙** — 에이전트가 "물체 옆에 그냥 머무는" 전략을 피하도록 reach를 낮게, lift를 압도적으로 높게 설정

---

### 1. `reach_object` — 접근 보상

**가중치**: `+1.0` | **함수**: `ee_object_distance`

```python
dist = norm(object_pos - ee_pos)
reward = 1.0 - tanh(dist / 0.10)
```

EE(팔 끝)와 물체 사이 거리를 tanh 커널로 변환한 밀집 보상이다. `std=0.10m`이므로 10cm 거리에서 보상값 ≈ 0.76, 20cm에서 ≈ 0.96(tanh 특성상 멀수록 1에 수렴 — 즉 **가까울수록 낮아짐**).

실제 반환값 범위: 0(멀 때) ~ 1(완전히 겹칠 때).  
× 가중치 1.0이므로 에피소드 내 최대 기여는 스텝당 1.0.

**가중치를 낮게 유지하는 이유**: 이 보상만 높으면 에이전트가 "물체 옆에 서 있기"가 최적 전략이 된다. reach를 낮게 두어 어디까지나 탐색 유도용 보조 보상으로만 사용.

---

### 2. `grasp_effort` — 파지 시도 보상

**가중치**: `+12.0` | **함수**: `grasp_effort`

```python
dist   = norm(object_pos - ee_pos)
near   = 1.0 - tanh(dist / 0.08)
gripper_signal = clamp(action[-1], 0, 1)   # 정책 출력의 마지막 차원
within = (dist < 0.15).float()

reward = near × gripper_signal × within
```

세 가지 조건이 **동시에** 충족될 때만 보상이 나온다:

1. **near**: EE가 물체에 가까울수록 (std=0.08m → 더 좁은 유효 반경)
2. **gripper_signal**: 정책이 그리퍼 닫기 신호(> 0)를 출력할 때
3. **within**: 물체까지 거리가 15cm 미만일 때

이 세 조건의 곱이므로 하나라도 0이면 보상 없음. 학습 초기에 에이전트가 "물체에 접근 + 그리퍼 닫기"를 우연히 함께 하면 보상이 주어져서 파지 동작을 발견하도록 유도한다.

**오브젝트 스폰 커리큘럼과의 상호작용**: 초반에 물체를 EE 기본 자세 근처(x≈0.42)에 스폰시키기 때문에, 에이전트가 탐색 중 우연히 이 조건을 충족할 확률이 높아진다.

---

### 3. `lift_object` — 들어올리기 보상 (핵심)

**가중치**: `+25.0` | **함수**: `object_lifted`

```python
reward = (object_z > 0.04).float()   # 4cm 초과 시 1.0, 아니면 0.0
```

가장 단순하면서 가장 중요한 보상. 물체의 Z 좌표가 4cm를 넘으면 `1.0`, 아니면 `0.0`인 이진값이다.

× 가중치 25.0이므로 물체를 들어올리고 있는 매 스텝마다 25점씩 쌓인다. 에피소드 길이가 20초, decimation=2, dt=0.01이면 최대 1000스텝 → 최대 기여 25,000점.

이 보상을 가장 높게 설정한 이유: 이전 508M 스텝 학습에서 에이전트가 reach는 잘 하지만 lift를 전혀 못 하는 병목이 확인됐다. lift 가중치를 높여서 일단 들어올리는 행동이 발생하면 압도적으로 유리하게 만든다.

**주의**: 물체를 "들고 있는 동안" 계속 보상이 나오기 때문에, 들어올린 채로 멈춰있어도 시간이 지날수록 보상이 쌓인다. `move_to_goal`이 함께 있어서 들고 이동해야 추가 보상을 받는다.

---

### 4. `move_to_goal` — 목표 이동 보상

**가중치**: `+10.0` | **함수**: `object_goal_distance`

```python
goal = [0.5, 0.35, 0.25]   # 고정 목표 위치 (월드 좌표, m)
dist = norm(object_pos - goal)
lifted = (object_z > 0.04).float()

reward = lifted × (1.0 - tanh(dist / 0.20))
```

`lifted` 마스크가 곱해지므로 **물체를 들어올린 상태일 때만 활성화**된다. 바닥에 있는 물체가 우연히 목표 위치 근처에 있어도 보상이 없다.

std=0.20m로 `reach_object`(std=0.10)보다 두 배 넓다 — 목표에서 20cm 내외부터 의미 있는 기울기를 제공하므로, 이동 방향 학습에 더 넓은 유효 범위를 준다.

목표 위치 `[0.5, 0.35, 0.25]`는 로봇 오른쪽 앞 공중에 고정. 에피소드마다 바뀌지 않는다.

---

### 5. `place_success` — 놓기 성공 보상

**가중치**: `+50.0` | **함수**: `place_success`

```python
dist = norm(object_pos - [0.5, 0.35, 0.25])
reward = (dist < 0.06).float()   # 6cm 이내면 1.0
```

이진 희소 보상. 물체가 목표 6cm 이내에 있으면 매 스텝 1.0 반환 → × 50 = 스텝당 50점. 실질적으로는 "물체를 목표 근처에 내려놓고 유지하는 동안" 연속으로 받는다.

가중치를 50으로 가장 높게 설정한 이유: 전체 태스크의 최종 목표이므로 다른 모든 보상보다 크게 설정해서 에이전트가 이 상태를 향해 학습하도록 유도.

**`move_to_goal`과의 관계**: move_to_goal이 목표에 가까워지도록 연속 기울기를 제공하고, place_success가 도달 시 큰 보너스를 준다. 두 보상이 협력해서 "목표 위치에 정확히 내려놓기"를 학습시킨다.

---

### 6. `obstacle_penalty` — 장애물 근접 패널티

**가중치**: `0.0` (커리큘럼으로 `-3.0`까지) | **함수**: `obstacle_proximity_penalty`

```python
penalty = 0
for obs in all_10_variants:
    dist  = norm(ee_pos - obs_pos)
    delta = clamp(0.12 - dist, min=0)      # safe_dist=0.12m 이내만 기여
    penalty += 1.0 - tanh(delta / 0.05)

return -(penalty / 2.0)   # 슬롯 수(2)로 정규화
```

10개 변형체(슬롯당 5종 × 2슬롯) 전부에 대해 계산한다. 파킹된 변형체는 EE까지 거리가 ~200m이므로 `delta = clamp(0.12 - 200, 0) = 0` → 기여 없음. 자동으로 active variant만 패널티를 낸다.

`safe_dist=0.12m` 이내에서 패널티가 급격히 증가. std=0.05m로 좁아서 12cm에서 갑자기 강해지는 포텐셜 장벽 형태다.

슬롯 2개로 나눠서 정규화하므로 장애물이 2개 활성화돼도 최대 패널티는 1개일 때와 비슷한 스케일로 유지된다.

**커리큘럼 적용**: 30만 스텝(장애물 등장 시점)에서도 가중치가 0에 가까워 패널티가 약하고, 50만 스텝에 걸쳐 -3.0으로 서서히 강화. 에이전트가 장애물에 익숙해진 후에 회피 압력을 올린다.

---

### 7. `action_rate` — 액션 변화율 패널티

**가중치**: `-1e-3` | **함수**: `action_rate_l2` (Isaac Lab 내장)

```python
reward = -sum((action_t - action_{t-1})^2)
```

연속된 두 스텝 간 액션 벡터의 L2 차이 제곱합에 음의 부호를 붙인다. 값이 크면(액션이 급격히 바뀌면) 패널티가 크다.

실제 영향: 에이전트가 매 스텝 전혀 다른 명령을 보내는 "진동" 동작을 억제한다. 이전 학습에서 로봇이 테이블에서 꾸물거리던 현상의 원인 중 하나.

가중치 -1e-3은 의도적으로 작게 설정. 너무 크면 에이전트가 아예 액션을 안 바꾸는(정지) 쪽으로 학습된다.

커리큘럼이 -1e-2로 강화하므로 학습 후반에 더 부드러운 동작을 유도한다.

---

### 8. `joint_vel` — 관절 속도 패널티

**가중치**: `-1e-3` | **함수**: `joint_vel_l2` (Isaac Lab 내장)

```python
reward = -sum(joint_velocity^2)
```

9개 관절(팔 7개 + 손가락 2개) 속도의 L2 제곱합. 빠르게 움직일수록 패널티.

`action_rate`가 "명령 변화량"을 억제한다면, `joint_vel`은 "실제 움직임 속도"를 억제한다. 두 보상이 함께 있어서 명령도 부드럽고 실제 동작도 느리게 유지되도록 학습된다.

---

### ~~`contact_penalty`~~ — 제거됨

이전에는 EE가 장애물에 닿을 때 -30점을 주는 항이 있었다. 하지만 `obstacle_contact` 종료 조건이 이미 에피소드를 끝내서 미래 보상을 차단하는 효과를 주므로 중복이었다. 제거해서 보상 함수를 단순화했다.

---

## 종료 조건 (Terminations) — 에피소드를 강제로 끝내는 벌점

보상 함수의 음수 항과 달리, 종료 조건은 에피소드 자체를 끊어버린다. PPO 입장에서는 "이후 모든 보상을 0으로 만드는" 암묵적 벌점이다.

---

### 1. `time_out` — 시간 초과

```python
time_out = DoneTerm(func=mdp.time_out, time_out=True)
```

에피소드 최대 길이(20초)에 도달하면 종료. `time_out=True`로 표시되어 있어서 **bootstrap(부트스트랩)** 처리됨 — PPO가 이 종료를 "실패"가 아닌 "시간이 다 됐을 뿐"으로 인식하고 마지막 상태의 가치를 예측값으로 사용한다. 즉, 패널티가 아니라 그냥 에피소드 끝.

에피소드 길이: `episode_length_s=20.0`, `dt=0.01`, `decimation=2` → 최대 **1000 스텝**.

---

### 2. `object_dropped` — 물체 낙하

```python
object_dropped = DoneTerm(
    func=mdp.root_height_below_minimum,
    params={"minimum_height": -0.10, "asset_cfg": SceneEntityCfg("object")},
)
```

물체 Z 좌표가 -10cm 미만이 되면 즉시 종료. 테이블 아래로 떨어졌다는 의미.

`time_out=True`가 없으므로 **실패 종료** — PPO가 미래 가치를 0으로 처리한다. 물체를 떨어뜨리면 남은 에피소드를 통째로 잃는 셈이라 강한 암묵적 패널티가 된다.

임계값이 -10cm인 이유: 물체가 테이블 표면(z≈0) 근처에서 튀거나 흔들릴 때 오탐지를 방지하기 위한 여유값.

---

### 3. `obstacle_contact` — 장애물 충돌

```python
obstacle_contact = DoneTerm(
    func=mdp.obstacle_contact,
    params={
        "contact_sensor_cfg": SceneEntityCfg("ee_contact"),
        "force_threshold": 1.0,
    },
)
```

두 가지 조건을 **동시에** 충족해야 종료가 트리거된다 (이중 게이트):

1. **접촉력 조건**: `panda_hand`의 ContactSensor에서 감지된 힘이 1N 초과
2. **기하학적 조건**: EE가 active 장애물 중 하나로부터 충분히 가까운 위치

두 조건 모두 필요한 이유: 접촉력만 보면 물체를 집을 때도 트리거될 수 있고, 기하학적 조건만 보면 실제 충돌 없이 스쳐 지나가도 종료될 수 있어서.

`time_out=True` 없음 → **실패 종료**. 장애물을 치면 에피소드가 끝나고 미래 보상 전체를 잃는다. `obstacle_penalty`가 "가까이 가지 마라"는 점진적 신호라면, 이 종료 조건은 "실제로 치면 그 에피소드는 끝"이라는 하드 경계다.

---

### 종료 조건 요약

| 조건 | 트리거 | time_out | 암묵적 패널티 |
|------|--------|----------|--------------|
| `time_out` | 20초 경과 | ✓ | 없음 (정상 종료) |
| `object_dropped` | 물체 Z < -10cm | ✗ | 강함 (미래 보상 0) |
| `obstacle_contact` | 접촉력 > 1N + 기하학 확인 | ✗ | 강함 (미래 보상 0) |

---

## 커리큘럼

```
0           20만        30만        60만        70만   (스텝)
│           │           │           │           │
├───────────┼───────────┼───────────┼───────────┤
│ Stage 0-A │ Stage 0-B │ Stage 1   │ Stage 2   │
│ 장애물 없음 │ 장애물 없음 │ 장애물 1개 │ 장애물 2개 │
│ 좁은 스폰  │ 스폰 확장  │ 전체 범위  │ penalty↑  │
```

- **오브젝트 스폰 좁은 범위**: x (0.38, 0.46), y (-0.05, 0.02)
- **오브젝트 스폰 전체 범위**: x (0.30, 0.55), y (-0.25, 0.05)
- **Play-v0**: stage threshold 모두 0 → 처음부터 장애물 2개 활성화

---

## 명령어

### 학습

```bash
# 처음부터
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-PickPlace-Obstacle-Franka-v0 \
    --headless \
    --num_envs 1024

# 체크포인트에서 이어서
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-PickPlace-Obstacle-Franka-v0 \
    --headless \
    --num_envs 1024 \
    --load_run 2026-08-28_00-16-05 \
    --checkpoint model_2999.pt
```

### 평가

```bash
# Play-v0 사용 — 처음부터 장애물 활성화
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_debug_ee.py \
    --task Isaac-PickPlace-Obstacle-Franka-Play-v0 \
    --num_envs 1 \
    --dashboard \
    --checkpoint logs/rsl_rl/franka_pick_place_obstacle/<날짜폴더>/model_XXXX.pt

# 최신 체크포인트 찾기
find logs/rsl_rl/franka_pick_place_obstacle -name "*.pt" | xargs ls -t | head -3
```

### 텐서보드

```bash
./isaaclab.sh -p -m tensorboard.main \
    --logdir logs/rsl_rl/franka_pick_place_obstacle \
    --port 6006

# 또는
tensorboard --logdir logs/rsl_rl/franka_pick_place_obstacle --port 6006
```

브라우저: `http://localhost:6006`

---

### 텐서보드 전체 지표 목록

#### Train/ — 학습 전체 현황

| 키 | 설명 | 잘 되고 있으면 |
|----|------|----------------|
| `Train/mean_reward` | 에피소드당 평균 누적 보상 | 꾸준히 우상향 |
| `Train/mean_episode_length` | 평균 에피소드 길이 (스텝 수) | 초반엔 짧다가, 태스크를 오래 수행할수록 길어짐 |
| `Train/mean_reward/time` | 위와 동일, x축이 스텝이 아닌 실제 시간(초) | — |
| `Train/mean_episode_length/time` | 위와 동일, x축 시간 기준 | — |

#### Loss/ — PPO 손실값

| 키 | 설명 | 정상 범위 / 해석 |
|----|------|-----------------|
| `Loss/value` | 가치 함수 손실 (MSE). Critic이 리턴을 얼마나 잘 예측하는지 | 초반엔 크다가 점점 감소해야 함 |
| `Loss/surrogate` | 정책 손실 (PPO Clip). Actor가 얼마나 좋은 방향으로 업데이트되는지 | 작은 양수 또는 음수 유지 |
| `Loss/entropy` | 정책 엔트로피. 행동의 다양성 지표 | 너무 빨리 0으로 떨어지면 조기 수렴 (탐색 부족) |
| `Loss/learning_rate` | 현재 학습률 (스케줄러 적용 시 변화) | 설정값 유지 또는 감소 |

#### Policy/ — 정책 파라미터

| 키 | 설명 | 해석 |
|----|------|------|
| `Policy/mean_std` | 정책 출력의 평균 표준편차. 액션의 불확실성 | 초반엔 높다가 학습될수록 낮아짐. 너무 빨리 0이 되면 탐색 부족 |

#### Episode/ — 에피소드별 보상 항목 (Isaac Lab RewardManager 자동 로깅)

각 보상 항목의 **에피소드 평균값**이 종료 시마다 기록된다.

| 키 | 대응 보상 | 해석 |
|----|----------|------|
| `Episode/reach_object` | `reach_object` (+1.0) | 물체에 얼마나 접근했는지. 초반부터 올라야 함 |
| `Episode/grasp_effort` | `grasp_effort` (+12.0) | 파지 시도 빈도. 학습 중반부터 증가해야 함 |
| `Episode/lift_object` | `lift_object` (+25.0) | **핵심 지표.** 0에서 양수로 올라가는 시점이 파지 학습 시점 |
| `Episode/move_to_goal` | `move_to_goal` (+10.0) | 들어올린 채로 목표 이동. lift 이후 등장 |
| `Episode/place_success` | `place_success` (+50.0) | 놓기 성공률. 가장 늦게 올라감 |
| `Episode/obstacle_penalty` | `obstacle_penalty` (0→-3.0) | 음수값. 0에 가까울수록 장애물 회피 잘 됨 |
| `Episode/action_rate` | `action_rate` (-1e-3) | 음수값. -0에 가까울수록 동작 부드러움 |
| `Episode/joint_vel` | `joint_vel` (-1e-3) | 음수값. -0에 가까울수록 천천히 움직임 |
| `Episode/time_outs` | `time_out` 종료 조건 | 0~1 비율. 1이면 모든 에피소드가 시간 초과로 끝남 (실패 없이 버팀) |

#### Perf/ — 연산 성능

| 키 | 설명 | 해석 |
|----|------|------|
| `Perf/total_fps` | 초당 처리 프레임 수 (num_envs × steps/sec) | 높을수록 좋음. GPU 포화 여부 확인용 |
| `Perf/collection_time` | 롤아웃 수집 시간 (초) | 낮을수록 좋음 |
| `Perf/learning_time` | 정책 업데이트 시간 (초) | 낮을수록 좋음 |

---

### 학습 진단 체크리스트

| 증상 | 원인 가능성 | 확인할 지표 |
|------|------------|-------------|
| `mean_reward`가 전혀 안 올라감 | 탐색 부족, 보상 설계 문제 | `Episode/reach_object`, `Loss/entropy` |
| `lift_object`가 0에서 안 움직임 | 파지 미발견 (가장 흔한 병목) | `Episode/grasp_effort`, 커리큘럼 스폰 범위 확인 |
| `Loss/entropy`가 급격히 0으로 | 조기 수렴, 탐색 중단 | `entropy_coef` 올리기 (현재 0.01) |
| `Loss/value`가 계속 크게 유지 | Critic이 리턴 예측 못함 | `num_learning_epochs`, `num_mini_batches` 조정 |
| `Perf/total_fps`가 갑자기 떨어짐 | GPU 메모리 부족, 다른 프로세스 간섭 | `nvidia-smi` 확인 |
| `Episode/time_outs`가 1.0 고정 | 에이전트가 아무것도 못 하고 버팀 | `reach_object` 값 확인, 환경 초기화 확인 |

### 실시간 대시보드

```bash
# 터미널 A — 평가 실행 (JSON 출력)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_debug_ee.py \
    --task Isaac-PickPlace-Obstacle-Franka-Play-v0 \
    --num_envs 1 \
    --dashboard \
    --checkpoint logs/rsl_rl/franka_pick_place_obstacle/<날짜폴더>/model_XXXX.pt

# 터미널 B — 대시보드 서버
python3 scripts/dashboard_server.py
```

브라우저: `http://localhost:8765`

표시 항목: 조인트 목표각도(7개) / 현재 각도 / 그리퍼 OPEN·CLOSE / EE·물체·장애물 위치 / 정책 원시 출력값
