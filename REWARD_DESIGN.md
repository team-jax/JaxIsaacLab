# Team JAX — 보상 함수 설계 전체 기록

**프로젝트:** 휴머노이드 Bimanual Box Grasp  
**마지막 업데이트:** 2026-07-09  
**목적:** v1~v5까지 보상 함수 설계 변천사를 코드 수준으로 기록

---

## 설계 원칙

> **태스크를 수행하지 않았을 때 보상 > 0 이면 잘못된 설계다.**

강화학습 에이전트는 보상을 최대화하는 가장 쉬운 방법을 찾는다.  
의도한 행동(박스 잡기)이 아닌 다른 방법으로도 보상을 받을 수 있으면, 에이전트는 반드시 그 방법을 찾아낸다.  
이것을 **Reward Hacking**이라 하며, 이 문서는 우리가 마주친 모든 reward hacking 사례와 해결 과정을 기록한다.

---

## v1 — 초기 설계 (직관 기반)

**시기:** 2026년 6월 초  
**결과:** 실패 — 로봇이 박스 근처에서 맴돌기만 함

### 보상 항목

| 항목 | Weight | 함수 | 접촉 필요? |
|------|--------|------|-----------|
| palm_approach_box_side | +2.0 | 손목↔박스측면 거리 기반 | ❌ 없음 |
| palm_facing_box | +2.0 | 손바닥 법선 방향 정렬 | ❌ 없음 |
| box_upright_bonus | +1.0 | 박스 쿼터니언 w 성분 | ❌ 없음 |
| box_not_fallen | +1.0 | 박스 z > 0.6m | ❌ 없음 |
| left_hand_contact | +1.0 | ContactSensor 힘 > 1N | ✅ 있음 |
| right_hand_contact | +1.0 | ContactSensor 힘 > 1N | ✅ 있음 |
| box_lift | +3.0 | 박스 z 높이 상승 | ❌ 없음 |
| action_rate | −0.05 | 연속 액션 변화량 L2 | — |

### 코드

```python
# palm_approach_box_side
def palm_approach_box_side(env, palm_cfg, box_cfg, side,
                            box_half_y=0.14, palm_offset=0.10, std=0.1):
    wrist_pos = robot.data.body_pos_w[:, palm_cfg.body_ids[0], :3]
    box_pos   = box.data.root_pos_w[:, :3]
    sign   = 1.0 if side == "left" else -1.0
    target = box_pos + torch.tensor([0.0, sign*(box_half_y+palm_offset), 0.0])
    dist = torch.norm(wrist_pos - target, dim=-1)
    return 1.0 - torch.tanh(dist / std)   # 가까울수록 1.0

# palm_facing_box
def palm_facing_box(env, palm_cfg, box_cfg, side):
    wrist_quat = robot.data.body_quat_w[:, palm_cfg.body_ids[0], :]
    palm_local = torch.zeros(env.num_envs, 3)
    palm_local[:, 1] = 1.0 if side=="left" else -1.0
    palm_world = quat_rotate(wrist_quat, palm_local)
    to_box = normalize(box_pos - wrist_pos)
    alignment = (palm_world * to_box).sum(dim=-1)
    return torch.clamp(alignment, 0.0, 1.0)  # 정렬될수록 1.0

# box_lift (접촉 무관)
def old_box_lift(env, box_cfg, initial_height=1.00, std=0.08):
    lift = torch.clamp(box.data.root_pos_w[:, 2] - initial_height, min=0.0)
    return 1.0 - torch.exp(-lift / std)
```

### 문제점

**① palm_approach_box_side — 박스 옆에 서 있기만 해도 보상**
```
기대: 손을 박스 측면으로 이동 → 접촉
실제: 손을 박스 옆 0.24m 위치에 고정 → 항상 높은 보상
     (dist≈0 이면 tanh(0)=0, 보상=1.0)
```

**② palm_facing_box — 팔을 위로 올리면 보상**
```
기대: 손바닥이 박스를 향해야 보상
실제: 팔을 박스 위쪽으로 올리면 wrist Y축이 아래(=박스 방향)를 향함
     → 박스를 안 잡아도 팔만 올려도 palm_facing 보상 획득
```

**③ box_lift — 밀어서 날리기 가능**
```
기대: 잡고 들어올리기
실제: 팔로 박스 측면을 밀면 박스가 공중에 뜸 → box_lift 보상 획득
     접촉 조건 없어서 차단 불가
```

**④ 보상 우선순위 역전**
```
비접촉 보상 합: palm_approach(2.0) + palm_facing(2.0) + box_upright(1.0) + box_alive(1.0) = 6.0
실제 태스크 보상: left_contact(1.0) + right_contact(1.0) = 2.0
→ 박스에 손을 안 대는 전략이 3배 이득
```

---

## v2 — 비접촉 보상 전부 제거

**시기:** 2026년 6월 중순  
**결과:** 부분 성공 → 새로운 문제 발견

### 변경 내용

비접촉 보상을 전부 제거하고 ContactSensor 기반 보상만 유지.

| 항목 | Weight | 변경 |
|------|--------|------|
| palm_approach_box_side | — | ❌ 제거 |
| palm_facing_box | — | ❌ 제거 |
| box_upright_bonus | — | ❌ 제거 |
| box_not_fallen | — | ❌ 제거 |
| left_hand_contact | +1.0 | ✅ 유지 |
| right_hand_contact | +1.0 | ✅ 유지 |
| no_contact_penalty | −3.0 | ✅ 추가 (접촉 없으면 패널티) |
| contact_gated_lift | +3.0 | ✅ 추가 (접촉 + 들기) |
| box_placed_on_table | +5.0 | ✅ 추가 (내려놓기 보상) |

### 코드

```python
# no_contact_penalty
def no_contact_penalty(env, contact_cfg, force_threshold=1.0):
    sensor = env.scene.sensors[contact_cfg.name]
    force = sensor.data.net_forces_w_history[:, 0, 0, :]
    not_touching = (torch.norm(force, dim=-1) < force_threshold).float()
    return -not_touching   # 안 닿으면 -1, 닿으면 0

# box_placed_on_table
def box_placed_on_table(env, box_cfg, left_contact_cfg, right_contact_cfg,
                         table_z_min=0.88, table_z_max=1.05, force_threshold=1.0):
    box_z = box.data.root_pos_w[:, 2]
    on_table = (box_z > table_z_min) & (box_z < table_z_max)
    # 손이 떨어진 상태 = 내려놓기 완료
    no_contact = (left_force < threshold) & (right_force < threshold)
    return (on_table & no_contact).float()
```

### 문제점

**① no_contact_penalty가 내려놓기를 불가능하게 만듦**
```
no_contact_penalty weight=-3.0
→ 손이 박스에서 떨어지면 매 스텝 -3점
→ "항상 잡고 있기"가 최적 전략
→ 내려놓기(place) 학습 불가능

box_placed_on_table은 no_contact 조건을 만족해야 하는데,
no_contact_penalty 때문에 손을 절대 안 놓음 → box_placed 보상 획득 불가능
```

**② box_placed_on_table이 에피소드 시작부터 만점**
```
초기 상태: 박스 z=1.00 (테이블 위), 양손 접촉 없음
→ on_table=True, no_contact=True
→ box_placed=5.0 (만점) 이 에피소드 시작부터 항상 나옴

텐서보드 확인:
  box_placed = 4.98 (최대 5.0)  ← 처음부터 항상 만점
  bimanual_grasp = 0.00          ← 전혀 학습 안 됨
```

**③ 한 손 밀기 전략**
```
left_contact(1.0) + right_contact(1.0) = 각각 독립 보상
→ 한 손으로만 박스에 대고 밀어서 반 보상 획득 가능
→ 양팔 협조 학습 안 됨
```

---

## v3 — AND 조건 + contact_gated_lift

**시기:** 2026년 6월 말  
**결과:** 부분 성공 → 윗면 누르기 꼼수 등장

### 변경 내용

| 항목 | Weight | 변경 |
|------|--------|------|
| no_contact_penalty | — | ❌ 제거 (내려놓기 방해) |
| box_placed_on_table | — | ❌ 제거 (초기 만점) |
| left_hand_contact | +1.0 | ✅ 유지 |
| right_hand_contact | +1.0 | ✅ 유지 |
| bimanual_contact_grasp | +3.0 | ✅ 추가 (양손 AND) |
| contact_gated_lift | +3.0 | ✅ 수정 (접촉 게이트) |
| hold_at_height | +5.0 | ✅ 추가 (목표 높이 유지) |
| sustained_hold | +2.0 | ✅ 추가 (지속 접촉) |

### 코드

```python
# bimanual_contact_grasp — AND 조건으로 한 손 전략 차단
def bimanual_contact_grasp(env, left_contact_cfg, right_contact_cfg, force_threshold=1.0):
    left_force  = left_sensor.data.net_forces_w_history[:, 0, 0, :]
    right_force = right_sensor.data.net_forces_w_history[:, 0, 0, :]
    both_contact = (
        (torch.norm(left_force,  dim=-1) > force_threshold) &
        (torch.norm(right_force, dim=-1) > force_threshold)
    )
    return both_contact.float()   # 둘 다 닿아야 1.0

# contact_gated_lift — 접촉 없으면 들기 보상 0
def contact_gated_lift(env, box_cfg, left_contact_cfg, right_contact_cfg,
                        initial_height=1.00, std=0.08, force_threshold=1.0):
    lift = torch.clamp(box.data.root_pos_w[:, 2] - initial_height, min=0.0)
    lift_reward = 1.0 - torch.exp(-lift / std)

    both_contact = (
        (left_force  > threshold) &
        (right_force > threshold)
    ).float()

    return lift_reward * both_contact   # 접촉 없으면 0
```

### 문제점

**① 윗면 누르기 꼼수**
```
박스 윗면을 양손으로 누르면:
  - both_contact = True (접촉 있음)
  - box_z 상승 (눌러서 아래로 가다가 물리 반발로 올라감)
  - contact_gated_lift = 보상 발생

윗면 누르기는 박스를 불안정하게 만들어 원하는 파지 전략이 아님.
또한 수직으로 들어올리지 않고 박스를 옆으로 밀어 올리는 동작도 관측됨.
```

**② 접촉 위치에 대한 보상 구분 없음**
```
어디서 잡든 동일한 보상
→ 측면, 윗면, 아래면 모두 동등
→ 실제 양손 파지에 필요한 "측면 접촉"을 따로 유도할 방법 없음
```

---

## v4 — 측면 접촉 필터 (엄격)

**시기:** 2026년 7월 6일  
**결과:** 실패 — 학습 신호 희소로 2000 iter 동안 아무것도 학습 못 함

### 변경 내용

박스 측면에 닿은 경우에만 보상을 주는 `_is_side_contact()` 헬퍼 추가.

| 항목 | 변경 |
|------|------|
| bimanual_contact_grasp | 양손 측면 접촉 시에만 보상 |
| contact_gated_lift | 양손 측면 접촉 시에만 보상 |
| hold_at_height | 양손 측면 접촉 시에만 보상 |

### 코드

```python
def _is_side_contact(wrist_pos, box_pos, side,
                     box_half_z=0.10, box_half_y=0.14):
    """
    조건 1: 손목 z < 박스 상단 z + box_half_z  (윗면 위치 아님)
    조건 2: 손목이 올바른 Y 방향(측면)에 있음
    """
    sign = 1.0 if side == "left" else -1.0
    height_diff = wrist_pos[:, 2] - box_pos[:, 2]
    not_on_top  = height_diff < box_half_z          # 윗면 차단
    at_side     = sign * (wrist_pos[:, 1] - box_pos[:, 1]) > 0  # 측면 확인
    return not_on_top & at_side

# bimanual_contact_grasp v4 — 측면만 인정
def bimanual_contact_grasp_v4(env, ...):
    both_contact = (left_force > thr) & (right_force > thr)
    both_side = (
        _is_side_contact(left_wrist,  box_pos, "left") &
        _is_side_contact(right_wrist, box_pos, "right")
    )
    # 측면 접촉이 아니면 0
    return (both_contact & both_side).float()
```

### 문제점

**학습 신호 희소 (Sparse Reward Problem)**

```
탐색 초반에 손이 정확히 박스 측면에 닿을 확률은 매우 낮음:
  - 박스 측면 면적 ≈ 0.28m × 0.20m = 0.056 m²
  - 전체 접촉 가능 면적 (윗면+측면+뒷면) ≈ 0.3 m²
  - 측면 확률 ≈ 19%
  - 양손 동시 측면: 0.19² ≈ 3.6%

텐서보드 결과 (2000 iteration 완료):
  bimanual_grasp = 0.01   ← 거의 0
  mean_reward    = -0.81  ← 항상 음수
  learning_rate  = 0.0000 ← 학습 멈춤

로봇 행동: 그냥 서 있음. 아무것도 안 하는 게 최선.
이유: 접촉 시도 → 측면 아님 → 보상 0
     + action_rate 패널티 → 움직이면 오히려 손해
```

---

## v5 — 등급제 (현재 최종)

**시기:** 2026년 7월 7일~  
**결과:** 성공 — mean_reward=145.59, 2249 iteration 완료

### 핵심 아이디어

완전 차단(측면만 1.0, 나머지 0.0) 대신 **등급제** 도입:
- 측면 접촉: `side_reward=1.0` × weight → 정상 보상
- 기타 접촉: `other_reward=0.2` × weight → 낮은 보상, 탐색은 허용

탐색 중에는 기타 접촉으로 작은 학습 신호를 받으면서,  
측면 접촉이 5배 이득이므로 자연스럽게 측면 접촉으로 수렴.

### 변경 내용

| 항목 | Weight | 측면 접촉 | 기타 접촉 | 미접촉 |
|------|--------|----------|----------|-------|
| left_hand_contact | +1.0 | 1.0 | 0.2 | 0.0 |
| right_hand_contact | +1.0 | 1.0 | 0.2 | 0.0 |
| bimanual_grasp | +3.0 | 1.0 | 0.2 | 0.0 |
| contact_gated_lift | +6.0 | lift_rew | 0.0 | 0.0 |
| hold_at_height | +5.0 | height_rew | 0.0 | 0.0 |
| sustained_hold | +2.0 | hist_rew | 0.0 | 0.0 |
| action_rate | −0.01 | — | — | — |
| action_l2 | −0.001 | — | — | — |
| joint_torques | −0.0001 | — | — | — |
| joint_vel | −0.005 | — | — | — |
| joint_limits | −0.5 | — | — | — |

> **참고:** `contact_gated_lift`, `hold_at_height`, `sustained_hold`는 측면 접촉이 아니면 완전히 0.  
> 들기/높이 보상은 엄격하게 측면만 허용. 탐색 허용은 `single_hand_contact`와 `bimanual_grasp`에서만.

### 최종 코드

```python
def _is_side_contact(wrist_pos, box_pos, side,
                     box_half_z=0.10, box_half_y=0.14):
    sign = 1.0 if side == "left" else -1.0
    not_on_top = (wrist_pos[:, 2] - box_pos[:, 2]) < box_half_z
    at_side    = sign * (wrist_pos[:, 1] - box_pos[:, 1]) > 0
    return not_on_top & at_side


def single_hand_contact(env, contact_cfg, palm_cfg, box_cfg, side,
                         force_threshold=1.0, side_reward=1.0, other_reward=0.2):
    force       = sensor.data.net_forces_w_history[:, 0, 0, :]
    has_contact = torch.norm(force, dim=-1) > force_threshold
    wrist_pos   = robot.data.body_pos_w[:, palm_cfg.body_ids[0], :3]
    box_pos     = box.data.root_pos_w[:, :3]
    is_side     = _is_side_contact(wrist_pos, box_pos, side)
    reward      = torch.where(is_side, side_reward, other_reward)  # 등급제
    return reward * has_contact.float()


def bimanual_contact_grasp(env, left_contact_cfg, right_contact_cfg,
                            left_palm_cfg, right_palm_cfg, box_cfg,
                            force_threshold=1.0, side_reward=1.0, other_reward=0.2):
    both_contact = (left_force > thr) & (right_force > thr)
    both_side    = (
        _is_side_contact(left_wrist,  box_pos, "left") &
        _is_side_contact(right_wrist, box_pos, "right")
    )
    reward = torch.where(both_side, side_reward, other_reward)
    return reward * both_contact.float()


def contact_gated_lift(env, box_cfg, left_contact_cfg, right_contact_cfg,
                        left_palm_cfg, right_palm_cfg,
                        initial_height=1.00, std=0.05, force_threshold=1.0):
    lift        = torch.clamp(box.data.root_pos_w[:, 2] - initial_height, min=0.0)
    lift_reward = 1.0 - torch.exp(-lift / std)  # std=0.05: 조금만 들어도 보상 급증
    both_contact = (
        (left_force  > thr) & (right_force > thr) &
        _is_side_contact(left_wrist,  box_pos, "left") &  # 엄격: 측면만
        _is_side_contact(right_wrist, box_pos, "right")
    ).float()
    return lift_reward * both_contact


def hold_at_height(env, box_cfg, left_contact_cfg, right_contact_cfg,
                    left_palm_cfg, right_palm_cfg,
                    target_height=1.50, std=0.08, force_threshold=1.0):
    both_contact = (
        (left_force  > thr) & (right_force > thr) &
        _is_side_contact(left_wrist,  box_pos, "left") &
        _is_side_contact(right_wrist, box_pos, "right")
    ).float()
    box_z         = box.data.root_pos_w[:, 2]
    height_reward = 1.0 - torch.tanh(torch.abs(box_z - target_height) / std)
    return height_reward * both_contact


def sustained_contact_hold(env, box_cfg, left_contact_cfg, right_contact_cfg,
                             target_height=1.50, height_tol=0.15, force_threshold=1.0):
    # history 전체에서 접촉이 유지돼야 보상 (순간 접촉 차단)
    left_all  = (torch.norm(left_hist,  dim=-1).squeeze(-1) > thr).all(dim=-1)
    right_all = (torch.norm(right_hist, dim=-1).squeeze(-1) > thr).all(dim=-1)
    box_z     = box.data.root_pos_w[:, 2]
    at_height = (box_z > target_height - height_tol) & (box_z < target_height + height_tol)
    return (left_all & right_all & at_height).float()
```

### cfg 설정

```python
# h1_upper_body_env_cfg.py RewardsCfg

left_hand_contact = RewTerm(
    func=mdp.single_hand_contact, weight=1.0,
    params={"contact_cfg": LEFT_CONTACT_CFG, "palm_cfg": LEFT_PALM_CFG,
            "box_cfg": BOX_CFG, "side": "left",
            "force_threshold": 1.0, "side_reward": 1.0, "other_reward": 0.2})

right_hand_contact = RewTerm(
    func=mdp.single_hand_contact, weight=1.0,
    params={"contact_cfg": RIGHT_CONTACT_CFG, "palm_cfg": RIGHT_PALM_CFG,
            "box_cfg": BOX_CFG, "side": "right",
            "force_threshold": 1.0, "side_reward": 1.0, "other_reward": 0.2})

bimanual_grasp = RewTerm(
    func=mdp.bimanual_contact_grasp, weight=3.0,
    params={"left_contact_cfg": LEFT_CONTACT_CFG, "right_contact_cfg": RIGHT_CONTACT_CFG,
            "left_palm_cfg": LEFT_PALM_CFG, "right_palm_cfg": RIGHT_PALM_CFG,
            "box_cfg": BOX_CFG, "force_threshold": 1.0,
            "side_reward": 1.0, "other_reward": 0.2})

box_lift = RewTerm(
    func=mdp.contact_gated_lift, weight=6.0,          # 3.0 → 6.0 (망설임 해소)
    params={"box_cfg": BOX_CFG,
            "left_contact_cfg": LEFT_CONTACT_CFG, "right_contact_cfg": RIGHT_CONTACT_CFG,
            "left_palm_cfg": LEFT_PALM_CFG, "right_palm_cfg": RIGHT_PALM_CFG,
            "initial_height": 1.00, "std": 0.05,      # 0.08 → 0.05 (초기 들기 보상 민감도↑)
            "force_threshold": 1.0})

hold_height = RewTerm(
    func=mdp.hold_at_height, weight=5.0,
    params={"box_cfg": BOX_CFG,
            "left_contact_cfg": LEFT_CONTACT_CFG, "right_contact_cfg": RIGHT_CONTACT_CFG,
            "left_palm_cfg": LEFT_PALM_CFG, "right_palm_cfg": RIGHT_PALM_CFG,
            "target_height": 1.50, "std": 0.08, "force_threshold": 1.0})

sustained_hold = RewTerm(
    func=mdp.sustained_contact_hold, weight=2.0,
    params={"box_cfg": BOX_CFG,
            "left_contact_cfg": LEFT_CONTACT_CFG, "right_contact_cfg": RIGHT_CONTACT_CFG,
            "target_height": 1.50, "height_tol": 0.15, "force_threshold": 1.0})

action_rate   = RewTerm(func=mdp.action_rate_l2, weight=-0.01)   # -0.05 → -0.01
action_l2     = RewTerm(func=mdp.action_l2,      weight=-0.001)
joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-1e-4,
                        params={"asset_cfg": ACTIVE_ARM_CFG})
joint_vel     = RewTerm(func=mdp.joint_vel_l2,   weight=-0.005,
                        params={"asset_cfg": ACTIVE_ARM_CFG})
joint_acc     = RewTerm(func=mdp.joint_acc_l2,   weight=-1e-5,
                        params={"asset_cfg": ACTIVE_ARM_CFG})
joint_limits  = RewTerm(func=mdp.joint_pos_limits, weight=-0.5,
                        params={"asset_cfg": ACTIVE_ARM_CFG})
```

### 학습 결과

| Run | Iterations | Mean Reward | Ep. Length | LR 최종 |
|-----|-----------|-------------|------------|---------|
| 2026-07-07_08-01-57 | 2249 (완료) | **145.59** | 600/600 | 0.0000 (소멸) |
| 2026-07-07_19-48-57 | 763 (중단) | 138.88 | 600/600 | 0.0001 |

`mean_episode_length = 600` = 타임아웃까지 에피소드 생존. 박스를 떨어뜨리지 않음.

---

## 버전별 비교 요약

| 버전 | 핵심 변경 | 결과 | 주요 실패 원인 |
|------|----------|------|--------------|
| v1 | 접근+방향+들기 보상 | ❌ 맴돌기 | 비접촉 보상이 더 쉬움 |
| v2 | 비접촉 보상 제거 | ⚠️ 부분 성공 | 초기 만점, 한 손 전략, no_contact_penalty 역효과 |
| v3 | AND 조건 + contact_gated | ⚠️ 부분 성공 | 윗면 누르기 꼼수 |
| v4 | 측면 필터 (엄격) | ❌ 학습 불가 | 보상 신호 희소 |
| v5 | 등급제 | ✅ 성공 | LR 소멸 (미해결) |

---

## 보상 설계 체크리스트

새 보상 추가 전 반드시 확인:

- [ ] 에피소드 시작 상태(t=0)에서 이 보상이 > 0인가? → YES면 제거
- [ ] 접촉 없이도 이 보상을 받을 수 있는가? → YES면 ContactSensor 조건 추가
- [ ] 이 보상을 받는 가장 쉬운 방법이 의도한 행동인가? → NO면 수정
- [ ] 이 보상이 없으면 에이전트가 다음 단계로 진행할 동기가 있는가? → NO면 이 보상 필요
- [ ] 패널티 보상이 초반 탐색을 막지 않는가? → 막으면 커리큘럼 적용
