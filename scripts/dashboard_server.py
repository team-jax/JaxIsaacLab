#!/usr/bin/env python3
"""Isaac-PnP 실시간 대시보드 서버

사용법:
  1. 터미널 A (평가 실행):
       ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_debug_ee.py \
           --task Isaac-PickPlace-Obstacle-Franka-v0 \
           --num_envs 1 --dashboard

  2. 터미널 B (대시보드 서버):
       python3 scripts/dashboard_server.py

  3. 브라우저: http://localhost:8765
"""

import http.server
import json
import os
import sys

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
STATE_FILE = os.path.join(os.path.dirname(__file__), "..", "logs", "dashboard.json")

HTML = r"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PnP 로봇 대시보드</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=IBM+Plex+Sans+KR:wght@400;500;600;700&display=swap">
<style>
:root {
  --bg: #0C1219;
  --surf: #131D2A;
  --surf2: #1A2538;
  --border: #243047;
  --text: #BDC9DC;
  --dim: #52637A;
  --accent: #3B8FFF;
  --green: #2ECC71;
  --yellow: #F0B429;
  --red: #E05D5D;
  --mono: 'JetBrains Mono', monospace;
  --sans: 'IBM Plex Sans KR', 'Apple SD Gothic Neo', sans-serif;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: var(--bg); color: var(--text); font-family: var(--sans); font-size: 14px; line-height: 1.6; }

/* ── 헤더 ── */
.hd {
  display: flex; align-items: center; gap: 10px;
  padding: 12px 20px; border-bottom: 1px solid var(--border);
  background: var(--surf);
}
.dot { width: 9px; height: 9px; border-radius: 50%; background: var(--dim); flex-shrink: 0; }
.dot.live { background: var(--green); box-shadow: 0 0 8px var(--green); animation: blink 1.6s infinite; }
.dot.dead { background: var(--red); }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.3} }
.hd-title { font-family: var(--mono); font-size: .9rem; font-weight: 700; color: var(--accent); }
.hd-meta { font-family: var(--mono); font-size: .72rem; color: var(--dim); margin-left: 6px; }
.hd-time { font-family: var(--mono); font-size: .72rem; color: var(--dim); margin-left: auto; }

/* ── 2단 레이아웃 ── */
.layout { display: grid; grid-template-columns: 1fr 1fr; gap: 1px; background: var(--border); }
.col { background: var(--bg); padding: 20px; }

/* ── 섹션 ── */
.section { margin-bottom: 28px; }
.section:last-child { margin-bottom: 0; }
.sec-title {
  font-family: var(--mono); font-size: .62rem; font-weight: 700;
  letter-spacing: .14em; text-transform: uppercase;
  color: var(--accent); margin-bottom: 4px;
}
.sec-desc {
  font-size: .72rem; color: var(--dim); margin-bottom: 14px;
  line-height: 1.5;
}

/* ── 조인트 명령 카드 ── */
.joint-card {
  background: var(--surf); border: 1px solid var(--border);
  border-radius: 6px; padding: 12px 14px; margin-bottom: 8px;
}
.joint-head {
  display: flex; justify-content: space-between; align-items: baseline;
  margin-bottom: 8px;
}
.joint-name { font-family: var(--mono); font-size: .75rem; color: var(--dim); }
.joint-name b { color: var(--text); font-size: .78rem; }
.joint-cmd {
  font-family: var(--mono); font-size: 1.15rem; font-weight: 700;
}
.joint-row2 {
  display: flex; gap: 16px; font-family: var(--mono); font-size: .68rem; color: var(--dim);
}
.joint-row2 span b { color: var(--text); }

/* 범위 바 */
.rangebar { position: relative; height: 8px; background: var(--surf2); border-radius: 4px; margin-top: 8px; }
.rangebar-cur { position: absolute; top: 1px; height: 6px; border-radius: 3px; transition: left .12s, width .12s; }
.rangebar-cmd { position: absolute; top: 0; width: 3px; height: 8px; border-radius: 2px; background: #fff; opacity:.85; transition: left .12s; }

/* ── 그리퍼 ── */
.gripper-big {
  border: 2px solid var(--border); border-radius: 8px; padding: 18px 20px;
  text-align: center; transition: border-color .3s, background .3s;
}
.gripper-big.open { border-color: var(--green); background: rgba(46,204,113,.07); }
.gripper-big.close { border-color: var(--red); background: rgba(224,93,93,.07); }
.gripper-state { font-family: var(--mono); font-size: 1.8rem; font-weight: 700; }
.gripper-big.open .gripper-state { color: var(--green); }
.gripper-big.close .gripper-state { color: var(--red); }
.gripper-sub { font-size: .72rem; color: var(--dim); margin-top: 6px; }

/* ── 위치 카드 ── */
.pos-card {
  background: var(--surf); border: 1px solid var(--border);
  border-radius: 6px; padding: 14px 16px; margin-bottom: 10px;
}
.pos-label { font-size: .72rem; font-weight: 600; color: var(--dim); margin-bottom: 10px; }
.pos-label b { color: var(--text); }
.pos-axes { display: flex; gap: 12px; }
.pos-axis-item { flex: 1; text-align: center; }
.pos-axis-letter {
  font-family: var(--mono); font-size: .6rem; color: var(--dim);
  display: block; margin-bottom: 3px;
}
.pos-axis-val {
  font-family: var(--mono); font-size: 1.0rem; font-weight: 600;
  display: block;
}

/* 거리 표시 */
.dist-box {
  display: flex; align-items: center; gap: 12px;
  background: var(--surf); border: 1px solid var(--border);
  border-radius: 6px; padding: 14px 16px; margin-bottom: 10px;
}
.dist-val { font-family: var(--mono); font-size: 1.6rem; font-weight: 700; }
.dist-unit { font-family: var(--mono); font-size: .78rem; color: var(--dim); }
.dist-bar-wrap { flex: 1; }
.dist-bar { height: 6px; background: var(--surf2); border-radius: 3px; overflow: hidden; }
.dist-bar-fill { height: 6px; border-radius: 3px; transition: width .15s, background .3s; }
.dist-meaning { font-size: .7rem; color: var(--dim); margin-top: 5px; }

/* ── 정책 출력 패널 ── */
.action-wrap { display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; }
.act-cell {
  background: var(--surf); border: 1px solid var(--border);
  border-radius: 4px; padding: 8px 6px; text-align: center;
}
.act-cell-name { font-family: var(--mono); font-size: .6rem; color: var(--dim); display: block; }
.act-cell-val { font-family: var(--mono); font-size: .82rem; font-weight: 600; display: block; margin-top: 2px; }
.act-mini-bar { height: 3px; background: var(--surf2); border-radius: 1px; margin-top: 5px; position: relative; overflow: hidden; }
.act-mini-fill { position: absolute; top: 0; height: 3px; border-radius: 1px; }

/* ── 도움말 박스 ── */
.help-box {
  background: var(--surf2); border-left: 3px solid var(--accent);
  border-radius: 0 4px 4px 0; padding: 10px 14px;
  font-size: .71rem; color: var(--dim); line-height: 1.7; margin-bottom: 14px;
}
.help-box b { color: var(--text); }

/* ── 반응형 ── */
@media (max-width: 900px) {
  .layout { grid-template-columns: 1fr; }
  .action-wrap { grid-template-columns: repeat(4, 1fr); }
}
</style>
</head>
<body>

<div class="hd">
  <div class="dot" id="dot"></div>
  <span class="hd-title">Isaac-PnP 로봇 대시보드</span>
  <span class="hd-meta" id="hd-meta">— 연결 대기 중</span>
  <span class="hd-time" id="hd-time"></span>
</div>

<div class="layout">

<!-- ═══════════════════════════════ 왼쪽 열 ═══════════════════════════════ -->
<div class="col">

  <!-- 조인트 명령 -->
  <div class="section">
    <div class="sec-title">▶ 조인트 명령 각도</div>
    <div class="sec-desc">
      이번 스텝에서 정책(신경망)이 각 관절에 보낸 <b style="color:var(--text)">목표 각도</b>입니다.<br>
      공식: <b style="color:var(--accent);font-family:var(--mono)">목표 = 기본자세 + 0.5 × 정책출력</b>
    </div>
    <div id="joints"></div>
  </div>

  <!-- 그리퍼 -->
  <div class="section">
    <div class="sec-title">▶ 그리퍼 명령</div>
    <div class="sec-desc">
      손가락을 <b style="color:var(--green)">열지</b> 아니면 <b style="color:var(--red)">닫을지</b>에 대한 명령입니다.<br>
      정책출력 ≥ 0 → 열기, &lt; 0 → 닫기
    </div>
    <div class="gripper-big" id="gripper-box">
      <div class="gripper-state" id="gripper-state">—</div>
      <div class="gripper-sub" id="gripper-sub"></div>
    </div>
  </div>

  <!-- 정책 원시 출력 -->
  <div class="section">
    <div class="sec-title">▶ 정책 원시 출력 (신경망 출력값 그대로)</div>
    <div class="sec-desc">
      신경망이 출력한 원시값. 이 값에 0.5를 곱하고 기본자세를 더하면 위의 목표각도가 됩니다.<br>
      범위는 보통 −1 ~ +1 사이.
    </div>
    <div class="action-wrap" id="actions"></div>
  </div>

</div>

<!-- ═══════════════════════════════ 오른쪽 열 ═══════════════════════════════ -->
<div class="col">

  <!-- EE 위치 -->
  <div class="section">
    <div class="sec-title">▶ 팔 끝(EE) 위치</div>

    <div class="help-box">
      <b>EE (End-Effector)</b> = 로봇 손목 끝의 파지 지점.<br>
      좌표계: <b>X</b> = 앞뒤(+앞), <b>Y</b> = 좌우(+왼쪽), <b>Z</b> = 위아래(+위). 단위: 미터(m).
    </div>
    <div class="pos-card" id="ee-card">
      <div class="pos-label"><b>팔 끝 현재 위치</b></div>
      <div class="pos-axes" id="ee-pos"></div>
    </div>

    <div class="pos-card" id="obj-card">
      <div class="pos-label"><b>물체 위치</b> — 집어야 할 대상</div>
      <div class="pos-axes" id="obj-pos"></div>
    </div>

    <div class="dist-box" id="dist-box">
      <div>
        <div style="font-size:.7rem;color:var(--dim);margin-bottom:4px">팔 끝 ↔ 물체 거리</div>
        <span class="dist-val" id="dist-val">—</span>
        <span class="dist-unit">m</span>
      </div>
      <div class="dist-bar-wrap">
        <div class="dist-bar"><div class="dist-bar-fill" id="dist-fill"></div></div>
        <div class="dist-meaning" id="dist-meaning"></div>
      </div>
    </div>

    <div class="pos-card">
      <div class="pos-label"><b>목표 위치</b> — 물체를 내려놓을 곳 (고정값)</div>
      <div class="pos-axes" id="goal-pos"></div>
    </div>
  </div>

  <!-- 장애물 -->
  <div class="section">
    <div class="sec-title">▶ 장애물 위치</div>
    <div class="sec-desc">작업 공간에 있는 장애물들의 현재 좌표입니다.</div>
    <div id="obstacles"></div>
  </div>

  <!-- 현재 조인트 상태 -->
  <div class="section">
    <div class="sec-title">▶ 현재 조인트 실제 각도</div>
    <div class="sec-desc">
      물리 시뮬레이션에서 측정한 <b style="color:var(--text)">실제 각도</b>.<br>
      위의 목표각도와 차이가 있으면 아직 이동 중인 관절입니다.
    </div>
    <div id="current-joints"></div>
  </div>

</div>
</div>

<script>
// 프란카 팬더 조인트 정보
const JOINTS = [
  { name: 'J1 (베이스 회전)',   limits: [-166, 166] },
  { name: 'J2 (어깨)',         limits: [-101, 101] },
  { name: 'J3 (팔꿈치 회전)',  limits: [-166, 166] },
  { name: 'J4 (팔꿈치 굽힘)',  limits: [-176,  -4] },
  { name: 'J5 (손목 회전1)',   limits: [-166, 166] },
  { name: 'J6 (손목 굽힘)',    limits: [  -1, 215] },
  { name: 'J7 (손목 회전2)',   limits: [-166, 166] },
];
const DEFAULT_DEG = [0, -32.6, 0, -161, 0, 174, 42.5];

function sgn(v){ return v >= 0 ? '+' : ''; }
function fd(v, n=2){ return sgn(v) + v.toFixed(n) + '°'; }
function fm(v, n=3){ return sgn(v) + v.toFixed(n) + ' m'; }
function fp(v, n=4){ return sgn(v) + v.toFixed(n); }

function limitColor(deg, lo, hi){
  const range = hi - lo;
  const prox = Math.min(Math.abs(deg - lo), Math.abs(deg - hi)) / (range * 0.5);
  return prox < 0.12 ? 'var(--red)' : prox < 0.30 ? 'var(--yellow)' : 'var(--green)';
}

function pct(deg, lo, hi){ return Math.max(0, Math.min(100, (deg - lo) / (hi - lo) * 100)); }

function renderJoints(cmdDegs, curDegs, container){
  container.innerHTML = JOINTS.map((j, i) => {
    const cmd = cmdDegs[i];
    const cur = curDegs[i];
    const diff = cmd - cur;
    const [lo, hi] = j.limits;
    const color = limitColor(cmd, lo, hi);
    const curPct = pct(cur, lo, hi);
    const cmdPct = pct(cmd, lo, hi);

    return `<div class="joint-card">
      <div class="joint-head">
        <span class="joint-name"><b>${j.name}</b> &nbsp;한계: ${lo}° ~ ${hi}°</span>
        <span class="joint-cmd" style="color:${color}">${fd(cmd, 1)}</span>
      </div>
      <div class="joint-row2">
        <span>현재: <b>${fd(cur, 1)}</b></span>
        <span>기본자세: <b>${fd(DEFAULT_DEG[i], 1)}</b></span>
        <span>목표−현재: <b style="color:${Math.abs(diff)>5?'var(--yellow)':'var(--dim)'}">${fd(diff, 1)}</b></span>
      </div>
      <div class="rangebar">
        <div class="rangebar-cur" style="
          left:${Math.min(curPct,cmdPct).toFixed(1)}%;
          width:${Math.abs(cmdPct-curPct).toFixed(1)}%;
          background:var(--surf2);
          border:1px dashed var(--dim);
        "></div>
        <div class="rangebar-cmd" style="left:calc(${cmdPct.toFixed(1)}% - 1px); background:${color};"></div>
      </div>
    </div>`;
  }).join('');
}

function renderCurrentJoints(curDegs){
  const el = document.getElementById('current-joints');
  el.innerHTML = JOINTS.map((j, i) => {
    const cur = curDegs[i];
    const [lo, hi] = j.limits;
    const color = limitColor(cur, lo, hi);
    return `<div style="display:flex;justify-content:space-between;align-items:center;
      padding:7px 10px;background:var(--surf);border-radius:4px;margin-bottom:5px;
      font-family:var(--mono);font-size:.75rem;">
      <span style="color:var(--dim)">${j.name}</span>
      <span style="color:${color};font-weight:600">${fd(cur, 1)}</span>
    </div>`;
  }).join('');
}

function renderPosAxes(container, arr){
  const axes = [
    { label: 'X (앞뒤)', color: '#FF6B6B' },
    { label: 'Y (좌우)', color: '#4ECDC4' },
    { label: 'Z (높이)', color: '#FFE66D' },
  ];
  container.innerHTML = axes.map((a, i) => `
    <div class="pos-axis-item">
      <span class="pos-axis-letter" style="color:${a.color}">${a.label}</span>
      <span class="pos-axis-val">${fp(arr[i], 3)}</span>
    </div>`).join('');
}

function renderActions(acts, gripperSig){
  const el = document.getElementById('actions');
  const cells = acts.map((v, i) => {
    const pct = Math.min(1, Math.abs(v) / 1.5) * 50;
    const fillLeft = v >= 0 ? 50 : 50 - pct;
    return `<div class="act-cell">
      <span class="act-cell-name">J${i+1} 원시값</span>
      <span class="act-cell-val">${fp(v, 3)}</span>
      <div class="act-mini-bar">
        <div class="act-mini-fill" style="left:${fillLeft.toFixed(0)}%;width:${pct.toFixed(0)}%;background:var(--accent)"></div>
      </div>
    </div>`;
  });
  // 그리퍼
  const gpct = Math.min(1, Math.abs(gripperSig) / 1.5) * 50;
  const gLeft = gripperSig >= 0 ? 50 : 50 - gpct;
  const gColor = gripperSig >= 0 ? 'var(--green)' : 'var(--red)';
  cells.push(`<div class="act-cell">
    <span class="act-cell-name">그리퍼 원시값</span>
    <span class="act-cell-val" style="color:${gColor}">${fp(gripperSig, 3)}</span>
    <div class="act-mini-bar">
      <div class="act-mini-fill" style="left:${gLeft.toFixed(0)}%;width:${gpct.toFixed(0)}%;background:${gColor}"></div>
    </div>
  </div>`);
  el.innerHTML = cells.join('');
}

function renderObstacles(obs0, obs1){
  const el = document.getElementById('obstacles');
  function obsCard(label, pos){
    const far = pos.every(v => Math.abs(v) > 50);
    if(far) return `<div style="padding:8px 12px;background:var(--surf);border-radius:4px;
      margin-bottom:8px;font-size:.73rem;color:var(--dim);">
      ${label}: <b>비활성 (작업 공간 밖)</b></div>`;
    return `<div class="pos-card">
      <div class="pos-label"><b>${label}</b></div>
      <div class="pos-axes">
        ${['X','Y','Z'].map((a,i)=>`<div class="pos-axis-item">
          <span class="pos-axis-letter">${a}</span>
          <span class="pos-axis-val">${fp(pos[i],3)}</span>
        </div>`).join('')}
      </div>
    </div>`;
  }
  el.innerHTML = obsCard('장애물 0', obs0) + obsCard('장애물 1', obs1);
}

async function poll(){
  try {
    const r = await fetch('/state');
    if(!r.ok) throw new Error();
    const d = await r.json();

    if(d.status === 'waiting'){
      document.getElementById('dot').className = 'dot';
      document.getElementById('hd-meta').textContent = '— play 스크립트 대기 중...';
      setTimeout(poll, 300); return;
    }

    document.getElementById('dot').className = 'dot live';
    document.getElementById('hd-meta').textContent =
      `에피소드 스텝: ${d.episode_step}  |  총 스텝: ${d.step}`;
    document.getElementById('hd-time').textContent = d.timestamp;

    // 명령 각도 = DEFAULT_DEG + 0.5 * action (도 단위)
    const cmdRad = d.actions_arm.map((a, i) => {
      const defaultRad = [0, -0.569, 0, -2.810, 0, 3.037, 0.741][i];
      return defaultRad + 0.5 * a;
    });
    const cmdDeg = cmdRad.map(r => r * 180 / Math.PI);
    const curDeg = d.joint_pos_deg.slice(0, 7);

    renderJoints(cmdDeg, curDeg, document.getElementById('joints'));
    renderCurrentJoints(curDeg);
    renderActions(d.actions_arm, d.action_gripper);

    // 그리퍼
    const gb = document.getElementById('gripper-box');
    gb.className = 'gripper-big ' + (d.gripper_state === 'OPEN' ? 'open' : 'close');
    document.getElementById('gripper-state').textContent =
      d.gripper_state === 'OPEN' ? '열림 (OPEN)' : '닫힘 (CLOSE)';
    document.getElementById('gripper-sub').textContent =
      `신호값: ${fp(d.action_gripper, 3)}  |  손가락 벌림: ${(d.joint_pos_rad[7]*1000).toFixed(1)} mm`;

    // 위치
    renderPosAxes(document.getElementById('ee-pos'), d.ee_pos);
    renderPosAxes(document.getElementById('obj-pos'), d.object_pos);
    renderPosAxes(document.getElementById('goal-pos'), d.goal_pos);

    // 거리
    const dist = d.dist_ee_obj;
    document.getElementById('dist-val').textContent = dist.toFixed(3);
    document.getElementById('dist-val').style.color =
      dist < 0.05 ? 'var(--green)' : dist < 0.15 ? 'var(--yellow)' : 'var(--red)';
    const fillW = Math.max(0, 100 - dist / 0.5 * 100);
    const fillC = dist < 0.05 ? 'var(--green)' : dist < 0.15 ? 'var(--yellow)' : 'var(--red)';
    document.getElementById('dist-fill').style.width = fillW.toFixed(0) + '%';
    document.getElementById('dist-fill').style.background = fillC;
    document.getElementById('dist-meaning').textContent =
      dist < 0.05 ? '✓ 파지 가능 거리' : dist < 0.15 ? '접근 중' : '아직 멀리 있음';

    renderObstacles(d.obstacle_0_pos, d.obstacle_1_pos);

  } catch(e) {
    document.getElementById('dot').className = 'dot dead';
    document.getElementById('hd-meta').textContent = '서버 연결 끊김 — python3 scripts/dashboard_server.py 확인';
  }
  setTimeout(poll, 150);
}

poll();
</script>
</body>
</html>"""


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/state":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            state_path = os.path.abspath(STATE_FILE)
            if os.path.exists(state_path):
                with open(state_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.wfile.write(b'{"status":"waiting"}')
        elif self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML.encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    server = http.server.HTTPServer(("", PORT), Handler)
    print(f"대시보드 서버: http://localhost:{PORT}")
    print(f"JSON 파일: {os.path.abspath(STATE_FILE)}")
    print("Ctrl+C로 종료")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n종료.")
