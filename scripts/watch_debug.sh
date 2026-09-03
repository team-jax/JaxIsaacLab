#!/usr/bin/env bash
# watch_debug.sh — 다른 터미널에서 디버그 로그를 실시간으로 모니터링합니다.
#
# 사용법:
#   bash scripts/watch_debug.sh                        # 기본 경로 사용
#   bash scripts/watch_debug.sh logs/debug_trace.log   # 경로 지정
#   bash scripts/watch_debug.sh logs/debug_trace.log TARGET_EE  # 특정 태그만 필터
#
# 지원 태그:
#   TARGET_EE   — EE 위치 (매 스텝)
#   OBJECT_POS  — 그립 오브젝트 위치 (매 스텝)
#   OBSTACLE_0  — 슬롯 0 장애물 위치 (매 스텝)
#   OBSTACLE_1  — 슬롯 1 장애물 위치 (매 스텝)
#   EPISODE     — 에피소드 시작/종료 이벤트
#
# 터미널 A (학습/평가 실행):
#   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_debug_ee.py \
#       --task Isaac-PnP-Obstacle-Franka-Play-v0 \
#       --num_envs 1 --verbose_ee --debug_log logs/debug_trace.log
#
# 터미널 B (이 스크립트):
#   bash scripts/watch_debug.sh logs/debug_trace.log

LOG_FILE="${1:-logs/debug_trace.log}"
FILTER="${2:-TARGET_EE|OBJECT_POS|OBSTACLE_0|OBSTACLE_1|EPISODE}"

# 로그 파일이 생길 때까지 대기
if [ ! -f "$LOG_FILE" ]; then
    echo "[watch_debug] Waiting for log file: $LOG_FILE ..."
    while [ ! -f "$LOG_FILE" ]; do
        sleep 0.5
    done
fi

echo "[watch_debug] Tailing $LOG_FILE (filter: $FILTER) — Ctrl+C to stop"
echo "--------------------------------------------------------------------"
tail -f "$LOG_FILE" | grep --line-buffered -E "$FILTER"
