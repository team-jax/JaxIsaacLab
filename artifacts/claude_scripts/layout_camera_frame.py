"""Top view of the Isaac (training) and team MuJoCo layouts in the camera frame (README 4: origin on the stand axis, floor + 1.215 m,
base axes). Run with any Python that has Pillow: python layout_camera_frame.py -> layout_camera_frame.png next to this file."""
import os
from PIL import Image, ImageDraw, ImageFont
S = 600.0
X0, X1, Y0, Y1 = -0.75, 0.85, -0.50, 1.15      # screen right = -x_c, screen up = -y_c (robot front up)
W, H = int((X1 - X0) * S), int((Y1 - Y0) * S) + 90
im = Image.new("RGB", (W, H), "white"); dr = ImageDraw.Draw(im)
F = lambda s: ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", s)
P = lambda xc, yc: ((-xc - X0) * S, H - 90 - ((-yc) - Y0) * S)


def rect(x0, x1, y0, y1, fill, outline, w=2):
    a, b = P(x0, y0), P(x1, y1)
    dr.rectangle([min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])], fill=fill, outline=outline, width=w)


for k in range(-8, 12):
    dr.line([P(-k / 10.0, 0)[0], 0, P(-k / 10.0, 0)[0], H - 90], fill=(235, 235, 235))
for k in range(-12, 6):
    dr.line([0, P(0, k / 10.0)[1], W, P(0, k / 10.0)[1]], fill=(235, 235, 235))
rect(-0.59, 0.61, -1.025, -0.225, (215, 225, 240), (60, 90, 160))
rect(-0.53, 0.03, -0.425, -0.265, (120, 160, 230), (30, 60, 150), 3)
rect(-0.65, -0.16, -0.355, 0.355, (240, 220, 190), (160, 110, 50))
rect(-0.5625, -0.4575, -0.2875, -0.0125, (245, 160, 80), (190, 90, 20), 3)
cx, cy = P(0, 0); dr.ellipse([cx - 9, cy - 9, cx + 9, cy + 9], fill=(30, 30, 30))
sx, sy = P(-0.15, 0); dr.ellipse([sx - 7, sy - 7, sx + 7, sy + 7], fill=(200, 40, 40))
dr.line([cx, cy, sx, sy], fill=(80, 80, 80), width=3)
t = F(19)
dr.text((cx - 300, cy + 12), "카메라 = 원점 (0, 0)\n모델 중앙, 바닥 1.215 m (책상 위 48 cm)", fill=(0, 0, 0), font=t)
dr.text((sx - 60, sy - 34), "어깨축 (−0.15, 0)", fill=(160, 30, 30), font=t)
x, y = P(0.55, -1.0); dr.text((x + 6, y + 6), "Isaac 책상 (로봇 정면, 지금 학습)\nx −0.59…+0.61, y −1.03…−0.23", fill=(40, 70, 140), font=t)
x, y = P(0.03, -0.425); dr.text((x + 4, y - 50), "Isaac 큐브 범위\nx −0.53…+0.03, y −0.43…−0.27", fill=(20, 50, 130), font=t)
x, y = P(-0.16, 0.355); dr.text((x + 6, y - 58), "팀 MuJoCo 책상 (로봇 옆)\nx −0.65…−0.16, y −0.36…+0.36", fill=(130, 85, 30), font=t)
x, y = P(-0.4575, 0.05); dr.text((x + 6, y), "팀 큐브·목적지\nx −0.56…−0.46\ny −0.29…−0.01", fill=(170, 80, 10), font=t)
dr.text((20, H - 82), "위에서 본 그림, 카메라 좌표 (m): x = 로봇 베이스 x (그림 왼쪽이 +), y = 베이스 y (그림 아래가 +), 격자 10 cm.\n"
        "파랑 = 지금 Isaac 학습 배치, 주황 = 팀 MuJoCo 배치. 겹치는 곳은 x −0.53…−0.46, y −0.29…−0.27뿐입니다.", fill=(0, 0, 0), font=F(18))
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "layout_camera_frame.png")
im.save(out); print(out)
