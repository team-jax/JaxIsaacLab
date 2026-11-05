"""Tile several rendered videos into one grid video with a label on each tile.
usage: montage.py OUT.mp4 COLS label1=path1.mp4 label2=path2.mp4 ..."""
import sys, numpy as np, imageio.v3 as iio, imageio
from PIL import Image, ImageDraw, ImageFont
out, cols = sys.argv[1], int(sys.argv[2])
items = [a.split("=", 1) for a in sys.argv[3:]]
vids = [list(iio.imiter(p)) for _, p in items]
fps = iio.immeta(items[0][1]).get("fps", 60)  # source rate (60 = real time); 30 played it at half speed
n = max(len(v) for v in vids)
h, w = vids[0][0].shape[:2]; SC = float(__import__("os").environ.get("SCALE", "0.5")); sh, sw = int(h * SC) // 2 * 2, int(w * SC) // 2 * 2
rows = (len(vids) + cols - 1) // cols
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
except OSError:
    font = ImageFont.load_default()
wr = imageio.get_writer(out, fps=fps, codec="libx264", quality=7, macro_block_size=1)
for k in range(n):
    canvas = Image.new("RGB", (sw * cols, sh * rows), (0, 0, 0))
    for i, ((lab, _), v) in enumerate(zip(items, vids)):
        f = Image.fromarray(v[min(k, len(v) - 1)]).resize((sw, sh))
        d = ImageDraw.Draw(f); d.rectangle((0, 0, 18 * len(lab) + 20, 40), fill=(0, 0, 0)); d.text((8, 3), lab, fill=(255, 255, 0), font=font)
        canvas.paste(f, ((i % cols) * sw, (i // cols) * sh))
    wr.append_data(np.asarray(canvas))
wr.close(); print("MONTAGE", out, n, "frames")
