"""Play several videos one after another in one file, each with a label at the top-left.
usage: concat_labeled.py OUT.mp4 label1=path1.mp4 label2=path2.mp4 ..."""
import sys, numpy as np, imageio.v3 as iio, imageio
from PIL import Image, ImageDraw, ImageFont
out = sys.argv[1]; items = [a.split("=", 1) for a in sys.argv[2:]]
fps = iio.immeta(items[0][1]).get("fps", 60)
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 34)
wr = imageio.get_writer(out, fps=fps, codec="libx264", quality=8, macro_block_size=1)
for lab, p in items:
    for fr in iio.imiter(p):
        im = Image.fromarray(fr); d = ImageDraw.Draw(im)
        tw = d.textlength(lab, font=font); d.rectangle((0, 0, tw + 28, 50), fill=(0, 0, 0)); d.text((14, 6), lab, fill=(255, 220, 0), font=font)
        wr.append_data(np.asarray(im))
wr.close(); print("CONCAT", out)
