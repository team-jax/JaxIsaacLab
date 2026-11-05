import cv2, numpy as np, sys
vid, out = sys.argv[1], sys.argv[2]; idx = [int(x) for x in sys.argv[3].split(',')]
c = cv2.VideoCapture(vid); fr = []
while True:
    ok, f = c.read()
    if not ok: break
    fr.append(f)
ims = [cv2.putText(cv2.resize(fr[min(i, len(fr)-1)], (480, 270)), f't={i/60:.2f}s', (8, 26), 0, 0.8, (0, 0, 255), 2) for i in idx]
cv2.imwrite(out, np.vstack([np.hstack(ims[k:k + 3]) for k in range(0, len(ims), 3)]))
