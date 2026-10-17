"""Run the trained detector on an image file (any size) and save an annotated copy.

usage: python infer.py input.png output.png
"""
import sys, cv2, torch
from data import IMG, CLASSES
from model import TinyDetector, decode

img = cv2.imread(sys.argv[1])
small = cv2.resize(img, (IMG, IMG))
x = torch.from_numpy(small[:, :, ::-1].copy()).permute(2, 0, 1).float()[None] / 255.0

model = TinyDetector()
model.load_state_dict(torch.load("checkpoints/tiny_detector.pt"))
model.eval()
from model import nms
dets = nms(decode(model(x), conf_thresh=0.4)[0])

sx, sy = img.shape[1] / IMG, img.shape[0] / IMG
for c, s, x1, y1, x2, y2 in dets:
    p1, p2 = (int(x1 * sx), int(y1 * sy)), (int(x2 * sx), int(y2 * sy))
    cv2.rectangle(img, p1, p2, (0, 0, 255), 2)
    cv2.putText(img, f"{CLASSES[c]} {s:.2f}", (p1[0], max(p1[1] - 5, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
cv2.imwrite(sys.argv[2], img)
print(f"{len(dets)} objects found ->", sys.argv[2])
