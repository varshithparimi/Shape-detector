"""Re-score the saved checkpoint on the test split, with and without NMS, and redraw samples."""
import json, numpy as np, torch, cv2
from torch.utils.data import DataLoader
from data import ShapeDataset, CLASSES
from model import TinyDetector, decode, nms
from evaluate import evaluate

test_ds = ShapeDataset(500, 3)          # same seed as train.py -> identical held-out set
model = TinyDetector(); model.load_state_dict(torch.load("checkpoints/tiny_detector.pt")); model.eval()

raw, post, gts = [], [], []
with torch.no_grad():
    for x, _, idx in DataLoader(test_ds, batch_size=100):
        for d, i in zip(decode(model(x), conf_thresh=0.3), idx.tolist()):
            raw.append(d); post.append(nms(d)); gts.append(test_ds.samples[i][1])

res = {"without_nms": evaluate(raw, gts), "with_nms": evaluate(post, gts)}
print(json.dumps(res, indent=2))
json.dump(res, open("outputs/metrics_nms_comparison.json", "w"), indent=2)

tiles = []
for i in range(8):
    img = cv2.resize(test_ds.samples[i][0].copy(), (256, 256), interpolation=cv2.INTER_NEAREST)
    for c, x1, y1, x2, y2 in test_ds.samples[i][1]:
        cv2.rectangle(img, (int(x1*2), int(y1*2)), (int(x2*2), int(y2*2)), (0, 255, 0), 1)
    for c, s, x1, y1, x2, y2 in post[i]:
        cv2.rectangle(img, (int(x1*2), int(y1*2)), (int(x2*2), int(y2*2)), (0, 0, 255), 2)
        cv2.putText(img, f"{CLASSES[c]} {s:.2f}", (int(x1*2), max(int(y1*2) - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    tiles.append(img)
cv2.imwrite("outputs/sample_predictions_nms.png", np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])]))
