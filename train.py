"""Train the detector, evaluate on a held-out split, save curves + sample predictions."""
import json, time, sys
import numpy as np, torch, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from data import ShapeDataset, CLASSES
from model import TinyDetector, detection_loss, decode
from evaluate import evaluate

torch.manual_seed(0)
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 25
N_TRAIN, N_VAL, N_TEST = 3000, 500, 500

train_ds, val_ds, test_ds = ShapeDataset(N_TRAIN, 1), ShapeDataset(N_VAL, 2), ShapeDataset(N_TEST, 3)
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=100)

model = TinyDetector()
print("parameters:", sum(p.numel() for p in model.parameters()))
opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, epochs=EPOCHS, steps_per_epoch=len(train_dl))

hist = {"train": [], "val": []}
t0 = time.time()
for ep in range(EPOCHS):
    model.train(); tl = 0
    for x, y, _ in train_dl:
        loss = detection_loss(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        tl += loss.item() * len(x)
    model.eval(); vl = 0
    with torch.no_grad():
        for x, y, _ in val_dl:
            vl += detection_loss(model(x), y).item() * len(x)
    hist["train"].append(tl / N_TRAIN); hist["val"].append(vl / N_VAL)
    print(f"epoch {ep+1:2d}/{EPOCHS}  train {hist['train'][-1]:.3f}  val {hist['val'][-1]:.3f}  ({time.time()-t0:.0f}s)", flush=True)

torch.save(model.state_dict(), "checkpoints/tiny_detector.pt")

# ---- evaluation on the untouched test split ----
model.eval()
all_dets, all_gts = [], []
test_dl = DataLoader(test_ds, batch_size=100)
with torch.no_grad():
    for x, _, idx in test_dl:
        dets = decode(model(x), conf_thresh=0.3)
        for d, i in zip(dets, idx.tolist()):
            all_dets.append(d); all_gts.append(test_ds.samples[i][1])
report = evaluate(all_dets, all_gts)
print(json.dumps(report, indent=2))
json.dump(report, open("outputs/metrics.json", "w"), indent=2)

# ---- loss curve ----
plt.figure(figsize=(5, 3.2))
plt.plot(hist["train"], label="train"); plt.plot(hist["val"], label="validation")
plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.title("Training curve"); plt.tight_layout()
plt.savefig("outputs/loss_curve.png", dpi=130); plt.close()

# ---- sample predictions grid (green = ground truth, red = prediction) ----
COL = {0: (255, 160, 0), 1: (0, 200, 255), 2: (255, 0, 200)}
tiles = []
for i in range(8):
    img = test_ds.samples[i][0].copy()
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_NEAREST)
    for c, x1, y1, x2, y2 in test_ds.samples[i][1]:
        cv2.rectangle(img, (int(x1*2), int(y1*2)), (int(x2*2), int(y2*2)), (0, 255, 0), 1)
    for c, s, x1, y1, x2, y2 in all_dets[i]:
        cv2.rectangle(img, (int(x1*2), int(y1*2)), (int(x2*2), int(y2*2)), (0, 0, 255), 2)
        cv2.putText(img, f"{CLASSES[c]} {s:.2f}", (int(x1*2), max(int(y1*2) - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    tiles.append(img)
grid = np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])])
cv2.imwrite("outputs/sample_predictions.png", grid)
print("done")
