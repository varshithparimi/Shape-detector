"""A tiny YOLO-v1-style detector: a small CNN that maps a 128x128 image to an
8x8 grid, where every cell predicts (objectness, x, y, w, h, class scores)."""
import torch
import torch.nn as nn
from data import S, C


def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.LeakyReLU(0.1, inplace=True),
        nn.MaxPool2d(2),
    )


class TinyDetector(nn.Module):
    def __init__(self):
        super().__init__()
        # 128 -> 64 -> 32 -> 16 -> 8 spatial resolution
        self.backbone = nn.Sequential(block(3, 16), block(16, 32), block(32, 64), block(64, 96))
        self.head = nn.Conv2d(96, 5 + C, 1)

    def forward(self, x):
        out = self.head(self.backbone(x))          # (B, 5+C, S, S)
        return out.permute(0, 2, 3, 1)             # (B, S, S, 5+C)


def detection_loss(pred, target, l_coord=5.0, l_noobj=0.5):
    obj = target[..., 0] == 1
    noobj = ~obj
    bce = nn.functional.binary_cross_entropy_with_logits
    # objectness: every cell, down-weighting empty cells (most cells are empty)
    loss_obj = bce(pred[..., 0][obj], target[..., 0][obj], reduction="sum")
    loss_noobj = bce(pred[..., 0][noobj], target[..., 0][noobj], reduction="sum")
    # box regression only where an object exists
    xy = torch.sigmoid(pred[..., 1:3])
    wh = torch.sigmoid(pred[..., 3:5])
    loss_xy = ((xy[obj] - target[..., 1:3][obj]) ** 2).sum()
    loss_wh = ((wh[obj].sqrt() - target[..., 3:5][obj].sqrt()) ** 2).sum()
    # classification only where an object exists
    loss_cls = bce(pred[..., 5:][obj], target[..., 5:][obj], reduction="sum")
    n = pred.shape[0]
    return (loss_obj + l_noobj * loss_noobj + l_coord * (loss_xy + loss_wh) + loss_cls) / n


@torch.no_grad()
def decode(pred, conf_thresh=0.5):
    """Raw network output -> list (per image) of (cls, score, x1, y1, x2, y2) in pixels."""
    from data import IMG
    B = pred.shape[0]
    results = []
    obj = torch.sigmoid(pred[..., 0])
    xy = torch.sigmoid(pred[..., 1:3])
    wh = torch.sigmoid(pred[..., 3:5])
    cls_p = torch.sigmoid(pred[..., 5:])
    for b in range(B):
        dets = []
        for j in range(S):
            for i in range(S):
                if obj[b, j, i] < conf_thresh:
                    continue
                cx = (i + xy[b, j, i, 0].item()) / S * IMG
                cy = (j + xy[b, j, i, 1].item()) / S * IMG
                w, h = wh[b, j, i, 0].item() * IMG, wh[b, j, i, 1].item() * IMG
                c = int(cls_p[b, j, i].argmax())
                score = obj[b, j, i].item() * cls_p[b, j, i, c].item()
                dets.append((c, score, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
        results.append(dets)
    return results


def nms(dets, iou_thresh=0.4):
    """Non-max suppression: among overlapping boxes, keep only the highest-scoring one.
    dets: list of (cls, score, x1, y1, x2, y2). Boxes of different classes can overlap
    legitimately only if the IoU is low; here we suppress across classes too, since a
    single physical object should get a single label."""
    from data import _iou_xyxy
    keep = []
    for d in sorted(dets, key=lambda t: -t[1]):
        if all(_iou_xyxy(d[2:], k[2:]) < iou_thresh for k in keep):
            keep.append(d)
    return keep
