"""Detection metrics: per-class Average Precision @ IoU 0.5, plus mAP, precision, recall."""
import numpy as np
from data import CLASSES, _iou_xyxy


def average_precision(rec, prec):
    """Area under the precision-recall curve (all-point interpolation, VOC style)."""
    rec = np.concatenate([[0], rec, [1]])
    prec = np.concatenate([[0], prec, [0]])
    for i in range(len(prec) - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])
    idx = np.where(rec[1:] != rec[:-1])[0]
    return float(np.sum((rec[idx + 1] - rec[idx]) * prec[idx + 1]))


def evaluate(all_dets, all_gts, iou_thresh=0.5):
    """all_dets[i]: list of (cls, score, x1,y1,x2,y2); all_gts[i]: list of (cls, x1,y1,x2,y2)."""
    aps, report = [], {}
    tot_tp = tot_fp = tot_fn = 0
    for c, name in enumerate(CLASSES):
        dets = [(i, d) for i, ds in enumerate(all_dets) for d in ds if d[0] == c]
        dets.sort(key=lambda t: -t[1][1])
        gts = {i: [g[1:] for g in gs if g[0] == c] for i, gs in enumerate(all_gts)}
        n_gt = sum(len(v) for v in gts.values())
        used = {i: [False] * len(v) for i, v in gts.items()}
        tp, fp = np.zeros(len(dets)), np.zeros(len(dets))
        for k, (i, d) in enumerate(dets):
            best, best_j = 0, -1
            for j, g in enumerate(gts[i]):
                iou = _iou_xyxy(d[2:], g)
                if iou > best:
                    best, best_j = iou, j
            if best >= iou_thresh and not used[i][best_j]:
                tp[k] = 1
                used[i][best_j] = True
            else:
                fp[k] = 1
        ctp, cfp = np.cumsum(tp), np.cumsum(fp)
        rec = ctp / max(n_gt, 1)
        prec = ctp / np.maximum(ctp + cfp, 1e-9)
        ap = average_precision(rec, prec) if len(dets) else 0.0
        aps.append(ap)
        report[name] = {"AP@0.5": round(ap, 4), "n_gt": n_gt}
        tot_tp += tp.sum(); tot_fp += fp.sum(); tot_fn += n_gt - tp.sum()
    report["mAP@0.5"] = round(float(np.mean(aps)), 4)
    report["precision"] = round(float(tot_tp / max(tot_tp + tot_fp, 1)), 4)
    report["recall"] = round(float(tot_tp / max(tot_tp + tot_fn, 1)), 4)
    return report
