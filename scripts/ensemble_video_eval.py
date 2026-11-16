from __future__ import annotations
import argparse, csv, json, math, sys
from pathlib import Path
sys.path.insert(0, 'src')
import cv2
import numpy as np
from firesight.config import PROJECT_ROOT, load_config
from firesight.temporal import TemporalConfirmer
from firesight.types import Box, Detection
from firesight.video_filter import FilterConfig, VideoFalseAlarmFilter

FPS = 2.0
P50 = {'fire': 0.29, 'smoke': 0.33}
SALAH = {'fire': 0.51, 'smoke': 0.57}


def wil(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / d
    return [round(max(0, c - h), 4), round(min(1, c + h), 4)]


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def tile_nms(dets, thr=0.5):
    out = []
    for d in sorted(dets, key=lambda x: -x['conf']):
        if any(d['label'] == k['label'] and iou(d['box'], k['box']) >= thr for k in out):
            continue
        out.append(d)
    return out


def above(dets, thr):
    return [d for d in dets if d['conf'] >= thr[d['label']]]


def fuse(mode, p, s):
    p_d = above(p, P50)
    s_d = above(s, SALAH)
    if mode == 'p50':
        return tile_nms(p_d)
    if mode == 'salah':
        return tile_nms(s_d)
    if mode == 'union':
        return tile_nms(p_d + s_d)
    if mode == 'split':
        return tile_nms([d for d in s_d if d['label'] == 'fire'] + [d for d in p_d if d['label'] == 'smoke'])
    if mode == 'split_fire_union':
        return tile_nms([d for d in p_d + s_d if d['label'] == 'fire'] + [d for d in p_d if d['label'] == 'smoke'])
    if mode == 'intersect':
        out = []
        for label in ('fire', 'smoke'):
            a = [d for d in p_d if d['label'] == label]
            b = [d for d in s_d if d['label'] == label]
            for x in a:
                m = [y for y in b if iou(x['box'], y['box']) >= 0.3]
                if m:
                    out.append(max([x] + m, key=lambda d: d['conf']))
        return tile_nms(out)
    if mode == 'maxscore':
        all_d = tile_nms(p + s)
        return [d for d in all_d if d['conf'] >= P50[d['label']]]
    raise ValueError(mode)


def cache_model(model_path, manifest, out_dir, name):
    from ultralytics import YOLO
    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(PROJECT_ROOT / model_path), task='detect')
    names = {int(k): str(v).lower() for k, v in (model.names or {}).items()}
    rows = [json.loads(l) for l in open(manifest, encoding='utf-8') if l.strip()]
    for c in rows:
        dest = out_dir / f"{c['id'].replace(':', '_')}.json"
        if dest.exists():
            print('cache exists', name, c['id'], flush=True)
            continue
        path = PROJECT_ROOT / c['path']
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        step = max(1, round(fps / FPS))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        limit = c.get('max_seconds') or (n / max(fps, 1))
        frames = []
        idx = sampled = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            t = idx / fps
            if t > limit:
                break
            if idx % step:
                idx += 1
                continue
            r = model.predict(fr, imgsz=640, conf=0.05, device='cpu', verbose=False)[0]
            dets = []
            for b in r.boxes:
                nm = names.get(int(b.cls.item()), '')
                lab = 'fire' if 'fire' in nm else 'smoke' if 'smoke' in nm else None
                if lab is None:
                    continue
                box = [round(float(v), 2) for v in b.xyxy[0].tolist()]
                dets.append({'label': lab, 'conf': round(float(b.conf.item()), 4), 'box': box})
            frames.append({'idx': sampled, 't': round(t, 3), 'dets': dets})
            sampled += 1
            idx += 1
        cap.release()
        rec = {'id': c['id'], 'label': c['label'], 'category': c['category'], 'title': c['title'],
               'eval_seconds': round(min(limit, n / max(fps, 1)), 1), 'sampled_frames': sampled,
               'w': w, 'h': h, 'frames': frames}
        dest.write_text(json.dumps(rec), encoding='utf-8')
        print('cached', name, c['id'], 'frames', sampled, 'dets', sum(len(f['dets']) for f in frames), flush=True)


def eval_mode(mode, rows, cdir_p, cdir_s, out_dir):
    cfg = load_config(); cfg.temporal.tracker = 'iou'
    results = []
    for c in rows:
        cp = json.loads((cdir_p / f"{c['id'].replace(':', '_')}.json").read_text(encoding='utf-8'))
        cs = json.loads((cdir_s / f"{c['id'].replace(':', '_')}.json").read_text(encoding='utf-8'))
        sf = {f['idx']: f for f in cs['frames']}
        tc = TemporalConfirmer(cfg.temporal)
        vf = VideoFalseAlarmFilter(FilterConfig())
        blank = np.zeros((cp['h'], cp['w'], 3), dtype=np.uint8)
        raw = temp = layer = False
        maxf = maxs = 0.0
        for pf in cp['frames']:
            fr = sf.get(pf['idx'], {'dets': []})
            fused = fuse(mode, pf['dets'], fr['dets'])
            dets = []
            for d in fused:
                maxf = max(maxf, d['conf']) if d['label'] == 'fire' else maxf
                maxs = max(maxs, d['conf']) if d['label'] == 'smoke' else maxs
                dets.append(Detection(d['label'], d['conf'], Box(*d['box'])))
            if dets:
                raw = True
            tconf = tc.update(pf['idx'], pf['t'], dets, size=(cp['h'], cp['w']))
            if tconf.confirmed:
                temp = True
                kept, _, _ = vf.update(c['id'], pf['t'], blank, tconf.confirmed)
                if kept:
                    layer = True
        results.append({'id': c['id'], 'label': c['label'], 'category': c['category'], 'title': c['title'],
                        'eval_seconds': cp['eval_seconds'], 'sampled_frames': cp['sampled_frames'],
                        'max_fire': round(maxf, 4), 'max_smoke': round(maxs, 4),
                        'raw_alert': raw, 'temporal_alert': temp, 'layer_alert': layer})
    pos = [r for r in results if r['label'] in ('fire', 'smoke')]
    neg = [r for r in results if r['label'] == 'nonfire']
    neg_all = [r for r in results if r['label'] not in ('fire', 'smoke')]
    metrics = {}
    for key in ('raw_alert', 'temporal_alert', 'layer_alert'):
        tp = sum(1 for r in pos if r[key]); fn = len(pos) - tp
        fp = sum(1 for r in neg if r[key]); tn = len(neg) - fp
        fpa = sum(1 for r in neg_all if r[key])
        prec = tp / (tp + fp) if tp + fp else None
        rec = tp / len(pos) if pos else None
        metrics[key] = {
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'precision': None if prec is None else round(prec, 4),
            'recall': None if rec is None else round(rec, 4),
            'recall_ci': wil(tp, len(pos)),
            'fpr': round(fp / len(neg), 4) if neg else None,
            'fpr_ci': wil(fp, len(neg)),
            'fpr_all_neg': round(fpa / len(neg_all), 4) if neg_all else None,
            'fa_all_neg': fpa,
            'accuracy': round((tp + tn) / (len(pos) + len(neg)), 4),
        }
    percat = {}
    for cat in sorted({r['category'] for r in results}):
        sub = [r for r in results if r['category'] == cat]
        percat[cat] = {'n': len(sub), 'layer_alerts': sum(1 for r in sub if r['layer_alert']),
                       'raw_alerts': sum(1 for r in sub if r['raw_alert'])}
    neg_hours = sum(r['eval_seconds'] for r in neg_all) / 3600
    summary = {'mode': mode, 'n': len(results), 'metrics': metrics, 'per_category': percat,
               'negative_hours_all': round(neg_hours, 3),
               'layer_fa_per_hour_all': round(sum(1 for r in neg_all if r['layer_alert']) / neg_hours, 3) if neg_hours else None}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f'{mode}_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    with (out_dir / f'{mode}_per_clip.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys())); w.writeheader(); w.writerows(results)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default='data/video_eval/public/curated_manifest.jsonl')
    ap.add_argument('--p50', default='build/kaggle_weights/runs/P50/weights/best.onnx')
    ap.add_argument('--salah', default='models/external/salah_yolov26_best.pt')
    ap.add_argument('--cache-p50', default='build/video_eval/cache_p50')
    ap.add_argument('--cache-salah', default='build/video_eval/cache_salah')
    ap.add_argument('--out', default='build/video_eval/ensemble')
    ap.add_argument('--cache-only', action='store_true')
    ap.add_argument('--modes', default='p50,salah,union,split,split_fire_union,intersect,maxscore')
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.manifest, encoding='utf-8') if l.strip()]
    cp = PROJECT_ROOT / args.cache_p50; cs = PROJECT_ROOT / args.cache_salah
    cache_model(args.p50, args.manifest, cp, 'p50')
    cache_model(args.salah, args.manifest, cs, 'salah')
    if args.cache_only:
        return
    out = PROJECT_ROOT / args.out
    summaries = []
    for mode in args.modes.split(','):
        s = eval_mode(mode.strip(), rows, cp, cs, out)
        summaries.append(s)
        m = s['metrics']['layer_alert']
        print(mode, 'layer prec/rec/fpr/acc', m['precision'], m['recall'], m['fpr'], m['accuracy'],
              'smoke', s['per_category'].get('smoke', {}).get('layer_alerts'), '/5',
              'fa_all_neg', m['fa_all_neg'], 'fa/hr', s['layer_fa_per_hour_all'], flush=True)
    (out / 'comparison.json').write_text(json.dumps(summaries, indent=2), encoding='utf-8')


if __name__ == '__main__':
    raise SystemExit(main())
