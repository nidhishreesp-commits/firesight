from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, 'src')
import cv2
import numpy as np
from firesight.config import PROJECT_ROOT, load_config
from firesight.temporal import TemporalConfirmer
from firesight.video_filter import FilterConfig, VideoFalseAlarmFilter
from ensemble_video_eval import fuse

FPS = 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='intersect')
    ap.add_argument('--manifest', default='data/video_eval/public/curated_manifest.jsonl')
    ap.add_argument('--cache-p50', default='build/video_eval/cache_p50')
    ap.add_argument('--cache-salah', default='build/video_eval/cache_salah')
    ap.add_argument('--out', default='build/video_eval/ensemble/intersect_annotated')
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.manifest, encoding='utf-8') if l.strip()]
    cp_dir = PROJECT_ROOT / args.cache_p50
    cs_dir = PROJECT_ROOT / args.cache_salah
    out = PROJECT_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(); cfg.temporal.tracker = 'iou'
    for c in rows:
        cp = json.loads((cp_dir / f"{c['id'].replace(':', '_')}.json").read_text(encoding='utf-8'))
        cs = json.loads((cs_dir / f"{c['id'].replace(':', '_')}.json").read_text(encoding='utf-8'))
        sf = {f['idx']: f for f in cs['frames']}
        tc = TemporalConfirmer(cfg.temporal)
        vf = VideoFalseAlarmFilter(FilterConfig())
        blank = np.zeros((cp['h'], cp['w'], 3), dtype=np.uint8)
        cap = cv2.VideoCapture(str(PROJECT_ROOT / c['path']))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        step = max(1, round(fps / FPS))
        dest = out / f"{c['id'].replace(':', '_')}_annotated.mp4"
        wr = cv2.VideoWriter(str(dest), cv2.VideoWriter_fourcc(*'mp4v'), FPS, (cp['w'], cp['h']))
        idx = 0
        layer_seen = False
        for f in cp['frames']:
            target = f['idx'] * step
            while idx < target:
                ok, _ = cap.read()
                if not ok:
                    break
                idx += 1
            ok, fr = cap.read()
            if not ok:
                break
            idx += 1
            fused = fuse(args.mode, f['dets'], sf.get(f['idx'], {'dets': []})['dets'])
            dets = []
            for d in fused:
                dets.append((d['label'], d['conf'], d['box']))
            from firesight.types import Box, Detection
            dd = [Detection(l, cf, Box(*b)) for l, cf, b in dets]
            tconf = tc.update(f['idx'], f['t'], dd, size=(cp['h'], cp['w']))
            if tconf.confirmed:
                kept, _, _ = vf.update(c['id'], f['t'], blank, tconf.confirmed)
                if kept:
                    layer_seen = True
            ann = fr.copy()
            for lab, conf, b in dets:
                x1, y1, x2, y2 = map(int, b)
                col = (0, 0, 255) if lab == 'fire' else (0, 140, 255)
                cv2.rectangle(ann, (x1, y1), (x2, y2), col, 2)
                cv2.putText(ann, f'{lab} {conf:.2f}', (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
            cv2.putText(ann, f"{args.mode}  {c['label']}  t={f['t']:.1f}s  layer={'YES' if layer_seen else 'no'}",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            wr.write(ann)
        cap.release(); wr.release()
        print('annotated', c['id'], flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
