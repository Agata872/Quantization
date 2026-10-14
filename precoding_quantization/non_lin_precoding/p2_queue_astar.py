"""Queue seeds 1 and 2 of the tuned original GNN (A*) for every (K, b) once its seed-0 lr choice is known.

A* = the better of A_orig (lr 5e-3, Feys et al.) and A_orig_lr1e-3 by the best VALIDATION rate of seed 0
(history.json 'best_val_rate_ls'; the frozen test set is not read). The two lines 'K b A* 1' and 'K b A* 2' are
inserted right before 'K b F_full 1' / 'K b F_full 2' in ablation_p2/queue.txt, so they run with the other seeds of
that configuration. The choice is logged to ablation_p2/astar_choice.json. Idempotent; --loop repeats every 10 min
until all four configurations are handled.

  nohup python p2_queue_astar.py --loop > ablation_p2/logs/queue_astar.log 2>&1 &
"""
import json
import os
import sys
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, 'ablation_p2')
QUEUE = os.path.join(ROOT, 'queue.txt')
CHOICE = os.path.join(ROOT, 'astar_choice.json')
CONFIGS = [(2, 1), (2, 3), (6, 1), (6, 3)]


def val_best(K, b, v):
    d = os.path.join(ROOT, 'runs', f'K{K}b{b}', f'{v}_s0')
    if os.path.exists(os.path.join(d, 'FAILED')) and not os.path.exists(os.path.join(d, 'DONE')):
        return float('-inf')            # collapsed / failed seed 0: never chosen
    if not os.path.exists(os.path.join(d, 'DONE')):
        return None
    return json.load(open(os.path.join(d, 'history.json')))['best_val_rate_ls']


def once():
    choice = json.load(open(CHOICE)) if os.path.exists(CHOICE) else {}
    lines = open(QUEUE).read().splitlines()
    changed = False
    for K, b in CONFIGS:
        key = f'K{K}b{b}'
        if key in choice:
            continue
        a, a3 = val_best(K, b, 'A_orig'), val_best(K, b, 'A_orig_lr1e-3')
        if a is None or a3 is None:
            continue
        best = 'A_orig' if a >= a3 else 'A_orig_lr1e-3'
        choice[key] = {'chosen': best, 'val_A_orig_lr5e-3': a, 'val_A_orig_lr1e-3': a3,
                       'rule': 'best validation LS rate (Htest[4096:6144], 20 dB) of seed 0', 'at': f'{datetime.now():%Y-%m-%d %H:%M:%S}'}
        for s in (1, 2):
            new = f'{K} {b} {best} {s}'
            if new in lines:
                continue
            anchor = f'{K} {b} F_full {s}'
            i = lines.index(anchor) if anchor in lines else len(lines)
            lines.insert(i, new)
        changed = True
        print(f'[{datetime.now():%H:%M:%S}] {key}: {best} (val {a:.4f} vs {a3:.4f}); seeds 1, 2 queued', flush=True)
    if changed:
        open(QUEUE + '.tmp', 'w').write('\n'.join(lines) + '\n')
        os.replace(QUEUE + '.tmp', QUEUE)
        json.dump(choice, open(CHOICE, 'w'), indent=1)
    return len(choice) == len(CONFIGS)


if __name__ == '__main__':
    while not once() and '--loop' in sys.argv:
        time.sleep(600)
