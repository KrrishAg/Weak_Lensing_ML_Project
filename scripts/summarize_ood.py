#!/usr/bin/env python
import argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--tag',default='proposed'); args=ap.parse_args()
    p=ROOT/'results'/f'{args.tag}_ood_metrics.json'
    with open(p,encoding='utf-8') as f:d=json.load(f)
    shifts=list(d)
    methods=[m for m in ['ensemble_disagreement','nuisance_prior','nuisance_prior_tuned','power_spectrum','hybrid','hybrid_tuned'] if m in d[shifts[0]]]
    print('| detector | '+' | '.join(shifts)+' | mean |')
    print('|---|'+'---:|'*(len(shifts)+1))
    for m in methods:
        vals=[d[s][m]['low_fpr_tpr'] for s in shifts]
        print('| '+m+' | '+' | '.join(f'{v:.3f}' for v in vals)+f' | {sum(vals)/len(vals):.3f} |')
if __name__=='__main__':main()
