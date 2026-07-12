#!/usr/bin/env python3
# plot every kalman step from sparxe_kf_trace.csv
# long format: time,filter,sensor,step,applied,nis,kind,idx,value
# kinds: z Hx r S R dx P x_pre x_post
import os, sys
import matplotlib; matplotlib.use('TkAgg')
import pandas as pd, matplotlib.pyplot as plt

CSV = '../sparxe_kf_trace.csv'
SLAB = {'trans': ['px','py','pz','vx','vy','vz','mu_r','mu_l'], 'mekf': ['th_x','th_y','th_z']}
MLAB = {'uwb': ['range'], 'accel': ['a0','a1','a2'], 'magnm': ['m0','m1','m2'], 'control': ['yaw_odo']}

path = CSV if os.path.exists(CSV) else CSV + '.csv'
if not os.path.exists(path):
    sys.exit(f"no '{CSV}' — run nav_node first")
df = pd.read_csv(path)
for c in ('time','step','applied','nis','idx','value'):
    df[c] = pd.to_numeric(df[c], errors='coerce')
df = df.dropna(subset=['time','value'])
df['t'] = df['time'] - df['time'].min()


def wide(d, kind):
    sub = d[d['kind'] == kind]
    if sub.empty:
        return None, None
    piv = sub.pivot_table(index='step', columns='idx', values='value')
    return piv, sub.groupby('step')['t'].first().reindex(piv.index)


def sl(f, i): return SLAB.get(f, [])[i] if i < len(SLAB.get(f, [])) else f's{i}'
def ml(s, i): return MLAB.get(s, [])[i] if i < len(MLAB.get(s, [])) else f'm{i}'


for f, s in sorted(set(df[['filter','sensor']].drop_duplicates().itertuples(index=False, name=None))):
    d = df[(df['filter'] == f) & (df['sensor'] == s)]
    lm, ls = (lambda i: ml(s, i)), (lambda i: sl(f, i))
    fig, ax = plt.subplots(6, 1, figsize=(15, 15), sharex=True)
    fig.suptitle(f'{f} / {s} — kalman pipeline', fontsize=13)

    zw, tz = wide(d, 'z'); hw, _ = wide(d, 'Hx')
    if zw is not None:
        for i in zw.columns:
            ax[0].plot(tz.values, zw[i].values, '.-', ms=3, lw=1, label=f'z[{lm(int(i))}]')
    if hw is not None:
        for i in hw.columns:
            ax[0].plot(tz.values, hw[i].values, '--', lw=1.2, label=f'Hx[{lm(int(i))}]')
    ax[0].set_title('z (meas) vs Hx (pred)', fontsize=10)

    for a, kind, lab, title, zero in [
        (ax[1], 'r',  lm, 'residual r = z - Hx', True),
        (ax[2], 'dx', ls, 'correction dx = K·r', True),
        (ax[3], 'P',  ls, 'state covariance P diag', False)]:
        w, t = wide(d, kind)
        if w is not None:
            for i in w.columns:
                a.plot(t.values, w[i].values, '.-', ms=3, lw=1, label=f'{kind}[{lab(int(i))}]')
        if zero:
            a.axhline(0, color='k', lw=0.8, alpha=0.5)
        a.set_title(title, fontsize=10)
    ax[3].set_yscale('log')

    xp, tp = wide(d, 'x_pre'); xq, _ = wide(d, 'x_post')
    if xp is not None:
        for i in xp.columns:
            line, = ax[4].plot(tp.values, xp[i].values, '-', lw=1, label=f'{ls(int(i))}')
            if xq is not None and i in xq.columns:
                ax[4].plot(tp.values, xq[i].values, '--', lw=1, color=line.get_color(), alpha=0.7)
    ax[4].set_title('state x (solid=pre, dashed=post)', fontsize=10)

    nis = d.groupby('step')['nis'].first(); tn = d.groupby('step')['t'].first()
    ax[5].plot(tn.values, nis.values, '.-', ms=3, lw=1, color='tab:red', label='nis')
    for kind, st in (('S', '-'), ('R', ':')):
        w, _ = wide(d, kind)
        if w is not None:
            for i in w.columns:
                ax[5].plot(tn.reindex(w.index).values, w[i].values, st, lw=1, label=f'{kind}[{lm(int(i))}]')
    for st in d[d['applied'] == 0]['step'].unique():
        if st in tn.index:
            ax[5].axvline(tn[st], color='gray', alpha=0.15, lw=1)
    ax[5].set_title('nis (red), S solid, R dotted; gray=rejected', fontsize=10)

    for a in ax:
        a.legend(loc='upper right', fontsize=7, ncol=4)
        a.grid(True, ls='--', alpha=0.4)
    ax[-1].set_xlabel('time (s)')
    fig.tight_layout(rect=(0, 0, 1, 0.98))

plt.show()
