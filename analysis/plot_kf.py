#!/usr/bin/env python3
# One figure per state; rows are the complete correction pipeline.
import os, sys
import matplotlib; matplotlib.use('TkAgg')
import pandas as pd, matplotlib.pyplot as plt

CSV='../sparxe_kf_trace.csv'
SLAB={'trans':['px','py','pz','vx','vy','vz','mu_r','mu_l'],
      'mekf':['th_x','th_y','th_z']}
MLAB={'uwb':['range'],'accel':['a0','a1','a2'],
      'magnm':['m0','m1','m2'],'control':['control_derived_yaw']}
path=CSV if os.path.exists(CSV) else CSV+'.csv'
if not os.path.exists(path): sys.exit(f"no '{CSV}' — run nav_node first")
df=pd.read_csv(path)
for c in ('time','step','applied','nis','idx','value'):
    df[c]=pd.to_numeric(df[c],errors='coerce')
df=df.dropna(subset=['time','value']); df['t']=df.time-df.time.min()

def wide(d,kind):
    sub=d[d.kind==kind]
    if sub.empty:return None,None
    w=sub.pivot_table(index='step',columns='idx',values='value')
    return w,sub.groupby('step').t.first().reindex(w.index)

def meas_label(sensor,i):
    labels=MLAB.get(sensor,[])
    return labels[i] if i<len(labels) else f'm{i}'

def plot_measurements(a,w,t,sensor,prefix,style='.-',root=False):
    if w is None:return
    for i in w.columns:
        v=w[i].clip(lower=0).pow(.5) if root else w[i]
        a.plot(t.values,v.values,style,ms=3,lw=1,
               label=f'{prefix}[{meas_label(sensor,int(i))}]')

def decorate(a,title,zero=False):
    a.set_title(title,fontsize=10)
    if zero:a.axhline(0,color='k',lw=.8,alpha=.5)
    a.legend(loc='upper right',fontsize=7,ncol=3)
    a.grid(True,ls='--',alpha=.4)

for f,sensor in sorted(set(df[['filter','sensor']].drop_duplicates().itertuples(index=False,name=None))):
    d=df[(df['filter']==f)&(df['sensor']==sensor)]
    z,tz=wide(d,'z'); hx,th=wide(d,'Hx'); resid,tr=wide(d,'r')
    dx,td=wide(d,'dx'); P,tP=wide(d,'P')
    xpre,tx=wide(d,'x_pre'); xpost,tq=wide(d,'x_post')
    S,tS=wide(d,'S'); R,tR=wide(d,'R')
    nis=d.groupby('step').nis.first(); tn=d.groupby('step').t.first().reindex(nis.index)
    rejected=d[d.applied==0].step.unique()

    groups = ([('position', [0,1,2]), ('velocity', [3,4,5]), ('wheel efficiency', [6,7])] if f == 'trans' else
              [('attitude error', list(range(len(SLAB.get(f, [])))) )])
    for group, state_indices in groups:
        fig,ax=plt.subplots(8,1,figsize=(15,20),sharex=True)
        fig.suptitle(f'{f} / {sensor} / {group} — complete correction pipeline',fontsize=13)

        plot_measurements(ax[0],z,tz,sensor,'z')
        plot_measurements(ax[0],hx,th,sensor,'Hx','--')
        decorate(ax[0],'measurement z vs predicted Hx')

        plot_measurements(ax[1],resid,tr,sensor,'z-Hx')
        decorate(ax[1],'measurement residual z - Hx',True)

        for state_i in state_indices:
            state = SLAB[f][state_i]
            if dx is not None and state_i in dx.columns:
                ax[2].plot(td.values,dx[state_i].values,'.-',ms=3,lw=1,label=f'K*r -> {state}')
        decorate(ax[2],f'correction into {group}: K*r',True)

        for state_i in state_indices:
            state = SLAB[f][state_i]
            if P is not None and state_i in P.columns:
                ax[3].plot(tP.values,P[state_i].clip(lower=0).pow(.5).values,'.-',ms=3,lw=1,label=f'sqrt(P_{state})')
        decorate(ax[3],f'{group} standard deviation sqrt(P)')

        for state_i in state_indices:
            state = SLAB[f][state_i]
            if xpre is not None and state_i in xpre.columns:
                ax[4].plot(tx.values,xpre[state_i].values,'-',lw=1,label=f'{state} pre')
            if xpost is not None and state_i in xpost.columns:
                ax[4].plot(tq.values,xpost[state_i].values,'--',lw=1,label=f'{state} post')
        decorate(ax[4],f'{group} before and after correction')

        ax[5].plot(tn.values,nis.values,'.-',ms=3,lw=1,color='tab:red',label='NIS')
        decorate(ax[5],'normalized innovation squared (NIS)')

        plot_measurements(ax[6],S,tS,sensor,'sqrt(S)',root=True)
        decorate(ax[6],'innovation standard deviation sqrt(S)')

        plot_measurements(ax[7],R,tR,sensor,'sqrt(R)',root=True)
        decorate(ax[7],'measurement noise standard deviation sqrt(R)')

        for step in rejected:
            if step in tn.index:
                for a in ax:a.axvline(tn[step],color='gray',alpha=.15,lw=1)
        ax[-1].set_xlabel('time (s)')
        fig.tight_layout(rect=(0,0,1,.98))
plt.show()
