import pandas as pd,numpy as np,sys
sys.argv=['x'];exec(open('feats.py').read().split("full=pd.concat")[0])
D='./'
tr=pd.read_pickle(D+'tr_f.pkl');te=pd.read_pickle(D+'te_f.pkl')
full=pd.concat([tr,te]).sort_values('t').reset_index(drop=True)
assert (full.t.diff().dropna().dt.total_seconds()==1800).all(), 'gaps'
LL={'london':(51.51,-0.13),'birmingham':(52.49,-1.89),'manchester':(53.48,-2.24),'leeds':(53.80,-1.55),'bristol':(51.45,-2.59),
'norwich':(52.63,1.30),'southampton':(50.90,-1.40),'plymouth':(50.38,-4.14),'cardiff':(51.48,-3.18),'newcastle':(54.98,-1.62),'glasgow':(55.86,-4.25),'edinburgh':(55.95,-3.19)}
t=full.t;mn=(t.dt.minute.values==0)
ww=np.array([W[c] for c in CITIES]);ww=ww/ww.sum()
new={}
def interp(x):
    # hourly value repeated over :00,:30. slot centre h:15 -> 0.75*cur+0.25*prev hour; h:45 -> 0.75*cur+0.25*next hour
    prev=x.shift(2);nxt=x.shift(-2)
    o=np.where(mn,0.75*x+0.25*prev.fillna(x),0.75*x+0.25*nxt.fillna(x))
    return pd.Series(o,index=x.index)
cs_sum=0;ph_sum=0;kt_sum=0
for c in CITIES:
    la,lo=LL[c];el,_=solar_pos(t,la,lo);cz=np.sin(np.radians(np.clip(el,0,None)))
    hz=np.where(cz>0.01,1098*cz*np.exp(-0.057/np.maximum(cz,0.01)),0)
    kt=(full[c+'_ghi']/(full[c+'_clearsky']+5)).clip(0,1.3)
    kti=interp(kt);cli=interp(full[c+'_cloud_pct'])
    new[c+'_kti']=kti.values;new[c+'_phys']=kti.values*hz
    new[c+'_cli']=cli.values
    cs_sum=cs_sum+W[c]*hz;ph_sum=ph_sum+W[c]*kti.values*hz;kt_sum=kt_sum+W[c]*kti.values
s=sum(W.values())
new['hz_w']=cs_sum/s;new['phys_w']=ph_sum/s;new['kti_w']=kt_sum/s
tm=interp(full.temp_mean).values;new['temp_i']=tm
new['phys_w_der']=new['phys_w']*(1-0.004*(tm-25)).clip(0.7,1.1)
new['ghi_mean_i']=interp(full.ghi_mean).values
N=pd.DataFrame(new,index=full.index)
for c in ['phys_w','kti_w']:
    for k in (1,2):
        N[f'{c}_l{k}']=N[c].shift(k);N[f'{c}_f{k}']=N[c].shift(-k)
full=pd.concat([full,N],axis=1)
tr=full[full.src=='tr'].copy();te=full[full.src=='te'].copy()
tr.to_pickle(D+'tr_f5.pkl');te.to_pickle(D+'te_f5.pkl');print('ok',tr.shape,te.shape)
print(np.corrcoef(tr.phys_w.fillna(0),tr.generation_mw/tr.installed_capacity_mwp)[0,1],np.corrcoef(tr.phys_cf_w,tr.cf)[0,1])
