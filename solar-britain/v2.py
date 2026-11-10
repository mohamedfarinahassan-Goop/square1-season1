import pandas as pd, numpy as np, lightgbm as lgb
D='./'
tr=pd.read_csv(D+'train.csv');te=pd.read_csv(D+'test.csv');T='generation_mw'
CITIES=[c[:-4] for c in tr if c.endswith('_ghi')]
def solar_pos(t,lat=52.5,lon=-1.5):
    # NOAA-style approximation; t = UTC timestamps (half-hour period start, use midpoint +15min)
    tt=t+pd.Timedelta(minutes=15);doy=tt.dt.dayofyear.values;hr=tt.dt.hour.values+tt.dt.minute.values/60
    g=2*np.pi/365*(doy-1+(hr-12)/24)
    eqt=229.18*(0.000075+0.001868*np.cos(g)-0.032077*np.sin(g)-0.014615*np.cos(2*g)-0.040849*np.sin(2*g))
    dec=0.006918-0.399912*np.cos(g)+0.070257*np.sin(g)-0.006758*np.cos(2*g)+0.000907*np.sin(2*g)-0.002697*np.cos(3*g)+0.00148*np.sin(3*g)
    ha=np.radians((hr*60+eqt+4*lon)/4-180);la=np.radians(lat)
    cz=np.sin(la)*np.sin(dec)+np.cos(la)*np.cos(dec)*np.cos(ha);el=np.degrees(np.arcsin(np.clip(cz,-1,1)))
    az=np.degrees(np.arctan2(np.sin(ha),np.cos(ha)*np.sin(la)-np.tan(dec)*np.cos(la)))
    return el,az
def fe(d):
    d=d.copy();t=pd.to_datetime(d.datetime_utc);d['t']=t
    d['month']=t.dt.month;d['hh']=t.dt.hour*2+t.dt.minute//30;d['doy']=t.dt.dayofyear
    g=[c for c in d if c.endswith('_ghi')];cs=[c for c in d if c.endswith('_clearsky')]
    d['ghi_mean']=d[g].mean(1);d['cs_mean']=d[cs].mean(1);d['ghi_std']=d[g].std(1)
    d['kt']=d.ghi_mean/(d.cs_mean+1);d['cloud_mean']=d[[c for c in d if c.endswith('_cloud_pct')]].mean(1)
    d['temp_mean']=d[[c for c in d if c.endswith('_temp_c')]].mean(1)
    # v2: solar geometry, per-city clearness, south-weighted GHI (most GB PV is in the south)
    d['sun_el'],d['sun_az']=solar_pos(t);d['sin_el']=np.sin(np.radians(d.sun_el.clip(lower=0)))
    for c in CITIES: d[c+'_kt']=d[c+'_ghi']/(d[c+'_clearsky']+1)
    south=['london','bristol','southampton','plymouth','cardiff','norwich','birmingham']
    d['ghi_south']=d[[c+'_ghi' for c in south]].mean(1);d['kt_south']=d.ghi_south/(d[[c+'_clearsky' for c in south]].mean(1)+1)
    d['ghi_x_cap']=d.ghi_mean*d.installed_capacity_mwp/1e4
    return d
full=pd.concat([tr,te],keys=['tr','te']).reset_index(level=0).rename(columns={'level_0':'src'}).reset_index(drop=True)
full=fe(full).sort_values('t')
# within-day lags/leads of weather (weather known on test rows; no target used). Only use true 30-min neighbours.
dt=full.t.diff().dt.total_seconds().values;dtf=-full.t.diff(-1).dt.total_seconds().values
for c in ['ghi_mean','kt','cloud_mean','ghi_south','kt_south']:
    for k in (1,2):
        l=full[c].shift(k);f=full[c].shift(-k)
        okl=full.t.diff(k).dt.total_seconds().values==1800*k;okf=(-full.t.diff(-k)).dt.total_seconds().values==1800*k
        full[f'{c}_l{k}']=np.where(okl,l,np.nan);full[f'{c}_f{k}']=np.where(okf,f,np.nan)
    full[c+'_d']=full.t.dt.date;full[c+'_daymean']=full.groupby(c+'_d')[c].transform('mean');full=full.drop(columns=c+'_d')
full=full.sort_index()
tr=full[full.src=='tr'].copy();te=full[full.src=='te'].copy()
tr['cf']=tr[T]/tr.installed_capacity_mwp
F=[c for c in tr if c not in ('id','datetime_utc','t',T,'cf','installed_capacity_mwp','src')]
P=dict(objective='l1',learning_rate=0.03,num_leaves=63,min_data_in_leaf=40,feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,verbose=-1,num_threads=4)
cut=pd.Timestamp('2024-07-01',tz='UTC');a=tr.t<cut;b=~a
SEEDS=[0,1,2];pv=[];its=[]
for s in SEEDS:
    m=lgb.train({**P,'seed':s},lgb.Dataset(tr[a][F],tr[a].cf,weight=tr[a].installed_capacity_mwp),4000,valid_sets=[lgb.Dataset(tr[b][F],tr[b].cf,weight=tr[b].installed_capacity_mwp)],callbacks=[lgb.early_stopping(200,verbose=False)])
    p=np.clip(m.predict(tr[b][F],num_iteration=m.best_iteration),0,None)*tr[b].installed_capacity_mwp
    pv.append(p);its.append(m.best_iteration);print('seed',s,'MAE',np.abs(p-tr[b][T]).mean(),'iters',m.best_iteration,flush=True)
pb=np.mean(pv,0);pb=np.where(tr[b].sun_el<-3,0,pb)
print('v2 CV MAE (seed avg)',np.abs(pb-tr[b][T]).mean())
imp=pd.Series(m.feature_importance('gain'),F).sort_values(ascending=False);print(imp.head(15))
pe=[]
for s,it in zip(SEEDS,its):
    m=lgb.train({**P,'seed':s},lgb.Dataset(tr[F],tr.cf,weight=tr.installed_capacity_mwp),int(it*1.1));pe.append(m.predict(te[F]))
te[T]=np.clip(np.mean(pe,0),0,None)*te.installed_capacity_mwp;te.loc[te.sun_el<-3,T]=0
sub=pd.read_csv(D+'sample_submission.csv')[['id']].merge(te[['id',T]],on='id',how='left')
assert sub[T].notna().all() and len(sub)==17520
sub.to_csv(D+'submissions/v2.csv',index=False);print('saved')
