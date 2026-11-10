import pandas as pd, numpy as np, lightgbm as lgb
D='./'
tr=pd.read_csv(D+'train.csv');te=pd.read_csv(D+'test.csv');T='pm25'
W=['wind_ms','precip_mm','temp_c','rh_pct','pressure_kpa','ghi_kwh_m2']
def fe(d):
    d=d.copy();t=pd.to_datetime(d.date);d['t']=t;d['month']=t.dt.month;d['doy']=t.dt.dayofyear;d['dow']=t.dt.dayofweek
    d['trange']=d.temp_max_c-d.temp_min_c;d['site_c']=d.site.str[1:].astype(int);d['div_c']=d.division.astype('category').cat.codes
    d=d.sort_values(['site','t'])
    for c in W:  # weather is a known input on test days: lags/leads/rolling within site use no target
        g=d.groupby('site')[c];d[c+'_l1']=g.shift(1);d[c+'_f1']=g.shift(-1)
        d[c+'_r3']=g.transform(lambda s:s.rolling(3,min_periods=1).mean())
        d[c+'_r7']=g.transform(lambda s:s.rolling(7,min_periods=1).mean())
        d[c+'_dev7']=d[c]-d[c+'_r7']
    d['dp']=d.groupby('site').pressure_kpa.diff()
    d['dry_days']=d.groupby('site').precip_mm.transform(lambda s:(s<0.5).astype(int).groupby((s>=0.5).cumsum()).cumsum())
    d['stag']=d.wind_ms_r3*(1+d.precip_mm_r3)
    # same-day division / national weather context
    for c in ['wind_ms','precip_mm','temp_c','pressure_kpa']:
        d[c+'_div']=d.groupby(['division','date'])[c].transform('mean');d[c+'_nat']=d.groupby('date')[c].transform('mean')
    return d.sort_index()
full=fe(pd.concat([tr,te],keys=['tr','te']).reset_index(level=0).rename(columns={'level_0':'src'}).reset_index(drop=True))
tr=full[full.src=='tr'].copy();te=full[full.src=='te'].copy()
def te_enc(fit,app):
    fit=fit[~((fit.t>='2023-05-15')&(fit.t<'2023-08-15'))]
    s=fit.groupby('site')[T].agg(['mean','median']).add_prefix('site_');sm=fit.groupby(['site','month'])[T].mean().rename('sm')
    sq=fit.groupby('site')[T].quantile(0.9).rename('site_q90')
    dm=fit.groupby(['division','month'])[T].median().rename('dm')
    o=app.merge(s,left_on='site',right_index=True,how='left').merge(sm,left_on=['site','month'],right_index=True,how='left')
    o=o.merge(sq,left_on='site',right_index=True,how='left').merge(dm,left_on=['division','month'],right_index=True,how='left')
    o['sm']=o.sm.fillna(o.site_mean);return o
F=['lat_r','lon_r','temp_c','temp_max_c','temp_min_c','rh_pct','wind_ms','precip_mm','pressure_kpa','ghi_kwh_m2','month','doy','dow','trange','site_c','div_c','site_mean','site_median','sm','site_q90','dm','dp','dry_days','stag']+[c for c in tr if c.endswith(('_l1','_f1','_r3','_r7','_dev7','_div','_nat'))]
P=dict(objective='l1',learning_rate=0.03,num_leaves=63,min_data_in_leaf=50,feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,verbose=-1,num_threads=4)
SEEDS=[0,1,2]
a=tr.t<'2024-04-01';b=(tr.t>='2024-04-01')&(tr.t<'2025-01-01')
A=te_enc(tr[a],tr[a]);B=te_enc(tr[a],tr[b])
pv=[];its=[]
for s in SEEDS:
    m=lgb.train({**P,'seed':s},lgb.Dataset(A[F],A[T],weight=np.where((A.t>='2023-05-15')&(A.t<'2023-08-15'),0.3,1.0)),5000,valid_sets=[lgb.Dataset(B[F],B[T])],callbacks=[lgb.early_stopping(200,verbose=False)])
    pv.append(m.predict(B[F],num_iteration=m.best_iteration));its.append(m.best_iteration)
    print('seed',s,'MAE',np.abs(pv[-1]-B[T]).mean(),'iters',m.best_iteration,flush=True)
print('v5 CV MAE (seed avg)',np.abs(np.mean(pv,0)-B[T]).mean())
A=te_enc(tr,tr);E=te_enc(tr,te);pe=[]
for s,it in zip(SEEDS,its):
    m=lgb.train({**P,'seed':s},lgb.Dataset(A[F],A[T],weight=np.where((A.t>='2023-05-15')&(A.t<'2023-08-15'),0.3,1.0)),int(it*1.1));pe.append(m.predict(E[F]))
E[T]=np.clip(np.mean(pe,0),0,None)
sub=pd.read_csv(D+'sample_submission.csv')[['id']].merge(E[['id',T]],on='id',how='left')
assert sub[T].notna().all() and len(sub)==66699
sub.to_csv(D+'submissions/v5.csv',index=False);print('saved')
v1=pd.read_csv(D+'submissions/v1.csv');bl=sub[['id']].merge(v1,on='id');bl[T]=0.5*sub[T].values+0.5*bl[T].values
bl.to_csv(D+'submissions/v5blend.csv',index=False);print('saved blend',bl[T].describe())
