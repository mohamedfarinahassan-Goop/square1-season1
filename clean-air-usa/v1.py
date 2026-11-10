import pandas as pd, numpy as np, lightgbm as lgb
D='./'
tr=pd.read_csv(D+'train.csv');te=pd.read_csv(D+'test.csv');T='pm25'
def fe(d):
    d=d.copy();t=pd.to_datetime(d.date);d['t']=t;d['month']=t.dt.month;d['doy']=t.dt.dayofyear;d['dow']=t.dt.dayofweek
    d['trange']=d.temp_max_c-d.temp_min_c;d['site_c']=d.site.str[1:].astype(int);d['div_c']=d.division.astype('category').cat.codes
    d=d.sort_values(['site','t'])
    for c in ['wind_ms','precip_mm','temp_c','rh_pct','pressure_kpa']:  # weather is known for test days, so lags/leads within site OK
        g=d.groupby('site')[c];d[c+'_l1']=g.shift(1);d[c+'_r3']=g.transform(lambda s:s.rolling(3,min_periods=1).mean())
    d['dp']=d.groupby('site').pressure_kpa.diff()
    return d.sort_index()
full=fe(pd.concat([tr,te],keys=['tr','te']).reset_index(level=0).rename(columns={'level_0':'src'}).reset_index(drop=True))
tr=full[full.src=='tr'].copy();te=full[full.src=='te'].copy()
def te_enc(fit,app):
    s=fit.groupby('site')[T].agg(['mean','median']).add_prefix('site_');sm=fit.groupby(['site','month'])[T].mean().rename('sm')
    o=app.merge(s,left_on='site',right_index=True,how='left').merge(sm,left_on=['site','month'],right_index=True,how='left')
    o['sm']=o.sm.fillna(o.site_mean);return o
F=['lat_r','lon_r','temp_c','temp_max_c','temp_min_c','rh_pct','wind_ms','precip_mm','pressure_kpa','ghi_kwh_m2','month','doy','dow','trange','site_c','div_c','site_mean','site_median','sm']+[c for c in tr if c.endswith(('_l1','_r3'))]+['dp']
P=dict(objective='l1',learning_rate=0.03,num_leaves=63,min_data_in_leaf=50,feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,verbose=-1,num_threads=4)
a=tr.t<'2024-07-01';b=~a
A=te_enc(tr[a],tr[a]);B=te_enc(tr[a],tr[b])
print('baseline-rule CV MAE',np.abs(B.sm-B[T]).mean())
m=lgb.train(P,lgb.Dataset(A[F],A[T]),5000,valid_sets=[lgb.Dataset(B[F],B[T])],callbacks=[lgb.early_stopping(200,verbose=False)])
print('model CV MAE',np.abs(m.predict(B[F])-B[T]).mean(),'iters',m.best_iteration)
A=te_enc(tr,tr);E=te_enc(tr,te)
m=lgb.train(P,lgb.Dataset(A[F],A[T]),int(m.best_iteration*1.1))
E[T]=np.clip(m.predict(E[F]),0,None)
E[['id',T]].to_csv(D+'submissions/v1.csv',index=False)
