import pandas as pd, numpy as np, lightgbm as lgb
D='./'
tr=pd.read_csv(D+'train.csv');te=pd.read_csv(D+'test.csv');T='generation_mw'
def fe(d):
    d=d.copy();t=pd.to_datetime(d.datetime_utc);d['t']=t
    d['month']=t.dt.month;d['hh']=t.dt.hour*2+t.dt.minute//30;d['doy']=t.dt.dayofyear
    g=[c for c in d if c.endswith('_ghi')];cs=[c for c in d if c.endswith('_clearsky')]
    d['ghi_mean']=d[g].mean(1);d['cs_mean']=d[cs].mean(1);d['ghi_std']=d[g].std(1)
    d['kt']=d.ghi_mean/(d.cs_mean+1);d['cloud_mean']=d[[c for c in d if c.endswith('_cloud_pct')]].mean(1)
    d['temp_mean']=d[[c for c in d if c.endswith('_temp_c')]].mean(1)
    return d
tr=fe(tr);te=fe(te)
tr['cf']=tr[T]/tr.installed_capacity_mwp
F=[c for c in tr if c not in ('id','datetime_utc','t',T,'cf','installed_capacity_mwp')]
P=dict(objective='l1',learning_rate=0.03,num_leaves=63,min_data_in_leaf=40,feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,verbose=-1,num_threads=4)
cut=pd.Timestamp('2024-07-01',tz='UTC');a=tr.t<cut;b=~a
def base(fit,app):
    m=fit.groupby(['month','hh']).cf.mean();return app[['month','hh']].merge(m.rename('m').reset_index(),how='left').m.fillna(0).values*app.installed_capacity_mwp.values
print('baseline-rule CV MAE',np.abs(base(tr[a],tr[b])-tr[b][T]).mean())
# model predicts capacity factor; weight by capacity so l1 on cf*cap ~ MAE in MW
m=lgb.train(P,lgb.Dataset(tr[a][F],tr[a].cf,weight=tr[a].installed_capacity_mwp),4000,valid_sets=[lgb.Dataset(tr[b][F],tr[b].cf,weight=tr[b].installed_capacity_mwp)],callbacks=[lgb.early_stopping(200,verbose=False)])
p=np.clip(m.predict(tr[b][F]),0,None)*tr[b].installed_capacity_mwp
print('model CV MAE',np.abs(p-tr[b][T]).mean(),'iters',m.best_iteration)
m=lgb.train(P,lgb.Dataset(tr[F],tr.cf,weight=tr.installed_capacity_mwp),int(m.best_iteration*1.1))
te[T]=np.clip(m.predict(te[F]),0,None)*te.installed_capacity_mwp
te[['id',T]].to_csv(D+'submissions/v1.csv',index=False)
