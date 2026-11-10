import pandas as pd, numpy as np, lightgbm as lgb
from sklearn.model_selection import GroupKFold
D='./'
tr=pd.read_csv(D+'train.csv');te=pd.read_csv(D+'test.csv');T='annual_co2e_t'
FU=sorted({f for s in pd.concat([tr.fuels,te.fuels]) for f in s.split(',')})
def fe(d):
    d=d.copy()
    for f in FU: d['has_'+f]=d.fuels.str.split(',').apply(lambda s:f in s).astype(int)
    d['logcap']=np.log1p(d.capacity_mw)
    for c in ['primary_fuel','country','fuels']: d[c]=d[c].astype('category')
    return d
al=pd.concat([tr,te],keys=['tr','te']);al=fe(al);tr=al.loc['tr'].copy();te=al.loc['te'].copy()
F=['year','primary_fuel','fuel_count','country','capacity_mw','logcap','lat_r','lon_r','fuels']+['has_'+f for f in FU]
y=np.log1p(tr[T].values)
rmsle=lambda p,t:np.sqrt(np.mean((np.log1p(p)-np.log1p(t))**2))
P=dict(objective='l2',learning_rate=0.03,num_leaves=31,min_data_in_leaf=30,feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,cat_smooth=20,lambda_l2=1,verbose=-1,num_threads=4)
oof=np.zeros(len(tr));ob=np.zeros(len(tr));its=[]
for i,(a,b) in enumerate(GroupKFold(5).split(tr,y,tr.plant)):
    ta,tb=tr.iloc[a],tr.iloc[b]
    r=(ta[T]/ta.capacity_mw.replace(0,np.nan)).groupby(ta.primary_fuel,observed=True).median()
    ob[b]=tb.capacity_mw.values*tb.primary_fuel.map(r).astype(float).fillna(r.median()).values
    m=lgb.train(P,lgb.Dataset(ta[F],y[a]),5000,valid_sets=[lgb.Dataset(tb[F],y[b])],callbacks=[lgb.early_stopping(200,verbose=False)])
    oof[b]=m.predict(tb[F]);its.append(m.best_iteration)
print('baseline-rule CV RMSLE',rmsle(np.clip(ob,0,None),tr[T].values))
print('model CV RMSLE',rmsle(np.clip(np.expm1(oof),0,None),tr[T].values),'iters',its)
m=lgb.train(P,lgb.Dataset(tr[F],y),int(np.mean(its)*1.1))
te[T]=np.clip(np.expm1(m.predict(te[F])),0,None)
te[['id',T]].to_csv(D+'submissions/v1.csv',index=False)
