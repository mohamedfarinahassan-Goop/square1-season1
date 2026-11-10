import pandas as pd,numpy as np,lightgbm as lgb,sys,json
D='./';T='generation_mw'
tr=pd.read_pickle(D+'tr_f5.pkl');te=pd.read_pickle(D+'te_f5.pkl')
tr['cf']=tr[T]/tr.installed_capacity_mwp
base=[c for c in pd.read_pickle(D+'tr_f.pkl') if c not in ('id','datetime_utc','t',T,'cf','installed_capacity_mwp','src')]
newf=[c for c in tr if c not in base and c not in ('id','datetime_utc','t',T,'cf','installed_capacity_mwp','src')]
cut=pd.Timestamp('2024-07-01',tz='UTC');a=(tr.t<cut).values;b=~a
def run(F,P,lr=0.05,seed=0,init=None,wfun=None,ret=False):
    Pp=dict(objective='l1',learning_rate=lr,num_leaves=127,min_data_in_leaf=30,feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,verbose=-1,num_threads=8,seed=seed);Pp.update(P)
    w=tr.installed_capacity_mwp.values.copy()
    if wfun is not None: w=w*wfun(tr)
    ia=None if init is None else tr[init].fillna(0).values[a]; ib=None if init is None else tr[init].fillna(0).values[b]
    da=lgb.Dataset(tr[F][a],tr.cf[a],weight=w[a],init_score=ia);db=lgb.Dataset(tr[F][b],tr.cf[b],weight=w[b],init_score=ib)
    m=lgb.train(Pp,da,6000,valid_sets=[db],callbacks=[lgb.early_stopping(200,verbose=False)])
    p=m.predict(tr[F][b],num_iteration=m.best_iteration)+(0 if ib is None else ib)
    p=np.clip(p,0,None)*tr.installed_capacity_mwp.values[b];p=np.where(tr.sun_el.values[b]<-3,0,p)
    mae=float(np.abs(p-tr[T].values[b]).mean())
    return (mae,m.best_iteration,p,m) if ret else (mae,m.best_iteration)
if __name__=='__main__':
    print('n new',len(newf),flush=True)
    for name,F,kw in [('v4',base,{}),('v4+new',base+newf,{})]:
        print(name,run(F,{},**kw),flush=True)
