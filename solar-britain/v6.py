# v6: equal mean of 5 LightGBM variants (v4 features, +interp physics features, ff0.5, extra_trees, 255 leaves), 3 seeds each.
# CV (holdout 2024-07..2025-06) of the blend 115.55 vs v4 118.70. Run after feats.py and v5feats.py.
import pickle,numpy as np,pandas as pd,lightgbm as lgb
from exp5 import tr,te,T,D
V=pickle.load(open(D+'exp5c.pkl','rb'))
keep=['base','new','ff5','xt','l255'];outs=[]
for k in keep:
    _,its,F,P=V[k];ps=[]
    for s,it in zip(range(3),its):
        Pp=dict(objective='l1',learning_rate=0.05,num_leaves=127,min_data_in_leaf=30,feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,verbose=-1,num_threads=6,seed=s);Pp.update(P)
        m=lgb.train(Pp,lgb.Dataset(tr[F],tr.cf,weight=tr.installed_capacity_mwp),int(it*1.1));ps.append(m.predict(te[F]))
    p=np.clip(np.mean(ps,0),0,None)*te.installed_capacity_mwp.values;p=np.where(te.sun_el.values<-3,0,p)
    outs.append(p);print(k,'done',flush=True)
te[T]=np.mean(outs,0)
sub=pd.read_csv(D+'sample_submission.csv')[['id']].merge(te[['id',T]],on='id',how='left')
assert sub[T].notna().all() and len(sub)==17520
sub.to_csv(D+'submissions/v6.csv',index=False);print('saved v6.csv')
