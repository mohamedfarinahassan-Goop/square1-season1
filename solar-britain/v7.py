# v7: equal mean of 9 LightGBM variants = v6's 5 + ff0.3, ff0.5/63 leaves, extra_trees ff0.5, ff0.5 lr0.03 (3 seeds each). CV 115.08 vs v6 115.55.
import pickle,numpy as np,pandas as pd,lightgbm as lgb
from exp5 import tr,te,T,D
V=pickle.load(open(D+'exp5d.pkl','rb'));outs=[]
for k in ['ff3','ff5_l63','xt_ff5','ff5_lr03']:
    _,its,F,P=V[k];P=dict(P);lr=P.pop('learning_rate',0.05);ps=[]
    for s,it in zip(range(3),its):
        Pp=dict(objective='l1',learning_rate=lr,num_leaves=127,min_data_in_leaf=30,feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,verbose=-1,num_threads=7,seed=s);Pp.update(P)
        ps.append(lgb.train(Pp,lgb.Dataset(tr[F],tr.cf,weight=tr.installed_capacity_mwp),int(it*1.1)).predict(te[F]))
    p=np.clip(np.mean(ps,0),0,None)*te.installed_capacity_mwp.values;outs.append(np.where(te.sun_el.values<-3,0,p));print(k,flush=True)
ss=pd.read_csv(D+'sample_submission.csv')[['id']]
v6=ss.merge(pd.read_csv(D+'submissions/v6.csv'),on='id')[T].values
new=ss.merge(pd.DataFrame({'id':te.id.values,T:np.sum(outs,0)}),on='id')[T].values
sub=ss.assign(**{T:(5*v6+new)/9});assert sub[T].notna().all() and len(sub)==17520
sub.to_csv(D+'submissions/v7.csv',index=False);print('saved v7.csv')
