# v3: mean (log1p space) of 4 LightGBM configs: v2, v2+plant-level feature-only aggregates, 63 leaves/ff0.5, huber. GroupKFold-by-plant CV 1.0060 vs v2 1.0157.
exec(open('./exp3.py').read().split("print(tr.groupby")[0])
r=pickle.load(open(D+'exp3.pkl','rb'))
A=tr.copy();A[EF]=enc_oof(tr).values;E=te.copy();E[EF]=enc(tr,te).values;outs=[]
for k,(oof,its,FF,PP) in r.items():
    pe=[lgb.train({**PP,'seed':s},lgb.Dataset(A[FF],y),int(np.mean(its)*1.1)).predict(E[FF]) for s in range(3)]
    outs.append(np.mean(pe,0));print(k,flush=True)
E[T]=np.clip(np.expm1(np.mean(outs,0)),0,None)
sub=pd.read_csv(D+'sample_submission.csv')[['id']].merge(E[['id',T]],on='id',how='left')
assert sub[T].notna().all() and len(sub)==12560
sub.to_csv(D+'submissions/v3.csv',index=False)
import os
if os.path.exists(D+'submissions/v2.csv'):  # optional comparison with v2
    v2=pd.read_csv(D+'submissions/v2.csv');print('saved; corr log vs v2',np.corrcoef(np.log1p(sub[T]),np.log1p(sub[['id']].merge(v2,on='id')[T]))[0,1])
