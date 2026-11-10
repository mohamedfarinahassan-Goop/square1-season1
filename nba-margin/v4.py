# v4: v2 pre-game features (no future elo), 5-seed LGB L1, trained on all NBA seasons; predictions rounded to integers
# (target is an integer and never 0, so MAE-optimal points are integers). CV 2010-12: raw 9.2327, rounded 9.2283, round(p-0.2) 9.2226 (v1 9.235).
import sys;sys.argv=['x']
exec(open('./v2.py').read().split("pl,q=lin")[0])
P['num_threads']=4
its=[402, 477, 320, 431, 381];pe=[]
for s,it in zip(range(5),its):
    m=lgb.train({**P,'seed':s},lgb.Dataset(tr[F],tr[T]),int(it*1.1));pe.append(m.predict(te[F]))
p=np.mean(pe,0)
sub=pd.read_csv(D+'sample_submission.csv')[['id']]
for name,v in [('v4_round',np.round(p)),('v4_round_m02',np.round(p-0.2))]:
    s=sub.merge(pd.DataFrame({'id':te.id.values,T:v}),on='id',how='left');assert s[T].notna().all() and len(s)==3944
    s.to_csv(D+'submissions/'+name+'.csv',index=False);print('saved',name,s[T].describe().to_dict())
import os
if os.path.exists(D+'submissions/v1.csv'):  # optional comparison with v1
    v1=pd.read_csv(D+'submissions/v1.csv');print('corr with v1',np.corrcoef(sub.merge(v1,on='id')[T],sub.merge(pd.DataFrame({'id':te.id.values,T:p}),on='id')[T])[0,1])
