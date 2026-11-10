# v7: v2 features, L1 on log1p(pm25 clipped at 0) (median is invariant to the log, but trees fit relative error better).
# CV (train<2024-07, valid 2024-07..2025-03): log_l127 2.564, log 2.576, huber_log 2.579 vs v2 2.636.
import pickle
exec(open('./v2.py').read().split("SEEDS=[0,1,2]")[0])
r=pickle.load(open(D+'exp7.pkl','rb'))
A=te_enc(tr,tr);E=te_enc(tr,te);ya=np.log1p(A[T].clip(lower=0));out={}
for k in ['log_l127','log','huber_log']:
    _,its,FF,PP,_=r[k];PP={**PP,'num_threads':6}
    pe=[np.expm1(lgb.train({**PP,'seed':s},lgb.Dataset(A[FF],ya),int(it*1.1)).predict(E[FF])) for s,it in zip(range(3),its)]
    out[k]=np.clip(np.mean(pe,0),0,None);print(k,flush=True)
    pickle.dump(out,open(D+'v7_preds.pkl','wb'))
ss=pd.read_csv(D+'sample_submission.csv')[['id']]
def save(name,v):
    s=ss.merge(pd.DataFrame({'id':E.id.values,T:v}),on='id',how='left');assert s[T].notna().all() and len(s)==66699;s.to_csv(D+'submissions/'+name,index=False);print('saved',name)
save('v7.csv',out['log_l127'])
save('v7mean3.csv',np.mean([out[k] for k in out],0))
import os
if os.path.exists(D+'submissions/v125mean.csv'):  # optional blend with the earlier v1/v2/v5 mean; not needed for v7.csv
    b=ss.merge(pd.read_csv(D+'submissions/v125mean.csv'),on='id');m=ss.merge(pd.DataFrame({'id':E.id.values,T:out['log_l127']}),on='id')
    save2=ss.assign(**{T:0.5*b[T].values+0.5*m[T].values});save2.to_csv(D+'submissions/v7_v125blend.csv',index=False);print('saved v7_v125blend.csv corr',np.corrcoef(b[T],m[T])[0,1])
