import pickle
src=open('./v2.py').read().split("SEEDS=[0,1,2]")[0]
exec(src)
P['num_threads']=3
a=tr.t<'2024-07-01';b=~a
A=te_enc(tr[a],tr[a]);B=te_enc(tr[a],tr[b])

def run(name,F,PP,logt=False,seeds=(0,1,2)):
    pv=[];its=[]
    ya=np.log1p(A[T].clip(lower=0)) if logt else A[T];yb=np.log1p(B[T].clip(lower=0)) if logt else B[T]
    for s in seeds:
        m=lgb.train({**PP,'seed':s},lgb.Dataset(A[F],ya),5000,valid_sets=[lgb.Dataset(B[F],yb)],callbacks=[lgb.early_stopping(200,verbose=False)])
        p=m.predict(B[F],num_iteration=m.best_iteration);p=np.expm1(p) if logt else p;pv.append(p);its.append(m.best_iteration)
    pm=np.mean(pv,0);res[name]=(pm,its,F,PP,logt);pickle.dump(res,open(D+'exp7.pkl','wb'))
    print(name,np.abs(pm-B[T]).mean(),its,flush=True)
import os
res=pickle.load(open(D+'exp7.pkl','rb')) if os.path.exists(D+'exp7.pkl') else {}
run('log',F,P,logt=True)
run('log_l127',F,{**P,'num_leaves':127,'min_data_in_leaf':30,'feature_fraction':0.5},logt=True)
run('huber_log',F,{**P,'objective':'huber','alpha':0.3},logt=True)
