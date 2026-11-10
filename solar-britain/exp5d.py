from exp5 import *
import pickle
V=pickle.load(open(D+'exp5c.pkl','rb'))
vars=[('ff3',base+newf,{'feature_fraction':0.3}),('ff5_l63',base+newf,{'feature_fraction':0.5,'num_leaves':63,'min_data_in_leaf':60}),('xt_ff5',base+newf,{'extra_trees':True,'min_data_in_leaf':20,'feature_fraction':0.5}),('ff5_lr03',base+newf,{'feature_fraction':0.5,'learning_rate':0.03})]
for name,F,P in vars:
    ps=[];its=[]
    for s in range(3):
        Pq=dict(P);lr=Pq.pop('learning_rate',0.05)
        mae,it,p,m=run(F,{**Pq,'num_threads':4},lr=lr,seed=s,ret=True);ps.append(p);its.append(it)
    pm=np.mean(ps,0);V[name]=(pm,its,F,{**P})
    print(name,'3seed',float(np.abs(pm-tr[T].values[b]).mean()),its,flush=True)
    pickle.dump(V,open(D+'exp5d.pkl','wb'))
