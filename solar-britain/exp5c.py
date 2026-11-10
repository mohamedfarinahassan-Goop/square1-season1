from exp5 import *
import pickle
V={}
vars=[('base',base,{}),('new',base+newf,{}),('huber',base,{'objective':'huber','alpha':0.02}),('l2',base,{'objective':'regression'}),
('ff5',base+newf,{'feature_fraction':0.5}),('xt',base+newf,{'extra_trees':True,'min_data_in_leaf':20}),('l255',base,{'num_leaves':255,'min_data_in_leaf':50,'lambda_l2':5})]
for name,F,P in vars:
    ps=[];its=[]
    for s in range(3):
        mae,it,p,m=run(F,P,seed=s,ret=True);ps.append(p);its.append(it)
    pm=np.mean(ps,0);V[name]=(pm,its,F,P)
    print(name,'3seed',float(np.abs(pm-tr[T].values[b]).mean()),its,flush=True)
    pickle.dump(V,open(D+'exp5c.pkl','wb'))
