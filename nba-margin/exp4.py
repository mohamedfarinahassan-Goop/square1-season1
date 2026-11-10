import sys;sys.argv=['x']
src=open('v2.py').read().split("pl,q=lin")[0]
exec(src)
P['num_threads']=2
y=tr[b][T].values
pv=[];its=[]
for s in range(5):
    m=lgb.train({**P,'seed':s},lgb.Dataset(tr[a][F],tr[a][T]),5000,valid_sets=[lgb.Dataset(tr[b][F],tr[b][T])],callbacks=[lgb.early_stopping(300,verbose=False)])
    pv.append(m.predict(tr[b][F],num_iteration=m.best_iteration));its.append(m.best_iteration)
pg=np.mean(pv,0);print('5seed',np.abs(pg-y).mean(),its)
mae=lambda p:np.abs(p-y).mean()
print('round',mae(np.round(pg)))
def snap(p): r=np.round(p);r=np.where(r==0,np.sign(p)+(p==0),r);return r
print('round+snap0',mae(snap(pg)))
p2=np.where(np.abs(pg)<1,np.sign(pg),pg);print('snap0 only',mae(p2))
for d in [-0.5,-0.3,-0.2,-0.1,0.1,0.2,0.3,0.5]: print('shift',d,mae(pg+d))
np.save('cvpred.npy',pg)
for d in [0,-0.2,-0.3,-0.5]: print('round shift',d,mae(np.round(pg+d)))
# residual median by season in validation
r=y-pg;print(pd.Series(r).groupby(tr[b].season.values).median(),pd.Series(r).groupby(tr[b].season.values).mean())
