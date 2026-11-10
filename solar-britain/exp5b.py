from exp5 import *
mae,it,p,m=run(base,{},ret=True)
y=tr[T].values[b];tb=tr.t[b]
print('mae',mae,'bias',(p-y).mean(),'sum ratio',p.sum()/y.sum())
for s in [0.97,0.98,0.99,1.01,1.02]: print(s,np.abs(p*s-y).mean())
df=pd.DataFrame({'m':tb.dt.month.values,'e':p-y,'ae':np.abs(p-y),'y':y})
print(df.groupby('m').agg(bias=('e','mean'),mae=('ae','mean'),y=('y','mean')).to_string())
drop=[c for c in base if c not in ('cap','ghi_x_cap')]
print('drop cap',run(drop,{}))
print('drop cap+doy',run([c for c in drop if c!='doy'],{}))
yr=lambda d:np.exp((d.t-pd.Timestamp('2024-07-01',tz='UTC')).dt.days.values/365*np.log(2))
print('recency x2/yr',run(base,{},wfun=yr))
