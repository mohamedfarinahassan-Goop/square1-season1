import pandas as pd, numpy as np, lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import BallTree
D='./'
tr=pd.read_csv(D+'train.csv');te=pd.read_csv(D+'test.csv');T='annual_co2e_t'
FU=sorted({f for s in pd.concat([tr.fuels,te.fuels]) for f in s.split(',')})
def fe(d):
    d=d.copy()
    for f in FU: d['has_'+f]=d.fuels.str.split(',').apply(lambda s:f in s).astype(int)
    d['logcap']=np.log1p(d.capacity_mw)
    d['cell']=(np.floor(d.lat_r/3)*1000+np.floor(d.lon_r/3)).astype(int)
    d['n_plants_country']=d.groupby('country').plant.transform('nunique')
    d['cap_rank_country']=d.groupby(['country','year']).capacity_mw.rank(pct=True)
    d['cap_share_country']=d.capacity_mw/d.groupby(['country','year']).capacity_mw.transform('sum')
    return d
al=fe(pd.concat([tr,te],keys=['tr','te']));
for c in ['primary_fuel','country','fuels']: al[c+'_cat']=al[c].astype('category')
tr=al.loc['tr'].reset_index(drop=True);te=al.loc['te'].reset_index(drop=True)
tr['li']=np.log1p(tr[T])-np.log1p(tr.capacity_mw)   # log intensity (t per MW)
KEYS={'te_cf':['country','primary_fuel'],'te_c':['country'],'te_f':['primary_fuel'],'te_fuels':['fuels'],'te_cellf':['cell','primary_fuel'],'te_fy':['primary_fuel','year'],'te_cfy':['country','primary_fuel','year']}
def enc(fit,app,k=10):
    o=pd.DataFrame(index=app.index);gm=fit.li.mean()
    for n,ks in KEYS.items():
        s=fit.groupby(ks).li.agg(['sum','count']);m=app[ks].merge(s,left_on=ks,right_index=True,how='left')
        o[n]=((m['sum'].fillna(0)+20*gm)/(m['count'].fillna(0)+20)).values;o[n+'_n']=m['count'].fillna(0).values
    # nearest plants (plant-level mean log intensity)
    pl=fit.groupby('plant').agg(lat=('lat_r','first'),lon=('lon_r','first'),li=('li','mean'),f=('primary_fuel','first'))
    q=np.radians(app[['lat_r','lon_r']].values)
    bt=BallTree(np.radians(pl[['lat','lon']].values),metric='haversine');dd,ii=bt.query(q,k=k)
    o['nn_li']=pl.li.values[ii].mean(1);o['nn_dist']=dd.mean(1)*6371
    o['nnf_li']=np.nan;o['nnf_dist']=np.nan
    for f,g in pl.groupby('f'):
        msk=(app.primary_fuel==f).values
        if not msk.any(): continue
        kk=min(k,len(g));bt=BallTree(np.radians(g[['lat','lon']].values),metric='haversine');dd,ii=bt.query(q[msk],k=kk)
        o.loc[msk,'nnf_li']=g.li.values[ii].mean(1);o.loc[msk,'nnf_dist']=dd.mean(1)*6371;o.loc[msk,'nnf1_li']=g.li.values[ii[:,0]]
    return o
def enc_oof(fit):  # out-of-fold encodings for the fit rows themselves (grouped by plant)
    o=[]
    for a,b in GroupKFold(5).split(fit,groups=fit.plant): o.append(enc(fit.iloc[a],fit.iloc[b]))
    return pd.concat(o).loc[fit.index]
BASE=['year','primary_fuel_cat','fuel_count','country_cat','capacity_mw','logcap','lat_r','lon_r','fuels_cat','n_plants_country','cap_rank_country','cap_share_country']+['has_'+f for f in FU]
EF=list(enc(tr.iloc[:50],tr.iloc[:5]).columns)
F=BASE+EF
y=np.log1p(tr[T].values)
rmsle=lambda p,t:np.sqrt(np.mean((np.log1p(p)-np.log1p(t))**2))
P=dict(objective='l2',learning_rate=0.02,num_leaves=31,min_data_in_leaf=60,feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,cat_smooth=50,cat_l2=10,min_data_per_group=50,lambda_l2=5,verbose=-1,num_threads=4)
oof=np.zeros(len(tr));its=[]
for i,(a,b) in enumerate(GroupKFold(5).split(tr,y,tr.plant)):
    ta,tb=tr.iloc[a].copy(),tr.iloc[b].copy()
    ta[EF]=enc_oof(ta).values;tb[EF]=enc(ta,tb).values
    m=lgb.train({**P,'seed':i},lgb.Dataset(ta[F],y[a]),5000,valid_sets=[lgb.Dataset(tb[F],y[b])],callbacks=[lgb.early_stopping(200,verbose=False)])
    oof[b]=m.predict(tb[F],num_iteration=m.best_iteration);its.append(m.best_iteration)
    print('fold',i,rmsle(np.clip(np.expm1(oof[b]),0,None),tr[T].values[b]),m.best_iteration,flush=True)
print('v2 CV RMSLE',rmsle(np.clip(np.expm1(oof),0,None),tr[T].values),'iters',its)
A=tr.copy();A[EF]=enc_oof(tr).values;E=te.copy();E[EF]=enc(tr,te).values;pe=[]
for s in [0,1,2]:
    m=lgb.train({**P,'seed':s},lgb.Dataset(A[F],y),int(np.mean(its)*1.1));pe.append(m.predict(E[F]))
E[T]=np.clip(np.expm1(np.mean(pe,0)),0,None)
sub=pd.read_csv(D+'sample_submission.csv')[['id']].merge(E[['id',T]],on='id',how='left')
assert sub[T].notna().all() and len(sub)==12560
sub.to_csv(D+'submissions/v2.csv',index=False);print('saved')
