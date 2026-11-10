import sys
src=open('./v2.py').read()
head=src.split("oof=np.zeros(len(tr))")[0]
# plant-level features from the feature columns only (no target)
head=head.replace("al=fe(pd.concat([tr,te],keys=['tr','te']));","al=fe(pd.concat([tr,te],keys=['tr','te']));\ng=al.groupby('plant')\nal['p_n']=g.year.transform('count');al['p_y0']=g.year.transform('min');al['p_y1']=g.year.transform('max');al['p_age']=al.year-al.p_y0;al['p_left']=al.p_y1-al.year\nal['p_capmax']=g.capacity_mw.transform('max');al['p_capmin']=g.capacity_mw.transform('min');al['p_capr']=al.capacity_mw/al.p_capmax;al['p_capchg']=al.capacity_mw-g.capacity_mw.shift(1).fillna(al.capacity_mw) if False else al.capacity_mw/g.capacity_mw.transform('mean')\n")
exec(head)
PF=['p_n','p_y0','p_y1','p_age','p_left','p_capr','p_capchg']
P['num_threads']=3
import pickle
res={}
def cv(F,PP,name):
    oof=np.zeros(len(tr));its=[]
    for i,(a,b) in enumerate(GroupKFold(5).split(tr,y,tr.plant)):
        ta,tb=tr.iloc[a].copy(),tr.iloc[b].copy()
        ta[EF]=enc_oof(ta).values;tb[EF]=enc(ta,tb).values
        m=lgb.train({**PP,'seed':i},lgb.Dataset(ta[F],y[a]),5000,valid_sets=[lgb.Dataset(tb[F],y[b])],callbacks=[lgb.early_stopping(200,verbose=False)])
        oof[b]=m.predict(tb[F],num_iteration=m.best_iteration);its.append(m.best_iteration)
    s=rmsle(np.clip(np.expm1(oof),0,None),tr[T].values);print(name,'CV',s,its,flush=True);res[name]=(oof,its,F,PP);pickle.dump(res,open(D+'exp3.pkl','wb'))
print(tr.groupby('plant').year.count().describe())
cv(F,P,'v2')
cv(F+PF,P,'v2+plant')
cv(F+PF,{**P,'num_leaves':63,'min_data_in_leaf':30,'feature_fraction':0.5},'plant_l63')
cv(F+PF,{**P,'objective':'huber','alpha':1.0},'plant_huber')
