import pandas as pd, numpy as np, lightgbm as lgb, sys
D='./'
tr=pd.read_csv(D+'train.csv');te=pd.read_csv(D+'test.csv');T='home_margin'
full=pd.concat([tr,te],keys=['tr','te']).reset_index(level=0).rename(columns={'level_0':'src'}).reset_index(drop=True)
full['elo_diff']=full.home_elo-full.away_elo
full['rest_diff']=full.home_rest.clip(0,10)-full.away_rest.clip(0,10)
full['logit_f']=np.log(full.home_forecast.clip(1e-6,1-1e-6)/(1-full.home_forecast.clip(1e-6,1-1e-6)))
full['b2b_h']=(full.home_rest<=1).astype(int);full['b2b_a']=(full.away_rest<=1).astype(int)
full['rest_x_elo']=full.rest_diff*full.elo_diff/400
full['home_team_c']=full.home_team.astype('category')
full['away_team_c']=full.away_team.astype('category')
full['home_fr_c']=full.home_franchise.astype('category')
full['away_fr_c']=full.away_franchise.astype('category')
tr=full[(full.src=='tr')&(full.league=='NBA')].copy();te=full[full.src=='te'].copy()
F=['elo_diff','logit_f','home_forecast','home_elo','away_elo','neutral','playoffs','season_day',
   'home_seasongame','away_seasongame','home_rest','away_rest','rest_diff','b2b_h','b2b_a','season',
   'rest_x_elo','home_team_c','away_team_c']
P=dict(objective='l1',learning_rate=0.02,num_leaves=15,min_data_in_leaf=200,feature_fraction=0.8,
       bagging_fraction=0.8,bagging_freq=1,verbose=-1,num_threads=4)
a=tr.season<=2009;b=tr.season>=2010
# calibrated mapping from 538 win probability (and a couple extras)
from sklearn.linear_model import QuantileRegressor
XL=['logit_f','rest_diff','playoffs','neutral']
def lin(fit,app):
    f=fit[fit.season>=fit.season.max()-9]
    q=QuantileRegressor(quantile=0.5,alpha=0,solver='highs').fit(f[XL],f[T])
    return q.predict(app[XL]),q
pl,q=lin(tr[a],tr[b]);print('linear-logit CV MAE',float(np.abs(pl-tr[b][T]).mean()),'coef',q.coef_,q.intercept_,flush=True)
pv=[];its=[]
for s in [0,1,2]:
    m=lgb.train({**P,'seed':s},lgb.Dataset(tr[a][F],tr[a][T],categorical_feature=['home_team_c','away_team_c']),
                5000,valid_sets=[lgb.Dataset(tr[b][F],tr[b][T],categorical_feature=['home_team_c','away_team_c'])],
                callbacks=[lgb.early_stopping(300,verbose=False)])
    pv.append(m.predict(tr[b][F],num_iteration=m.best_iteration));its.append(m.best_iteration)
    print('seed',s,float(np.abs(pv[-1]-tr[b][T]).mean()),m.best_iteration,flush=True)
pg=np.mean(pv,0);print('lgb CV MAE',float(np.abs(pg-tr[b][T]).mean()),flush=True)
best=(9e9,1.0)
for w in np.arange(0,1.01,0.1):
    e=float(np.abs(w*pg+(1-w)*pl-tr[b][T]).mean());best=min(best,(e,w))
print('v3 CV MAE (blend)',best[0],'w_lgb',best[1],flush=True)
if best[0]>=9.234:
    print('no CV gain vs v2 9.234; not writing');sys.exit(0)
w=best[1];pe=[]
for s,it in zip([0,1,2],its):
    m=lgb.train({**P,'seed':s},lgb.Dataset(tr[F],tr[T],categorical_feature=['home_team_c','away_team_c']),max(100,int(it*1.1)))
    pe.append(m.predict(te[F]))
plt,_=lin(tr,te);te[T]=w*np.mean(pe,0)+(1-w)*plt
sub=pd.read_csv(D+'sample_submission.csv')[['id']].merge(te[['id',T]],on='id',how='left')
assert sub[T].notna().all() and len(sub)==3944
sub.to_csv(D+'submissions/v3.csv',index=False);print('saved submissions/v3.csv',best[0],flush=True)
