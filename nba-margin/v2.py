import pandas as pd, numpy as np, lightgbm as lgb, sys
D='./'
FUTURE='--future' in sys.argv  # also use the team's NEXT pre-game elo (post-game info; leak-like, kept optional)
tr=pd.read_csv(D+'train.csv');te=pd.read_csv(D+'test.csv');T='home_margin'
full=pd.concat([tr,te],keys=['tr','te']).reset_index(level=0).rename(columns={'level_0':'src'}).reset_index(drop=True)
def fe(d):
    d=d.copy();d['elo_diff']=d.home_elo-d.away_elo;d['rest_diff']=d.home_rest.clip(0,10)-d.away_rest.clip(0,10)
    d['logit_f']=np.log(d.home_forecast/(1-d.home_forecast));d['b2b_h']=(d.home_rest<=1).astype(int);d['b2b_a']=(d.away_rest<=1).astype(int)
    return d
full=fe(full)
# team-game long table: elo dynamics from the given pre-game elos only (no target used)
L=pd.concat([full[['id','season','season_day','home_team','home_elo']].set_axis(['id','season','day','team','elo'],axis=1).assign(side='h'),
             full[['id','season','season_day','away_team','away_elo']].set_axis(['id','season','day','team','elo'],axis=1).assign(side='a')])
L=L.sort_values(['team','season','day']);g=L.groupby(['team','season']).elo
L['d_prev']=L.elo-g.shift(1)            # elo change caused by the team's previous game (its result) -> known pre-game
L['d_prev2']=g.shift(1)-g.shift(2)
L['form5']=L.elo-g.shift(5)              # elo change over the last 5 games
L['form10']=L.elo-g.shift(10)
L['elo_start']=L.elo-g.transform('first')  # change since season start
L['d_next']=g.shift(-1)-L.elo            # elo change caused by THIS game (post-game; only with --future)
cols=['d_prev','d_prev2','form5','form10','elo_start','d_next']
for s in 'ha':
    x=L[L.side==s].set_index('id')[cols].add_prefix(s+'_');full=full.join(x,on='id')
full['form5_diff']=full.h_form5-full.a_form5;full['dprev_diff']=full.h_d_prev-full.a_d_prev
full['dnext_diff']=full.h_d_next-full.a_d_next
tr=full[(full.src=='tr')&(full.league=='NBA')].copy();te=full[full.src=='te'].copy()
F=['elo_diff','logit_f','home_forecast','home_elo','away_elo','neutral','playoffs','season_day','home_seasongame','away_seasongame','home_rest','away_rest','rest_diff','b2b_h','b2b_a','season',
   'h_d_prev','a_d_prev','h_d_prev2','a_d_prev2','h_form5','a_form5','h_form10','a_form10','h_elo_start','a_elo_start','form5_diff','dprev_diff']
if FUTURE: F+=['h_d_next','a_d_next','dnext_diff']
P=dict(objective='l1',learning_rate=0.02,num_leaves=15,min_data_in_leaf=200,feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,verbose=-1,num_threads=4)
a=tr.season<=2009;b=tr.season>=2010
# linear l1-ish model on elo_diff with era-specific home advantage: fit on recent seasons of the fit fold
def lin(fit,app):
    f=fit[fit.season>=fit.season.max()-9]
    X=lambda d:np.c_[d.elo_diff/28+np.where(d.neutral==1,0,1)*0+0,1-d.neutral]
    from sklearn.linear_model import QuantileRegressor
    q=QuantileRegressor(quantile=0.5,alpha=0,solver='highs').fit(X(f),f[T]);return q.predict(X(app)),q
pl,q=lin(tr[a],tr[b]);print('linear CV MAE',np.abs(pl-tr[b][T]).mean(),'coef',q.coef_,q.intercept_)
pv=[];its=[]
for s in [0,1,2]:
    m=lgb.train({**P,'seed':s},lgb.Dataset(tr[a][F],tr[a][T]),5000,valid_sets=[lgb.Dataset(tr[b][F],tr[b][T])],callbacks=[lgb.early_stopping(300,verbose=False)])
    pv.append(m.predict(tr[b][F],num_iteration=m.best_iteration));its.append(m.best_iteration);print('seed',s,np.abs(pv[-1]-tr[b][T]).mean(),m.best_iteration,flush=True)
pg=np.mean(pv,0);print('lgb CV MAE',np.abs(pg-tr[b][T]).mean())
best=(9e9,1)
for w in np.arange(0,1.01,0.1):
    e=np.abs(w*pg+(1-w)*pl-tr[b][T]).mean();print(f'w_lgb={w:.1f} MAE {e:.4f}');best=min(best,(e,w))
print('v2 CV MAE (blend)',best[0],'w_lgb',best[1])
w=best[1];pe=[]
for s,it in zip([0,1,2],its):
    m=lgb.train({**P,'seed':s},lgb.Dataset(tr[F],tr[T]),max(100,int(it*1.1)));pe.append(m.predict(te[F]))
plt,_=lin(tr,te);te[T]=w*np.mean(pe,0)+(1-w)*plt
out='v2b_future_elo.csv' if FUTURE else 'v2.csv'
sub=pd.read_csv(D+'sample_submission.csv')[['id']].merge(te[['id',T]],on='id',how='left')
assert sub[T].notna().all() and len(sub)==3944
sub.to_csv(D+'submissions/'+out,index=False);print('saved',out)
