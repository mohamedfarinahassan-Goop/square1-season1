import pandas as pd, numpy as np, lightgbm as lgb
D='./'
tr=pd.read_csv(D+'train.csv');te=pd.read_csv(D+'test.csv');T='home_margin'
# pre-game only: elo/forecast are 538 pre-game values; no team-ID target encoding (teams/eras shift)
def fe(d):
    d=d.copy();d['elo_diff']=d.home_elo-d.away_elo;d['rest_diff']=d.home_rest.clip(0,10)-d.away_rest.clip(0,10)
    d['logit_f']=np.log(d.home_forecast/(1-d.home_forecast));d['b2b_h']=(d.home_rest<=1).astype(int);d['b2b_a']=(d.away_rest<=1).astype(int)
    return d
tr=fe(tr);te=fe(te);tr=tr[tr.league=='NBA']
F=['elo_diff','logit_f','home_forecast','home_elo','away_elo','neutral','playoffs','season_day','home_seasongame','away_seasongame','home_rest','away_rest','rest_diff','b2b_h','b2b_a','season']
P=dict(objective='l1',learning_rate=0.02,num_leaves=15,min_data_in_leaf=200,feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,verbose=-1,num_threads=4)
a=tr.season<=2009;b=tr.season>=2010
bl=np.where(tr[b].neutral==1,0,3)
print('baseline-rule CV MAE',np.abs(bl-tr[b][T]).mean())
m=lgb.train(P,lgb.Dataset(tr[a][F],tr[a][T]),5000,valid_sets=[lgb.Dataset(tr[b][F],tr[b][T])],callbacks=[lgb.early_stopping(300,verbose=False)])
print('model CV MAE',np.abs(m.predict(tr[b][F])-tr[b][T]).mean(),'iters',m.best_iteration)
m=lgb.train(P,lgb.Dataset(tr[F],tr[T]),max(100,int(m.best_iteration*1.1)))
te[T]=m.predict(te[F])
te[['id',T]].to_csv(D+'submissions/v1.csv',index=False)
