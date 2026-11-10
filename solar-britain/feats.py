import pandas as pd, numpy as np, lightgbm as lgb, sys
D='./'
tr=pd.read_csv(D+'train.csv');te=pd.read_csv(D+'test.csv');T='generation_mw'
CITIES=[c[:-4] for c in tr if c.endswith('_ghi')]
# Approximate GB PV capacity shares by city (south/east heavy). Modelling assumption, not external target data.
W={
    'london':0.18,'southampton':0.10,'norwich':0.11,'plymouth':0.12,
    'bristol':0.09,'cardiff':0.08,'birmingham':0.11,'manchester':0.07,
    'leeds':0.06,'newcastle':0.04,'glasgow':0.02,'edinburgh':0.02,
}
def solar_pos(t,lat=52.5,lon=-1.5):
    tt=t+pd.Timedelta(minutes=15);doy=tt.dt.dayofyear.values;hr=tt.dt.hour.values+tt.dt.minute.values/60
    g=2*np.pi/365*(doy-1+(hr-12)/24)
    eqt=229.18*(0.000075+0.001868*np.cos(g)-0.032077*np.sin(g)-0.014615*np.cos(2*g)-0.040849*np.sin(2*g))
    dec=0.006918-0.399912*np.cos(g)+0.070257*np.sin(g)-0.006758*np.cos(2*g)+0.000907*np.sin(2*g)-0.002697*np.cos(3*g)+0.00148*np.sin(3*g)
    ha=np.radians((hr*60+eqt+4*lon)/4-180);la=np.radians(lat)
    cz=np.sin(la)*np.sin(dec)+np.cos(la)*np.cos(dec)*np.cos(ha);el=np.degrees(np.arcsin(np.clip(cz,-1,1)))
    az=np.degrees(np.arctan2(np.sin(ha),np.cos(ha)*np.sin(la)-np.tan(dec)*np.cos(la)))
    return el,az
def fe(d):
    d=d.copy();t=pd.to_datetime(d.datetime_utc);d['t']=t
    d['month']=t.dt.month;d['hh']=t.dt.hour*2+t.dt.minute//30;d['doy']=t.dt.dayofyear
    g=[c for c in d if c.endswith('_ghi')];cs=[c for c in d if c.endswith('_clearsky')]
    d['ghi_mean']=d[g].mean(1);d['cs_mean']=d[cs].mean(1);d['ghi_std']=d[g].std(1)
    d['kt']=d.ghi_mean/(d.cs_mean+1);d['cloud_mean']=d[[c for c in d if c.endswith('_cloud_pct')]].mean(1)
    d['temp_mean']=d[[c for c in d if c.endswith('_temp_c')]].mean(1)
    d['wind_mean']=d[[c for c in d if c.endswith('_wind_ms')]].mean(1)
    d['rh_mean']=d[[c for c in d if c.endswith('_rh_pct')]].mean(1)
    d['sun_el'],d['sun_az']=solar_pos(t);d['sin_el']=np.sin(np.radians(d.sun_el.clip(lower=0)))
    for c in CITIES: d[c+'_kt']=d[c+'_ghi']/(d[c+'_clearsky']+1)
    south=['london','bristol','southampton','plymouth','cardiff','norwich','birmingham']
    mid=['birmingham','manchester','leeds']
    north=['newcastle','glasgow','edinburgh']
    d['ghi_south']=d[[c+'_ghi' for c in south]].mean(1);d['kt_south']=d.ghi_south/(d[[c+'_clearsky' for c in south]].mean(1)+1)
    d['ghi_mid']=d[[c+'_ghi' for c in mid]].mean(1);d['kt_mid']=d.ghi_mid/(d[[c+'_clearsky' for c in mid]].mean(1)+1)
    d['ghi_north']=d[[c+'_ghi' for c in north]].mean(1);d['kt_north']=d.ghi_north/(d[[c+'_clearsky' for c in north]].mean(1)+1)
    ww=np.array([W[c] for c in CITIES]); ww=ww/ww.sum()
    d['ghi_w']=d[[c+'_ghi' for c in CITIES]].values.dot(ww)
    d['cs_w']=d[[c+'_clearsky' for c in CITIES]].values.dot(ww)
    d['kt_w']=d.ghi_w/(d.cs_w+1)
    d['sun_el_m15'],_=solar_pos(t-pd.Timedelta(minutes=30));d['sun_el_p15'],_=solar_pos(t+pd.Timedelta(minutes=30))
    d['sin_el_m']=np.sin(np.radians(np.clip(d.sun_el_m15,0,None)))
    d['sin_el_p']=np.sin(np.radians(np.clip(d.sun_el_p15,0,None)))
    # NASA POWER GHI is hourly and repeated for both half-hours; allocate by sun elevation share.
    sib=np.where(t.dt.minute.values==0,d.sin_el_p.values,d.sin_el_m.values)
    d['el_share']=d.sin_el.values/(d.sin_el.values+sib+1e-6)
    d['temp_derate']=(1-0.004*(d.temp_mean-25)).clip(0.7,1.1)
    d['phys_cf']=d.kt*d.sin_el*d.temp_derate
    d['phys_cf_w']=d.kt_w*d.sin_el*d.temp_derate
    d['phys_stc']=(d.ghi_w/1000.0)*d.temp_derate*2*d.el_share
    d['phys_stc_mean']=(d.ghi_mean/1000.0)*d.temp_derate*2*d.el_share
    d['ghi_x_cap']=d.ghi_mean*d.installed_capacity_mwp/1e4
    d['cap']=d.installed_capacity_mwp  # fleet-size / newer-panel proxy
    return d
full=pd.concat([tr,te],keys=['tr','te']).reset_index(level=0).rename(columns={'level_0':'src'}).reset_index(drop=True)
full=fe(full).sort_values('t')
for c in ['ghi_mean','kt','cloud_mean','ghi_south','kt_south','ghi_w','kt_w','phys_cf_w']:
    for k in (1,2):
        l=full[c].shift(k);f=full[c].shift(-k)
        okl=full.t.diff(k).dt.total_seconds().values==1800*k;okf=(-full.t.diff(-k)).dt.total_seconds().values==1800*k
        full[f'{c}_l{k}']=np.where(okl,l,np.nan);full[f'{c}_f{k}']=np.where(okf,f,np.nan)
    full[c+'_d']=full.t.dt.date;full[c+'_daymean']=full.groupby(c+'_d')[c].transform('mean');full=full.drop(columns=c+'_d')
full=full.sort_index()
tr=full[full.src=='tr'].copy();te=full[full.src=='te'].copy()
tr['cf']=tr[T]/tr.installed_capacity_mwp
# shifted physical feature now that lags exist
tr['phys_cf_l1']=tr.get('kt_l1',tr.kt)*tr.sin_el*tr.temp_derate
te['phys_cf_l1']=te.get('kt_l1',te.kt)*te.sin_el*te.temp_derate
tr.to_pickle(D+'tr_f.pkl');te.to_pickle(D+'te_f.pkl');print('ok',tr.shape)
