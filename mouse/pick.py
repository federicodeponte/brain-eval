import pandas as pd
d='data/'
u=pd.read_csv(d+'units.csv',usecols=['id','ecephys_channel_id','isi_violations','amplitude_cutoff','presence_ratio','snr','firing_rate'])
c=pd.read_csv(d+'channels.csv',usecols=['id','ecephys_probe_id','ecephys_structure_acronym'])
p=pd.read_csv(d+'probes.csv',usecols=['id','ecephys_session_id'])
s=pd.read_csv(d+'sessions.csv',usecols=['id','session_type'])
u=u.merge(c.rename(columns={'id':'ecephys_channel_id'}),on='ecephys_channel_id').merge(p.rename(columns={'id':'ecephys_probe_id'}),on='ecephys_probe_id').merge(s.rename(columns={'id':'ecephys_session_id'}),on='ecephys_session_id')
q=u[(u.isi_violations<0.5)&(u.amplitude_cutoff<0.1)&(u.presence_ratio>0.95)&(u.ecephys_structure_acronym=='VISp')&(u.session_type=='brain_observatory_1.1')]
print(q.groupby('ecephys_session_id').size().sort_values(ascending=False).head(6))
