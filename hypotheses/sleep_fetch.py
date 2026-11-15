import mne, os
from mne.datasets.sleep_physionet.age import fetch_data
p=os.path.expanduser("~/work/vals-hyp/_data/mne")
os.makedirs(p,exist_ok=True)
files=fetch_data(subjects=list(range(12)),recording=[1],path=p,on_missing="warn")
print(len(files))
