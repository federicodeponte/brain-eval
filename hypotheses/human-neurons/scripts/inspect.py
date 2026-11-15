import sys, h5py
sys.path.insert(0, sys.argv[3])
from common import open_remote
f=open_remote(sys.argv[1], sys.argv[2])
def show(name,obj):
    if isinstance(obj,h5py.Dataset):
        if "electrodes/" in name: return
        if obj.size>1e6: print("  BIG",name,obj.shape); return
        print(name,obj.shape,obj.dtype)
f.visititems(show)
