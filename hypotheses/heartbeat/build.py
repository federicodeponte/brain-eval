import os, json, numpy as np, wfdb, scipy.signal as ss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
R=os.path.expanduser("~/work/vals-hyp/_data/cinc/training2017")
OUT=os.path.expanduser("~/work/vals-hyp/heartbeat")
rng=np.random.default_rng(7)
ref=[l.strip().split(",") for l in open(R+"/REFERENCE.csv")]
CLS={"N":0,"A":1,"O":2,"~":3}; NAMES=["Normal sinus rhythm","Atrial fibrillation","Other rhythm (not normal, not AF)","Too noisy to classify"]
L="ABCD"; WIN=10; OFF=2
def load(rec,fs):
    x=wfdb.rdrecord(f"{R}/{rec}").p_signal[:,0]
    seg=x[OFF*300:(OFF+WIN)*300]
    if len(seg)<WIN*300: return None
    y=ss.resample_poly(seg,fs,300)
    return np.round(y*1000).astype(int)  # microvolts
def feats(y,fs):
    b,a=ss.butter(2,[5/(fs/2),min(20,fs/2-1)/(fs/2)],"band"); f=ss.filtfilt(b,a,y.astype(float))
    e=np.abs(f); pk,_=ss.find_peaks(e,distance=int(0.25*fs),height=np.percentile(e,90)*0.4)
    rr=np.diff(pk)/fs if len(pk)>2 else np.array([1.0,1.0])
    d=np.diff(rr) if len(rr)>1 else np.array([0.0])
    hf=np.abs(np.fft.rfft(y-y.mean())); fr=np.fft.rfftfreq(len(y),1/fs)
    return [len(pk),rr.mean(),rr.std(),rr.std()/rr.mean(),np.sqrt((d**2).mean()),(np.abs(d)>0.05).mean(),
            np.median(rr),rr.min(),rr.max(),y.std(),np.abs(y).max(),
            hf[fr>15].sum()/hf.sum(),hf[fr<1].sum()/hf.sum(),ss.welch(y,fs)[1].argmax()]
recs=[(r,l) for r,l in ref if wfdb.rdheader(f"{R}/{r}").sig_len>=(OFF+WIN)*300]
by={c:[r for r,l in recs if l==c] for c in CLS}
items={c:list(rng.choice(by[c],10,replace=False)) for c in CLS}
shots={c:list(rng.choice([r for r in by[c] if r not in items[c]],2,replace=False)) for c in CLS}
held=set(sum(items.values(),[]))|set(sum(shots.values(),[]))
def ser(v): return " ".join(map(str,v))
def build(fs,tag,fewshot):
    rows=[]; order=[(c,r) for c in CLS for r in items[c]]; rng2=np.random.default_rng(1); rng2.shuffle(order)
    head=(f"Below is a raw single-lead ECG recording (handheld AliveCor device, roughly lead I), {WIN} seconds sampled at {fs} Hz, "
          f"values in microvolts as integers, in time order.\n")
    q="Which best describes this recording?\n"+"\n".join(f"{L[i]}) {NAMES[i]}" for i in range(4))+"\nAnswer with a single letter."
    ex=""
    if fewshot:
        ex="Labeled reference recordings (same device and format):\n"
        for c in CLS:
            for r in shots[c]: ex+=f"[Example, label {L[CLS[c]]}) {NAMES[CLS[c]]}]\n{ser(load(r,fs))}\n"
        ex+="\nNow the recording to classify:\n"
    for k,(c,r) in enumerate(order):
        y=load(r,fs)
        p=head+ex+ser(y)+"\n\n"+q
        rows.append({"id":f"heartbeat-{tag}-{k:03d}","hypothesis":"heartbeat","prompt":p,"answer":L[CLS[c]],
          "choices":{L[i]:NAMES[i] for i in range(4)},"meta":{"record":r,"label":c,"fs":fs,"win_s":WIN,"offset_s":OFF,"variant":tag}})
    return rows
zs=build(100,"zs",False); fsr=build(50,"fs",True)
for fn,rows in [("tasks.jsonl",zs),("tasks_fewshot.jsonl",fsr)]:
    with open(f"{OUT}/{fn}","w") as f:
        for r in rows: f.write(json.dumps(r)+"\n")
    print(fn,len(rows),"max chars",max(len(r["prompt"]) for r in rows), "answers",{a:sum(r["answer"]==a for r in rows) for a in L})
open(f"{OUT}/examples/example_zeroshot.txt","w").write(zs[0]["prompt"]+"\n\n[ANSWER] "+zs[0]["answer"])
open(f"{OUT}/examples/example_fewshot.txt","w").write(fsr[1]["prompt"]+"\n\n[ANSWER] "+fsr[1]["answer"])
# baselines
res={}
for fs,rows in [(100,zs),(50,fsr)]:
    tr=[(r,l) for r,l in recs if r not in held]
    idx=rng.choice(len(tr),3000,replace=False)
    X=[];Y=[]
    for i in idx:
        r,l=tr[i]; y=load(r,fs)
        if y is None: continue
        X.append(feats(y,fs)); Y.append(CLS[l])
    X=np.array(X);Y=np.array(Y)
    Xq=np.array([feats(load(r["meta"]["record"],fs),fs) for r in rows]); Yq=np.array([CLS[r["meta"]["label"]] for r in rows])
    lr=make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000,class_weight="balanced")).fit(X,Y)
    rf=RandomForestClassifier(500,class_weight="balanced",random_state=0,min_samples_leaf=3).fit(X,Y)
    res[f"LR_{fs}Hz"]=(lr.predict(Xq)==Yq).mean(); res[f"RF_{fs}Hz"]=(rf.predict(Xq)==Yq).mean()
    if fs==50:
        Xs=np.array([feats(load(r,fs),fs) for c in CLS for r in shots[c]]); Ys=np.array([CLS[c] for c in CLS for r in shots[c]])
        sc=StandardScaler().fit(X)
        knn=KNeighborsClassifier(1).fit(sc.transform(Xs),Ys); res["1NN_prompt_examples_only_50Hz"]=(knn.predict(sc.transform(Xq))==Yq).mean()
        sc2=StandardScaler().fit(Xs); knn2=KNeighborsClassifier(1).fit(sc2.transform(Xs),Ys); res["1NN_prompt_only_selfscaled"]=(knn2.predict(sc2.transform(Xq))==Yq).mean()
    pr=rf.predict(Xq); print(fs,"RF per-class acc",[round((pr[Yq==k]==k).mean(),2) for k in range(4)])
print(json.dumps({k:round(float(v),3) for k,v in res.items()}))
json.dump({k:float(v) for k,v in res.items()},open(f"{OUT}/baseline.json","w"),indent=1)
