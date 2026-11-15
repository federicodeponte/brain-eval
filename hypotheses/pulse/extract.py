import sys, os, subprocess, numpy as np, cv2, csv
B="https://huggingface.co/datasets/kyegorov/mcd_rppg/resolve/main/"
D=os.path.expanduser("~/work/vals-hyp/_data/mcd"); os.makedirs(D+"/rgb2",exist_ok=True); os.makedirs(D+"/tmp",exist_ok=True)
casc=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
def run(pid,step):
    out=f"{D}/rgb2/{pid}_{step}.npz"
    if os.path.exists(out): return
    v=f"{D}/tmp/{pid}_{step}.avi"
    subprocess.run(["curl","-sL","-o",v,B+f"video/{pid}_FullHDwebcam_{step}.avi"],check=True)
    subprocess.run(["curl","-sL","-o",v+".ppg",B+f"ppg_sync/{pid}_FullHDwebcam_{step}.txt"],check=True)
    cap=cv2.VideoCapture(v); fps=cap.get(cv2.CAP_PROP_FPS); rgb=[]; box=None; i=0; dets=[]
    while True:
        ok,fr=cap.read()
        if not ok: break
        if box is None and i%10==0 and i<=150:
            g=cv2.cvtColor(cv2.resize(fr,None,fx=0.5,fy=0.5),cv2.COLOR_BGR2GRAY)
            f=casc.detectMultiScale(g,1.1,5,minSize=(25,25))
            if len(f): x,y,w,h=[int(a*2) for a in max(f,key=lambda r:r[2]*r[3])]; dets.append((x+int(.2*w),y+int(.1*h),int(.6*w),int(.8*h)))
            if len(dets)>=8 or (i==150 and dets): box=tuple(int(v) for v in np.median(np.array(dets),0))
        if box is None: rgb.append([np.nan]*3)
        else:
            x,y,w,h=box; r=fr[y:y+h,x:x+w].reshape(-1,3).mean(0)[::-1]; rgb.append(r)
        i+=1
    cap.release()
    ppg=np.loadtxt(v+".ppg")
    np.savez_compressed(out,rgb=np.array(rgb,dtype=np.float32),fps=fps,ppg=ppg)
    os.remove(v); os.remove(v+".ppg")
    print(pid,step,len(rgb),fps,ppg.shape,flush=True)
if __name__=="__main__":
    pid,step=sys.argv[1],sys.argv[2]
    try: run(pid,step)
    except Exception as e: print("ERR",pid,step,e,flush=True)
