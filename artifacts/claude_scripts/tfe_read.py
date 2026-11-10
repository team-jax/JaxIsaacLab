import sys, glob, struct
def recs(p):
    with open(p,'rb') as f:
        while True:
            h=f.read(12)
            if len(h)<12: return
            n=struct.unpack('<Q',h[:8])[0]; d=f.read(n); f.read(4); yield d
def varint(b,i):
    r=s=0
    while True:
        c=b[i]; i+=1; r|=(c&0x7f)<<s; s+=7
        if c<0x80: return r,i
def fields(b):
    i=0; out=[]
    while i<len(b):
        k,i=varint(b,i); t=k&7; fn=k>>3
        if t==0: v,i=varint(b,i)
        elif t==1: v=b[i:i+8]; i+=8
        elif t==5: v=b[i:i+4]; i+=4
        elif t==2: l,i=varint(b,i); v=b[i:i+l]; i+=l
        else: return out
        out.append((fn,t,v))
    return out
S={}
for p in glob.glob(sys.argv[1]+'/events*'):
    for d in recs(p):
        step=None
        for fn,t,v in fields(d):
            if fn==2 and t==0: step=v
            if fn==5 and t==2:
                for fn2,t2,v2 in fields(v):
                    if fn2==1 and t2==2:
                        tag=None; sv=None
                        for fn3,t3,v3 in fields(v2):
                            if fn3==1: tag=v3.decode()
                            if fn3==2 and t3==5: sv=struct.unpack('<f',v3)[0]
                        if tag and sv is not None: S.setdefault(tag,[]).append((step,sv))
pre=sys.argv[2] if len(sys.argv)>2 else ''
for t in sorted(S):
    if t.startswith(pre):
        v=S[t]; k=max(1,len(v)//20); last=sum(x for _,x in v[-k:])/k
        print(f"{t:45s} last {last:9.3f}")
