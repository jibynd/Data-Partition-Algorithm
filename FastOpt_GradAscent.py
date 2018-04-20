def OptGradAscent(M,Pt,c):
    n = len(M); lk=[]; cumPt = np.cumsum(Pt); cumPt = np.concatenate(([0],cumPt))
    opt = np.zeros(n+1); cumM = np.cumsum(M); cumM = np.concatenate(([0],cumM)); lCp=np.zeros(n+1);
    opt[1]= obj(Pt[0],M[0],c)
    for i in np.array(range(1,n))+1:
        x = int(lCp[i-1])
        N = np.ones(i-x)*cumPt[i]-cumPt[x:i]; # sN = np.sum(Pt[x:i]); 
        M = np.ones(i-x)*cumM[i]-cumM[x:i];# sM = np.sum(M[x:i]);  #print M, N
        conv= False; j = (i-x)/2; ix = None;
        while not conv:
            v = obji(N[j],M[j],c)+obji(N[0]-N[j],M[0]-M[j],c)  # middle opt value
            #print N[j], M[j], N[0], M[0]
            try:
                vl = obji(N[j-1],M[j-1],c)+obji(N[0]-N[j-1],M[0]-M[j-1],c)  # left opt value
                vr = obji(N[j+1],M[j+1],c)+obji(N[0]-N[j+1],M[0]-M[j+1],c)  # right opt value
            except:
                #print 'err', i
                if j==0: vl = v; vr = obji(N[j+1],M[j+1],c)+obji(N[0]-N[j+1],M[0]-M[j+1],c)
                if j==(i-x-1): vr = v; vl = obji(N[j-1],M[j-1],c)+obji(N[0]-N[j-1],M[0]-M[j-1],c)
            if vl<=v and vr<=v: 
                ix = j
                conv = True
            if vl>v and vr>v:
                j+= 1*(vr>vl)-1*(vr<vl)
                #print i, obj(N,M,c)+np.concatenate(([0],obj((N[0]-N)[1:],(M[0]-M)[1:],c)))
            else: 
                j += 1*(vr>v)-1*(vl>v)
            one_chunk = obj(N[0],M[0],c)
            if one_chunk > v:
                v = one_chunk
                ix = 0
        opt[i] = v + opt[x]
        lCp[i]= x+ix;
        #if i==9: print N,M, 'gh', (M[0]-M)
        #if i+100>=n-1: print i, obj(N,M,c)+np.concatenate(([0],obj((N[0]-N)[1:],(M[0]-M)[1:],c)))# + opt[x]
    lCp = lCp[1:]+1
    return (lCp,opt[-1])
