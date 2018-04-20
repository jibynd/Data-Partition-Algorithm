import matplotlib.pyplot as plt
import numpy as np
from timeit import default_timer as timer

def obj(n,m,c): # n is the number of data points, and m the density of a data cell. c is a penalty constant
    return n*np.log(n/m)-c
def obji(n,m,c):
    if n==0 or m==0: return 0
    return n*np.log(n/m)-c

## M is an array of the density of data cells, Pt is the number of points in each of them, c is a penalty constant

# optint is the regular algorithm previously developed
def optint(M,Pt,c):
    n = len(M); cumPt = np.cumsum(Pt); cumPt = np.concatenate(([0],cumPt))
    opt = np.zeros(n+1); cumM = np.cumsum(M); cumM = np.concatenate(([0],cumM)); lCp=np.zeros(n+1);
    # cumPt and cumM are the cumulative sums of Pt and M respectively
    # lCp contains the change points
    for i in np.array(range(n))+1:
        x = int(lCp[i-1]); # x is the last change point
        N = np.ones(i-x)*cumPt[i]-cumPt[x:i] # N is the inverted cumulative sum from the last change point to the ith point of Pt
        Mm = np.ones(i-x)*cumM[i]-cumM[x:i]; # Mm is the inverted cumulative sum from the last change point to the ith point of M
        val = obj(N,Mm,c)+opt[x:i] ;
        #print i, list(val-opt[x])
        opt[i] = max(val); lCp[i]= np.argmax(val)+x
    lCp = lCp[1:]+1
    return (lCp,opt[-1])


# fastopt behaves like optint but follows the key assumption that change points are likely to remain unchanged
def fastopt(M,Pt,c):
    n = len(M); cumPt = np.concatenate(([0],np.cumsum(Pt)))
    opt = np.zeros(n+1); cumM = np.concatenate(([0],np.cumsum(M))); lCp=np.zeros(n+1);
    opt[1]= obj(Pt[0],M[0],c)
    for i in np.array(range(1,n))+1:
        x = int(lCp[i-1])
        N = np.ones(2)*cumPt[i]-cumPt[x:x+2]; # Same N above but only first two values
        Mm = np.ones(2)*cumM[i]-cumM[x:x+2];  # Same Mm above but only first two values
        v = obji(N[0],Mm[0],c); vr = obji(N[1],Mm[1],c)+obji(N[0]-N[1],Mm[0]-Mm[1],c)
        if v>=vr: lCp[i] = x # Unchanged last change point
        else:
            N = np.ones(i-x)*cumPt[i]-cumPt[x:i]
            Mm = np.ones(i-x)*cumM[i]-cumM[x:i];
            lav = vr
            conv= False; j = 1; ix = None; # ix is the amount by which the last change point increases
            while not conv:
                if j==len(N)-1: ix = j;v = lav; conv = True; break
                vr = obji(N[j+1],Mm[j+1],c)+obji(N[0]-N[j+1],Mm[0]-Mm[j+1],c)
                if vr<lav: ix = j;v = lav; conv = True; break
                lav = vr; j+=1
            lCp[i]= x+ix;
        opt[i] = v + opt[x]
    lCp = lCp[1:]+1
    return (lCp,opt[-1])


def cout(M,lCp,c):  # Calculate the value of the given partition
    n = len(lCp); part = []; k = n
    while k > 0:
        k = int(k)
        k = lCp[k-1]; part.append(int(k)); k -= 1
    part = np.array(part[::-1])
    if part[-1]!=n: part = np.concatenate((part,[n]))
    part = np.concatenate(([0],part))
    cout = 0
    for i in range(len(part)-1):
        cout += obji((part[i+1]-part[i])/1.0,np.sum(M[part[i]:part[i+1]]),c)
    return cout


# jointopt consist of splitting the whole dataset into overlapping blocks having the same size. It takes an additional parameter a that specifies the size of the overlap 
def jointopt(M,Pt,c,a): # Joint partition algorithm that uses 'cout' and 'optint'
    n = len(M); lCp=np.zeros(n); a = int(1000*a); T = 1000; s = T-a # T is the size of each of the overlapping blocks
    t = int((n-a)/s); lCp1 = optint(M[0:T],Pt[0:T],c)[0] # t is the number of overlapping blocks
    for i in np.array(range(t-1))+1:
        lCp2 = optint(M[s*i:s*i+T],Pt[s*i:s*i+T],c)[0]; LL = lCp2 # s*i to s*i+T specifies the successive overlapping blocks
        cost1 = cout(M[s*(i-1):s*(i-1)+T],lCp1,c)
        cost2 = cout(M[s*(i):s*(i)+T],lCp2,c)
        lCp1 = lCp1+s*(i-1); lCp2 = lCp2+ s*i
        if cost1 >= cost2:
            lCp[s*(i-1):s*(i-1)+s+T] = np.concatenate((lCp1,lCp2[a:T]))
        else:
            lCp[s*(i-1):s*(i-1)+s+T] = np.concatenate((lCp1[0:s],lCp2))
        lCp1 =LL
    #lCp = lCp[1:]+1
    return (lCp,cout(M,lCp,c))
