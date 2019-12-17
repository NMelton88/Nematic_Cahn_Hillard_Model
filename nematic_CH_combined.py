import matplotlib.pyplot as plt
import numpy as np
from math import sin, cos, acos, pi, sqrt
import matplotlib.pyplot as plt
import matplotlib.cm as cm

N = 40
m = 40
g = 2.0
kbT = 2.0
dt = dt2 = 0.0125
tfinal = 10.0
final_kbT = 2.0
tempjump = 0.2
D = 1.0
g2 = 0.5

Nx = Ny = m

dx = N/Nx
dy = N/Ny

def torque(x, y, nx, ny, nz):
    xplus = x+1
    xminus = x-1
    yplus = y+1
    yminus = y-1

    if (xplus == N):  xplus = 0
    if (xminus == -1):  xminus = N-1
    if (yplus == N):  yplus = 0
    if (yminus == -1):  yminus = N-1
    
    ntest = [nx[x][y], ny[x][y], nz[x][y]]
    n1 = [nx[xplus][y], ny[xplus][y], nz[xplus][y]]
    n2 = [nx[xminus][y], ny[xminus][y], nz[xminus][y]]
    n3 = [nx[x][yplus], ny[x][yplus], nz[x][yplus]]
    n4 = [nx[x][yminus], ny[x][yminus], nz[x][yminus]]
    n5 = [nx[xplus][yplus], ny[xplus][yplus], nz[xplus][yplus]]
    n6 = [nx[xplus][yminus], ny[xplus][yminus], nz[xplus][yminus]]
    n7 = [nx[xminus][yplus], ny[xminus][yplus], nz[xminus][yplus]]
    n8 = [nx[xminus][yminus], ny[xminus][yminus], nz[xminus][yminus]]
    
    dot_xplus = np.dot(ntest,n1)
    dot_xminus = np.dot(ntest,n2)
    dot_yplus = np.dot(ntest,n3)
    dot_yminus = np.dot(ntest,n4)
    dot5 = np.dot(ntest,n5)
    dot6 = np.dot(ntest,n6)
    dot7 = np.dot(ntest,n7)
    dot8 = np.dot(ntest,n8)
    
    cross_xplus = np.cross(ntest,n1)
    cross_xminus = np.cross(ntest,n2)
    cross_yplus = np.cross(ntest,n3)
    cross_yminus = np.cross(ntest,n4)
    cross5 = np.cross(ntest,n5)
    cross6 = np.cross(ntest,n6)
    cross7 = np.cross(ntest,n7)
    cross8 = np.cross(ntest,n8)

    total_torque = np.array(dot_xplus*cross_xplus) + np.array(dot_xminus*cross_xminus) + np.array(dot_yplus*cross_yplus) + np.array(dot_yminus*cross_yminus) + np.array(dot5*cross5) + np.array(dot6*cross6) + np.array(dot7*cross7) + np.array(dot8*cross8)
    
    np.random.seed(np.random.randint(0,4000000))
    theta2 = acos(2.0 * np.random.rand() - 1.0)
    phi2 = 2.0 * pi * np.random.rand()
    random_direction = [sin(theta2)*cos(phi2), sin(theta2)*sin(phi2), cos(theta2)]

    return (1.0/g)*total_torque + sqrt(2 * kbT / (g * dt)) * np.random.normal(0,1) * np.array(random_direction)
    
def localorder(x, y, nx, ny ,nz):
    xplus = x+1
    xplus2 = x+2
    xplus3 = x+3
    xminus = x-1
    xminus2 = x-2
    xminus3 = x-3
    yplus = y+1
    yplus2 = y+2
    yplus3 = y+3
    yminus = y-1
    yminus2 = y-2
    yminus3 = y-3

    if (xplus == N):  xplus = 0
    if (xplus2 == N):  xplus2 = 0
    if (xplus2 == N+1):  xplus2 = 1
    if (xplus3 == N):  xplus3 = 0
    if (xplus3 == N+1):  xplus3 = 1
    if (xplus3 == N+2):  xplus3 = 2
    if (xminus == -1):  xminus = N-1
    if (xminus2 == -1):  xminus2 = N-1
    if (xminus2 == -2):  xminus2 = N-2
    if (xminus3 == -1):  xminus3 = N-1
    if (xminus3 == -2):  xminus3 = N-2
    if (xminus3 == -3):  xminus3 = N-3
    if (yplus == N):  yplus = 0
    if (yplus2 == N):  yplus2 = 0
    if (yplus2 == N+1):  yplus2 = 1
    if (yplus3 == N):  yplus3 = 0
    if (yplus3 == N+1):  yplus3 = 1
    if (yplus3 == N+2):  yplus3 = 2
    if (yminus == -1):  yminus = N-1
    if (yminus2 == -1):  yminus2 = N-1
    if (yminus2 == -2):  yminus2 == N-2
    if (yminus3 == -1):  yminus3 = N-1
    if (yminus3 == -2):  yminus3 = N-2
    if (yminus3 == -3):  yminus3 = N-3
    
    Qxx = (nx[x][y]*nx[x][y] + nx[xplus][y]*nx[xplus][y] + nx[xplus2][y]*nx[xplus2][y] + nx[xplus3][y]*nx[xplus3][y] + nx[xminus][y]*nx[xminus][y] + nx[xminus2][y]*nx[xminus2][y] + nx[xminus3][y]*nx[xminus3][y] + nx[x][yplus]*nx[x][yplus] + nx[x][yplus2]*nx[x][yplus2] + nx[x][yplus3]*nx[x][yplus3]+ nx[x][yminus]*nx[x][yminus] + nx[x][yminus2]*nx[x][yminus2] + nx[x][yminus3]*nx[x][yminus3] + nx[xplus][yplus]*nx[xplus][yplus] + nx[xplus][yplus2]*nx[xplus][yplus2] + nx[xplus2][yplus]*nx[xplus2][yplus] + nx[xplus][yminus]*nx[xplus][yminus] + nx[xplus][yminus2]*nx[xplus][yminus2]+ nx[xplus2][yminus]*nx[xplus2][yminus] + nx[xminus][yplus]*nx[xminus][yplus] + nx[xminus][yplus2]*nx[xminus][yplus2] + nx[xminus][yminus]*nx[xminus][yminus] + nx[xminus][yminus2]*nx[xminus][yminus2] + nx[xminus2][yplus]*nx[xminus2][yplus] + nx[xminus2][yminus]*nx[xminus2][yminus] - 25.0*0.33333)/25.0 
    Qyy = (ny[x][y]*ny[x][y] + ny[xplus][y]*ny[xplus][y] + ny[xplus2][y]*ny[xplus2][y] + ny[xplus3][y]*ny[xplus3][y] + ny[xminus][y]*ny[xminus][y] + ny[xminus2][y]*ny[xminus2][y] + ny[xminus3][y]*ny[xminus3][y] + ny[x][yplus]*ny[x][yplus] + ny[x][yplus2]*ny[x][yplus2] + ny[x][yplus3]*ny[x][yplus3]+ ny[x][yminus]*ny[x][yminus] + ny[x][yminus2]*ny[x][yminus2] + ny[x][yminus3]*ny[x][yminus3] + ny[xplus][yplus]*ny[xplus][yplus] + ny[xplus][yplus2]*ny[xplus][yplus2] + ny[xplus2][yplus]*ny[xplus2][yplus] + ny[xplus][yminus]*ny[xplus][yminus] + ny[xplus][yminus2]*ny[xplus][yminus2]+ ny[xplus2][yminus]*ny[xplus2][yminus] + ny[xminus][yplus]*ny[xminus][yplus] + ny[xminus][yplus2]*ny[xminus][yplus2] + ny[xminus][yminus]*ny[xminus][yminus] + ny[xminus][yminus2]*ny[xminus][yminus2] + ny[xminus2][yplus]*ny[xminus2][yplus] + ny[xminus2][yminus]*ny[xminus2][yminus] - 25.0*0.33333)/25.0 
    Qzz = (nz[x][y]*nz[x][y] + nz[xplus][y]*nz[xplus][y] + nz[xplus2][y]*nz[xplus2][y] + nz[xplus3][y]*nz[xplus3][y] + nz[xminus][y]*nz[xminus][y] + nz[xminus2][y]*nz[xminus2][y] + nz[xminus3][y]*nz[xminus3][y] + nz[x][yplus]*nz[x][yplus] + nz[x][yplus2]*nz[x][yplus2] + nz[x][yplus3]*nz[x][yplus3]+ nz[x][yminus]*nz[x][yminus] + nz[x][yminus2]*nz[x][yminus2] + nz[x][yminus3]*nz[x][yminus3] + nz[xplus][yplus]*nz[xplus][yplus] + nz[xplus][yplus2]*nz[xplus][yplus2] + nz[xplus2][yplus]*nz[xplus2][yplus] + nz[xplus][yminus]*nz[xplus][yminus] + nz[xplus][yminus2]*nz[xplus][yminus2]+ nz[xplus2][yminus]*nz[xplus2][yminus] + nz[xminus][yplus]*nz[xminus][yplus] + nz[xminus][yplus2]*nz[xminus][yplus2] + nz[xminus][yminus]*nz[xminus][yminus] + nz[xminus][yminus2]*nz[xminus][yminus2] + nz[xminus2][yplus]*nz[xminus2][yplus] + nz[xminus2][yminus]*nz[xminus2][yminus] - 25.0*0.33333)/25.0 
    Qxy = (nx[x][y]*ny[x][y] + nx[xplus][y]*ny[xplus][y] + nx[xplus2][y]*ny[xplus2][y] + nx[xplus3][y]*ny[xplus3][y] + nx[xminus][y]*ny[xminus][y] + nx[xminus2][y]*ny[xminus2][y] + nx[xminus3][y]*ny[xminus3][y] + nx[x][yplus]*ny[x][yplus] + nx[x][yplus2]*ny[x][yplus2] + nx[x][yplus3]*ny[x][yplus3]+ nx[x][yminus]*ny[x][yminus] + nx[x][yminus2]*ny[x][yminus2] + nx[x][yminus3]*ny[x][yminus3] + nx[xplus][yplus]*ny[xplus][yplus] + nx[xplus][yplus2]*ny[xplus][yplus2] + nx[xplus2][yplus]*ny[xplus2][yplus] + nx[xplus][yminus]*ny[xplus][yminus] + nx[xplus][yminus2]*ny[xplus][yminus2]+ nx[xplus2][yminus]*ny[xplus2][yminus] + nx[xminus][yplus]*ny[xminus][yplus] + nx[xminus][yplus2]*ny[xminus][yplus2] + nx[xminus][yminus]*ny[xminus][yminus] + nx[xminus][yminus2]*ny[xminus][yminus2] + nx[xminus2][yplus]*ny[xminus2][yplus] + nx[xminus2][yminus]*ny[xminus2][yminus])/25.0 
    Qxz = (nx[x][y]*nz[x][y] + nx[xplus][y]*nz[xplus][y] + nx[xplus2][y]*nz[xplus2][y] + nx[xplus3][y]*nz[xplus3][y] + nx[xminus][y]*nz[xminus][y] + nx[xminus2][y]*nz[xminus2][y] + nx[xminus3][y]*nz[xminus3][y] + nx[x][yplus]*nz[x][yplus] + nx[x][yplus2]*nz[x][yplus2] + nx[x][yplus3]*nz[x][yplus3]+ nx[x][yminus]*nz[x][yminus] + nx[x][yminus2]*nz[x][yminus2] + nx[x][yminus3]*nz[x][yminus3] + nx[xplus][yplus]*nz[xplus][yplus] + nx[xplus][yplus2]*nz[xplus][yplus2] + nx[xplus2][yplus]*nz[xplus2][yplus] + nx[xplus][yminus]*nz[xplus][yminus] + nx[xplus][yminus2]*nz[xplus][yminus2]+ nx[xplus2][yminus]*nz[xplus2][yminus] + nx[xminus][yplus]*nz[xminus][yplus] + nx[xminus][yplus2]*nz[xminus][yplus2] + nx[xminus][yminus]*nz[xminus][yminus] + nx[xminus][yminus2]*nz[xminus][yminus2] + nx[xminus2][yplus]*nz[xminus2][yplus] + nx[xminus2][yminus]*nz[xminus2][yminus])/25.0 
    Qyz = (ny[x][y]*nz[x][y] + ny[xplus][y]*nz[xplus][y] + ny[xplus2][y]*nz[xplus2][y] + ny[xplus3][y]*nz[xplus3][y] + ny[xminus][y]*nz[xminus][y] + ny[xminus2][y]*nz[xminus2][y] + ny[xminus3][y]*nz[xminus3][y] + ny[x][yplus]*nz[x][yplus] + ny[x][yplus2]*nz[x][yplus2] + ny[x][yplus3]*nz[x][yplus3]+ ny[x][yminus]*nz[x][yminus] + ny[x][yminus2]*nz[x][yminus2] + ny[x][yminus3]*nz[x][yminus3] + ny[xplus][yplus]*nz[xplus][yplus] + ny[xplus][yplus2]*nz[xplus][yplus2] + ny[xplus2][yplus]*nz[xplus2][yplus] + ny[xplus][yminus]*nz[xplus][yminus] + ny[xplus][yminus2]*nz[xplus][yminus2]+ ny[xplus2][yminus]*nz[xplus2][yminus] + ny[xminus][yplus]*nz[xminus][yplus] + ny[xminus][yplus2]*nz[xminus][yplus2] + ny[xminus][yminus]*nz[xminus][yminus] + ny[xminus][yminus2]*nz[xminus][yminus2] + ny[xminus2][yplus]*nz[xminus2][yplus] + ny[xminus2][yminus]*nz[xminus2][yminus])/25.0 

    Q = [[Qxx, Qxy, Qxz], [Qxy, Qyy, Qyz], [Qxz, Qyz, Qzz]]
    
    eigenvalues = np.linalg.eigvals(Q)
    return 1.5 * np.max(eigenvalues.real)

def Laplacian(concentration, dx, dy):
    for x in range(0,N):
        for y in range(0,N):
            xplus = x+1
            if (xplus == N):   xplus = 0
            xminus = x-1
            if (xminus == -1):   xminus = N-1
            yplus = y+1
            if (yplus == N):   yplus = 0
            yminus = y-1
            if (yminus == -1):   yminus = N-1
            dc[x][y] = (concentration[xplus][y] + concentration[xminus][y] - 2*concentration[x][y])/dx/dx + (concentration[x][yplus] + concentration[x][yminus] - 2*concentration[x][y])/dy/dy
    
    return dc

def Gradient(concentration, dx, dy):
    for x in range(0,N):
        for y in range(0,N):
            xplus = x+1
            if (xplus == N):   xplus = 0
            xminus = x-1
            if (xminus == -1):   xminus = N-1
            yplus = y+1
            if (yplus == N):   yplus = 0
            yminus = y-1
            if (yminus == -1):   yminus = N-1
            dS[x][y] = (concentration[xplus][y] - concentration[xminus][y])/2.0/dx + (concentration[x][yplus] - concentration[x][yminus])/2.0/dy
    
    return dS


nx = [[0.0 for i in range(N)] for j in range(N)]
ny = [[0.0 for i in range(N)] for j in range(N)]
nz = [[0.0 for i in range(N)] for j in range(N)]
omega = [[0.0 for i in range(N)] for j in range(N)]
ordermap = [[0.0 for i in range(N)] for j in range(N)]
X, Y = np.meshgrid(np.arange(0,N,1), np.arange(0,N,1))
c = [[np.random.normal(-0.75,0.1) for i in range(N)] for j in range(N)]
cstep = [[0 for i in range(N)] for j in range(N)]
dc = [[0 for i in range(N)] for j in range(N)]
dS = [[0 for i in range(N)] for j in range(N)]

for x in range(0,N):
    for y in range(0,N):
            np.random.seed(np.random.randint(0,4000000))
            theta = acos(2.0 * np.random.rand() - 1.0)
            phi = 2.0 * pi * np.random.rand()
            nx[x][y] = sin(theta)*cos(phi)
            ny[x][y] = sin(theta)*sin(phi)
            nz[x][y] = cos(theta)

while (kbT >= final_kbT): 

    t = t2 = 0
    while (t <= tfinal):
        print (t, kbT)
        for i in range(0,N):
            for j in range(0,N):
                omega[i][j] = torque(i, j, nx, ny, nz)

        for i in range(0,N):
            for j in range(0,N):
                    n = [0,0,0]
                    n = [nx[i][j], ny[i][j], nz[i][j]]
                    n = np.array(n) + np.array(np.cross(omega[i][j],n)*dt)
                    nx[i][j] = n[0]/sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2])
                    ny[i][j] = n[1]/sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2])
                    nz[i][j] = n[2]/sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2])
                    
        #t += dt

        for a in range(0,N):
                for b in range(0,N):
                        ordermap[a][b] = localorder(a, b, nx, ny, nz)

         #while (t <= tfinal):
        mu = ((np.power(c,3) - c)*np.array(ordermap) - g2*np.array(Laplacian(c, dx, dy)))
        cstep = Laplacian(mu, dx, dy)
        c = c + np.array(cstep)*dt*D

        t += dt
   
    kbT -= tempjump

#plt.figure(1)
#plt.quiver(X, Y, nx, ny,headaxislength=0)
#plt.xlim([-1,N+1])
#plt.ylim([-1,N+1])
#plt.show()
plt.figure(2)
plt.imshow(ordermap, origin="lower")
cbar = plt.colorbar()
plt.show() 
plt.figure(3)
plt.ion()
plt.imshow(c, origin="lower", animated = True, cmap=cm.seismic)
plt.colorbar()
plt.show() 