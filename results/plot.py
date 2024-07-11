import numpy as np
import matplotlib.pyplot as plt
import matplotlib  
from mpl_toolkits import mplot3d
import matplotlib.ticker

# matplotlib.rcParams.update({'font.size': 14}) #Default Font Size
# matplotlib.rcParams['text.usetex'] = True

# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')

def sin(a):
    return np.sin(a)

def cos(a):
    return np.cos(a)

def get_data(fname,col,nx,ny):
    q_data = np.genfromtxt(fname, skip_header=3,delimiter=",")
    x1 = q_data[:,0]
    y1 = q_data[:,1]
    q1 = q_data[:,col-1]
    xq = np.reshape(x1,(nx,ny),order='F') # order F for fortran is important for proper reshape
    yq = np.reshape(y1,(nx,ny),order='F')
    data  = np.reshape(q1,(nx,ny),order='F')
    return xq,yq,data

def plot_contour(xq,yq,data,title,fname):
    plt.figure(figsize=(8,6.5))
    plt.contourf(xq,yq,data)
    plt.colorbar()
    # plt.title(title)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.axis('equal')
    plt.savefig(fname)
    
    # ax = plt.axes(projection='3d')
    # ax.contour3D(xq,yq,data,100)
    # ax.set_xlabel(r'$x$')
    # ax.set_ylabel(r'$y$')
    # ax.set_zlabel(r'$u$')
    # ax.set_title(title)
    # plt.show()  # show the plot     
    return

def plot_data(a,b,title,fname,label):
    plt.plot(a,b,lw=2,label=label)
    plt.title(title)
    plt.legend()
    # plt.ylabel(r'Residual')
    # plt.xlabel(r'Iterations')
    plt.ylabel(r'$G_k$')
    plt.xlabel(r'$\beta_k$') 
    # plt.locator_params(axis="x", nbins=12)   
    # plt.yscale('log')
    # matplotlib.yaxis.set_major_locator(y_major)
    # y_minor = matplotlib.ticker.LogLocator(base = 10.0, subs = numpy.arange(1.0, 10.0) * 0.1, numticks = 10)
    # matplotlib.yaxis.set_minor_locator(y_minor)
    # matplotlib.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    return

xq,yq,data = get_data('p.dat',3,182,130)
plot_contour(xq,yq,data,r'$u$ ($256\times 256$ Grid - Gauss-Seidel Method)  ','t.png')