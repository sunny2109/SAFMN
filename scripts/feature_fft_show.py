import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

def feature_show(x,name):
    flag = 0
    x = x.squeeze().cpu().numpy()
    x = np.mean(x,axis=0)
  #  x = (x-np.min(x))/(np.max(x)-np.min(x))
    
    f = np.fft.ifft2(x)
    fshift = np.fft.fftshift(f)
    x = np.log(np.abs(fshift)+6e-6)
 
    plt.figure()
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])

    if flag == 1 :
      vnorm = mpl.colors.Normalize(vmin=-13,vmax=-3)
      plt.imshow(x, cmap='jet',norm=vnorm)
    else:
      plt.imshow(x , cmap='jet')
    plt.colorbar()
    plt.savefig(os.path.join("plt",name),dpi=200)
    plt.close()
    
