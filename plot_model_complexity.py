import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import MultipleLocator
import mpl_toolkits.axisartist as axisartist

def doule_subfigures():
    font_size = 15

    fig, (ax, ax2) = plt.subplots(1,2,sharey=True,figsize=(15, 10), gridspec_kw={'width_ratios': [2, 1]})

    # flops
    x =       [10.46, 63.05, 37.69]
    # psnr
    y =       [31.07, 33.71, 34.01]

    # params
    params =  [2.22, 67.89, 9.76]

    methods = ['ESTRNN', 'NAFNet', 'BasicVSR++']

    colors =  ['#797000',  '#8E236B',  '#238E68']

    x2 = [357.90, 760.43]
    y2 = [31.67, 32.73]
    params2 = [16.19, 20.13]
    methods2 = ['CDVD-TSP', 'MPRNet']
    colors2 = ['#264653', '#6F4242']


    for i in range(len(y)):
        area = 35 * params[i]

        if i == 1:
            ax.scatter(x[i], y[i], s=area, alpha=0.8, marker='.', c=colors[i], edgecolors='white', linewidths=2.0)
            ax.annotate(methods[i], xy=(x[i], y[i]), xytext=(x[i], y[i]), fontproperties='Times New Roman', fontsize=20)
        elif i == 5:
            ax.scatter(x[i], y[i], s=area, alpha=0.8, marker='.', c=colors[i], edgecolors='white',linewidths=2.0)
            ax.annotate(methods[i], xy=(x[i], y[i]+0.02), xytext=(x[i], y[i]+0.02), fontproperties='Times New Roman', fontsize=20)
        elif i==7:
            ax.scatter(x[i], y[i], s=area, alpha=0.8, marker='.', c=colors[i], edgecolors='white', linewidths=2.0)
            ax.annotate(methods[i], xy=(x[i], y[i]), xytext=(x[i], y[i]), fontproperties='Times New Roman', fontsize=20, weight='bold')
        else:
            ax.scatter(x[i], y[i], s=area, alpha=0.8, marker='.', c=colors[i], edgecolors='white', linewidths=2.0)
            ax.annotate(methods[i], xy=(x[i], y[i]), xytext=(x[i], y[i]), fontproperties='Times New Roman', fontsize=20)

    for i in range(len(y2)):
        area = 80 * params2[i]
        ax2.scatter(x2[i], y2[i], s=area, alpha=0.8, marker='.', c=colors2[i], edgecolors='white', linewidths=2.0)
        plt.annotate(methods2[i], xy=(x2[i], y2[i]), xytext=(x2[i], y2[i]), fontproperties='Times New Roman', fontsize=20)


    ax.grid()
    ax.set_xlim(10, 200)
    ax2.grid()
    ax2.set_xlim(400, 800)
    ax.spines['right'].set_visible(False) #关闭子图1中底部脊
    ax2.spines['left'].set_visible(False) #关闭子图2中顶部脊
    d = .55
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=15,
              linestyle='none', color='r', mec='r', mew=1, clip_on=False)
    ax.plot([1, 1], [1, 0],transform=ax.transAxes, **kwargs)
    ax2.plot([0, 0], [0, 1], transform=ax2.transAxes, **kwargs)

    ax.set_xticks([0, 50, 100, 150, 200])
    ax2.set_xticks([350, 700])
    ax.set_xticklabels([0, 50, 100, 150, 200], fontproperties='Times New Roman', size=30)
    ax2.set_xticklabels([400, 800], fontproperties='Times New Roman', size=30)

    ax.set_yticks([31, 32, 33, 34, 35])
    ax.set_yticklabels([31, 32, 33, 34, 35], fontproperties='Times New Roman', size=30)
    
    ax.set_ylabel('PSNR (dB)',fontproperties='Times New Roman',  fontsize=35)

    plt.xlabel('FLOPs (G)', fontproperties='Times New Roman', fontsize=35)

    plt.suptitle('PSNR vs. FLOPs vs. Params', fontproperties='Times New Roman', fontsize=35)
    plt.show()


def single_figure():
    font_size = 15

    fig, ax = plt.subplots(1, 1, figsize=(15, 10))

    '''0 - 10'''
    # params
    x =       [13.57, 10.51, 16.19]
    # psnr
    y =       [34.92, 31.52, 31.67]

    # flops
    flops =  [559.45, 43.04, 357.90]

    methods = ['RVRT', 'PVDNet', 'CDVD-TSP']
 
    colors =  ['#CCFF99', '#66CCCC', '#339999']

    for i in range(len(y)):
        area = 35 * flops[i]

        if i == 2 or i == 3:
            ax.scatter(x[i], y[i], s=area, alpha=0.8, marker='.', c=colors[i], edgecolors='white', linewidths=2.0)
            ax.annotate(methods[i], xy=(x[i], y[i]), xytext=(x[i], y[i]), fontproperties='Times New Roman', fontsize=20, weight='bold')
        else:
            ax.scatter(x[i], y[i], s=area, alpha=0.8, marker='.', c=colors[i], edgecolors='white', linewidths=2.0)
            ax.annotate(methods[i], xy=(x[i], y[i]), xytext=(x[i], y[i]), fontproperties='Times New Roman', fontsize=20)
    
    ax.grid()
    ax.set_xlim(3, 21)

    ax.set_xticks([3, 6, 9, 12, 15, 18, 21])
    ax.set_xticklabels([3, 6, 9, 12, 15, 18, 21], fontproperties='Times New Roman', size=30)


    ax.set_yticks([31, 32, 33, 34, 35])
    ax.set_yticklabels([31, 32, 33, 34, 35], fontproperties='Times New Roman', size=30)
    
    ax.set_ylabel('PSNR (dB)',fontproperties='Times New Roman',  fontsize=35)

    plt.xlabel('Params (M)', fontproperties='Times New Roman', fontsize=35)

    plt.suptitle('PSNR vs. Params vs. FLOPs', fontproperties='Times New Roman', fontsize=35)
    plt.show()


if __name__ == '__main__':
    single_figure()
