import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from math import sqrt

frame = int(raw_input('Frame number:'))
row = int(sqrt(frame/2))
colume = int(sqrt(frame/2)*2)
print('row, colume', row, colume)
mode = raw_input('Left[left] or Right[right], default as None[none]') or None

fig=plt.figure(frameon=False, figsize=(20, 10))
# fig.set_size_inches(32,16)
ax=plt.Axes(fig,[0.,0.,1.,1.])
ax.set_axis_off()
fig.add_axes(ax)

# plt.figure(figsize = (10,10))
gs1 = gridspec.GridSpec(row, colume)
gs1.update(wspace=0, hspace=0) # set the spacing between axes.

for i in range(frame):
   # i = i + 1 # grid spec indexes from 0
    if mode is not None:
        img = plt.imread('equirectangular-%s.%05d.png'%(mode, i))
    else:
        img = plt.imread('equirectangular.%05d.png'%i)
    ax1=plt.subplot(gs1[row-1-i%row, colume-1-i/row])
    print('which row and colume', row-1-i%row, colume-1-i/row)
    # ax1 = plt.imshow(img)
    plt.axis('off')
    
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_aspect('equal')
    # ax1.set_aspect('equal')
    plt.imshow(img)
# plt.show()
# .imshow(origin='lower', vmin=0, vmax=5,aspect='normal', extent=[-180,180,-90,90])
plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0) 
if mode is not None:
    fig.savefig('stereoscopic_panaroma-%s.JPG'%mode, dpi=150, transparent=True)
    print('write tereoscopic_panaroma-%s.JPG'%mode)
else:
    fig.savefig('stereoscopic_panaroma.JPG', dpi=150, transparent=True)
    print('write tereoscopic_panaroma.JPG')
