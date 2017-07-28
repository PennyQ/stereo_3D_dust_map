import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

row = 3
colume = 6
fig=plt.figure(frameon=False, figsize=(20, 10))
# fig.set_size_inches(32,16)
ax=plt.Axes(fig,[0.,0.,1.,1.])
ax.set_axis_off()
fig.add_axes(ax)

# plt.figure(figsize = (10,10))
gs1 = gridspec.GridSpec(row, colume)
gs1.update(wspace=0, hspace=0) # set the spacing between axes.

for i in range(18):
   # i = i + 1 # grid spec indexes from 0
    img = plt.imread('equirectangular.%05d.png'%i)
    ax1=plt.subplot(gs1[row-1-i%3, colume-1-i/3])
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
fig.savefig('stereoscopic_panaroma.JPG', dpi=150, transparent=True)
