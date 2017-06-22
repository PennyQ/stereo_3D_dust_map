from __future__ import print_function

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import astropy.units as units
from astropy.coordinates import SkyCoord

from dustmaps.sfd import SFDQuery
from dustmaps.planck import PlanckQuery
from dustmaps.bayestar import BayestarQuery


def SaveFigureAsImage(fileName,fig=None,**kwargs):
    ''' Save a Matplotlib figure as an image without borders or frames.
       Args:
            fileName (str): String that ends in .png etc.

            fig (Matplotlib figure instance): figure you want to save as the image
        Keyword Args:
            orig_size (tuple): width, height of the original image used to maintain 
            aspect ratio.
    '''
    fig_size = fig.get_size_inches()
    w,h = fig_size[0], fig_size[1]
    fig.patch.set_alpha(0)
    if kwargs.has_key('orig_size'): # Aspect ratio scaling if required
        w,h = kwargs['orig_size']
        w2,h2 = fig_size[0],fig_size[1]
        fig.set_size_inches([(w2/w)*w,(w2/w)*h])
        fig.set_dpi((w2/w)*fig.get_dpi())
    a=fig.gca()
    a.set_frame_on(False)
    a.set_xticks([]); a.set_yticks([])
    plt.axis('off')
    plt.xlim(0,h); plt.ylim(w,0)
    fig.savefig(fileName, transparent=True, bbox_inches='tight', \
                        pad_inches=0)
                        
fig=plt.figure(frameon=False)
fig.set_size_inches(32,16)
ax=plt.Axes(fig,[0.,0.,1.,1.])
ax.set_axis_off()
fig.add_axes(ax)

l0, b0 = (0., 0.)
l = -1*np.arange(l0 - 180., l0 + 180, 0.1)
b = np.arange(b0 - 90., b0 + 90, 0.1)
l, b = np.meshgrid(l, b)
coords = SkyCoord(l*units.deg, b*units.deg,
                  distance=1.*units.kpc, frame='galactic')

sfd = SFDQuery()
Av_sfd = 2.742 * sfd(coords)

print('av_sfd', Av_sfd.shape)
# to get rid of axis and margins
#fig = plt.figure(figsize=(34, 17))

ax.imshow(Av_sfd,origin='lower', vmin=0, vmax=5,aspect='normal')#extent=[-180,180,-90,90])



#plt.axis('off')
#ax.axes.get_xaxis().set_visible(False)
#ax.axes.get_yaxis().set_visible(False)

# plt.savefig('sfd-dust-map-test.JPG', dpi=150, transparent=True, bbox_inches='tight', pad_inches=0)
fig.savefig('2D_dust_map.JPG', dpi=150, transparent=True)
#SaveFigureAsImage('sfd-dust-map-test.JPG', plt.gcf())