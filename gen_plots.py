#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  gen_plots.py
#  
#  Copyright 2014 greg <greg@greg-UX301LAA>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  

import numpy as np

import matplotlib
matplotlib.use('Agg')
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patheffects as PathEffects
from mpl_toolkits.axes_grid1 import ImageGrid, make_axes_locatable

import healpy as hp
import h5py
import glob
import os

import hputils, maptools, model


# Locate pan1
pan1 = '/nfs_pan1/www/ggreen/'
if not os.path.isdir(pan1):
    pan1 = '/n/pan1/www/ggreen/'


def dust_priors():
    img_fname = pan1 + 'paper-plots/dust_priors.svg'
    dpi = 400
    figsize = (9, 4)
    img_shape = (4000, 2000)
    
    nside = 16
    n_regions = 30
    sigma_bin = 1.4
    
    gal_model = model.TGalacticModel()
    
    n_pix = hp.pixelfunc.nside2npix(nside)
    pix_idx = np.arange(n_pix)
    l, b = hputils.pix2lb(nside, pix_idx)
    
    log_Delta_EBV = np.empty((n_pix, n_regions+1), dtype='f8')
    
    # Determine log(Delta EBV) in each pixel and bin
    for i,ll,bb in zip(pix_idx, l, b):
        tmp = gal_model.EBV_prior(ll, bb, n_regions=n_regions, sigma_bin=1.4)
        log_Delta_EBV[i,:] = tmp[1][:]
    
    scatter = sigma_bin * np.random.normal(size=log_Delta_EBV.shape)
    EBV_rand = np.sum(np.exp(log_Delta_EBV + scatter), axis=1)
    EBV_smooth = np.sum(np.exp(log_Delta_EBV), axis=1) * np.exp(0.5*sigma_bin**2.)
    
    # Load SFD dust map
    EBV_SFD = model.get_SFD_map(nside=nside)
    
    # Rasterize maps
    proj = hputils.Hammer_projection()
    nside_arr = nside * np.ones(n_pix, dtype='i8')
    rasterizer = hputils.MapRasterizer(nside_arr, pix_idx, img_shape,
                                       proj=proj)
    bounds = rasterizer.get_lb_bounds()
    
    log_SFD = np.log10(EBV_SFD)
    log_rand = np.log10(EBV_rand)
    log_smooth = np.log10(EBV_smooth)
    
    log_SFD_diff = log_SFD - log_smooth
    log_rand_diff = log_rand - log_smooth
    
    print np.percentile(log_SFD_diff, [5., 10., 50., 90., 95.])
    
    vmin_top = min([np.min(log_SFD), np.min(log_rand)])
    vmax_top = max([np.max(log_SFD), np.max(log_rand)])
    
    vmax_bot = max([np.max(np.abs(log_SFD_diff)), np.max(np.abs(log_rand_diff))])
    vmin_bot = -vmax_bot
    
    img_SFD = rasterizer(log_SFD)
    img_rand = rasterizer(log_rand)
    img_SFD_diff = rasterizer(log_SFD_diff)
    img_rand_diff = rasterizer(log_rand_diff)
    
    # Rasterize regions to shade on maps
    shade_fn = lambda l, b: hputils.lb_in_bounds(l, b, [-60., 60., -5., 5.]).astype('f8')
    img_shade = rasterizer.rasterize_func(shade_fn)
    
    nside_hidpi = 256
    n_pix_hidpi = hp.pixelfunc.nside2npix(nside_hidpi)
    pix_idx_hidpi = np.arange(n_pix_hidpi)
    nside_arr_hidpi = nside_hidpi * np.ones(n_pix_hidpi, dtype='i8')
    rasterizer_hidpi = hputils.MapRasterizer(nside_arr_hidpi, pix_idx_hidpi,
                                             img_shape, proj=proj)
    bounds_hidpi = rasterizer.get_lb_bounds()
    
    l, b = hputils.pix2lb(nside_hidpi, pix_idx_hidpi)
    clipped = np.empty(n_pix_hidpi, dtype='f8')
    
    # Determine log(Delta EBV) in each pixel and bin
    for i,(ll,bb) in enumerate(zip(l, b)):
        tmp = gal_model.EBV_prior(ll, bb, n_regions=n_regions, sigma_bin=1.4)
        clipped[i] = np.max(tmp[1][:]) >= -4. - 1.e-5
    
    #clipped = (np.max(log_Delta_EBV, axis=1) >= -4. - 1.e-5).astype('f8')
    img_clipped = rasterizer_hidpi(clipped)
    
    # Plot in 2x2 grid
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = [fig.add_subplot(2,2,i+1) for i in xrange(4)]
    
    
    im_11 = ax[0].imshow(img_SFD.T, extent=bounds, aspect='auto',
                            origin='lower', interpolation='nearest',
                            vmin=vmin_top, vmax=vmax_top,
                            rasterized=True)
    
    im_12 = ax[1].imshow(img_rand.T, extent=bounds, aspect='auto',
                            origin='lower', interpolation='nearest',
                            vmin=vmin_top, vmax=vmax_top,
                            rasterized=True)
    
    im_21 = ax[2].imshow(img_SFD_diff.T, extent=bounds, aspect='auto',
                            origin='lower', interpolation='nearest',
                            vmin=vmin_bot, vmax=vmax_bot,
                            cmap='bwr',
                            rasterized=True)
    
    im_22 = ax[3].imshow(img_rand_diff.T, extent=bounds, aspect='auto',
                            origin='lower', interpolation='nearest',
                            vmin=vmin_bot, vmax=vmax_bot,
                            cmap='bwr',
                            rasterized=True)
    
    for i in xrange(4):
        #ax[i].imshow(img_shade.T, extent=bounds, aspect='auto',
        #                          origin='lower', interpolation='nearest',
        #                          vmin=0., vmax=1., cmap='binary',
        #                          alpha=0.25)
        ax[i].imshow(img_clipped.T, extent=bounds_hidpi, aspect='auto',
                                  origin='lower', interpolation='nearest',
                                  vmin=0., vmax=1., cmap='binary',
                                  alpha=0.25, rasterized=True)
    
    fig.subplots_adjust(left=0.02, right=0.82,
                        bottom=0.02, top=0.92,
                        hspace=0.02, wspace=0.02)
    
    # Remove axis frames
    for a in ax:
        a.axis('off')
    
    # Color bars
    x0,y0,w,h = ax[1].get_position().bounds
    cax_top = fig.add_axes([x0 + w + 0.01, y0, 0.03, h])
    
    x0,y0,w,h = ax[3].get_position().bounds
    cax_bot = fig.add_axes([x0 + w + 0.01, y0, 0.03, h])
    
    cbar_top = fig.colorbar(im_12, cax=cax_top)
    cbar_bot = fig.colorbar(im_22, cax=cax_bot)
    
    # Labels
    x0,y0,w,h = ax[0].get_position().bounds
    fig.text(x0+0.5*w, y0+h+0.01, r'$\mathrm{SFD}$',
            fontsize=16, ha='center', va='bottom')
    
    x0,y0,w,h = ax[1].get_position().bounds
    fig.text(x0+0.5*w, y0+h+0.01, r'$\mathrm{Priors}$',
            fontsize=16, ha='center', va='bottom')
    
    cbar_top.set_label(r'$\log_{10} \mathrm{E} \left( B - V \right)$',
                       fontsize=14)
    cbar_bot.set_label(r'$\log_{10} \left[ \frac{ \mathrm{E} \left( B - V \right) }{ \left< \mathrm{E} \left( B - V \right) \right> } \right]$',
                       fontsize=14)
    
    cbar_top.locator = ticker.MaxNLocator(nbins=6)
    cbar_top.update_ticks()
    
    cbar_bot.locator = ticker.MaxNLocator(nbins=6)
    cbar_bot.update_ticks()
    
    cbar_bot.solids.set_edgecolor('face') # Fix for matplotlib svg-rendering bug
    cbar_top.solids.set_edgecolor('face') # Fix for matplotlib svg-rendering bug
    
    fig.savefig(img_fname, dpi=dpi, bbox_inches='tight')
    plt.show()


def map_slices():
    img_fname = pan1 + 'paper-plots/dust_slices'
    
    figsize = (8.5, 9.)
    img_shape = (8000, 4000)
    dpi = 500
    gamma = 1.5
    
    dists = [316.23, 794.33, 1995.26, 7943.28]
    l_0, b_0 = 130., 0.
    
    # Load in data cube
    fnames = ['/n/fink1/ggreen/bayestar/output/allsky_2MASS/compact_10samp.h5']
    mapper = maptools.LOSMapper(fnames, load_stacked_pdfs=False)
    
    # Generate the rasterizer
    print 'Generating rasterizer ...'
    proj = hputils.Hammer_projection()
    rasterizer = mapper.gen_rasterizer(img_shape, proj=proj,
                                       l_cent=l_0, b_cent=b_0)
    bounds = rasterizer.get_lb_bounds()
    
    fig_list = [plt.figure(figsize=figsize, dpi=dpi) for k in xrange(2)]
    
    # Project meridians and parallels
    l_lines = [0., 90., 180., 270.]
    b_lines = [-60., -30., 0., 30., 60.]
    
    l_lines_minor = [30., 60., 120., 150., 210., 240., 300., 330.]
    b_lines_minor = [-75., -45., -15., 15., 45., 75.]
    
    medblue = '#57808a'
    lightblue = '#bdd6db'
    stroke = [PathEffects.withStroke(linewidth=0.5, foreground=lightblue)]
    
    # Plot each distance slice
    mu = 5. * (np.log10(dists) - 1.)
    im = None
    #img = []
    
    for k,(m,d) in enumerate(zip(mu,dists)):
        print 'Plotting map to %d pc ...' % d
        
        txt = None
        ds = None
        
        if k == 0:
            txt = r'$< %d \, \mathrm{pc}$' % d
        else:
            txt = r'$%d \! - \! %d \, \mathrm{pc}$' % (dists[k-1], d)
            d0 = 0.5 * (d + dists[k-1])
            m = 5. * np.log10(d0/10.)
            ds = -(d - dists[k-1])/1000.
        
        # Rasterize image
        tmp, tmp, pix_val = mapper.gen_EBV_map(m, fit='piecewise',
                                                  method='median',
                                                  reduce_nside=False,
                                                  delta_mu=ds)
        
        if ds != None:
            pix_val *= abs(ds)
        
        img = rasterizer(pix_val)
        
        # Generate figure
        ax = fig_list[k/2].add_subplot(2, 1, (k % 2) + 1)
        
        # Display map
        im = ax.imshow(np.power(img, 1./gamma).T, extent=bounds, aspect='auto',
                              origin='lower', interpolation='nearest',
                              vmin=0., vmax=1., cmap='binary', rasterized=True)
        
        # Meridians/parallels
        hputils.plot_graticules(
            ax, rasterizer,
            l_lines_minor, b_lines_minor,
            meridian_style=None,
            parallel_style=None,
            #thick_c=medblue,
            thin_c=medblue,
            thick_alpha=0.,
            thin_alpha=0.2
        )
        
        hputils.plot_graticules(
            ax, rasterizer,
            l_lines, b_lines,
            meridian_style=70.,
            parallel_style='h',
            fontsize=12,
            x_excise=7., y_excise=7.,
            txt_path_effects=stroke,
            thick_c=medblue,
            thin_c=medblue,
            label_ang_tol=15.,
            label_dist=2.5
        )
        
        # Distance label
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        x = xlim[0] - 0.03 * (xlim[1] - xlim[0])
        y = ylim[0] + 0.5 * (ylim[1] - ylim[0])
        
        ax.text(x, y, txt,
                fontsize=24, rotation=90.,
                ha='right', va='center',
                color='k', path_effects=stroke)
        
        ax.axis('off')
    
    for fig,suffix in zip(fig_list, ['near', 'far']):
        # Colorbar
        fig.subplots_adjust(left=0.05, right=0.85,
                            bottom=0.05, top=0.95,
                            wspace=0., hspace=-0.07)
        
        cax = fig.add_axes([0.86, 0.05, 0.03, 0.90])
        cbar = fig.colorbar(im, cax=cax)
        
        cbar.set_label(r'$\mathrm{E} \left( B - V \right)$',
                       fontsize=22, labelpad=10)
        
        cticks = [0., 0.2, 0.4, 0.6, 0.8, 1.]
        cticks_sqrt = [x**(1./gamma) for x in cticks]
        cbar.set_ticks(cticks_sqrt)
        cbar.set_ticklabels([r'$%.1f$' % x for x in cticks])
        cbar.ax.tick_params(labelsize=18, direction='out')
        
        cbar.solids.set_edgecolor('face') # Fix for matplotlib svg-rendering bug
        
        # Save image
        fig.savefig('%s_%s.svg' % (img_fname, suffix),
                    dpi=dpi, bbox_inches='tight')
    
    print 'Done.'


def minmax_reliable_dists(n_close=2., n_far=10.,
                          pct_close=0., pct_far=0.,
                          n_blocks=20, block=0):
    infiles = glob.glob('/n/fink1/ggreen/bayestar/output/allsky_2MASS/rw_stellar_data.*.h5')
    
    #block = 19
    #n_blocks = 20
    
    n_files = len(infiles)
    n_per_block = int(np.ceil(n_files / float(n_blocks)))
    
    k_start = block * n_per_block
    k_end = (block+1) * n_per_block
    
    infiles.sort()
    infiles = infiles[k_start:k_end]
    
    loc_block_list = []
    DM_block_list = []
    loc_dtype = [('nside', 'i4'), ('healpix_index', 'i8')]
    DM_dtype = [('DM_min', 'f4'), ('DM_max', 'f4')]
    
    #print k_start, k_end
    #print infiles
    #print 'reliable_dists.%02d.h5' % block
    #return
    
    for j, infile in enumerate(infiles):
        print '%d of %d: Loading %s ...' % (j+1, len(infiles), infile)
        
        tmp_locs = []
        
        f = h5py.File(infile, 'r')
        
        keys = f['/locs'].keys()
        
        loc_block = np.empty(len(keys), dtype=loc_dtype)
        DM_block = np.empty((len(keys), 2), dtype='f4')
        DM_block[:] = np.nan
        
        for k, pix_label in enumerate(keys):
            nside, healpix_idx = [int(s) for s in pix_label.split('-')]
            loc_block['nside'][k] = nside
            loc_block['healpix_index'][k] = healpix_idx
            
            data = f['/samples/' + pix_label][:]
            attrib = f['/locs/' + pix_label][:]
            
            idx = np.nonzero((   (attrib['conv'] == 1)
                               & (attrib['lnZ'] > -10.)
                               & (attrib['rw_chisq_min'] < 1.)
                             ))[0]
            
            n_stars = idx.size
            
            if n_stars == 0:
                continue
            
            #print '    loc, nstars: (%d, %d), %d' % (nside, healpix_idx, n_stars)
            
            threshold_close = max([n_close, pct_close/100.*n_stars])
            threshold_far = max([n_far, pct_far/100.*n_stars])
            
            #print threshold_close, threshold_far
            
            #if n_stars <= max([threshold_close, threshold_far]):
            #    #print '       --> pass'
            #    DM_block[k,0] = np.nan
            #    DM_block[k,1] = np.nan
            #    continue
            
            Mr = data['Mr'][:]
            giant_idx = Mr < 4.
            ln_w = data['ln_w'][:]
            ln_w[giant_idx] = -1.e10
            
            #w = np.exp(data['ln_w'][idx]).flatten()
            w = np.exp(ln_w[idx]).flatten()
            DM = data['DM'][idx].flatten()
            Mr = data['Mr'][idx].flatten()
            
            sort_idx = np.argsort(DM)
            
            W = np.cumsum(w[sort_idx])
            #W *= n_stars / W[-1]
            
            DM = DM[sort_idx]
            idx_min = np.sum(W < threshold_close)
            idx_max = np.sum(W < W[-1] - threshold_far) - 1
            
            '''
            print n_stars, W[-1]
            print ''
            print DM[:500:20]
            print ''
            print W[:500:20]
            print ''
            print ''
            '''
            
            if (idx_min >= 0) and (idx_min < DM.size):
                DM_block[k,0] = DM[idx_min]
            
            if (idx_max >= 0) and (idx_max < DM.size):
                DM_block[k,1] = DM[idx_max]
            
            #DM_block[k,1] = DM[idx_max]
            
            #print 'n_stars/n_effective:      %6d  %6d' % (n_stars, W[-1])
            #print 'min/max (00000, 00000): %
            
            #l, b = hputils.pix2lb_scalar(nside, healpix_idx)
            #print 'min/max (%6.1f, %6.1f): %6.2f  %6.2f' % (l, b, DM_block[k,0], DM_block[k,1])
        
        f.close()
        
        loc_block_list.append(loc_block)
        DM_block_list.append(DM_block)
    
    loc_data = np.hstack(loc_block_list)
    DM_data = np.concatenate(DM_block_list, axis=0)
    
    print '# of NaNs: %d' % (np.sum(np.isnan(DM_data)))
    print 'Writing data ...'
    
    f = h5py.File('reliable_dists.%02d.h5' % block, 'w')
    f.create_dataset('/locs', data=loc_data,
                     chunks=True, compression='gzip',
                     compression_opts=9)
    f.create_dataset('/distmod', data=DM_data,
                     chunks=True, compression='gzip',
                     compression_opts=9) #scaleoffset=2)
    f.close()


def extract_stellar_pos():
    infiles = glob.glob('/n/fink1/ggreen/bayestar/output/allsky_2MASS/rw_stellar_data.*.h5')
    
    #infiles = infiles[4:5]
    
    sparsity = 1000
    
    locs = []
    dtype = [('l', 'f4'), ('b', 'f4'), ('DM', 'f4')]
    
    for j, infile in enumerate(infiles):
        print '%d of %d: Loading %s ...' % (j+1, len(infiles), infile)
        
        tmp_locs = []
        
        f = h5py.File(infile, 'r')
        
        keys = f['/locs'].keys()
        
        for k, pix_label in enumerate(keys):
            #print ' --> (%d of %d): %s ...' % (k+1, len(keys), pix_label)
            
            data = f['/samples/' + pix_label][:]
            attrib = f['/locs/' + pix_label][:]
            
            idx = np.nonzero((   (attrib['conv'] == 1)
                               & (attrib['lnZ'] > -15.)
                               & (attrib['rw_chisq_min'] < 5.)
                             ))[0]
            
            #print '      ', idx.size, attrib.size
            
            if idx.size == 0:
                #print '       --> pass'
                continue
            
            W = np.cumsum(np.exp(data['ln_w'][idx]), axis=1)
            #print W[:,-1]
            W /= W[:,-1][:, np.newaxis]
            #print W[:,-1]
            #print ''
            
            samp_W = np.random.random(idx.size)
            samp_idx = np.sum(W < samp_W[:, np.newaxis], axis=1)
            
            tmp = np.empty(idx.size, dtype=dtype)
            tmp['l'] = attrib['l'][idx]
            tmp['b'] = attrib['b'][idx]
            tmp['DM'] = data['DM'][idx, samp_idx]
            
            tmp_locs.append(tmp)
        
        f.close()
        
        tmp_locs = np.hstack(tmp_locs)
        idx = np.arange(tmp_locs.size)
        np.random.shuffle(idx)
        locs.append(tmp_locs[idx[::sparsity]])
    
    print 'Writing locations ...'
    locs = np.hstack(locs)
    
    f = h5py.File('stellar_locs.h5', 'w')
    f.create_dataset('/locs', data=locs,
                     chunks=True, compression='gzip',
                     compression_opts=9)
    f.close()


def load_stellar_locs(fname):
    f = h5py.File(fname, 'r')
    locs = f['/locs'][:]
    f.close()
    
    l = np.radians(locs['l'])
    b = np.radians(locs['b'])
    
    cl, sl = np.cos(l), np.sin(l)
    cb, sb = np.cos(b), np.sin(b)
    d = np.power(10., locs['DM']/5. + 1.)
    
    xyz = np.empty((l.size, 3), dtype='f8')
    xyz[:,0] = d * cl * cb
    xyz[:,1] = d * sl * cb
    xyz[:,2] = d * sb
    
    return xyz


def downsample_by_two(img, times=1):
    for k in xrange(times):
        img = 0.25 * (  img[:-1:2, :-1:2]
                      + img[1::2,  :-1:2]
                      + img[:-1:2, 1::2]
                      + img[1::2,  1::2] )
    
    return img


def ortho_proj():
    # Plot properties
    cumulative = False
    add_DM = False#1.682
    plot_stars = False
    
    img_dir = '/n/pan1/www/ggreen/paper-plots/'
    img_fname = img_dir + 'ortho_slices_hq'
    map_fname = '/n/fink1/ggreen/bayestar/output/allsky_2MASS/compact_10samp.h5'
    stellar_loc_fname = '/n/home09/ggreen/documents/3d-map-release/scripts/stellar_locs.h5'
    
    if cumulative:
        img_fname += '_cumulative'
    
    if plot_stars:
        img_fname += '_stars'
    
    if add_DM:
        img_fname += '_DM'
    
    dpi = 400
    
    n_averaged = 5
    
    beta = 90. + np.hstack([0., 0., np.arange(0., 180., 15.)])
    alpha = np.hstack([180, 180., 90.*np.ones(beta.size-2)])
    
    degrade = 1
    
    #scale = np.array([0.25, 0.25, 0.25]) * degrade
    scale = np.array([0.5, 0.5, 0.5]) * degrade
    
    n_x = np.hstack([12000, 3000 * np.ones(beta.size, dtype='i4')]) / degrade
    n_y = np.hstack([12000, 3000, 1000 * np.ones(beta.size-1, dtype='i4')]) / degrade
    n_z = 10 * np.ones(beta.size, dtype='i4') / degrade
    
    # Load map
    print 'Loading map ...'
    mapper = maptools.LOSMapper([map_fname], max_samples=10)
    nside = mapper.data.nside[0]
    pix_idx = mapper.data.pix_idx[0]
    los_EBV = mapper.data.los_EBV[0]
    DM_min, DM_max = mapper.data.DM_EBV_lim[:2]
    
    mapper3d = maptools.Mapper3D(nside, pix_idx, los_EBV, DM_min, DM_max,
                                 remove_nan=False, keep_cumulative=cumulative)
    
    # Load stellar locations
    star_pos = load_stellar_locs(stellar_loc_fname)
    
    # Normal vector to projection
    ca = np.cos(np.radians(alpha))
    sa = np.sin(np.radians(alpha))
    cb = np.cos(np.radians(beta))
    sb = np.sin(np.radians(beta))
    
    # Render orthographic projections of map samples and take median
    np.seterr(divide='ignore')
    
    img_list = []
    wh_list = []
    vmax_list = []
    
    for a, b, x, y, z in zip(alpha, beta, n_x, n_y, n_z):
        print 'Rendering (a, b) = (%.1f deg, %.1f deg) ...' % (a, b)
        
        img = np.empty((n_averaged, 2*y+1, 2*x+1), dtype='f8')
        
        for i in xrange(n_averaged):
            print 'Rendering %d of %d sampled maps ...' % (i+1, n_averaged)
            img[i] = mapper3d.proj_map_in_slices('ortho', 2*z, 'sample',
                                                  a, b, y, x, z, scale,
                                                  cumulative=cumulative,
                                                  add_DM=add_DM,
                                                  verbose=True)
        
        img = np.median(img, axis=0)[::-1,::-1]
        img[img < 1.e-30] = np.nan
        
        ds_factor = 1
        
        if (img.shape[0] > 12000) or (img.shape[1] > 12000):
            ds_factor = 3
        elif (img.shape[0] > 6000) or (img.shape[1] > 6000):
            ds_factor = 2
        
        img = downsample_by_two(img, times=ds_factor)
        
        if not cumulative:
            img *= 1000.
        
        img_list.append(img)
        
        w = scale[0] * x
        h = scale[1] * y
        wh_list.append([w, h])
        
        idx = np.isfinite(img)
        vmax_list.append(np.percentile(img[idx], 98.))
        print vmax_list[-1], np.max(img[idx])
    
    vmax_list = np.array(vmax_list)
    
    # Plot results
    print 'Plotting ...'
    
    #c_bg = '#8DC9C1'
    #c_bg = '#9FCFC8'
    c_bg = '#CADEDB'
    
    fig = plt.figure(figsize=(9, 8), dpi=dpi)
    im = None
    
    stroke = [PathEffects.withStroke(linewidth=0.5, foreground='w')]
    stroke_thick = [PathEffects.withStroke(linewidth=1.5, foreground='w')]
    
    for k, (img, (w, h)) in enumerate(zip(img_list, wh_list)):
        extent = [-w, w, -h, h]
        
        if k == 1:
            fig = plt.figure(figsize=(9, 8), dpi=dpi)
        elif k == 2:
            fig = plt.figure(figsize=(18, 8), dpi=dpi)
        
        vmax = None
        
        # Add subplot
        if k in [0, 1]:
            vmax = vmax_list[k]
            ax = fig.add_subplot(1, 1, 1, axisbg=c_bg)
        else:
            vmax = np.max(vmax_list[2:])
            ax = fig.add_subplot(4, 3, k-1, axisbg=c_bg)
        
        # Solar-centric distance labels
        if k in [0, 1]:
            theta = np.linspace(0., 2.*np.pi, 1000)
            circ_x = np.cos(theta)
            circ_y = np.sin(theta)
            
            dists = np.arange(300., 2100.1, 300.)
            delta = 50.
            
            if k == 0:
                dists *= 4.
                delta *= 4.
            
            for d in dists:
                ax.plot(d*circ_x, d*circ_y,
                        lw=2., ls='--',
                        c='k', alpha=0.2,
                        zorder=-1)
                ax.plot(d*circ_x, d*circ_y,
                        lw=0.8, ls='--',
                        c='k', alpha=0.5,
                        zorder=-2)
                
                a = 0.5 * np.sqrt(2.)
                x_txt = a * (d + delta)
                y_txt = -a * (d + delta)
                
                if (x_txt < w) and (y_txt > -h):
                    ax.text(x_txt, y_txt,
                            r'$%d \, \mathrm{pc}$' % d,
                            fontsize=14, rotation=45.,
                            ha='center', va='center',
                            zorder=-3)
        
        # Dust density image
        cmap = 'Greys'
        
        vmin = 0.
        
        if add_DM:
            vmin = 4.
        
        im = ax.imshow(img, origin='lower', cmap=cmap,
                            interpolation='bilinear', aspect='auto',
                            extent=extent, vmin=vmin, vmax=vmax,
                            rasterized=True)
        
        prune = None
        
        if k in [12, 13]:
            prune = 'lower'
        
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6, prune=prune))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6, prune=prune))
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        
        # Stellar locations
        if plot_stars:
            u = np.array([[ca[k]*cb[k], -sb[k], sa[k]*cb[k]],
                          [ca[k]*sb[k],  cb[k], sa[k]*sb[k]],
                          [     -sa[k],      0, ca[k]      ]])
            
            plane_dist = np.einsum('in,ki->nk', u, star_pos)
            idx = np.abs(plane_dist[2]) < 5.
            
            print '# of stars: %d' % (np.sum(idx))
            
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            
            s = 2.
            lw = 0.5
            alpha = 0.7
            
            if k == 0:
                s = 2.
                lw = 0.5
                alpha = 1.
            elif k == 1:
                s = 4.
                lw = 0.5
                alpha = 1.
            
            ax.scatter(-plane_dist[1,idx], -plane_dist[0,idx],
                       facecolor='#246AB5', edgecolor='#82BEFF',
                       s=s, lw=lw, alpha=alpha)
            
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        
        # Axis labeling
        if k in [0, 1]:
            ax.set_xlabel(r'$\mathrm{pc}$', fontsize=14)
            ax.set_ylabel(r'$\mathrm{pc}$', fontsize=14)
        elif k == 2:
            fig.text(0.475, 0.07, r'$\mathrm{pc}$',
                     fontsize=14, ha='center', va='top')
            fig.text(0.04, 0.525, r'$\mathrm{pc}$',
                     fontsize=14, ha='right', va='center',
                     rotation=90., alpha=0.8)
        
        # \ell Arrows
        if k in [0, 1]:
            ax.arrow(0.60*w, 0., 0.30*w, 0.,
                     head_width=0.04*h, head_length=0.05*w,
                     width=0.015*h, ec='w', fc='k', lw=1.)
            ax.text(0.75*w, 0.03*h, r'${\boldmath \ell = 0^{\circ}}$',
                    fontsize=20, ha='center', va='bottom',
                    path_effects=stroke_thick)
            
            ax.arrow(0., 0.60*h, 0., 0.30*h,
                     head_width=0.04*w, head_length=0.05*h,
                     width=0.015*w, ec='w', fc='k', lw=1.)
            ax.text(-0.03*w, 0.75*h, r'${\boldmath \ell = 90^{\circ}}$',
                    fontsize=20, ha='right', va='center',
                    path_effects=stroke_thick, rotation=90)
        else:
            ax.arrow(0.60*w, 0.6*h, 0.30*w, 0.,
                     head_width=0.04*h, head_length=0.05*w,
                     ec='k', fc='k')
            ax.text(0.75*w, 0.65*h, r'$\ell = %d^{\circ}$' % (beta[k]-90.),
                    fontsize=10, ha='center', va='bottom')
            
            ax.arrow(-0.60*w, 0.6*h, -0.30*w, 0.,
                     head_width=0.04*h, head_length=0.05*w,
                     ec='k', fc='k')
            ax.text(-0.75*w, 0.65*h, r'$\ell = %d^{\circ}$' % (np.mod(beta[k]+90., 360.)),
                    fontsize=10, ha='center', va='bottom')
        
        # Remove extraneous tick labels
        if k not in [0, 1, 11, 12, 13]:
            ax.set_xticklabels([])
        
        if k not in [0, 1, 2, 5, 8, 11]:
            ax.set_yticklabels([])
        
        # Color bar, save figure
        if k in [0, 1, 13]:
            fig.subplots_adjust(left=0.08, right=0.87,
                                bottom=0.10, top=0.95,
                                wspace=0.02, hspace=0.02)
            
            cax = fig.add_axes([0.88, 0.10, 0.02, 0.85])
            
            formatter = ticker.ScalarFormatter()
            formatter.set_scientific(True)
            formatter.set_powerlimits((-2,2))
            
            cbar = fig.colorbar(im, cax=cax, format=formatter)
            
            quantity = r'\frac{\mathrm{d}}{\mathrm{d} s} \mathrm{E} \left( B - V \right)'
            units = r'\left( \mathrm{mag} / \mathrm{kpc} \right)'
            
            if cumulative:
                quantity = r'\mathrm{E} \left( B - V \right)'
                units = r'\left( \mathrm{mag} \right)'
                
                if add_DM:
                    quantity = r'R_{i} \, ' + quantity
                    quantity += r' + \mu'
            
            cbar.set_label(r'$%s \ %s$' % (quantity, units), fontsize=14)
            cbar.ax.tick_params(labelsize=11)
            
            cbar.solids.set_edgecolor('face') # Fix for matplotlib svg-rendering bug
            
            suffix = ''
            
            if k == 0:
                suffix = '_z0_wide'
            elif k == 1:
                suffix = '_z0'
            else:
                suffix = '_vertical'
            
            fig.savefig(img_fname + suffix + '.svg', dpi=dpi, bbox_inches='tight')


def main():
    #dust_priors()
    map_slices()
    #extract_stellar_pos()
    #ortho_proj()
    
    '''
    from multiprocessing import Pool
    pool = Pool(20)
    
    for n in xrange(20):
        kw = {'n_blocks':20, 'block':n}
        pool.apply_async(minmax_reliable_dists, (), kw)
    
    pool.close()
    pool.join()
    '''
    
    return 0

if __name__ == '__main__':
    main()

