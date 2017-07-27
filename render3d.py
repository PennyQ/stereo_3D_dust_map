#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  render3d.py
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
import scipy
import scipy.interpolate

import matplotlib
matplotlib.use('Agg')
matplotlib.rc('text', usetex=False)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patheffects as PathEffects
from mpl_toolkits.axes_grid1 import inset_locator
from matplotlib.font_manager import FontProperties

import healpy as hp
import h5py, pyfits

import os, sys, time
import os.path

import multiprocessing
import Queue

from PIL import Image

import hputils, maptools
from gen_plots import downsample_by_two

from alphastacker import AlphaStacker
from config import pan1

# Locate pan1
# pan1 = '/nfs_pan1/www/ggreen/'


def pm_ang_formatter(theta, pos):
    if np.abs(theta) < 1.e-5:
        return r'$+0^{\circ}$'
    elif theta > 0.:
        return r'$+%d^{\circ}$' % theta
    else:
        return r'$%d^{\circ}$' % theta

def proj_points(x, y, z, r_0,
                alpha, beta,
                proj_name, fov,
                dth=0, dph=0):
    # Center coordinates on camera
    x = x - r_0[0]
    y = y - r_0[1]
    z = z - r_0[2]
    
    # Convert coordinates to spherical
    rho = np.sqrt(x**2. + y**2. + z**2.)
    lon = np.arctan2(y, x)
    lat = np.pi/2. - np.arccos(z / rho)
   
    if (dth != 0) or (dph != 0):
        # Offset points in space
        ph_hat = [np.sin(lon), -np.cos(lon), np.zeros(lon.shape)]
        th_hat = [-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)]
        
        dx = dth * th_hat[0] + dph * ph_hat[0]
        dy = dth * th_hat[1] + dph * ph_hat[1]
        dz = dth * th_hat[2] + dph * ph_hat[2]
        
        #txt = '\ndx: %s\ndy: %s\ndz: %s\n' % (str(dx), str(dy), str(dz))
        #print txt
        
        x += dx
        y += dy
        z += dz
        
        # Convert coordinates to spherical again
        rho = np.sqrt(x**2. + y**2. + z**2.)
        lon = np.arctan2(y, x)
        lat = np.pi/2. - np.arccos(z / rho)
    
    # Set up projection
    phi_0 = 90. - alpha
    lam_0 = beta
    proj, proj2 = None, None
    
    if proj_name.lower() in ['rect', 'rectilinear', 'gnomonic']:
        proj = hputils.Gnomonic_projection(phi_0=phi_0, lam_0=lam_0)
        proj2 = hputils.Gnomonic_projection(phi_0=0., lam_0=0.)
    elif proj_name.lower() in ['stereo', 'stereographic']:
        proj = hputils.Stereographic_projection(phi_0=phi_0, lam_0=lam_0)
        proj2 = hputils.Stereographic_projection(phi_0=0., lam_0=0.)
    else:
        raise ValueError('Projection not implemented: "%s"' % proj_name)
    
    X, Y, oob = proj.proj(lat, lon, ret_bounds=True)
    
    # Scale projected coordinates
    df = np.radians(fov) / 2.
    phi = np.array([0., 0.])
    lam = np.array([-df, df])
    X_0, Y_0 = proj2.proj(phi, lam)
    
    #print 'pos:', X, Y
    
    scale = fov / (X_0[1] - X_0[0])
    
    #print 'scale:', scale
    
    X *= -scale
    Y *= scale
    
    return X, Y, ~oob


def lbd2xyz(l, b, d):
    l = np.radians(l)
    b = np.radians(b)
    
    x = d * np.cos(l) * np.cos(b)
    y = d * np.sin(l) * np.cos(b)
    z = d * np.sin(b)
    
    return x, y, z


def gen_movie_frames(map_fname, plot_props,
                     camera_pos, camera_props,
                     label_props, labels, axis_on,
                     **kwargs):
    n_procs = kwargs.pop('n_procs', 1)
    
    # Set up queue for workers to pull frame numbers from
    frame_q = multiprocessing.Queue()
    
    n_frames = len(camera_pos['alpha'])
    
    for k in xrange(n_frames):
        frame_q.put(k)
    
    # Set up lock to allow first image to be written without interference btw/ processes
    lock = multiprocessing.Lock()
    
    # Spawn worker processes to plot images
    procs = []
    
    # get mapper here
    # Load 3D map
    fname = [map_fname]
    mapper = maptools.LOSMapper(fname, max_samples=5) # load data and map to pixel
    nside = mapper.data.nside[0]
    pix_idx = mapper.data.pix_idx[0]
    los_EBV = mapper.data.los_EBV[0]
    DM_min, DM_max = mapper.data.DM_EBV_lim[:2]
    
    mapper3d = maptools.Mapper3D(nside, pix_idx, los_EBV,
                                 DM_min, DM_max)  # map from pixel to cart
    
    
    for i in xrange(n_procs):
        
        # setup a list of processes that we want to run
        p = multiprocessing.Process(
                target=gen_frame_worker,
                args=(mapper3d, frame_q, lock,
                      map_fname, plot_props,
                      camera_pos, camera_props,
                      label_props, labels, axis_on),
                kwargs=kwargs
            )
        
        procs.append(p)

    # Run processes
    for p in procs:
        p.start()

    # Exit the completed processes    
    for p in procs:
        p.join()
    
    print 'Done.'

# TODO: remove loading data from the Queue, make it a buffer 
# share loaded data in memory
def gen_frame_worker(mapper3d, frame_q, lock,
                     map_fname, plot_props,
                     camera_pos, camera_props,
                     label_props, labels, axis_on,
                     **kwargs):
    # Copy to avoid overwriting original
    plot_props = plot_props.copy()
    fname_base = plot_props['fname']
    
    if fname_base.endswith('.png'):
        fname_base = fname_base[:-4]
    
    # Reseed random number generator
    t = time.time()
    t_after_dec = int(1.e9*(t - np.floor(t)))
    seed = np.bitwise_xor([t_after_dec], [os.getpid()])
    
    np.random.seed(seed=seed)
    


    
    # Loop through queue
    first_img = True
    np.seterr(all='ignore')
    
    while True:
        try:
            k = frame_q.get_nowait()
            t_start = time.time()
            print 'Projecting frame %d ...' % k
            
            # Copy/modify keyword arguments
            # (to avoid pop() from removing keywords)
            cam_props_cpy = camera_props.copy()
            plot_props_cpy = plot_props.copy()
            label_props_cpy = label_props.copy()
            kwargs_cpy = kwargs.copy()
            
            plot_props_cpy['fname'] = fname_base + '.%05d.png' % k
            camera_pos_frame = {
                'xyz': camera_pos['xyz'][k],
                'alpha': camera_pos['alpha'][k],
                'beta': camera_pos['beta'][k]
            }
            
            if first_img:
                first_img = False
                kwargs['lock'] = lock
            else:
                kwargs['lock'] = None
            
            # Generate frame
            logfile=open('log.txt', 'a')
            logfile.write('\n')
            logfile.write('Frame%d, time except gen_frame %.1f s\n' % (k, time.time()-t_start))
            gen_frame(mapper3d, camera_pos_frame, cam_props_cpy,
                                plot_props_cpy, label_props_cpy,
                                labels, axis_on,**kwargs_cpy)
            
            t_end = time.time()
            print 't = %.1f s' % (t_end - t_start)
            # logfile.write('\n')
            logfile.write('Frame%d = %.1f s \n' %(k, t_end-t_start))
            logfile.close()
            
        except Queue.Empty:
            print 'Worker finished.'
            return



def gen_frame(mapper3d, camera_pos, camera_props,
                        plot_props, label_props,
                        labels, axis_on, **kwargs):
    
    if not axis_on and 'Sol' in labels:
        # remove sol, 0 and 90 from labels
        no_axis_labels = dict(labels)
        
        labels.pop('Sol')
        labels.pop(u'0°')
        labels.pop(u'90°')
        print('-----------labes are ', labels)

        # 'Sol': ((0., 0., 0.), ('left', 'top', 1., -0.75)),
        # u'0°': ((0, -32.01, 47.17), ('center', 'center', 0., 0.)),
        # u'90°': ((90, -32.01, 47.17), ('center', 'center', 0., 0.)),
    
    #
    # Read settings
    #
    
    # Read camera position/orientation
    alpha = camera_pos['alpha']
    beta = camera_pos['beta']
    r_cam = camera_pos['xyz']
    
    # Read camera properties
    proj_name = camera_props['proj_name']
    fov = camera_props['fov']
    n_x = camera_props['n_x']
    n_y = camera_props['n_y']
    n_z = camera_props['n_z']
    dr = camera_props['dr']
    z_0 = camera_props['z_0']
    
    # If z_0 is a location (x,y,z), then use the
    # distance to that location as the starting distance
    if hasattr(z_0, '__len__'):
        displacement = np.array(r_cam) - np.array(z_0)
        z_0 = np.sqrt(np.sum(displacement**2))
        n_z = int(round(n_z - z_0/dr))
        
        print 'z_0 = %.3f, n_z = %d' % (z_0, n_z)
    
    # Read additional image/plotting settings
    plt_fname = plot_props['fname']
    figsize = plot_props['figsize']
    dpi = plot_props.pop('dpi', 400)
    n_averaged = plot_props.pop('n_averaged', 1)
    reduction = plot_props.pop('reduction', 'sample')
    gamma = plot_props.pop('gamma', 1.)
    R = plot_props.pop('R', 3.)
    scale_opacity = plot_props.pop('scale_opacity', 1.)
    sigma = plot_props.pop('sigma', 0.15)
    oversample = plot_props.pop('oversample', 2)
    n_stack = plot_props.pop('n_stack', 20)
    randomize_dist = plot_props.pop('randomize_dist', False)
    randomize_ang = plot_props.pop('randomize_ang', False)
    foreground = plot_props.pop('foreground', (0, 0, 0))
    background = plot_props.pop('background', (255, 255, 255))
    
    R *= np.log(10.) / 5.
    sigma *= (2.*n_x+1.) / fov
    
    # Read general label properties
    c_txt = label_props.pop('text_color', (0, 166, 255))
    c_stroke = label_props.pop('stroke_color', (255, 148, 54))
    font = label_props.pop('font', 'fonts/cmunss.ttf')
    fontsize_base = label_props.pop('fontsize_base', 0.04)
    
    fontsize_base *= 2.*n_x+1.
    
    # Misc settings
    verbose = kwargs.pop('verbose', False)
    
    #
    # Generate image stack
    #
    
    n_images = n_z / n_stack + (1 if n_z % n_stack else 0)
    d_images = z_0 + np.linspace(0., (n_z-1.)*dr, n_images)
    img = np.empty((n_averaged, n_images, 2*n_y+1, 2*n_x+1), dtype='f8')
    
    np.seterr(all='ignore')
    
    for k in xrange(n_averaged):
        if verbose:
            print 'Rendering image %d of %d ...' % (k+1, n_averaged)
        
        img[k] = mapper3d.proj_map_in_slices(proj_name, n_z, reduction,
                                             alpha, beta, n_x, n_y, fov,
                                             r_cam, dr, z_0, stack=n_stack,
                                             randomize_dist=randomize_dist,
                                             randomize_ang=randomize_ang,
                                             verbose=verbose)
    
    img = np.mean(img, axis=0)
    img *= dr  # Convert from mean dE(B-V)/ds to E(B-V)
    
    # Smooth image
    if sigma > 1.e-10:
        #print 'Smoothing with sigma = %.3f' % sigma
        img = scipy.ndimage.filters.gaussian_filter(img, [0,sigma,sigma])
    
    # Convert image to opacity
    # As R -> 0, img -> optical depth
    img = 1. - np.exp(-R*img)
    img *= scale_opacity
    
    idx = img > 1.
    img[idx] = 1.
    
    # Gamma bending
    if gamma != 1.:
        img = np.power(img, 1./gamma)
    
    # Flip images properly
    img = np.swapaxes(img, 1, 2)[:,::-1,:]
    
    # Initialize class to stack images and text
    stacker = AlphaStacker(img, d_images)
    
    logfile = open('log.txt', 'a')
    start = time.time()
    
    #
    # Set up labels
    #
    
    for key, ((l, b, d), (ha, va, dph, dth)) in labels.iteritems():
        r_pts = np.array([lbd2xyz(l, b, d)])
        
        tmp = proj_points(r_pts[:,0], r_pts[:,1], r_pts[:,2],
                          r_cam, alpha, beta, proj_name, fov,
                          dph=dph, dth=dth)
        in_bounds = tmp[2]
        x_proj = tmp[0]
        y_proj = tmp[1]
        
        if ( (not in_bounds[0]) or
             (abs(x_proj[0]) > fov/2.-2.) or
             (abs(y_proj[0]) > float(n_y)/float(n_x)*fov/2.-2.) ):
            continue
        
        x = x_proj[0]
        y = y_proj[0]
        
        x += fov/2.
        y += float(n_y)/float(n_x) * fov/2.
        
        x *= (2.*n_x+1.) / float(fov)
        y *= (2.*n_y+1.) / (float(n_y) / float(n_x) * float(fov))
        y = (2*n_y+1) - y
        
        d = np.sqrt(np.sum( (r_pts[0]-r_cam)**2 ))
        
        if (not np.isfinite(d)) or (d < 10.):
            continue
        
        fontsize = fontsize_base / np.sqrt(d / 100.)
        stroke_width = fontsize / 26.
        #print stroke_width
        
        stacker.insert_text(key, (x, y), d,
                            ha=ha, va=va,
                            font=font,
                            fontsize=fontsize,
                            fontcolor=c_txt,
                            stroke_width=stroke_width,
                            stroke_color=c_stroke)
                            
    # Plot image
    w = fov/2.
    h = float(n_y)/float(n_x) * w
    extent = [-w, w, h, -h]
    
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(1,1,1)
    
    scene_time = time.time()
    # Render scene
    img_rendered = stacker.render(oversample=oversample,
                                  fg=foreground,
                                  bg=background)
                                  
    logfile.write('render scene time %.2f' % (time.time()-scene_time))
    
    
    # Save PIL image
    #pimg = Image.fromarray((255.*img_rendered[:,:,:3]).astype('u1'), mode='RGB')
    #
    #tmp_fname = ''
    #
    #if plt_fname.endswith('.png'):
    #    tmp_fname += plt_fname[:-4] + '_raw.png'
    #else:
    #    tmp_fname += plt_fname + '_raw.png'
    #
    #pimg.save(tmp_fname, 'PNG')
    
    # Plot slices
    ax.imshow(img_rendered, origin='lower', interpolation='none',
                            aspect='equal', extent=extent,
                            zorder=0)
    
    ax.set_xlim([-w, w])
    ax.set_ylim([-h, h])
    
    if axis_on: 
        # Project position of Sun and coord-system axes
        r_ovplt = {}
    
        r_ovplt['Sun'] = np.array([[0., 0., 0.]])
    
        range_tmp = np.linspace(-0.2, 1., 1000)
        zeros_tmp = np.zeros(range_tmp.size)
        const_tmp = -25. * np.ones(range_tmp.size)
    
        r_ovplt['xaxis'] = np.vstack([25.*range_tmp, zeros_tmp, const_tmp]).T
        r_ovplt['yaxis'] = np.vstack([zeros_tmp, 25.*range_tmp, const_tmp]).T
        r_ovplt['zaxis'] = np.vstack([zeros_tmp, zeros_tmp, 25.*(range_tmp-1.)]).T
    
        x_proj = {}
        y_proj = {}
    
        for key, r_pts in r_ovplt.iteritems():
            tmp = proj_points(r_pts[:,0], r_pts[:,1], r_pts[:,2],
                              r_cam, alpha, beta, proj_name, fov)
            in_bounds = tmp[2]
            x_proj[key] = tmp[0][in_bounds]
            y_proj[key] = tmp[1][in_bounds]
        
            #print '%s: %d in bounds' % (key, np.sum(in_bounds))
    
        # Dot for the Sun
        ax.scatter(x_proj['Sun'], y_proj['Sun'],
                   s=24, c='#116999',
                   edgecolor='none', alpha=0.75,
                   zorder=5002)
        ax.scatter(x_proj['Sun'], y_proj['Sun'],
                   s=8, c='#499ECC',
                   edgecolor='none', alpha=1.00,
                   zorder=5003)
    
        # Plot axes of Galactic coordinate system
        for key in ['xaxis', 'yaxis', 'zaxis']:
            ax.plot(x_proj[key], y_proj[key], ls='-', lw=2.,
                                              c='b', alpha=0.25,
                                              zorder=5000)
            ax.plot(x_proj[key], y_proj[key], ls='-', lw=1.25,
                                              c='#1E9EE3', alpha=0.9,
                                              zorder=5001)
        print('axis_on?', axis_on)
    
        # Formatting
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(pm_ang_formatter))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(pm_ang_formatter))
    
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    
    title = r'$\left( \alpha, \, \beta, \, x, \, y,  \, z \right) = \left( '
    title += r'%.1f^{\circ} \ %.1f^{\circ} \ \ ' % (alpha, beta)
    title += r'%d \, \mathrm{pc} \ ' % (r_cam[0])
    title += r'%d \, \mathrm{pc} \ ' % (r_cam[1])
    title += r'%d \, \mathrm{pc}'    % (r_cam[2])
    title += r'\right)$'
    
    ax.set_title(title, fontsize=20)
    
    fig.subplots_adjust(left=0.05, right=0.98, bottom=0.05, top=0.95)
    
    lock = kwargs.pop('lock', None)
    
    if lock != None:
        lock.acquire()
    
    fig.savefig(plt_fname, dpi=dpi, bbox_inches='tight')
    
    if lock != None:
        lock.release()
    
    plt.close(fig)
    del img
    


def main():
    #grand_tour_path(n_frames=100)
    #circle_local()
    from config import map_fname, plot_props, camera_props, label_props, camera_pos, n_procs, axis_on, stop_f
    
    # Points to project to camera coordinates
    labels = {
        'Sol': ((0., 0., 0.), ('left', 'top', 1., -0.75)),
        u'0°': ((0, -32.01, 47.17), ('center', 'center', 0., 0.)),
        u'90°': ((90, -32.01, 47.17), ('center', 'center', 0., 0.)),
        'Ori A': ((213., -20.5, 450.), ('center', 'center', 0., -15.)),
        'Ori B': ((205., -16., 450.), ('center', 'center', 15., -6.)),
        'Mon R2': ((215., -12., 800.), ('center', 'center', 35., 10.)),
        u'λ Ori': ((195.5, -12., 400.), ('center', 'center', 0., 0.)),
        'Perseus': ((160., -19., 250.), ('center', 'center', 30., -8.)),
        'California': ((162., -9.5, 430.), ('center', 'center', 8., 23.)),
        'Taurus': ((171., -16., 210.), ('center', 'center', -8., -10.)),
        'Rosette': ((207., -2.5, 1500.), ('center', 'center', 85., 0.)),
        'Maddalena': ((216., -2.5, 2200.), ('center', 'center', 0., -70.)),
        'CMa OB1': ((223., -0.2, 1450.), ('center', 'center', 0., 0.)),
        'Mon OB1': ((200., 1., 750.), ('center', 'center', 0., 26.)),
        #'Pegasus': ((100., -30., 130.), ('center', 'bottom', 0., 0.3))
        'Aquila S': ((38., -17., 160.), ('center', 'center', -25., 0.)),
        u'ρ Oph': ((-5., 17., 200.), ('center', 'center', 25., 0.)),
        'Galactic Center': ((0., 0., 8000.), ('center', 'center', 0., 0.)),
    }
    print('||||||||||||carera_pos', type(camera_pos))
    if type(camera_pos) is dict:
        # Generate frame
        n_procs = min([n_procs, len(camera_pos['alpha'])])
        
        if stop_f: 
            f = plot_props['fname']
            plot_props['fname'] = f.split('.png')[0]+'-stop.png'
        gen_movie_frames(map_fname, plot_props,
                         camera_pos, camera_props,
                         label_props, labels,
                         n_procs=n_procs, verbose=True, axis_on=axis_on)  
                           
    elif type(camera_pos) is list:
        # render left camera
        f = plot_props['fname']
        plot_props['fname'] = f.split('.png')[0]+'-left.png'
        if stop_f:
            plot_props['fname'] = f.split('.png')[0]+'-left-stop.png'
        n_procs = min([n_procs, len(camera_pos[0]['alpha'])])
        gen_movie_frames(map_fname, plot_props,
                         camera_pos[0], camera_props,
                         label_props, labels,
                         n_procs=n_procs, verbose=True, axis_on=axis_on)

        # render right camera                 
        plot_props['fname'] = f.split('.png')[0]+'-right.png'
        if stop_f:
            plot_props['fname'] = f.split('.png')[0]+'-right-stop.png'
        n_procs = min([n_procs, len(camera_pos[1]['alpha'])])                         
        gen_movie_frames(map_fname, plot_props,
                         camera_pos[1], camera_props,
                         label_props, labels,
                         n_procs=n_procs, verbose=True, axis_on=axis_on)
    
    # rename stop files with correct frame number
    if stop_f:
        plot_dir = plot_props['fname'].split(plot_props['fname'].split('/')[-1])[0]
        for fn in os.listdir(plot_dir):
            if 'stop' in fn:
                pre_name  = fn.split('-stop')[0]
                print(fn.split('.')[1])
                img_num = int(fn.split('.')[1])
                img_num = img_num + stop_f  # this part should goes to render3d.py
                os.rename(plot_dir+fn, plot_dir+pre_name + '.%05d.png' % img_num)
    return 0


if __name__ == '__main__':
    main()
