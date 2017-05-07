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


# Locate pan1
pan1 = '/nfs_pan1/www/ggreen/'
if not os.path.isdir(pan1):
    pan1 = '/n/pan1/www/ggreen/'


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
                     label_props, labels,
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
    
    for i in xrange(n_procs):
        p = multiprocessing.Process(
                target=gen_frame_worker,
                args=(frame_q, lock,
                      map_fname, plot_props,
                      camera_pos, camera_props,
                      label_props, labels),
                kwargs=kwargs
            )
        
        procs.append(p)
    
    for p in procs:
        p.start()
    
    for p in procs:
        p.join()
    
    print 'Done.'


def gen_frame_worker(frame_q, lock,
                     map_fname, plot_props,
                     camera_pos, camera_props,
                     label_props, labels,
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
    
    # Load 3D map
    fname = [map_fname]
    mapper = maptools.LOSMapper(fname, max_samples=5)
    nside = mapper.data.nside[0]
    pix_idx = mapper.data.pix_idx[0]
    los_EBV = mapper.data.los_EBV[0]
    DM_min, DM_max = mapper.data.DM_EBV_lim[:2]
    
    mapper3d = maptools.Mapper3D(nside, pix_idx, los_EBV,
                                 DM_min, DM_max)
    
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
            
            gen_frame(mapper3d, camera_pos_frame, cam_props_cpy,
                                plot_props_cpy, label_props_cpy,
                                labels, **kwargs_cpy)
            
            t_end = time.time()
            print 't = %.1f s' % (t_end - t_start)
            
        except Queue.Empty:
            print 'Worker finished.'
            return



def gen_frame(mapper3d, camera_pos, camera_props,
                        plot_props, label_props,
                        labels, **kwargs):
    
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
    
    # Render scene
    img_rendered = stacker.render(oversample=oversample,
                                  fg=foreground,
                                  bg=background)
    
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


def Orion_flythrough(n_frames=10):
    '''
    Fly through the Orion ring.
    '''
    
    l_0, b_0 = -148., -13.
    a = (90. - b_0) * np.ones(n_frames, dtype='f8')
    b = l_0 * np.ones(n_frames, dtype='f8')
    
    cl, sl = np.cos(np.radians(l_0)), np.sin(np.radians(l_0))
    cb, sb = np.cos(np.radians(b_0)), np.sin(np.radians(b_0))
    
    d = np.linspace(0., 1000., n_frames)
    x = d * cl * cb
    y = d * sl * cb
    z = d * sb
    r = np.array([x, y, z]).T
    
    camera_pos = {
        'xyz': r,
        'alpha': a,
        'beta': b
    }
    
    return camera_pos


def paper_renderings():
    '''
    A list of camera positions/orientations for
    possible use in the paper.
    '''
    
    r_0 = np.array([[  0.,  0.,  0.],
                    [147., 26., 63.],
                    [144., 41., 67.]])
    a = np.array([ 96.0, 103.4, 104.1])
    b = np.array([185.0, 190.2, 196.0])
    
    camera_pos = {
        'xyz': r_0,
        'alpha': a,
        'beta': b
    }
    
    return camera_pos


def local_dust_path(n_frames=10):
    '''
    Construct camera path that:
        * begins looking at the anticenter
        * zooms back about 150 pc
        * pans around by about 20 degrees
    
    This path focuses on the Orion molecular cloud complex,
    Taurus, California and Perseus, i.e. most of the large
    dust complexes in the Solar neighborhood.
    '''
    
    A = 1.
    
    # Zoom out
    r_0 = np.zeros((n_frames/4, 3), dtype='f8')
    r_0[:,0] = np.linspace(0., A*150., r_0.shape[0])
    r_0[:,2] = np.linspace(0., A*55., r_0.shape[0])
    
    dz = 20.
    dR = 200.
    
    a_0 = 180./np.pi * np.arctan((r_0[-1,2] + dz) / (r_0[-1,0] + dR))
    b_0 = 180.
    
    a_0 = np.linspace(90. + a_0/2., 90. + a_0, r_0.shape[0])
    b_0 = b_0 * np.ones(r_0.shape[0])
    
    # Rotate around azimuthally, while bobbing in z
    phi = 25. * np.pi/180. * np.sin(np.linspace(0., 2.*np.pi, int(3./4.*n_frames)))
    theta = np.linspace(0., 2.*np.pi, phi.size)
    #phi = np.linspace(0., 2.*np.pi, int(3./4.*n_frames))
    R = r_0[-1,0]
    Z = r_0[-1,2]
    
    r_1 = np.zeros((phi.size, 3), dtype='f8')
    r_1[:,0] = R * np.cos(phi)
    r_1[:,1] = R * np.sin(phi)
    r_1[:,2] = Z + 20. * np.sin(theta)
    #r_1[:,2] = Z * np.cos(phi/2.)
    
    a_1 = 90. + 180./np.pi * np.arctan((r_1[:,2] + dz) / (R + dR))
    b_1 = np.mod(180. + 180./np.pi * phi, 360.)
    
    camera_pos = {'xyz': np.concatenate([r_0, r_1], axis=0),
                  'alpha': np.hstack([a_0, a_1]),
                  'beta': np.hstack([b_0, b_1])}
    
    #print camera_pos['alpha']
    
    return camera_pos


def Cart2sph(xyz):
    '''
    Input:
        xyz  :  (n_points, 3), where the second axis is ordered (x, y, z)
    
    Output:
        sph  :  (n_points, 3), where the second axis is ordered (r, theta, phi).
                               Here, theta is the latitude, and phi is the
                               longitude.
    '''
    
    sph = np.empty(xyz.shape, dtype='f8')
    
    xy2 = xyz[:,0]**2 + xyz[:,1]**2
    
    sph[:,0] = np.sqrt(xy2 + xyz[:,2]**2)
    sph[:,1] = np.arctan2(xyz[:,2], np.sqrt(xy2))
    sph[:,2] = np.arctan2(xyz[:,1], xyz[:,0])
    
    return sph


class SplWrapper:
    def __init__(self, x, y, **kwargs):
        self._s = [scipy.interpolate.UnivariateSpline(x, y[:,k], **kwargs) for k in xrange(y.shape[1])]
    
    def __call__(self, x, **kwargs):
        return np.array([sk(x, **kwargs) for sk in self._s]).T


def camera_attractor(r_cam, r_att, R0, Z0):
    dr = r_cam - r_att
    R = np.sqrt(dr[0]**2 + dr[1]**2)
    Z = dr[2]
    rho = np.sqrt(R**2 + Z**2)
    
    #U_R = np.exp(R/R0)
    #U_Z = np.exp(np.abs(Z)/Z0)
    U = 0.13533 * np.exp(R/R0 + np.abs(Z)/Z0)
    
    F = -U * dr/rho
    
    return F


def camera_speed(r_cam, r_cent, R0, Z0, scaling=0.5):
    dr = r_cam[:,:] - r_cent[np.newaxis,:]
    R = np.sqrt(dr[:,0]**2 + dr[:,1]**2)
    Z = dr[:,2]
    rho = np.sqrt((R/R0)**2 + (Z/Z0)**2)
    
    v = 1. + np.power(rho, scaling)
    
    return v



def gen_interpolated_path(r_anchor, n_frames=10,
                          cam_k=0.0007, cam_gamma=0.1,
                          r_att=(0.,0.,0.), R0_att=1000., Z0_att=300.,
                          r_vcent=(0.,0.,0.), R0_v=1000., Z0_v=1000., v_scaling=0.5,
                          close_path=False, close_dist=100.,
                          path_img=None):
    
    if close_path:
        r_anchor = np.vstack([r_anchor, r_anchor[0]])
    
    dr = np.diff(r_anchor, axis=0)
    r_anchor_ext = np.vstack([
        r_anchor[0] - dr[0],
        r_anchor,
        r_anchor[-1] + dr[-1]
    ])
    
    x = np.arange(r_anchor_ext.shape[0]).astype('f8')
    #s = scipy.interpolate.interp1d(x, r_anchor_ext,
    #                               kind='cubic', axis=0)
    spl = SplWrapper(x, r_anchor_ext, k=3, s=1.5*x.size)
    
    # Determine path length and derivatives along curve
    x_fine = np.linspace(x[1], x[-2], 100000)
    r_fine = spl(x_fine)
    
    dr_tmp = np.diff(r_fine, axis=0)
    ds = np.sqrt(np.sum(dr_tmp**2, axis=1))
    
    r_vcent = np.array(r_vcent)
    v = camera_speed(r_fine, r_vcent,
                     R0_v, Z0_v,
                     scaling=v_scaling)
    v = 0.5 * (v[:-1] + v[1:])
    
    print 'velocity:'
    print v[::1000]
    
    ds /= v
    
    s_tot = np.hstack([0., np.cumsum(ds)])
    
    dr = np.zeros(r_fine.shape)
    dr[:-1] += dr_tmp
    dr[1:] += dr_tmp
    dr /= np.sqrt(np.sum(dr**2, axis=1))[:, np.newaxis]
    
    # Determine what values of x to put frames at
    s_insert = np.linspace(0., s_tot[-1], n_frames)
    k_insert = np.searchsorted(s_tot, s_insert, side='left')
    k_insert = np.clip(k_insert, 0, s_tot.size-1)
    
    if close_path:
        #print s_tot[-1], close_dist
        k = np.sum(s_tot < s_tot[-1] - close_dist)
        dr[k:] = dr[0]
        
        #print k, s_tot.size
        #for i in xrange(k, s_tot.size, 100):
        #    print dr[i]
    
    # Integrate camera direction
    r_cam = np.empty(dr.shape, dtype='f8')
    r_cam[0] = dr[0]
    v_cam = np.zeros(3, dtype='f8')
    r_att = np.array(r_att)
    
    for i in xrange(1, r_cam.shape[0]):
        F = cam_k * (dr[i] - r_cam[i-1])
        F -= cam_gamma * v_cam
        F += cam_k * camera_attractor(r_fine[i-1], r_att, R0_att, Z0_att)
        
        #if i % 100 == 0:
        #    print 'F_att =', cam_k * camera_attractor(r_cam[i-1], r_att, R0, Z0)
        #    print 'F_k   =', cam_k * (dr[i] - r_cam[i-1])
        #    print ''
        
        v_cam += F * ds[i-1]
        r_cam[i] = r_cam[i-1] + v_cam * ds[i-1]
        
        #print v_cam, r_cam[i]
        
        r_cam[i] /= np.sqrt(np.sum(r_cam[i]**2))
        v_cam = (r_cam[i] - r_cam[i-1]) / ds[i-1]
    
    # Extract camera positions and orientations
    x_frame = x_fine[k_insert]
    r_frame = spl(x_frame)
    r_cam_frame = r_cam[k_insert]
    
    if path_img != None:
        fig = plt.figure(figsize=(8,24), dpi=200)
        
        for k, (a1, a2) in enumerate([(0,1), (0,2), (1,2)]):
            ax = fig.add_subplot(3,1,k+1)
            ax.plot(r_fine[:,a1], r_fine[:,a2], 'b-')
            ax.scatter(r_anchor[:,a1], r_anchor[:,a2], c='k')
            
            for i in xrange(r_frame.shape[0]):
                z = r_frame[i]
                dz = 50. * r_cam_frame[i]
                ax.arrow(z[a1], z[a2], dz[a1], dz[a2])
            
            xlabel = 'xyz'[a1]
            ylabel = 'xyz'[a2]
            
            ax.set_xlabel(r'$%s$' % xlabel, fontsize=12)
            ax.set_ylabel(r'$%s$' % ylabel, fontsize=12)
        
        fig.savefig(path_img,
                    dpi=200, bbox_inches='tight')
    
    sph = Cart2sph(r_cam_frame)
    
    camera_pos = {
        'xyz': r_frame,
        'alpha': 90. - np.degrees(sph[:,1]),
        'beta': np.degrees(sph[:,2])
    }
    
    return camera_pos


def grand_tour_path(n_frames=10):
    '''
    r = np.array([
        #[-250., -250.,    0.],
        [-300., -230.,  -50.],
        [-200., -200.,  -40.],
        [  50., -100.,    0.],
        [ 150.,    0.,  100.],
        [ 250.,  100.,  150.],
        [ 400.,    0.,  150.],
        [ 300.,  -50.,  100.],
        [ 100.,   50.,   40.],
        [   0.,   20.,   10.],
        [   0., -100.,  -10.],
        #[-100., -150.,    0.],
        [-300., -160., -100.],
        #[-400., -200.,    0.],
        [-450., -100.,  -80.],
        [-300.,  -50.,  -50.],
        [-120.,  -70.,    0.],
        [   0.,  -50.,    0.],
        [  50.,   50.,  -20.],
        [ 300.,   20.,  -60.],
        [ 200., -100.,  -90.],
        [-200., -150., -100.],
        [-400., -200., -100.],
        [-400., -230.,  -80.],
        #[-300., -250.,  -50.]
    ])
    '''
    #r = np.array([
    #    [ 300.,  -60.,  100.],
    #    [ 280.,  -30.,   95.],
    #    [ 200.,    0.,   80.],
    #    [ 100.,   15.,   50.],
    #    [   0.,   20.,   10.],
    #    #[ -20.,   20.,   -5.],
    #    #[ -55.,   10.,  -10.],
    #    [ -60.,    0.,  -10.],
    #    [ -80.,  -20.,  -15.],
    #    [ -90.,  -40.,  -18.],
    #    [-120.,  -80.,  -25.],
    #    [-200., -120.,  -55.],
    #    [-280., -180.,  -75.],
    #    [-390., -240., -105.],
    #    #[-430., -230.,  -20.],
    #    #[-420., -200.,   10.],
    #    #[-380., -150.,    0.],
    #    #[-300., -100.,  -10.],
    #    #[-200.,  -50.,  -20.],
    #    #[-100.,    0.,  -20.],
    #    #[   0.,   30.,  -20.]
    #])
    
    '''
    r = np.array([
        [ 400.,  300.,  300.],
        [   0.,  -50.,   10.],
        [-400., -230., -100.],
        [-450., -100., -100.],
        [   0.,   20.,    0.],
        [ 300.,  300., -100.],
        [ 350.,  500., -100.],
        [ 200.,  500.,  -80.],
        [   0.,  100.,  -50.],
        [-300., -100.,  -30.],
        [-350.,    0.,  -20.],
        [   0.,  100.,    0.],
        [ 200.,  400.,  150.],
        [ 400.,  450.,  300.],
        [ 415.,  330.,  305.]
    ])
    '''
    
    r = np.array([
        [ 700.,   70.,   70.],
        [ 500.,   60.,   50.],
        [ 150.,   40.,   10.],
        [   5.,   10.,  -10.],
        [ -80.,  -40.,  -30.],
        [-400., -230., -100.],
        [-700., -200., -100.],
        [-800., -100.,    0.],
        [-700.,   50.,  150.],
        [-200.,  300.,  400.],
        [ 400.,  400.,  450.],
        [1200.,  350.,  400.],
        [1100.,  150.,  200.],
        [ 850.,   85.,  100.]
        #[-350.,    0.,  -20.],
        #[   0.,  100.,    0.],
        #[ 200.,  400.,  150.],
        #[ 400.,  450.,  300.],
        #[ 415.,  330.,  305.]
    ])
    
    return gen_interpolated_path(r, n_frames=n_frames,
                                 cam_k=0.001, cam_gamma=0.15,
                                 r_att=(-300.,0.,-50.), R0_att=500., Z0_att=200.,
                                 r_vcent=(0.,0.,0.), R0_v=500., Z0_v=500., v_scaling=1.,
                                 close_path=True, close_dist=500.,
                                 path_img=pan1+'3d/allsky_2MASS/grand-tour/simple-loop-att-v2.png')


def circle_local(n_frames=20, r_x=50., r_y=50.,
                 l_0=180., b_0=-10., d_stare=500.):
    '''
    Circle near the Sun.
    '''
    
    theta = np.linspace(0., 2.*np.pi, n_frames+1)[:-1]
    x = r_x * np.cos(theta)
    y = r_y * np.sin(theta)
    z = np.zeros(n_frames, dtype='f8')
    
    r = np.array([x, y, z]).T
    
    l_0 = np.radians(l_0)
    b_0 = np.radians(b_0)
    x_0 = d_stare * np.cos(l_0) * np.cos(b_0)
    y_0 = d_stare * np.sin(l_0) * np.cos(b_0)
    z_0 = d_stare * np.sin(b_0)
    
    dr = np.empty((n_frames,3), dtype='f8')
    dr[:,0] = x_0 - x
    dr[:,1] = y_0 - y
    dr[:,2] = z_0 - z
    
    sph = Cart2sph(dr)
    a = 90. - np.degrees(sph[:,1])
    b = np.degrees(sph[:,2])
    
    #a = 90. * np.ones(n_frames, dtype='f8')
    #b = np.degrees(np.arctan2(y_0-y, x_0-x))
    
    #print l_0
    #print 90. - a
    #print ''
    #print b
    #print ''
    
    camera_pos = {
        'xyz': r,
        'alpha': a,
        'beta': b
    }
    
    return camera_pos

def stereo_pair(l_0=180., b_0=-10., d_separation=50., d_converge=1000.):
    '''
    Stereo image pair.
    '''
    
    a = 90. * np.ones(2, dtype='f8')
    b = 180. * np.ones(2, dtype='f8') + 0. * np.array([-1., 1.])
    
    x = np.zeros(2, dtype='f8')
    y = 10. * np.array([-1., 1.])
    z = np.zeros(2, dtype='f8')
    
    r = np.array([x, y, z]).T
    
    camera_pos = {
        'xyz': r,
        'alpha': a,
        'beta': b
    }
    
    return camera_pos


def gen_movie():
    # Misc settings
    plot_props = {
        'fname': pan1 + '3d/allsky_2MASS/AqS/AqS-loop-hq.png', #'3d/allsky_2MASS/grand-tour/simple-loop-att-v2-lq.png',
        'figsize': (10, 7),
        'dpi': 30,    # TODO: dpi
        'n_averaged': 7,
        'gamma': 1.,
        'R': 3.1,
        'scale_opacity': 1.,
        'sigma': 0,
        'oversample': 2,
        'n_stack': 10,
        'randomize_dist': True,
        'randomize_ang': True,
        'foreground': (255, 255, 255),
        'background': (0, 0, 0)
    }
    
    # Camera path/orientation
    #camera_pos = local_dust_path(n_frames=400)
    #camera_pos = paper_renderings()
    #camera_pos = Orion_flythrough(n_frames=400)
    #camera_pos = grand_tour_path(n_frames=20)#1600)
    camera_pos = circle_local(n_frames=4, l_0=30., b_0=5.) # TODO: n_frames
    #camera_pos = stereo_pair()
    
    #for key in camera_pos:
    #    camera_pos[key] = camera_pos[key][:20]
    
    # Camera properties
    camera_props = {
        'proj_name': 'stereo',
        'fov': 140.,
        'n_x': 200*2,
        'n_y': 140*2,
        'n_z': 500*2,
        'dr': 10./2,
        'z_0': 1., #(0., 0., 0.)
    }
    
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
    
    # General label properties
    label_props = {
        'text_color': (0, 166, 255),#(255, 255, 255),
        'stroke_color': (9, 73, 92),#(0, 0, 0) #(192, 225, 235)
    }
    
    # Generate frame
    n_procs = 10
    n_procs = min([n_procs, len(camera_pos['alpha'])])
    # map_fname = '/n/fink1/ggreen/bayestar/output/allsky_2MASS/compact/dust-map-3d-uncompressed.h5'#compact_10samp.h5'
    map_fname = 'dust-map-3d-uncompressed.h5'

    gen_movie_frames(map_fname, plot_props,
                     camera_pos, camera_props,
                     label_props, labels,
                     n_procs=n_procs, verbose=True)


def main():
    #grand_tour_path(n_frames=100)
    #circle_local()
    gen_movie()
    
    return 0


if __name__ == '__main__':
    main()
