# This script includes all codes related to camera route calculation
# TODO: add another camera should also goes here
import numpy as np
import scipy
from math import sqrt, radians
import matplotlib.pyplot as plt

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

def gen_side_by_side(n_frames, AF, camera_pos, stop_f):
    # AF = r_cam_frame
    # \AF\
    from numpy import linalg as LA 
    AF = AF*500
    # AF_norm = np.sqrt(AF[:, 0]**2 + AF[:, 1]**2 + AF[:, 2]**2)
    
    # distance between two cameras
    cam_d = 10  

    # vector to north pole    
    north = [0, 0, 1] 
    
    # AA1 vector or delta = N x AF / (\N\*\AF\*sin(theta)) * |AA1\
    delta = np.empty((n_frames, 3), dtype='f8')
    c = np.cross(AF, north)
    #sin_theta = np.sin(np.radians(camera_pos['alpha']))
    delta = c /LA.norm(c, axis=1)[:, None]* (cam_d/2)
    
    print('-------delta', delta)
    print('c', c)
    print('AF', AF)
    #print('sin_theta', sin_theta)
    # left camera
    A1 = np.empty((n_frames, 3), dtype='f8')    
    A1[:, 0] = camera_pos['xyz'][:,0] - delta[:, 0]
    A1[:, 1] = camera_pos['xyz'][:,1] - delta[:, 1]
    A1[:, 2] = camera_pos['xyz'][:,2] - delta[:, 2]
    
    #right camrea
    A2 = np.empty((n_frames, 3), dtype='f8') 
    A2[:, 0] = camera_pos['xyz'][:,0] + delta[:, 0]
    A2[:, 1] = camera_pos['xyz'][:,1] + delta[:, 1]
    A2[:, 2] = camera_pos['xyz'][:,2] + delta[:, 2]
    
    A1F = -delta + AF
    A2F = delta + AF
    
    sph1 = Cart2sph(A1F)
    sph2 = Cart2sph(A2F)
        
    # print('------AF is', AF)
  #   print('------A from AF is', camera_pos['xyz'])
  #   print('------F from AF is', camera_pos['xyz']-AF)
  #   print('------F from A1F is', A1-A1F)
  #   print('------F from A2F is', A2-A2F)
    # three F are the same!
    
    # cross viewing
    camera_pos1 = {
         'xyz': A1,
         'alpha': 90. - np.degrees(sph1[:,1]),
         'beta': np.degrees(sph1[:,2])
    }
    camera_pos2 = {
             'xyz': A2,
             'alpha': 90. - np.degrees(sph2[:,1]),
             'beta': np.degrees(sph2[:,2])
    }
    # parallel viewing
    #camera_pos1 = {
    #    'xyz': A1,
    #    'alpha': camera_pos['alpha'],
    #    'beta': camera_pos['beta']
    #}
    #camera_pos2 = {
    #        'xyz': A2,
    #        'alpha': camera_pos['alpha'],
    #        'beta': camera_pos['beta']
    #}
    print('stop_f in gen_side by side', stop_f)
    if stop_f:
        print('xyz, alpha, beta', camera_pos1['xyz'].shape, camera_pos1['alpha'].shape, camera_pos1['beta'].shape)
        camera_pos1 = {
            'xyz': A1[stop_f:, :],
            'alpha': (90. - np.degrees(sph1[:,1]))[stop_f:],
            'beta': np.degrees(sph1[:,2])[stop_f:]
        }
        camera_pos2 = {
            'xyz': A2[stop_f:, :],
            'alpha': (90. - np.degrees(sph2[:,1]))[stop_f:],
            'beta': np.degrees(sph2[:,2])[stop_f:]
        }
        print('camera pos 1 2 in gen_side by side', camera_pos1, camera_pos1['xyz'].shape)
    return [camera_pos1, camera_pos2]

#---------------grand_tour_path--------------------
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

def grand_tour_path(n_frames=10,
                     cam_k=0.001, cam_gamma=0.15,
                     r_att=(-300.,0.,-50.), R0_att=500., Z0_att=200.,
                     r_vcent=(0.,0.,0.), R0_v=500., Z0_v=500., v_scaling=1.,
                     close_path=True, close_dist=500.,
                     # path_img=pan1+'3d/allsky_2MASS/grand-tour/simple-loop-att-v2.png')
                     path_img=None, side_by_side=False, stop_f=None):
#def gen_interpolated_path(r_anchor, n_frames=10,
                          # cam_k=0.0007, cam_gamma=0.1,
                         #  r_att=(0.,0.,0.), R0_att=1000., Z0_att=300.,
                         #  r_vcent=(0.,0.,0.), R0_v=1000., Z0_v=1000., v_scaling=0.5,
                         #  close_path=False, close_dist=100.,
                         #  path_img=None):
    # TODO: anchor?                     
    r_anchor = np.array([
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
    
    if close_path:
        r_anchor = np.vstack([r_anchor, r_anchor[0]])
    
    dr = np.diff(r_anchor, axis=0)
    
    # add starting and ending point the the r_anchor
    r_anchor_ext = np.vstack([
        r_anchor[0] - dr[0],
        r_anchor,
        r_anchor[-1] + dr[-1]
    ]) 
    # distance alone the path
    x = np.arange(r_anchor_ext.shape[0]).astype('f8')
    #s = scipy.interpolate.interp1d(x, r_anchor_ext,
    #                               kind='cubic', axis=0)
    # new path funtion, for each two points which lies on a plane
    spl = SplWrapper(x, r_anchor_ext, k=3, s=1.5*x.size)
    
    # Determine path length and derivatives along curve
    x_fine = np.linspace(x[1], x[-2], 100000)
    r_fine = spl(x_fine)
    
    dr_tmp = np.diff(r_fine, axis=0)
    ds = np.sqrt(np.sum(dr_tmp**2, axis=1))
    
    # determine camera speed
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
    
    # Seems not used here
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
        # print('save in path_img?', path_img)
    
    sph = Cart2sph(r_cam_frame)
    
    camera_pos = {
        'xyz': r_frame,
        'alpha': 90. - np.degrees(sph[:,1]),
        'beta': np.degrees(sph[:,2])
    }
    
    if not side_by_side:
        if stop_f:
            camera_pos = {
                'xyz': r_frame[stop_f:, :],
                'alpha': (90. - np.degrees(sph[:,1]))[stop_f:],
                'beta': np.degrees(sph[:,2])[stop_f:]
            }
        return camera_pos
    else:
        return gen_side_by_side(n_frames=n_frames, AF=r_cam_frame, camera_pos=camera_pos, stop_f=stop_f)


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
    

def circle_local(n_frames=20, r_x=50., r_y=50.,
                 l_0=180., b_0=-10., d_stare=500., side_by_side=False, stop_f=None):
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
    # print('----------sph is: ', sph)
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
    print('stop_f', stop_f)
    if side_by_side:
        return gen_side_by_side(n_frames=n_frames, AF=dr, camera_pos=camera_pos, stop_f=stop_f)
    else:
        if stop_f:
            camera_pos = {
                'xyz': r[stop_f:, :],
                'alpha': a[stop_f:],
                'beta': b[stop_f:]
            }
            print('stopped camera_pos', camera_pos)
        return camera_pos   

    
def nw_270(n_frames=20, d_stare=500.):
    # spherical coordinates in physics, cnetered on sun
    phi = np.linspace(0., 2.*np.pi, n_frames+1)[:-1]
    theta = np.zeros(n_frames)
    xyz = np.zeros((n_frames, 3))
    print(np.degrees(theta))
    
    camera_pos = {
        'xyz': xyz,
        'alpha': 90.-theta,
        'beta': np.degrees(phi)
    }
    return camera_pos
    
def equirectangular_route(n_frames=72, d_stare=500., side_by_side=False, stop_f=None):  # 30*30 for each piece
    # spherical coordinates in physics, cnetered on sun
    piece_l = radians(sqrt(360*180/n_frames))

    '''
    we only consider piece height=width
    as the frame here is 72, each piece will be 30*30, so theta will have 6 = srqt(72/2) steps, phi will have 12 steps
    so the array of theta and phi will match:
    165
    135
    105
    --0--30--...---360
    75
    45
    15
    the scan will start from [165, 0] -> [135, 0] ->...[15, 0] -> [175, 30]...
    we can change the scan order by change the tile and repeat below
    '''
    theta = np.tile(np.linspace(-np.pi/2+piece_l/2, np.pi/2-piece_l/2, sqrt(n_frames/2)),int(sqrt(n_frames/2)*2))
    phi = np.repeat(np.linspace(0., 2.*np.pi, sqrt(n_frames/2)*2+1)[:-1], int(sqrt(n_frames/2)))
    # theta = np.zeros(n_frames)
    xyz = np.zeros((n_frames, 3))
    print(np.degrees(theta))
    
    camera_pos = {
        'xyz': xyz,
        'alpha': 90.-np.degrees(theta),
        'beta': np.degrees(phi)
    }
    print('alpha', camera_pos['alpha'], camera_pos['alpha'].shape)
    print('beta', camera_pos['beta'], camera_pos['beta'].shape)
    # return camera_pos

    x_0 = d_stare * np.sin(theta) * np.sin(phi)
    y_0 = d_stare * np.sin(theta) * np.cos(phi)
    z_0 = d_stare * np.cos(theta)
    
    dr = np.empty((n_frames,3), dtype='f8')
    dr[:,0] = x_0 
    dr[:,1] = y_0 
    dr[:,2] = z_0
    
    if side_by_side:
        return gen_side_by_side(n_frames=n_frames, AF=dr, camera_pos=camera_pos, stop_f=stop_f)
    else:
        if stop_f:
            camera_pos = {
                'xyz': r[stop_f:, :],
                'alpha': a[stop_f:],
                'beta': b[stop_f:]
            }
            print('stopped camera_pos', camera_pos)
        return camera_pos
