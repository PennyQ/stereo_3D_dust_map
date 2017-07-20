#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  alphastacker.py
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
import matplotlib.pyplot as plt

from font_rendering import rasterize_text
from point_rendering import rasterize_point
from alpha import *


def upsample_image(img, nx, ny):
    return np.kron(img, np.ones((nx, ny)))


class AlphaStacker:
    def __init__(self, alpha_stack, alpha_dist):
        self.alpha_stack = np.array(alpha_stack)
        self.alpha_dist = np.array(alpha_dist)
        self.obj = []
        self.obj_dist = []

    def insert_text(self, txt, pos, dist, **kwargs):
        args = (txt, pos)
        self.obj.append((rasterize_text, self._transform_text,
                         args, kwargs.copy()))
        self.obj_dist.append(dist)

    def _transform_text(self, args, kwargs, oversample):
        '''
        Transform text settings for oversampling.
        '''

        txt, pos = args
        kwargs_cpy = kwargs.copy()

        pos_upsampled = [oversample*p for p in pos]
        args = (txt, pos_upsampled)

        fontsize = kwargs_cpy.pop('fontsize', 24)
        kwargs_cpy['fontsize'] = oversample * fontsize

        stroke_width = kwargs_cpy.pop('stroke_width', 0)
        kwargs_cpy['stroke_width'] = oversample * stroke_width

        #tmp = 'Text: ' + txt.encode('utf-8').strip()
        #tmp += '\n  stroke_width = %.3f' % stroke_width
        #print tmp

        return args, kwargs_cpy

    def insert_point(self, pos, dist, **kwargs):
        args = (pos,)
        self.obj.append((rasterize_point, self._transform_point,
                        args, kwargs.copy()))
        self.obj_dist.append(dist)

    def _transform_point(self, args, kwargs, oversample):
        '''
        Transform point settings for oversampling.
        '''

        pos = args[0]
        kwargs_cpy = kwargs.copy()

        pos_upsampled = [oversample*p for p in pos]
        args = (pos_upsampled,)

        radius = kwargs_cpy.pop('radius', 5)

        if isinstance(radius, int) or isinstance(radius, float):
            kwargs_cpy['radius'] = oversample * radius
        else:
            kwargs_cpy['radius'] = [oversample*r for r in radius]

        return args, kwargs_cpy

    def _alpha_blocks(self):
        # Determine ordering of alpha and rendered layers in stack
        n_a = self.alpha_dist.size
       	n_o = len(self.obj_dist)

        a_idx = np.hstack([np.arange(n_a), -np.ones(n_o)]).astype('i8')
        d = np.hstack([self.alpha_dist, np.array(self.obj_dist)])
        idx = np.argsort(d, kind='mergesort')[::-1]
        a_idx = a_idx[idx]
        d = d[idx]

        #print 'a_idx:'
        #print a_idx
        #print ''

        # Split alpha stack into contiguous blocks
        split_idx = np.nonzero(a_idx == -1)[0]
        a_idx = np.split(a_idx, split_idx)
        d_a = np.split(d, split_idx)

        #print 'split_idx:'
        #print split_idx
        #print ''

        #print 'a_idx:'
        #print a_idx
        #print ''

        #print 'd_a:'
        #print d_a
        #print ''

        n_blocks = 0
        dist_block = []

        for i,(k_block,d_block) in enumerate(zip(a_idx, d_a)):
            idx = (k_block != -1)
            a_idx[i] = k_block[idx]

            if np.sum(idx) != 0:
                n_blocks += 1
                dist_block.append(d_block[idx][0])

        dist_block = np.array(dist_block)

        s = self.alpha_stack.shape[1:]
        alpha_block = np.empty((n_blocks, s[0], s[1]), dtype=self.alpha_stack.dtype)

        counter = 0

        for k_block in a_idx:
            if k_block.size == 0:
                continue

            alpha_block[counter] = 1. - np.prod(1. - self.alpha_stack[k_block], axis=0)
            counter += 1

        #print dist_block
        #print n_blocks

        return dist_block, alpha_block

    def render(self, oversample=1, fg=(0,0,0), bg=(255, 255, 255, 255)):
        a_shape = self.alpha_stack.shape[1:]
        canvas = (oversample * a_shape[1], oversample * a_shape[0])

        # Split alpha stack into contiguous blocks
        a_dist, a_block = self._alpha_blocks()

        # Determine ordering of objects in stack
        #n_a = self.alpha_dist.size
        n_a = a_dist.size
        n_o = len(self.obj_dist)

        a_idx = np.hstack([np.arange(n_a), -np.ones(n_o)]).astype('i8')
        o_idx = np.hstack([-np.ones(n_a), np.arange(n_o)]).astype('i8')
        #d = np.hstack([self.alpha_dist, np.array(self.obj_dist)])
        d = np.hstack([a_dist, np.array(self.obj_dist)])
        idx = np.argsort(d, kind='mergesort')[::-1]
        a_idx = a_idx[idx]
        o_idx = o_idx[idx]

        # Stack images and text
        stacked = np.empty((canvas[0], canvas[1], 4), dtype='f8')
        stacked[:,:,3] = 1.
        for k in xrange(len(bg)):
            stacked[:,:,k] = bg[k] / 255.

        fg_color = [c/255. for c in fg]

        import time
        start = time.time()
        for dd, ka, ko in zip(d[idx], a_idx, o_idx):
            # print('stack loop, dd, ka, ko', dd, ka, ko)
            img = None

            #print 'Stacking d = %d ...' % dd
            alpha2img_time = time.time()
            if ka != -1:
                img = alpha2img(upsample_image(a_block[ka], oversample, oversample), color=fg_color)
                # alpha2img_end_time = time.time()
                # print('aplha2img function time', alpha2img_end_time - alpha2img_time)
                #img = alpha2img(upsample_image(self.alpha_stack[ka], oversample, oversample))
            elif ko != -1:
                f_render, f_oversample, args, kwargs = self.obj[ko]
                sample_start = time.time()
                args, kwargs = f_oversample(args, kwargs, oversample)
                oversample_time = time.time()
                # print('f_oversample time', oversample_time-sample_start)
                
                img = f_render(canvas[::-1], *args, **kwargs)
                # print('f_render time', time.time()-oversample_time)
                img = np.swapaxes(img, 0, 1)
            else:
                raise ValueError('Both ka and ko are -1')

            stacked = blend_images(stacked, img)
            # print('blend_images time', time.time() - alpha2img_end_time)

        logfile = open('log.txt', 'a')
        logfile.write('blend images %.2f' % (time.time()-start))
        logfile.close()

        return stacked


def test_alpha_stacker():
    n_stack = 10
    shape = (n_stack, 500, 400)

    x = np.linspace(0., 2.*np.pi, shape[1])
    y = np.linspace(0., 2.*np.pi, shape[2])
    y, x = np.meshgrid(y, x)

    img_stack = 5. * np.random.random(shape) / float(n_stack)
    img_stack *= np.cos(x)**2 * np.sin(y)**2

    #for k in xrange(n_stack):
    #    img_stack[k] *= np.power(np.cos(x)**2 * np.sin(y)**2, 2. - k/float(n_stack))

    img_dist = np.linspace(0., 1000., n_stack)

    stacker = AlphaStacker(img_stack, img_dist)

    stacker.insert_text('Nearby', (0, 0), 300.,
                        fontcolor=(28, 255, 172, 255),
                        fontsize=40,
                        stroke_color=(255, 255, 255, 255),
                        stroke_width=2,
                        ha='left',
                        va='top')

    stacker.insert_text('Far Away', (500, 400), 900.,
                        fontcolor=(28, 255, 172, 255),
                        fontsize=30,
                        stroke_color=(255, 255, 255, 255),
                        stroke_width=1,
                        ha='right',
                        va='bottom')

    stacker.insert_point((150, 250), 200.,
                         radius=(100,50),
                         color=(0, 255, 255, 255))

    stacker.insert_point((150, 250), 200.,
                         radius=(110,60),
                         color=(255,0,255,100))

    print 'Stacking ...'
    img = stacker.render(oversample=1)
    img = img[:,:,:3]
    print 'Done.'

    fig = plt.figure(figsize=(8,4), dpi=200)

    ax = fig.add_subplot(1,2,1)

    print 'Imshow stacking ...'

    for img_k in img_stack:
        ax.imshow(alpha2img(img_k), aspect='equal', interpolation='none')

    print 'Done.'

    ax.set_title(r'$\mathtt{imshow} \ \mathrm{stacking}$', fontsize=24)

    ax = fig.add_subplot(1,2,2)
    ax.imshow(img, aspect='equal', interpolation='none')

    ax.set_title(r'$\mathtt{AlphaStacker} \ \mathrm{stacking}$', fontsize=24)

    fig.subplots_adjust(wspace=0.15)

    fig.savefig('AlphaStacker-demo.png', dpi=400)
    plt.show()


def main():
    test_alpha_stacker()

    return 0

if __name__ == '__main__':
    main()

