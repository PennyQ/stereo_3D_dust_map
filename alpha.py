#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  alpha.py
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


def calc_alpha_add(alpha_current, alpha_target):
    return (alpha_target - alpha_current) / (1. - alpha_current)


def blend_alpha(alpha_current, alpha_add):
    alpha = alpha_add + (1. - alpha_add) * alpha_current
    
    return alpha


def calc_alpha_add_tau(tau_current, tau_target):
    alpha_current = 1. - np.exp(tau_current)
    alpha_target = 1. - np.exp(tau_target)
    return calc_alpha_add(alpha_current, alpha_target)


def alpha2img(alpha, color=(0,0,0)):
    s = alpha.shape
    img = np.zeros((s[1], s[0], 4), dtype='f8')
    img[:,:,3] = alpha.T
    
    for k,c in enumerate(color):
        if c != 0:
            img[:,:,k] = c
    
    return img


def blend_images(img_under, img_over, limit_alpha=False):
    img = np.empty(img_under.shape, dtype='f8')
    
    # Blend alphas
    img[:,:,3] = img_over[:,:,3] + (1. - img_over[:,:,3]) * img_under[:,:,3]
    
    # Blend colors
    over = np.einsum('ijk,ij->ijk', img_over[:,:,:3], img_over[:,:,3])
    under = np.einsum('ijk,ij->ijk', img_under[:,:,:3],
                                     img_under[:,:,3] * (1. - img_over[:,:,3]))
    
    norm = 1. / img[:,:,3]
    norm[img[:,:,3] == 0.] = 0.
    img[:,:,:3] = np.einsum('ijk,ij->ijk', over+under, norm)
    
    # Limit alpha to maximum of constituent alphas
    if limit_alpha:
        img[:,:,3] = np.maximum(img_over[:,:,3], img_under[:,:,3])
    
    idx = img > 1.
    img[idx] = 1.
    
    return img
