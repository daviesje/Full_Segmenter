# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 17:47:34 2020

@author: jed12
"""
from PIL import Image
import numpy as np
import sys
import glob
from os import remove

brightness_const = 16
in_dim = 4096
out_dim = 256
n_tiles = in_dim**2 // out_dim**2

#Manage proportions of Train/Validation/Test dataset
val_split = np.array([0.8,0.1,0.1])

n_split = np.array([0,0,1])
val_indices = np.array([])

#mask_prefix = 'Bleaching_glare_polygon_big_'
#tile_prefix = 'Bleaching_glare_map_big_'
mask_prefix = 'mask_'
tile_prefix = 'ortho_'

def set_constants(val_split=val_split),in_dim_par=in_dim,out_dim_par=out_dir):
    global in_dim, out_dim, n_tiles, val_indices, n_split
    in_dim = in_dim_par
    out_dim = out_dim_par
    n_tiles = in_dim**2 // out_dim**2
    n_split = (val_split*n_tiles).astype(int)
    val_indices = np.arange(n_tiles)
    np.random.shuffle(val_indices)

    print(f'constants: in_dim = {in_dim} out_dim = {out_dim} n_tiles = {n_tiles} val_split = {n_split}')

    return

def blockshaped(arr,subh,subw):
    h = arr.shape[0]
    w = arr.shape[1]
    return (arr.reshape(h//subh,subh,-1,subw)
            .swapaxes(1,2)
            .reshape(-1,subh,subw))

def unblockshaped(arr, h, w):
    """
    Return an array of shape (h, w) where
    h * w = arr.size

    If arr is of shape (n, nrows, ncols), n sublocks of shape (nrows, ncols),
    then the returned array preserves the "physical" layout of the sublocks.
    """
    n, nrows, ncols = arr.shape
    return (arr.reshape(h//nrows, -1, nrows, ncols)
               .swapaxes(1,2)
               .reshape(h, w))

def read_image(imfile):
    reader = Image.open(imfile,mode='r')
    raster = np.array(reader)
    if len(raster.shape) > 2:
        raster = raster[...,:3]
    reader.close()
    return raster

def tile_images(indir,outdir,tilenum):
    fullimage = read_image(indir+'/'+tile_prefix+tilenum+'.png')

    if fullimage.shape[0] != in_dim or fullimage.shape[1] != in_dim:
        print(f'dimension mismatch for {tilenum}')
        return 1

    #TODO: remove this with new datasets, one dataset was skewed
    if "try_these" in indir:
        fullimage[:6,...] = 0

    buf = np.zeros((n_tiles,out_dim,out_dim,3))
    buf[...,0] = blockshaped(fullimage[...,0],out_dim,out_dim)
    buf[...,1] = blockshaped(fullimage[...,1],out_dim,out_dim)
    buf[...,2] = blockshaped(fullimage[...,2],out_dim,out_dim)

    buf = buf.astype('uint8')

    for i in range(n_tiles):
        im = Image.fromarray(buf[val_indices[i],...])
        if i < n_split[0]:
            im.save(f'{outdir}/train_tiles/tile_{tilenum}_{val_indices[i]:03d}.png')
        elif i < n_split[0] + n_split[1]:
            im.save(f'{outdir}/val_tiles/tile_{tilenum}_{val_indices[i]:03d}.png')
        else:
            im.save(f'{outdir}/test_tiles/tile_{tilenum}_{val_indices[i]:03d}.png')
    
    return 0

def tile_masks(indir,outdir,tilenum):
    fullimage = read_image(indir+'/'+mask_prefix+tilenum+'.png')
    
    #fullimage = np.loadtxt(indir+mask_prefix+tilenum,dtype=int,delimiter=' ',comments='#',ndmin=2)

    if fullimage.shape[0] != in_dim or fullimage.shape[1] != in_dim:
        print(f'dimension mismatch for {tilenum}')
        return 1
    
    #TODO: remove this with new datasets, the "try_these" dataset was skewed
    if "try_these" in indir:
        print("WARNING: SHIFTING MASKS BY 6 TO CORRECT")
        fullimage = np.roll(fullimage,6,axis=0)
        fullimage[:6,...] = 0

    #colour case
    if len(fullimage.shape) == 3:
        buf = np.zeros((n_tiles,out_dim,out_dim,3))
        buf[...,0] = blockshaped(fullimage[...,0],out_dim,out_dim)
        buf[...,1] = blockshaped(fullimage[...,1],out_dim,out_dim)
        buf[...,2] = blockshaped(fullimage[...,2],out_dim,out_dim)
    #grayscale case
    elif len(fullimage.shape) == 2:
        buf = blockshaped(fullimage,out_dim,out_dim)
    else:
        print('images are not 2/3 dimensional')
        quit()

    #buf = blockshaped(fullimage,out_dim,out_dim) * brightness_const
    buf = buf.astype('uint8')
    #print(np.unique(buf,return_counts=True))

    for i in range(n_tiles):
        im = Image.fromarray(buf[val_indices[i],...])
        if i < n_split[0]:
            im.save(f'{outdir}/train_masks/mask_{tilenum}_{val_indices[i]:03d}.png')
        elif i < n_split[0] + n_split[1]:
            im.save(f'{outdir}/val_masks/mask_{tilenum}_{val_indices[i]:03d}.png')
        else:
            im.save(f'{outdir}/test_masks/mask_{tilenum}_{val_indices[i]:03d}.png')

    
    return 0

#check that each tile has a mask counterpart, and vice versa
#then remove standalone images
def check_dirs(outdir):
    subdir_list = [['train_tiles/tile_','train_masks/mask_']
        ,['val_tiles/tile_','val_masks/mask_']
        ,['test_tiles/tile_','test_masks/mask_']]
    subdir_names = ['train','val','test']

    full_list = np.full((3,2,20000),'NaN',dtype=str)
    n_ims = np.zeros((3,2))
    #build list of image indices in each subdir
    for i,subdir in enumerate(subdir_list):
        for j,imtype in enumerate(subdir):
            tile_list = glob.glob(outdir+'/'+imtype+"*.png")
            n_ims[i,j] = len(tile_list)
            if n_ims[i,j] > full_list.shape[-1]:
                print('cannot check for duplicates, increase buffer size')
                return
            for k,f in enumerate(tile_list):
                f = f.replace('\\','/')
                f = f.split(".")[-2] #remove file extension
                f = f.split(imtype)[1] #get the number
                full_list[i,j,k] = f
        
    removed = np.zeros((3,2))
    for i,row in enumerate(full_list):
        diff_arr = np.setdiff1d(row[0],row[1])
        for diff in diff_arr:
            print(f'{subdir_names[i]} tile {diff} not in masks')
            remove(outdir+"train_tiles/tile_"+diff+".png")
            removed[i,0] += 1
        diff_arr = np.setdiff1d(row[1],row[0])
        for diff in diff_arr:
            print(f'{subdir_names[i]} mask {diff} not in tiles')
            remove(outdir+"train_tiles/mask_"+diff+".png")
            removed[i,1] += 1

    print(f'train || tiles: {n_ims[0,0]} | masks: {n_ims[0,1]}')
    print(f'val || tiles: {n_ims[1,0]} | masks: {n_ims[1,1]}')
    print(f'test || tiles: {n_ims[2,0]} | masks: {n_ims[2,1]}')
    
    print(f'removed [train,val,test] [tiles,masks] = {removed}')

    return

#remove tiles and masks that contain only white/black pixels
def cleanup_dir(outdir):
    for prefix in ['train','val','test']:
        train_tile_list = glob.glob(outdir+'/'+prefix+"_tiles/tile*.png")
        n_tiles = len(train_tile_list)
        removed_count = 0
        removed_list = np.array([])
        for i,f in enumerate(train_tile_list):
            f = f.replace('\\','/')
            image = read_image(f)
            
            if np.all(np.logical_or(image==255,image==0)):
                #remove file
                print(f'removing {f}')
                remove(f)
                
                buf = f.split('_tiles/tile_')[1] #remove the filenames (just number left)
                removed_list = np.append(removed_list,prefix + ' | ' + buf)
                mfile = outdir+'/'+prefix+'_masks/mask_'+buf
                mfile = mfile.replace('\\','/')
                #remove corresponding mask
                print(f'removing {mfile}')
                remove(mfile)
                removed_count += 1
        print(f'removed {removed_count} out of {n_tiles} for being blank')
    
    np.savetxt(f'removed_bg.txt',removed_list,fmt='%s')
        
    return

#remove tiles and masks that contain only white/black pixels
def cleanup_masks(outdir):
    for prefix in ['train','val','test']:
        train_tile_list = glob.glob(outdir+'/'+prefix+"_masks/mask*.png")
        n_tiles = len(train_tile_list)
        removed_count = 0
        removed_list = np.array([])
        for i,f in enumerate(train_tile_list):
            f = f.replace('\\','/')
            image = read_image(f)
            
            if np.all(image==255) or np.all(image==0):
                #remove file
                print(f'removing {f}')
                remove(f)
                
                buf = f.split('_masks/mask_')[1] #remove the filenames (just number left)
                removed_list = np.append(removed_list,prefix + ' | ' + buf)
                mfile = outdir+'/'+prefix+'_tiles/tile_'+buf
                mfile = mfile.replace('\\','/')
                #remove corresponding mask
                print(f'removing {mfile}')
                remove(mfile)
                removed_count += 1
        print(f'removed {removed_count} out of {n_tiles} for being blank')
    
    np.savetxt(f'removed_bg.txt',removed_list,fmt='%s')
        
    return

def do_all_tiling(indir,outdir):
    print('start')
    print(indir+'/'+tile_prefix+"*.png")
    files = glob.glob(indir+'/'+tile_prefix+"*.png")

    for file in files:
        tilenum = file.split(".")[-2].split(tile_prefix)[-1]
        set_constants()
        print(f'tiling {tilenum}...')
        err = tile_images(indir,outdir,tilenum)
        if not err:
            tile_masks(indir,outdir,tilenum)


if __name__ == "__main__":
    indir = sys.argv[1]

    if sys.argv[2]:
        outdir = sys.argv[2]
    else:
        outdir = f'../input/train_images/'

    do_all_tiling(indir,outdir)
    
    check_dirs(outdir)
    cleanup_dir(outdir)
    #cleanup_masks(outdir)
    check_dirs(outdir)
