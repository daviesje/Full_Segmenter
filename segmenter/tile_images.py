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
val_split = np.array([0.8,0.1,0.1])
n_split = np.array([0,0,1])
val_indices = np.array([])

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
    reader.close()
    return raster

def tile_images(indir,outdir,tilenum):
    fullimage = read_image(indir+'/main'+tilenum+'.png')

    if fullimage.shape[0] != in_dim or fullimage.shape[1] != in_dim:
        print(f'dimension mismatch for {tilenum}')
        return

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

def tile_masks(indir,outdir,tilenum):
    #fullimage = read_image(indir+'/mask_'+tilenum+'.png')
    fullimage = np.loadtxt(indir+'/mask_cube'+tilenum,dtype=int,delimiter=' ',comments='#',ndmin=2)

    if fullimage.shape[0] != in_dim or fullimage.shape[1] != in_dim:
        print(f'dimension mismatch for {tilenum}')
        return

    buf = np.zeros((n_tiles,out_dim,out_dim))
    buf = blockshaped(fullimage,out_dim,out_dim) * brightness_const
    buf = buf.astype('uint8')
    #print(np.unique(buf,return_counts=True))
    #print(f'after tiles = {buf.shape}')

    for i in range(n_tiles):
        im = Image.fromarray(buf[val_indices[i],...])
        if i < n_split[0]:
            im.save(f'{outdir}/train_masks/mask_{tilenum}_{val_indices[i]:03d}.png')
        elif i < n_split[0] + n_split[1]:
            im.save(f'{outdir}/val_masks/mask_{tilenum}_{val_indices[i]:03d}.png')
        else:
            im.save(f'{outdir}/test_masks/mask_{tilenum}_{val_indices[i]:03d}.png')

def set_constants(val_split=np.array([0.8,0.1,0.1]),in_dim_par=4096,out_dim_par=256):
    global in_dim, out_dim, n_tiles, val_indices, n_split
    in_dim = in_dim_par
    out_dim = out_dim_par
    n_tiles = in_dim**2 // out_dim**2
    n_split = (val_split*n_tiles).astype(int)
    val_indices = np.arange(n_tiles)
    np.random.shuffle(val_indices)

    print(f'constants: in_dim = {in_dim} out_dim = {out_dim} n_tiles = {n_tiles} val_split = {n_split}')

    return

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
            tile_list = glob.glob(outdir+imtype+"*.png")
            n_ims[i,j] = len(tile_list)
            if n_ims[i,j] > 20000:
                print('cannot check for duplicates, increase buffer size')
                return
            for k,f in enumerate(tile_list):
                f = f.split(".")[-2]
                f = f.split("_")[-2] + "_" + f.split("_")[-1]
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
        train_tile_list = glob.glob(outdir+prefix+"_tiles/tile*.png")
        n_tiles = len(train_tile_list)
        removed_count = 0
        removed_list = np.array([])
        for i,f in enumerate(train_tile_list):
            image = read_image(f)
            
            if np.all(image==255) or np.all(image==0):
                #remove file
                print(f'removing {f}')
                remove(f)
                
                buf = f.split(".")[-2]
                buf = buf.split("_")[-2] + "_" + buf.split("_")[-1]
                removed_list = np.append(removed_list,prefix + ' | ' + buf)
                mfile = outdir+prefix+'_masks\mask_'+buf+".png"
                #remove corresponding mask
                print(f'removing {mfile}')
                remove(mfile)
                removed_count += 1
        print(f'removed {removed_count} out of {n_tiles} for being blank')
    
    np.savetxt(f'removed_bg.txt',removed_list,fmt='%s')
        
    return


def do_all_tiling(indir,outdir):
    files = glob.glob(indir+"main*.png")

    for file in files:
        tilenum = file.split(".")[-2].split("n")[-1]
        set_constants()
        print(f'tiling {tilenum}...')
        tile_images(indir,outdir,tilenum)
        tile_masks(indir,outdir,tilenum)
    
    #cleanup_dir(outdir)
    #check_dirs(outdir)


if __name__ == "__main__":
    indir = sys.argv[1]
    outdir = sys.argv[2]
    #tilenum = sys.argv[3]

    do_all_tiling(indir,outdir)
    check_dirs(outdir)
    cleanup_dir(outdir)
    check_dirs(outdir)