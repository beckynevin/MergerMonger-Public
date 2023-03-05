####################################################
# This is helper code to download individual
# SDSS galaxies and make plots of their images
####################################################

import numpy as np
import os
import astropy.io.fits as fits
import matplotlib
import matplotlib.pyplot as plt
import astropy
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy import coordinates as coords
from astropy import units as u
from astropy.nddata import Cutout2D


def download_galaxy(ID, ra, dec, prefix_frames, size, remove=True):
   decode=SDSS_objid_to_values(ID)
   if decode[2] < 1000:
       pref_run = '000'
   else:
       pref_run = '00'
       
   if decode[4] > 100:
       pref_field = '0'
   else:
       pref_field = '00'
   
   name = prefix_frames+'frame-r-'+pref_run+str(decode[2])+'-'+str(decode[3])+'-'+pref_field+str(decode[4])+'.fits'

   
   os.system('wget -O '+str(prefix_frames)+'frame-r-'+pref_run+str(decode[2])+'-'+str(decode[3])+'-'+pref_field+str(decode[4])+'.fits.bz2  https://data.sdss.org/sas/dr12/boss/photoObj/frames/301/'+str(decode[2])+'/'+str(decode[3])+'/frame-r-'+pref_run+str(decode[2])+'-'+str(decode[3])+'-'+pref_field+str(decode[4])+'.fits.bz2')

   

   os.system('bunzip2 '+str(prefix_frames)+'frame-r-'+pref_run+str(decode[2])+'-'+str(decode[3])+'-'+pref_field+str(decode[4])+'.fits.bz2')
   try:
        im=fits.open(prefix_frames + 'frame-r-'+pref_run+str(decode[2])+'-'+str(decode[3])+'-'+pref_field+str(decode[4])+'.fits')
   except FileNotFoundError:
        # Try removing it and restarting:
        os.system('rm '+prefix_frames+'frame*')
        os.system('wget -O '+str(prefix_frames)+'frame-r-'+pref_run+str(decode[2])+'-'+str(decode[3])+'-'+pref_field+str(decode[4])+'.fits.bz2  https://data.sdss.org/sas/dr12/boss/photoObj/frames/301/'+str(decode[2])+'/'+str(decode[3])+'/frame-r-'+pref_run+str(decode[2])+'-'+str(decode[3])+'-'+pref_field+str(decode[4])+'.fits.bz2')
        os.system('bunzip2 '+str(prefix_frames)+'frame-r-'+pref_run+str(decode[2])+'-'+str(decode[3])+'-'+pref_field+str(decode[4])+'.fits.bz2')
        im=fits.open(prefix_frames + 'frame-r-'+pref_run+str(decode[2])+'-'+str(decode[3])+'-'+pref_field+str(decode[4])+'.fits')

   print('trying to open this')
   print(prefix_frames + 'frame-r-'+pref_run+str(decode[2])+'-'+str(decode[3])+'-'+pref_field+str(decode[4])+'.fits')
   
   


   try:
      obj_coords = SkyCoord(str(ra)+' '+str(dec),unit=(u.deg, u.deg), frame='icrs')
   except NameError:
      print(ra)
      STOP
   size = u.Quantity((size,size), u.arcsec)#was 80,80
   wcs_a = WCS(im[0].header)
   try:
      stamp_a = Cutout2D(np.ma.masked_invalid(im[0].data), obj_coords, size, wcs=wcs_a)#was image_a[0].data
   except ValueError:
      print(im[0].data)
      plt.clf()
      plt.imshow(im[0].data, norm=matplotlib.colors.LogNorm())
      plt.colorbar()
      plt.title('failed for some reason')
      plt.show()
      return 0
   camera_data=(np.fliplr(np.rot90(stamp_a.data))/0.005)
   

   im.close()

   
   
   if remove:
      os.system('rm '+prefix_frames+'frame*')

   return camera_data

def SDSS_objid_to_values(objid):

    # Determined from http://skyserver.sdss.org/dr7/en/help/docs/algorithm.asp?key=objID                                                                                         \
                                                                                                                                                                                  

    bin_objid = bin(objid)
    bin_objid = bin_objid[2:len(bin_objid)]
    bin_objid = bin_objid.zfill(64)

    empty = int( '0b' + bin_objid[0], base=0)
    skyVersion = int( '0b' + bin_objid[1:4+1], base=0)
    rerun = int( '0b' + bin_objid[5:15+1], base=0)
    run = int( '0b' + bin_objid[16:31+1], base=0)
    camcol = int( '0b' + bin_objid[32:34+1], base=0)
    firstField = int( '0b' + bin_objid[35+1], base=0)
    field = int( '0b' + bin_objid[36:47+1], base=0)
    object_num = int( '0b' + bin_objid[48:63+1], base=0)

    return skyVersion, rerun, run, camcol, field, object_num

def download_sdss_ra_dec_table(path):
    file_path = path+'five_sigma_detection_saturated_mode1_beckynevin.csv'
    f = open(file_path, 'r+')
    data=f.readlines()[1:]


    sdss = []
    ra = []
    dec = []
    
    for line in data:
        line_split = line.split(',')



        sdss.append(int(line_split[0]))
        ra.append(float(line_split[1]))
        dec.append(float(line_split[2]))
    return sdss, ra, dec
    
def download_frame_open_image(prefix, decode):
    prefix_frames = prefix
    os.chdir(os.path.expanduser(prefix_frames))

    if decode[2] < 1000:
        pref_run = '000'
    else:
        pref_run = '00'

    if decode[4] > 100:
        pref_field = '0'
    else:
        pref_field = '00'

    name = prefix_frames + 'frame-r-'+pref_run+str(decode[2])+'-'+str(decode[3])+'-'+pref_field+str(decode[4])+'.fits'
    name_end = 'frame-r-'+pref_run+str(decode[2])+'-'+str(decode[3])+'-'+pref_field+str(decode[4])+'.fits'
    try:

        im=fits.open(name)



    except FileNotFoundError:


        '''Here is where it is necessary to know the SDSS data password and username'''
        try:
            os.system('wget https://data.sdss.org/sas/dr12/boss/photoObj/frames/301/'+str(decode[2])+'/'+str(decode[3])+'/frame-r-'+pref_run+str(decode[2])+'-'+str(decode[3])+'-'+pref_field+str(decode[4])+'.fits.bz2')
            os.system('bunzip2 frame-r-'+pref_run+str(decode[2])+'-'+str(decode[3])+'-'+pref_field+str(decode[4])+'.fits.bz2')

        except FileNotFoundError:
            return 0



    try:
        im=fits.open(prefix_frames + 'frame-r-'+pref_run+str(decode[2])+'-'+str(decode[3])+'-'+pref_field+str(decode[4])+'.fits')
    except FileNotFoundError:
        return 0
        
    return im

    

def plot_individual(id, ra, dec, prob, run, prefix_frames):
    decode=SDSS_objid_to_values(id)
    # prefix for where the frame images live
    fits_img = download_frame_open_image('/Users/beckynevin/CfA_Code/Kinematics_and_Imaging_Merger_Identification/sdss/', decode)
    if fits_img ==0:
        camera_data = download_galaxy(id, ra, dec, prefix_frames, 40, remove==True)
    else:
        obj_coords = SkyCoord(str(ra)+' '+str(dec),unit=(u.deg, u.deg), frame='icrs')
        
        # Create 2D cutouts of the object in each band in a 6 by 6 arcsec box
        size = u.Quantity((40,40), u.arcsec)#was 80,80
        #print(np.shape(im[0].data))

        wcs_a = WCS(fits_img[0].header)
        
        stamp_a = Cutout2D(fits_img[0].data, obj_coords, size, wcs=wcs_a)
        fits_img.close()
        
        camera_data=(np.fliplr(np.rot90(stamp_a.data))/0.005)
    
    plt.clf()
    fig = plt.figure()
    ax0 = fig.add_subplot(111)
    ax0.imshow(abs(camera_data), norm=matplotlib.colors.LogNorm(vmin=10**0,vmax=10**4), cmap='afmhot')
    ax0.axis('off')
    
    
    plt.savefig(prefix_frames+str(run)+'/'+str(id)+'_'+str(round(prob,2))+'.png', dpi=200, bbox_inches = 'tight',pad_inches = 0)



        
    

