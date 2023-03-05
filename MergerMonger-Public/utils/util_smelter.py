'''

GalaxySmelter - This is the 'Real' edition to be applied to a real galaxy survey to extract predictors and place them in a table.

Things to install: Source Extractor, Galfit, statmorph (cite Vicente Rodriguez-Gomez)

The first step is to determine how to obtain galaxy images, ivar images, and psf; here I show an example for MaNGA preimaging,
utilizing wget - you can make your own directory that contains files, ivar files, and psfs, just name them camera_data, camera_data_ivar,
and psf, respectively.

These images need to be in units of counts.

Here, I run everything from within a folder that has subfolders imaging/ and preim/
which contain the output check images and the input images, respectively.
'''

'''
Import things, these first few lines are so that this version of python plays nicely with plotting on the supercomputer
'''


import numpy as np
from astropy.io import fits
import os
import matplotlib
#matplotlib.use('agg') Use this if running on a supercomputer
import matplotlib.pyplot as plt
from astropy.cosmology import WMAP9
import photutils
#import statmorph #credit to Vicente Rodriguez-Gomez ()
import astropy.io.fits as pyfits
import math

from astropy import units as u
from astropy.nddata import Cutout2D
import scipy.ndimage as ndi
import statmorph
from astropy.visualization import simple_norm

"""
This file defines the `make_figure` function, which can be useful for
debugging and/or examining the morphology of a source in detail.
"""
# Author: Vicente Rodriguez-Gomez <v.rodriguez@irya.unam.mx>
# Licensed under a 3-Clause BSD License.

import numpy as np
import scipy.signal
import scipy.ndimage as ndi
from astropy.io import fits
import shlex
import subprocess

import time

__all__ = ['make_figure']

def write_fails_txt(prefix, sdss, num, failcode):
    file2=open(prefix+'Tables/SDSS_DR12_predictors_MPI_fails_'+str(num)+'.txt','a+')
    file2.write(str(sdss)+'\t'+str(failcode)+'\n')
#write_fails_txt(prefix, num, 'No file')

def rm_files(prefix, sdss):
    os.system('rm '+prefix+'imaging/*'+str(sdss)+'*')

    return

def rm_frame(pref, string):
    os.system('rm '+pref+string+'*')
    return

def get_predictors(id, image, prefix, verbose=False, just_segmap=False):

    num = 'troubleshoot_predictors'

    camera_data=(np.fliplr(np.rot90(image))/0.005)
   
    camera_data_sigma = np.sqrt(abs(image))

    '''This is vicente's method'''
    threshold = photutils.detect_threshold(camera_data, nsigma=1.5)
    npixels = 5  # minimum number of connected pixels 
    segm = photutils.detect_sources(camera_data, threshold, npixels)


    try:
        label = np.argmax(segm.areas) + 1
    except ValueError:
        rm_files(prefix, id)
        write_fails_txt(prefix, sdss, num, 'Segmentation')
        return
        
  
    segmap = segm.data == label
    

    segmap_float = ndi.uniform_filter(np.float64(segmap), size=10)
    segmap = segmap_float > 0.5

    if just_segmap:
        return camera_data, segmap
    
   

    '''
    The first step is creating some .fits files for use by galfit and Source extractor
    '''
    
    
    
    
    
    outfile = prefix+'imaging/pet_radius_'+str(id)+'.fits'
    hdu = fits.PrimaryHDU(abs(camera_data))#was d[0] (the unconvolved version)                                                                                      
    hdu_number = 0
    hdu.writeto(outfile, overwrite=True)
    hdr=fits.getheader(outfile, hdu_number)

    hdr['EXPTIME'] = 1
    hdr['EXPTIME']

    hdu.writeto(outfile, overwrite=True)

    # now make a psf file for galfit:
    
  
    outfile = prefix + 'imaging/out_convolved_'+str(id)+'.fits'
    hdu = fits.PrimaryHDU(abs(camera_data))
    hdu_number = 0
    hdu.writeto(outfile, overwrite=True)
    hdr=fits.getheader(outfile, hdu_number)
    
    hdr['EXPTIME'] = 1
    hdr['EXPTIME']
    
    hdu.writeto(outfile, overwrite=True)
    
    # and an error file for galfit:

    outfile = prefix + 'imaging/out_sigma_convolved_'+str(id)+'.fits'
    hdu = fits.PrimaryHDU(abs(camera_data_sigma))
    hdu_number = 0
    hdu.writeto(outfile, overwrite=True)
    hdr=fits.getheader(outfile, hdu_number)
    
    hdr['EXPTIME'] = 1
    hdr['EXPTIME']
    hdu.writeto(outfile, overwrite=True)
    
    
 
    
    write_sex_default(str(id), prefix)
    run_sex(str(id), prefix)



    sex_out = sex_params_galfit(str(id), prefix)
    

    #output if there are 2 bright sources aka 2 stellar bulges:
        #return sex_pet_r, minor, flux, x_max, y_max, n_bulges, eff_radius_1, B_A_1, PA_1, back_1, x_max_2, y_max_2, mag_max_2, eff_radius_2, B_A_2, PA_2
        #

    #output if 1 bulge:
        #return sex_pet_r, minor, flux, x_max, y_max, n_bulges, eff_radius_1, B_A_1, PA_1, back_1
    
    bg_data = plot_sex(str(id), sex_out[3], sex_out[4], prefix)
    length_gal=np.shape(camera_data)[0]
    
    
    

    '''now put together all of the necessary Galfit inputs - it is a little finnicky and needs to know
    magnitude guesses, which is annoying because everything is in counts and often there's no exposure time
    provided, but you can guess the magnitude as below by setting the mag_zpt and then calculating the flux ratio
    given the source extractor background and main source counts'''

    
 

    '''This is scaled up using the background flux and the flux from the petrosian output'''
    
    '''Now use the output of source extractor to also determine the petrosian radius,
    the semimajor and semiminor axes, the flux guess, the bg level, the PA, and the axis ratio'''
    r_sex=sex_out[0]#petrosian radius (aka semimajor axis because it is an elliptical aperture)
    b_sex=sex_out[1]#minor axis of the above
    flux_sex=sex_out[2]#flux of the brightest aperture in counts
    x_pos_1=sex_out[3]
    y_pos_1=sex_out[4]
    num_bulges=sex_out[5]
    eff_rad_1=sex_out[6]#effective radius
    AR_1=sex_out[7]#ellipticity
    PA1=sex_out[8]#position angle on sky
    bg_level=sex_out[9]#bg counts

    
  

 
    
    #Define an arbitrary magnitude zeropoint for Galfit
    mag_zpt=26.563

    #try setting mag_zpt as the background magnitude and scale up from there with the flux
    try:
        mag_guess=(-2.5*math.log10((flux_sex)/bg_level)+mag_zpt)
    except ValueError:
        write_fails_txt(prefix, id, num, 'MagGuess')
        
        return 

    if num_bulges==2:
        #further define some things from sourcextractor output:
        x_pos_2=sex_out[10]
        y_pos_2=sex_out[11]
        flux_sex_2=sex_out[12]
        eff_rad_2=sex_out[13]
        AR_2=sex_out[14]
        PA2=sex_out[15]
        
        
        mag_guess=(-2.5*math.log10((flux_sex)/bg_level)+mag_zpt)

        mag_guess_2=(-2.5*math.log10((flux_sex_2)/bg_level)+mag_zpt)

        #Prepares the galfit feedme file:
        #file = open(prefix+'imaging/galfit.feedme_'+str(sdss),'w')
        #file.write('WRITE')
        #file.close()
        #STOP
        f = write_galfit_feedme(id, prefix, x_pos_1, y_pos_1, x_pos_2, y_pos_2, mag_guess, mag_zpt, num_bulges, length_gal, eff_rad_1, eff_rad_2, mag_guess_2, AR_1, PA1, AR_2, PA2, bg_level)
        
    else:
        
        f = write_galfit_feedme(id, prefix, x_pos_1, y_pos_1, 0,0, mag_guess, mag_zpt, num_bulges, length_gal, eff_rad_1,0,0, AR_1, PA1, 0, 0, bg_level)
   
    '''Runs galfit inline and creates an output file'''
    # But first check if the file exists
    if os.path.isfile(prefix+'imaging/galfit.feedme_'+str(id)):
        pass
    else:
        write_fails_txt(prefix, id, num, 'NoFeedme')
        rm_files(prefix, sdss)
        
        return 
    
    run_galfit(str(id), prefix)
    
    
    output=prefix+'imaging/out_'+str(id)+'.fits'
    try:
        #This is the output result fit from Galfit:
        out=pyfits.open(output)
    except FileNotFoundError:
        #sometimes Galfit fails and you need to move on with your life.
        #It should really work for the majority of cases though.
        write_fails_txt(prefix, id, num, 'NoGalfitOut')
        #rm_files(prefix,id)

        
        return 
    
    
    # This measures useful parameters from the galfit output file
    h = galfit_params(id,num_bulges, prefix)
    if h[2]==999:
        rm_files(prefix, id)
        write_fails_txt(prefix, id, num, 'Galfitparams')
        return 

    
    #Re-adjust so the important parameters are now Galfit outputs:
    sep=h[0]
    flux_r=h[1]
    ser=h[2]# you can use many of these predictors to check the output, but the only important one is the sersic profile
    inc=h[3]


    

    

    #call statmorph
  
    try:
        segmap = 1*segmap
        
        
        source_morphs = statmorph.source_morphology(camera_data, segmap, gain=100, skybox_size=10)#,weightmap=camera_data_sigma)#psf=psf,
        
    except ValueError:
        rm_files(prefix, id)
        write_fails_txt(prefix, id, num, 'Statmorph')
        
        return
        
  
  
    try:
        n = img_assy_shape(segmap,id)
    except IndexError:
        rm_files(prefix, id)
        write_fails_txt(prefix, id, num, 'As')
        return

        

        
    
    morph=source_morphs[0]
    if verbose:
        make_figure(morph)
        plt.show()

    
    gini=morph.gini
    m20=morph.m20
    con=morph.concentration
    asy=morph.asymmetry
    clu=morph.smoothness
    
    ser_stat = morph.sersic_n
    
    s_n = morph.sn_per_pixel
    
    # Delete a lot of the saved files:
    rm_files(prefix, id)
    

    n_stat = morph.shape_asymmetry

    # what it was:
    write_string = str(id)+'\t'+str(sep)+'\t'+str(flux_r)+'\t'+str(gini)+'\t'+str(m20)+'\t'+str(con)+'\t'+str(asy)+'\t'+str(clu)+'\t'+str(ser)+'\t'+str(n)+'\t'+str(inc)+'\t'+str(s_n)+'\t'+str(ser_stat)+'\t'+str(n_stat)+'\n'
    
    #write_string = str(sdss)+'\t'+str(gini)+'\t'+str(m20)+'\t'+str(con)+'\t'+str(asy)+'\t'+str(clu)+'\t'+str(ser_stat)+'\t'+str(n)+'\t'+str(s_n)+'\n'

    file1=open(prefix+'Tables/SDSS_DR12_predictors_MPI_'+str(num)+'.txt','a+')
    #if m ==0 and num==0:
    #    file1.write('ID'+'\t'+'Sep'+'\t'+'Flux_ratio'+'\t'+'Gini'+'\t'+'M20'+'\t'+'C'+'\t'+'A'+'\t'+'S'+'\t'+'Sersic n'+'\t'+'A_s'+'\t'+'inc'+'\t'+'S/N'+'\t'+'Sersic n statmorph'+'\t'+'A_s_stat'+'\n')
    file1.write(write_string)

    return camera_data, segmap, gini, m20, con, asy, clu, ser, n, inc, s_n
#args = sdss_list[1], ra_list[1], dec_list[1]
#out = mainFunc(sdss_list[1], ra_list[1], dec_list[1])
#print('output', out)
#print('writing here', prefix+'Tables/SDSS_DR12_predictors_MPI.txt')
#file1=open(prefix+'Tables/SDSS_DR12_predictors_MPI.txt','w')
#file1.write('ID'+'\t'+'Gini'+'\t'+'M20'+'\t'+'C'+'\t'+'A'+'\t'+'S'+'\t'+'Sersic n'+'\t'+'A_s'+'\t'+'S/N'+'\n')
#file1.write(out)
#file1.close()



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#          Also including all of the stuff from 'utilities.py' on Cannon
#          This is stuff like write_galfit_feedme, tools to run source
#          extractor and galfit and statmorph, etc....
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
import astropy.io.fits as pyfits
import subprocess
import numpy as np
import shlex
import skimage

def normalize(image, m=None, M=None):
    if m is None:
        m = np.min(image)
    if M is None:
        M = np.max(image)

    retval = (image-m) / (M-m)
    retval[image <= m] = 0.0
    retval[image >= M] = 1.0

    return retval
def _get_ax(fig, row, col, nrows, ncols, wpanel, hpanel, htop, eps, wfig, hfig):
    x_ax = (col+1)*eps + col*wpanel
    y_ax = eps + (nrows-1-row)*(hpanel+htop)
    return fig.add_axes([x_ax/wfig, y_ax/hfig, wpanel/wfig, hpanel/hfig])

def make_figure(morph):
    """
    Creates a figure analogous to Fig. 4 from Rodriguez-Gomez et al. (2019)
    for a given ``SourceMorphology`` object.
    
    Parameters
    ----------
    morph : ``statmorph.SourceMorphology``
        An object containing the morphological measurements of a single
        source.
    Returns
    -------
    fig : ``matplotlib.figure.Figure``
        The figure.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors
    import matplotlib.cm

    if not isinstance(morph, statmorph.SourceMorphology):
        raise TypeError('Input must be of type SourceMorphology.')

    #assert morph.flag_catastrophic == 0  # some cases are not even worth plotting

    # I'm tired of dealing with plt.add_subplot, plt.subplots, plg.GridSpec,
    # plt.subplot2grid, etc. and never getting the vertical and horizontal
    # inter-panel spacings to have the same size, so instead let's do
    # everything manually:
    nrows = 2
    ncols = 4
    wpanel = 4.0  # panel width
    hpanel = 4.0  # panel height
    htop = 0.05*nrows*hpanel  # top margin and vertical space between panels
    eps = 0.005*nrows*hpanel  # all other margins
    wfig = ncols*wpanel + (ncols+1)*eps  # total figure width
    hfig = nrows*(hpanel+htop) + eps  # total figure height
    fig = plt.figure(figsize=(wfig, hfig))

    # For drawing circles/ellipses
    theta_vec = np.linspace(0.0, 2.0*np.pi, 200)

    # Add black to pastel colormap
    cmap_orig = matplotlib.cm.Pastel1
    colors = ((0.0, 0.0, 0.0), *cmap_orig.colors)
    cmap = matplotlib.colors.ListedColormap(colors)

    # Get some general info about the image
    image = np.float64(morph._cutout_stamp_maskzeroed)  # skimage wants double
    ny, nx = image.shape
    xc, yc = morph._xc_stamp, morph._yc_stamp  # centroid
    xca, yca = morph._asymmetry_center  # asym. center
    xcs, ycs = morph._sersic_model.x_0.value, morph._sersic_model.y_0.value  # Sersic center

    ##################
    # Original image #
    ##################
    ax = _get_ax(fig, 0, 0, nrows, ncols, wpanel, hpanel, htop, eps, wfig, hfig)
    ax.imshow(image, cmap='gray', origin='lower',
              norm=simple_norm(image, stretch='log', log_a=10000))
    ax.plot(xc, yc, 'go', markersize=5, label='Centroid')
    R = np.sqrt(nx**2 + ny**2)
    theta = morph.orientation_centroid
    x0, x1 = xc - R*np.cos(theta), xc + R*np.cos(theta)
    y0, y1 = yc - R*np.sin(theta), yc + R*np.sin(theta)
    ax.plot([x0, x1], [y0, y1], 'g--', lw=1.5, label='Major Axis (Centroid)')
    ax.plot(xca, yca, 'bo', markersize=5, label='Asym. Center')
    R = np.sqrt(nx**2 + ny**2)
    theta = morph.orientation_asymmetry
    x0, x1 = xca - R*np.cos(theta), xca + R*np.cos(theta)
    y0, y1 = yca - R*np.sin(theta), yca + R*np.sin(theta)
    ax.plot([x0, x1], [y0, y1], 'b--', lw=1.5, label='Major Axis (Asym.)')
    # Half-radius ellipse
    a = morph.rhalf_ellip
    b = a / morph.elongation_asymmetry
    theta = morph.orientation_asymmetry
    xprime, yprime = a*np.cos(theta_vec), b*np.sin(theta_vec)
    x = xca + (xprime*np.cos(theta) - yprime*np.sin(theta))
    y = yca + (xprime*np.sin(theta) + yprime*np.cos(theta))
    ax.plot(x, y, 'b', label='Half-Light Ellipse')
    # Some text
    text = 'flag = %d\nEllip. (Centroid) = %.4f\nEllip. (Asym.) = %.4f' % (
        morph.flag, morph.ellipticity_centroid, morph.ellipticity_asymmetry)
    ax.text(0.034, 0.966, text,
            horizontalalignment='left', verticalalignment='top',
            transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=1.0, boxstyle='round'))
    # Finish plot
    ax.legend(loc=4, fontsize=12, facecolor='w', framealpha=1.0, edgecolor='k')
    ax.set_xlim(-0.5, nx-0.5)
    ax.set_ylim(-0.5, ny-0.5)
    ax.set_title('Original Image (Log Stretch)', fontsize=14)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ##############
    # Sersic fit #
    ##############
    ax = _get_ax(fig, 0, 1, nrows, ncols, wpanel, hpanel, htop, eps, wfig, hfig)
    y, x = np.mgrid[0:ny, 0:nx]
    sersic_model = morph._sersic_model(x, y)
    # Add background noise (for realism)
    if morph.sky_sigma > 0:
        sersic_model += np.random.normal(scale=morph.sky_sigma, size=(ny, nx))
    ax.imshow(sersic_model, cmap='gray', origin='lower',
              norm=simple_norm(image, stretch='log', log_a=10000))
    ax.plot(xcs, ycs, 'ro', markersize=5, label='Sérsic Center')
    R = np.sqrt(nx**2 + ny**2)
    theta = morph.sersic_theta
    x0, x1 = xcs - R*np.cos(theta), xcs + R*np.cos(theta)
    y0, y1 = ycs - R*np.sin(theta), ycs + R*np.sin(theta)
    ax.plot([x0, x1], [y0, y1], 'r--', lw=1.5, label='Major Axis (Sérsic)')
    # Half-radius ellipse
    a = morph.sersic_rhalf
    b = a * (1.0 - morph.sersic_ellip)
    xprime, yprime = a*np.cos(theta_vec), b*np.sin(theta_vec)
    x = xcs + (xprime*np.cos(theta) - yprime*np.sin(theta))
    y = ycs + (xprime*np.sin(theta) + yprime*np.cos(theta))
    ax.plot(x, y, 'r', label='Half-Light Ellipse (Sérsic)')
    # Some text
    text = ('flag_sersic = %d' % (morph.flag_sersic,) + '\n' +
            'Ellip. (Sérsic) = %.4f' % (morph.sersic_ellip,) + '\n' +
            r'$n = %.4f$' % (morph.sersic_n,))
    ax.text(0.034, 0.966, text,
            horizontalalignment='left', verticalalignment='top',
            transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=1.0, boxstyle='round'))
    # Finish plot
    ax.legend(loc=4, fontsize=12, facecolor='w', framealpha=1.0, edgecolor='k')
    ax.set_title('Sérsic Model + Noise', fontsize=14)
    ax.set_xlim(-0.5, nx-0.5)
    ax.set_ylim(-0.5, ny-0.5)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ###################
    # Sersic residual #
    ###################
    ax = _get_ax(fig, 0, 2, nrows, ncols, wpanel, hpanel, htop, eps, wfig, hfig)
    y, x = np.mgrid[0:ny, 0:nx]
    sersic_res = morph._cutout_stamp_maskzeroed - morph._sersic_model(x, y)
    sersic_res[morph._mask_stamp] = 0.0
    ax.imshow(sersic_res, cmap='gray', origin='lower',
              norm=simple_norm(sersic_res, stretch='linear'))
    ax.set_title('Sérsic Residual, ' + r'$I - I_{\rm model}$', fontsize=14)
    ax.set_xlim(-0.5, nx-0.5)
    ax.set_ylim(-0.5, ny-0.5)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ######################
    # Asymmetry residual #
    ######################
    ax = _get_ax(fig, 0, 3, nrows, ncols, wpanel, hpanel, htop, eps, wfig, hfig)
    # Rotate image around asym. center
    # (note that skimage expects pixel positions at lower-left corners)
    image_180 = skimage.transform.rotate(image, 180.0, center=(xca, yca))
    image_res = image - image_180
    # Apply symmetric mask
    mask = morph._mask_stamp.copy()
    mask_180 = skimage.transform.rotate(mask, 180.0, center=(xca, yca))
    mask_180 = mask_180 >= 0.5  # convert back to bool
    mask_symmetric = mask | mask_180
    image_res = np.where(~mask_symmetric, image_res, 0.0)
    ax.imshow(image_res, cmap='gray', origin='lower',
              norm=simple_norm(image_res, stretch='linear'))
    ax.set_title('Asymmetry Residual, ' + r'$I - I_{180}$', fontsize=14)
    ax.set_xlim(-0.5, nx-0.5)
    ax.set_ylim(-0.5, ny-0.5)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ###################
    # Original segmap #
    ###################
    ax = _get_ax(fig, 1, 0, nrows, ncols, wpanel, hpanel, htop, eps, wfig, hfig)
    ax.imshow(image, cmap='gray', origin='lower',
              norm=simple_norm(image, stretch='log', log_a=10000))
    # Show original segmap
    contour_levels = [0.5]
    contour_colors = [(0, 0, 0)]
    segmap_stamp = morph._segmap.data[morph._slice_stamp]
    Z = np.float64(segmap_stamp == morph.label)
    ax.contour(Z, contour_levels, colors=contour_colors, linewidths=1.5)
    # Show skybox
    xmin = morph._slice_skybox[1].start
    ymin = morph._slice_skybox[0].start
    xmax = morph._slice_skybox[1].stop - 1
    ymax = morph._slice_skybox[0].stop - 1
    ax.plot(np.array([xmin, xmax, xmax, xmin, xmin]),
            np.array([ymin, ymin, ymax, ymax, ymin]),
            'b', lw=1.5, label='Skybox')
    # Some text
    text = ('Sky Mean = %.4e' % (morph.sky_mean,) + '\n' +
            'Sky Median = %.4e' % (morph.sky_median,) + '\n' +
            'Sky Sigma = %.4e' % (morph.sky_sigma,))
    ax.text(0.034, 0.966, text,
            horizontalalignment='left', verticalalignment='top',
            transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=1.0, boxstyle='round'))
    # Finish plot
    ax.set_xlim(-0.5, nx-0.5)
    ax.set_ylim(-0.5, ny-0.5)
    ax.legend(loc=4, fontsize=12, facecolor='w', framealpha=1.0, edgecolor='k')
    ax.set_title('Original Segmap', fontsize=14)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ###############
    # Gini segmap #
    ###############
    ax = _get_ax(fig, 1, 1, nrows, ncols, wpanel, hpanel, htop, eps, wfig, hfig)
    ax.imshow(image, cmap='gray', origin='lower',
              norm=simple_norm(image, stretch='log', log_a=10000))
    # Show Gini segmap
    contour_levels = [0.5]
    contour_colors = [(0, 0, 0)]
    Z = np.float64(morph._segmap_gini)
    ax.contour(Z, contour_levels, colors=contour_colors, linewidths=1.5)
    # Some text
    text = r'$\left\langle {\rm S/N} \right\rangle = %.4f$' % (morph.sn_per_pixel,)
    ax.text(0.034, 0.966, text, fontsize=12,
            horizontalalignment='left', verticalalignment='top',
            transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=1.0, boxstyle='round'))
    text = (r'$G = %.4f$' % (morph.gini,) + '\n' +
            r'$M_{20} = %.4f$' % (morph.m20,) + '\n' +
            r'$F(G, M_{20}) = %.4f$' % (morph.gini_m20_bulge,) + '\n' +
            r'$S(G, M_{20}) = %.4f$' % (morph.gini_m20_merger,))
    ax.text(0.034, 0.034, text, fontsize=12,
            horizontalalignment='left', verticalalignment='bottom',
            transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=1.0, boxstyle='round'))
    text = (r'$C = %.4f$' % (morph.concentration,) + '\n' +
            r'$A = %.4f$' % (morph.asymmetry,) + '\n' +
            r'$S = %.4f$' % (morph.smoothness,))
    ax.text(0.966, 0.034, text, fontsize=12,
            horizontalalignment='right', verticalalignment='bottom',
            transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=1.0, boxstyle='round'))
    # Finish plot
    ax.set_xlim(-0.5, nx-0.5)
    ax.set_ylim(-0.5, ny-0.5)
    ax.set_title('Gini Segmap', fontsize=14)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ####################
    # Watershed segmap #
    ####################
    ax = _get_ax(fig, 1, 2, nrows, ncols, wpanel, hpanel, htop, eps, wfig, hfig)
    labeled_array, peak_labels, xpeak, ypeak = morph._watershed_mid
    labeled_array_plot = (labeled_array % (cmap.N-1)) + 1
    labeled_array_plot[labeled_array == 0] = 0.0  # background is black
    ax.imshow(labeled_array_plot, cmap=cmap, origin='lower',
              norm=matplotlib.colors.NoNorm())
    sorted_flux_sums, sorted_xpeak, sorted_ypeak = morph._intensity_sums
    if len(sorted_flux_sums) > 0:
        ax.plot(sorted_xpeak[0], sorted_ypeak[0], 'bo', markersize=2,
                label='First Peak')
    if len(sorted_flux_sums) > 1:
        ax.plot(sorted_xpeak[1], sorted_ypeak[1], 'ro', markersize=2,
                label='Second Peak')
    # Some text
    text = (r'$M = %.4f$' % (morph.multimode,) + '\n' +
            r'$I = %.4f$' % (morph.intensity,) + '\n' +
            r'$D = %.4f$' % (morph.deviation,))
    ax.text(0.034, 0.034, text, fontsize=12,
            horizontalalignment='left', verticalalignment='bottom',
            transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=1.0, boxstyle='round'))
    ax.legend(loc=4, fontsize=12, facecolor='w', framealpha=1.0, edgecolor='k')
    ax.set_title('Watershed Segmap (' + r'$I$' + ' statistic)', fontsize=14)
    ax.set_xlim(-0.5, nx-0.5)
    ax.set_ylim(-0.5, ny-0.5)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ##########################
    # Shape asymmetry segmap #
    ##########################
    ax = _get_ax(fig, 1, 3, nrows, ncols, wpanel, hpanel, htop, eps, wfig, hfig)
    ax.imshow(morph._segmap_shape_asym, cmap='gray', origin='lower')
    ax.plot(xca, yca, 'bo', markersize=5, label='Asym. Center')
    r = morph.rpetro_circ
    ax.plot(xca + r*np.cos(theta_vec), yca + r*np.sin(theta_vec), 'b',
            label=r'$r_{\rm petro, circ}$')
    r = morph.rpetro_ellip
    ax.plot(xca + r*np.cos(theta_vec), yca + r*np.sin(theta_vec), 'r',
            label=r'$r_{\rm petro, ellip}$')
    r = morph.rmax_circ
    ax.plot(xca + r*np.cos(theta_vec),
            yca + r*np.sin(theta_vec),
            'c', lw=1.5, label=r'$r_{\rm max}$')
    text = (r'$A_S = %.4f$' % (morph.shape_asymmetry,))
    ax.text(0.034, 0.034, text, fontsize=12,
            horizontalalignment='left', verticalalignment='bottom',
            transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=1.0, boxstyle='round'))
    ax.legend(loc=4, fontsize=12, facecolor='w', framealpha=1.0, edgecolor='k')
    ax.set_xlim(-0.5, nx-0.5)
    ax.set_ylim(-0.5, ny-0.5)
    ax.set_title('Shape Asymmetry Segmap', fontsize=14)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    fig.subplots_adjust(left=eps/wfig, right=1-eps/wfig, bottom=eps/hfig,
                        top=1.0-htop/hfig, wspace=eps/wfig, hspace=htop/hfig)

    return fig

def make_figure_mine(morph):
    """                                                                                                              \
                                                                                                                      
    Creates a figure analogous to Fig. 4 from Rodriguez-Gomez et al. (2018)                                          \
                                                                                                                      
    for a given ``SourceMorphology`` object.                                                                         \
                                                                                                                      
                                                                                                                     \
                                                                                                                      
    Parameters                                                                                                       \
                                                                                                                      
    ----------                                                                                                       \
                                                                                                                      
    morph : ``statmorph.SourceMorphology``                                                                           \
                                                                                                                      
        An object containing the morphological measurements of a single                                              \
                                                                                                                      
        source.                                                                                                      \
                                                                                                                      
    Returns                                                                                                          \
                                                                                                                      
    -------                                                                                                          \
                                                                                                                      
    fig : ``matplotlib.figure.Figure``                                                                               \
                                                                                                                      
        The figure.                                                                                                  \
                                                                                                                      
    """
    # I'm tired of dealing with plt.add_subplot, plt.subplots, plg.GridSpec,                                         \
                                                                                                                      
    # plt.subplot2grid, etc. and never getting the vertical and horizontal                                           \
                                                                                                                      
    # inter-panel spacings to have the same size, so instead let's do
    # everything manually:                                                                                                                                     \
                                                                                                                                                                
    nrows = 2
    ncols = 4
    wpanel = 4.0  # panel width                                                                                                                                \
                                                                                                                                                                
    hpanel = 4.0  # panel height                                                                                                                               \
                                                                                                                                                                
    htop = 0.05*nrows*hpanel  # top margin and vertical space between panels                                                                                   \
                                                                                                                                                                
    eps = 0.005*nrows*hpanel  # all other margins                                                                                                              \
                                                                                                                                                                
    wfig = ncols*wpanel + (ncols+1)*eps  # total figure width                                                                                                  \
                                                                                                                                                                
    hfig = nrows*(hpanel+htop) + eps  # total figure height                                                                                                    \
                                                                                                                                                                
    fig = plt.figure(figsize=(wfig, hfig))

    # For drawing circles/ellipses                                                                                                                             \
                                                                                                                                                                
    theta_vec = np.linspace(0.0, 2.0*np.pi, 200)

    # Add black to pastel colormap                                                                                                                             \
                                                                                                                                                                
    cmap_orig = matplotlib.cm.Pastel1
    colors = ((0.0, 0.0, 0.0), cmap_orig.colors)
    cmap = matplotlib.colors.ListedColormap(colors)



    # Get some general info about the image                                                                                                                    \
                                                                                                                                                                
    image = np.float64(morph._cutout_stamp_maskzeroed)  # skimage wants double                                                                                 \
                                                                                                                                                                
    ny, nx = image.shape
    m = np.min(image)
    M = np.max(image)

    xc, yc = morph._xc_stamp, morph._yc_stamp  # centroid                                                                                                      \
                                                                                                                                                                
    xca, yca = morph._asymmetry_center  # asym. centroid                                                                                                       \
                                                                                                                                                                

    ##################                                                                                                                                         \
                                                                                                                                                                
    # Original image #                                                                                                                                         \
                                                                                                                                                                
    ##################                                                                                                                                         \
                                                                                                                                                                
    ax = get_ax(fig, 0, 0, nrows, ncols, wpanel, hpanel, htop, eps, wfig, hfig)
    ax.imshow(image, cmap='gray', origin='lower',
              norm=simple_norm(image, stretch='log', log_a=10000))
    ax.plot(xc, yc, 'go', markersize=5, label='Centroid')
    R = float(nx**2 + ny**2)
    theta = morph.orientation_centroid
    x0, x1 = xc - R*np.cos(theta), xc + R*np.cos(theta)
    y0, y1 = yc - R*np.sin(theta), yc + R*np.sin(theta)
    ax.plot([x0, x1], [y0, y1], 'g--', lw=1.5, label='Major Axis (Centroid)')
    ax.plot(xca, yca, 'bo', markersize=5, label='Asym. Center')
    R = float(nx**2 + ny**2)
    theta = morph.orientation_asymmetry
    x0, x1 = xca - R*np.cos(theta), xca + R*np.cos(theta)
    y0, y1 = yca - R*np.sin(theta), yca + R*np.sin(theta)
    ax.plot([x0, x1], [y0, y1], 'b--', lw=1.5, label='Major Axis (Asym.)')
    # Half-radius ellipse                                                                                                                                      \
                                                                                                                                                                
    a = morph.rhalf_ellip
    b = a / morph.elongation_asymmetry
    theta = morph.orientation_asymmetry
    xprime, yprime = a*np.cos(theta_vec), b*np.sin(theta_vec)
    x = xca + (xprime*np.cos(theta) - yprime*np.sin(theta))
    y = yca + (xprime*np.sin(theta) + yprime*np.cos(theta))
    ax.plot(x, y, 'b', label='Half-Light Ellipse')
    # Some text                                                                                                                                                \
                                                                                                                                                                
    text = 'flag = %d\nEllip. (Centroid) = %.4f\nEllip. (Asym.) = %.4f' % (
        morph.flag, morph.ellipticity_centroid, morph.ellipticity_asymmetry)
    ax.text(0.034, 0.966, text,
        horizontalalignment='left', verticalalignment='top',
        transform=ax.transAxes,
        bbox=dict(facecolor='white', alpha=1.0, boxstyle='round'))
    # Finish plot                                                                                                                                              \
                                                                                                                                                                
    ax.legend(loc=4, fontsize=12, facecolor='w', framealpha=1.0, edgecolor='k')
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_title('Original Image (Log Stretch)', fontsize=14)
    #ax.get_xaxis().set_visible(False)                                                                                                                         \
                                                                                                                                                                
    #ax.get_yaxis().set_visible(False)                                                                                                                          
    ##############                                                                                                                                             \
                                                                                                                                                                
    # Sersic fit #                                                                                                                                             \
                                                                                                                                                                
    ##############                                                                                                                                             \
    ax = get_ax(fig, 0, 1, nrows, ncols, wpanel, hpanel, htop, eps, wfig, hfig)
    y, x = np.mgrid[0:ny, 0:nx]
    sersic_model = morph._sersic_model(x, y)
    # Add background noise (for realism)                                                                                                                       \
                                                                                                                                                                
    if morph.sky_sigma > 0:
        sersic_model += np.random.normal(scale=morph.sky_sigma, size=(ny, nx))
    ax.imshow(sersic_model, cmap='gray', origin='lower',
             norm=simple_norm(sersic_model, stretch='log', log_a=10000))
    
    # Sersic center (within postage stamp)                                                                                                                     \
                                                                                                                                                                
    xcs, ycs = morph._sersic_model.x_0.value, morph._sersic_model.y_0.value
    ax.plot(xcs, ycs, 'ro', markersize=5, label='Sersic Center')
    R = float(nx**2 + ny**2)
    theta = morph.sersic_theta
    x0, x1 = xcs - R*np.cos(theta), xcs + R*np.cos(theta)
    y0, y1 = ycs - R*np.sin(theta), ycs + R*np.sin(theta)
    ax.plot([x0, x1], [y0, y1], 'r--', lw=1.5, label='Major Axis (Sersic)')
    # Half-radius ellipse                                                                                                                                      \
                                                                                                                                                                
    a = morph.sersic_rhalf
    b = a * (1.0 - morph.sersic_ellip)
    xprime, yprime = a*np.cos(theta_vec), b*np.sin(theta_vec)
    x = xc + (xprime*np.cos(theta) - yprime*np.sin(theta))
    y = yc + (xprime*np.sin(theta) + yprime*np.cos(theta))
    ax.plot(x, y, 'r', label='Half-Light Ellipse (Sersic)')
    # Some text                                                                                                                                                \
    text = ('flag_sersic = %d' % (morph.flag_sersic) + '\n' +
            'Ellip. (Sersic) = %.4f' % (morph.sersic_ellip) + '\n' +
            r'$n = %.4f$' % (morph.sersic_n))
    ax.text(0.034, 0.966, text,
        horizontalalignment='left', verticalalignment='top',
        transform=ax.transAxes,
        bbox=dict(facecolor='white', alpha=1.0, boxstyle='round'))
    # Finish plot                                                                                                                                              \
                                                                                                                                                                
    ax.legend(loc=4, fontsize=12, facecolor='w', framealpha=1.0, edgecolor='k')
    ax.set_title('Sersic Model + Noise', fontsize=14)
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ###################                                                                                                                                        \
                                                                                                                                                                
    # Sersic residual #                                                                                                                                        \
                                                                                                                                                                
    ###################                                                                                                                                        \
    ax = get_ax(fig, 0, 2, nrows, ncols, wpanel, hpanel, htop, eps, wfig, hfig)
    y, x = np.mgrid[0:ny, 0:nx]
    sersic_res = morph._cutout_stamp_maskzeroed - morph._sersic_model(x, y)
    sersic_res[morph._mask_stamp] = 0.0
    ax.imshow(normalize(sersic_res), cmap='gray', origin='lower')
    ax.set_title('Sersic Residual, ' + r'$I - I_{\rm model}$', fontsize=14)
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ######################                                                                                                                                     \
                                                                                                                                                                
    # Asymmetry residual #                                                                                                                                     \
                                                                                                                                                                
    ######################                                                                                                                                     \
                                                                                                                                                                
    ax = get_ax(fig, 0, 3, nrows, ncols, wpanel, hpanel, htop, eps, wfig, hfig)
    # Rotate image around asym. center                                                                                                                         \
                                                                                                                                                                
    image_180 = skimage.transform.rotate(image, 180.0, center=(xca, yca))
    image_res = image - image_180
    # Apply symmetric mask                                                                                                                                     \
                                                                                                                                                                
    mask = morph._mask_stamp.copy()
    mask_180 = skimage.transform.rotate(mask, 180.0, center=(xca, yca))
    mask_180 = mask_180 >= 0.5  # convert back to bool                                                                                                         \
                                                                                                                                                                
    mask_symmetric = mask | mask_180
    image_res = np.where(~mask_symmetric, image_res, 0.0)
    ax.imshow(normalize(image_res), cmap='gray', origin='lower')
    ax.set_title('Asymmetry Residual, ' + r'$I - I_{180}$', fontsize=14)
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ###################                                                                                                                                        \
                                                                                                                                                                
    # Original segmap #                                                                                                                                        \
                                                                                                                                                                
    ###################                                                                                                                                        \
                                                                                                                                                                
    ax = get_ax(fig, 1, 0, nrows, ncols, wpanel, hpanel, htop, eps, wfig, hfig)
    ax.imshow(image, cmap='gray', origin='lower',norm=simple_norm(image, stretch='log', log_a=10000))
    # Show original segmap                                                                                                                                     \
                                                                                                                                                                
    contour_levels = [0.5]
    contour_colors = [(0,0,0)]
    segmap_stamp = morph._segmap.data[morph._slice_stamp]
    Z = np.float64(segmap_stamp == morph.label)
    C = ax.contour(Z, contour_levels, colors=contour_colors, linewidths=1.5)
    # Show skybox                                                                                                                                              \
    xmin = morph._slice_skybox[1].start
    ymin = morph._slice_skybox[0].start
    xmax = morph._slice_skybox[1].stop - 1
    ymax = morph._slice_skybox[0].stop - 1
    ax.plot(np.array([xmin, xmax, xmax, xmin, xmin]) + 0.5,
            np.array([ymin, ymin, ymax, ymax, ymin]) + 0.5,
            'b', lw=1.5, label='Skybox')
    # Some text                                                                                                                                                \
                                                                                                                                                                
    text = ('Sky Mean = %.4f' % (morph.sky_mean) + '\n' +
            'Sky Median = %.4f' % (morph.sky_median) + '\n' +
            'Sky Sigma = %.4f' % (morph.sky_sigma)+'\n'+
            'Image min = %.4f' % (m) + '\n' +
            'Image max = %.4f' % (M))
    ax.text(0.034, 0.966, text,
        horizontalalignment='left', verticalalignment='top',
        transform=ax.transAxes,
        bbox=dict(facecolor='white', alpha=1.0, boxstyle='round'))
    # Finish plot                                                                                                                                              \
                                                                                                                                                                
    ax.legend(loc=4, fontsize=12, facecolor='w', framealpha=1.0, edgecolor='k')
    ax.set_title('Original Segmap', fontsize=14)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ###############                                                                                                                                            \
                                                                                                                                                                
    # Gini segmap #                                                                                                                                            \
                                                                                                                                                                
    ###############                                                                                                                                            \
                                                                                                                                                                
    ax = get_ax(fig, 1, 1, nrows, ncols, wpanel, hpanel, htop, eps, wfig, hfig)
    ax.imshow(image,
              cmap='gray', origin='lower', norm=simple_norm(image, stretch='log', log_a=10000))
    # Show Gini segmap                                                                                                                                         \
                                                                                                                                                                
    contour_levels = [0.5]
    contour_colors = [(1,0,1)]
    Z = np.float64(morph._segmap_gini)
    C = ax.contour(Z, contour_levels, colors=contour_colors, linewidths=1.5)
    # Some text                                                                                                                                                \
                                                                                                                                                                
    text = r'$\left\langle {\rm S/N} \right\rangle = %.4f$' % (morph.sn_per_pixel)
    ax.text(0.034, 0.966, text, fontsize=12,
        horizontalalignment='left', verticalalignment='top',
        transform=ax.transAxes,
        bbox=dict(facecolor='white', alpha=1.0, boxstyle='round'))
    text = (r'$G = %.4f$' % (morph.gini) + '\n' +
            r'$M_{20} = %.4f$' % (morph.m20) + '\n' +
            r'$F(G, M_{20}) = %.4f$' % (morph.gini_m20_bulge) + '\n' +
            r'$S(G, M_{20}) = %.4f$' % (morph.gini_m20_merger))
    ax.text(0.034, 0.034, text, fontsize=12,
        horizontalalignment='left', verticalalignment='bottom',
        transform=ax.transAxes,
        bbox=dict(facecolor='white', alpha=1.0, boxstyle='round'))
    text = (r'$C = %.4f$' % (morph.concentration) + '\n' +
            r'$A = %.4f$' % (morph.asymmetry) + '\n' +
            r'$S = %.4f$' % (morph.smoothness))
    ax.text(0.966, 0.034, text, fontsize=12,
        horizontalalignment='right', verticalalignment='bottom',
        transform=ax.transAxes,
        bbox=dict(facecolor='white', alpha=1.0, boxstyle='round'))
    # Finish plot                                                                                                                                              \
                                                                                                                                                                
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_title('Gini Segmap', fontsize=14)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ####################                                                                                                                                       \
                                                                                                                                                                
    # Watershed segmap #                                                                                                                                       \
                                                                                                                                                                
    ####################                                                                                                                                       \
    ax = get_ax(fig, 1, 2, nrows, ncols, wpanel, hpanel, htop, eps, wfig, hfig)
    labeled_array, peak_labels, xpeak, ypeak = morph._watershed_mid
    labeled_array_plot = (labeled_array % (cmap.N-1)) + 1
    labeled_array_plot[labeled_array == 0] = 0.0  # background is black                                                                                        \
                                                                                                                                                                
    ax.imshow(labeled_array_plot, cmap=cmap, origin='lower',
              norm=matplotlib.colors.NoNorm())
    sorted_flux_sums, sorted_xpeak, sorted_ypeak = morph._intensity_sums
    if len(sorted_flux_sums) > 0:
        ax.plot(sorted_xpeak[0] + 0.5, sorted_ypeak[0] + 0.5, 'bo', markersize=2, label='First Peak')
    if len(sorted_flux_sums) > 1:
        ax.plot(sorted_xpeak[1] + 0.5, sorted_ypeak[1] + 0.5, 'ro', markersize=2, label='Second Peak')
    # Some text                                                                                                                                                \
                                                                                                                                                                
    text = (r'$M = %.4f$' % (morph.multimode) + '\n' +
            r'$I = %.4f$' % (morph.intensity) + '\n' +
            r'$D = %.4f$' % (morph.deviation))
    ax.text(0.034, 0.034, text, fontsize=12,
        horizontalalignment='left', verticalalignment='bottom',
        transform=ax.transAxes,
        bbox=dict(facecolor='white', alpha=1.0, boxstyle='round'))
    ax.legend(loc=4, fontsize=12, facecolor='w', framealpha=1.0, edgecolor='k')
    ax.set_title('Watershed Segmap (' + r'$I$' + ' statistic)', fontsize=14)
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ##########################                                                                                                                                 \
                                                                                                                                                                
    # Shape asymmetry segmap #                                                                                                                                 \
                                                                                                                                                                
    ##########################                                                                                                                                 \
                                                                                                                                                                
    ax = get_ax(fig, 1, 3, nrows, ncols, wpanel, hpanel, htop, eps, wfig, hfig)
    ax.imshow(morph._segmap_shape_asym, cmap='gray', origin='lower')
    ax.plot(xca, yca, 'bo', markersize=5, label='Asym. Center')
    r = morph.rpetro_circ
    ax.plot(xca + r*np.cos(theta_vec), yca + r*np.sin(theta_vec), 'b', label=r'$r_{\rm petro, circ}$')
    r = morph.rpetro_ellip
    ax.plot(xca + r*np.cos(theta_vec), yca + r*np.sin(theta_vec), 'r', label=r'$r_{\rm petro, ellip}$')
    r = morph.rmax_circ
    ax.plot(np.floor(xca) + r*np.cos(theta_vec), np.floor(yca) + r*np.sin(theta_vec), 'c', lw=1.5, label=r'$r_{\rm max}$')
    # ~ r = morph._petro_extent_flux * morph.rpetro_ellip                                                                                                      \
                                                                                                                                                                
    # ~ ax.plot(xca + r*np.cos(theta_vec), yca + r*np.sin(theta_vec), 'r--', label='%g*rpet_ellip' % (morph._petro_extent_flux))                               \
                                                                                                                                                                
    text = (r'$A_S = %.4f$' % (morph.shape_asymmetry))
    ax.text(0.034, 0.034, text, fontsize=12,
        horizontalalignment='left', verticalalignment='bottom',
        transform=ax.transAxes,
        bbox=dict(facecolor='white', alpha=1.0, boxstyle='round'))
    ax.legend(loc=4, fontsize=12, facecolor='w', framealpha=1.0, edgecolor='k')
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_title('Shape Asymmetry Segmap', fontsize=14)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # defaults: left = 0.125, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.2, hspace = 0.2                                                                 \
                                                                                                                                                                
    fig.subplots_adjust(left=eps/wfig, right=1-eps/wfig, bottom=eps/hfig, top=1.0-htop/hfig, wspace=eps/wfig, hspace=htop/hfig)

    #fig.savefig('test_segmap.png', dpi=150)                                                                                                                   \
                                                                                                                                                                

    return fig
'''Prepares the default file for source extractor to read (I played with the parameters to the point where they work well:'''
def write_sex_default(name, prefix):
    try:
        file = open(prefix+'imaging/default_'+name+'.sex', "w")
    except IOError:
        print('Path is Incorrect')
    file.write('# Default configuration file for SExtractor 2.5.0'+'\n')

    file.write('CATALOG_NAME     imaging/test_'+name+'.cat       # name of the output catalog'+'\n')
    file.write('CATALOG_TYPE     ASCII_HEAD     # '+'\n')
    file.write('PARAMETERS_NAME  imaging/default.param  # name of the file containing catalog contents'+'\n')

    file.write('#------------------------------- Extraction ----------------------------------'+'\n')

    file.write('DETECT_TYPE      CCD            # CCD (linear) or PHOTO (with gamma correction)'+'\n')
    file.write('DETECT_MINAREA   2            # minimum number of pixels above threshold'+'\n')
    file.write('DETECT_THRESH    1.5            # <sigmas> or <threshold>,<ZP> in mag.arcsec-2'+'\n')
    file.write('ANALYSIS_THRESH  2            # <sigmas> or <threshold>,<ZP> in mag.arcsec-2'+'\n')

    file.write('FILTER           Y              # apply filter for detection (Y or N)?'+'\n')
    file.write('FILTER_NAME      imaging/default.conv   # name of the file containing the filter'+'\n')

    file.write('DEBLEND_NTHRESH  32             # Number of deblending sub-thresholds'+'\n')
    file.write('DEBLEND_MINCONT  0.005          # Minimum contrast parameter for deblending'+'\n')

    file.write('CLEAN            Y              # Clean spurious detections? (Y or N)?'+'\n')
    file.write('CLEAN_PARAM      1.0            # Cleaning efficiency'+'\n')

    file.write('MASK_TYPE        CORRECT        # type of detection MASKing: can be one of'+'\n')
                                # NONE, BLANK or CORRECT                                                                                                       \
                                                                                                                                                                
    file.write('WEIGHT_TYPE      NONE'+'\n')

    file.write('#------------------------------ Photometry -----------------------------------'+'\n')

    file.write('PHOT_APERTURES   5              # MAG_APER aperture diameter(s) in pixels'+'\n')
    file.write('PHOT_AUTOPARAMS  2.5, 3.5       # MAG_AUTO parameters: <Kron_fact>,<min_radius>'+'\n')
    file.write('PHOT_PETROPARAMS 2.0, 3.5       # MAG_PETRO parameters: <Petrosian_fact>,'+'\n')
                                # <min_radius>                                                                                                                 \
                                                                                                                                                                

    file.write('SATUR_LEVEL      50000.0        # level (in ADUs) at which arises saturation'+'\n')

    file.write('MAG_ZEROPOINT    0.0            # magnitude zero-point'+'\n')
    file.write('MAG_GAMMA        4.0            # gamma of emulsion (for photographic scans)'+'\n')
    file.write('GAIN             0.0            # detector gain in e-/ADU'+'\n')
    file.write('PIXEL_SCALE      1.0            # size of pixel in arcsec (0=use FITS WCS info)'+'\n')
    file.write('#------------------------- Star/Galaxy Separation ----------------------------'+'\n')

    file.write('SEEING_FWHM      1.2            # stellar FWHM in arcsec'+'\n')
    file.write('STARNNW_NAME     imaging/default.nnw    # Neural-Network_Weight table filename'+'\n')

    file.write('#------------------------------ Background -----------------------------------'+'\n')

    file.write('BACK_SIZE        30             # Background mesh: <size> or <width>,<height>'+'\n')
    file.write('BACK_FILTERSIZE  3              # Background filter: <size> or <width>,<height>'+'\n')
    file.write('BACKPHOTO_TYPE   LOCAL         # can be GLOBAL or LOCAL'+'\n')

    file.write('#------------------------------ Check Image ----------------------------------'+'\n')
    file.write('CHECKIMAGE_TYPE APERTURES         # can be NONE, BACKGROUND, BACKGROUND_RMS,'+'\n')
                                # MINIBACKGROUND, MINIBACK_RMS, -BACKGROUND,                                                                                   \
                                                                                                                                                                
                                # FILTERED, OBJECTS, -OBJECTS, SEGMENTATION,                                                                                   \
                                                                                                                                                                
                                # or APERTURES                                                                                                                 \
                                                                                                                                                                
    file.write('CHECKIMAGE_NAME  imaging/aps_'+name+'.fits     # Filename for the check-image'+'\n')

    file.write('#--------------------- Memory (change with caution!) -------------------------'+'\n')

    file.write('MEMORY_OBJSTACK  3000           # number of objects in stack'+'\n')
    file.write('MEMORY_PIXSTACK  300000         # number of pixels in stack'+'\n')
    file.write('MEMORY_BUFSIZE   1024           # number of lines in buffer'+'\n')

    file.write('#----------------------------- Miscellaneous ---------------------------------'+'\n')

    file.write('VERBOSE_TYPE     QUIET         # can be QUIET, NORMAL or FULL'+'\n')
    file.write('WRITE_XML        Y              # Write XML file (Y/N)?'+'\n')
    file.write('XML_NAME         imaging/sex.xml        # Filename for XML output'+'\n')


    file.close()
'''Runs source extractor from within python'''

def run_sex(name, prefix):
    #Save where you are                                                                                                                                         
    cwd = os.getcwd()
    os.chdir(prefix)
    os.system("sex -c imaging/default_"+name+".sex "+"imaging/pet_radius_"+name+".fits")
    os.chdir(cwd)
'''Extracts parameters from source extractor that are useful as a Galfit input/ first guess'''
def sex_params_galfit(name, prefix):


    file_path = prefix+'imaging/test_'+name+'.cat'
    f = open(file_path, 'r+')
    data=f.readlines()
    A=[]
    B=[]
    B_mult=[]
    pet_mult=[]
    flux_pet=[]
    x_pos=[]
    y_pos=[]
    mag_auto=[]
    eff_radius=[]
    PA=[]
    back=[]
    for line in data:
        words = line.split()
        if words[0] !='#':
            x_pos.append(float(words[1]))
            y_pos.append(float(words[2]))
            A.append(float(words[3]))#major axis RMS                                                                                                           \
                                                                                                                                                                
            B.append(float(words[4]))#minor axis RMS                                                                                                           \
                                                                                                                                                                
            eff_radius.append(float(words[7]))#this is the effective radius                                                                                    \
                                                                                                                                                                
            pet_mult.append(float(words[8]))#this is the petrosian radius                                                                                      \
                                                                                                                                                                
            flux_pet.append(float(words[10]))#flux within the petrosian aperture in counts                                                                     \
                                                                                                                                                                
            PA.append(float(words[12]))#position angle of the source                                                                                           \
                                                                                                                                                                
            back.append(float(words[13]))#background at centroid position in counts                                                                            \
                                                                                                                                                                

    max_i=flux_pet.index(max(flux_pet))
    x_max=x_pos[max_i]
    y_max=y_pos[max_i]
    sex_pet_r=(pet_mult[max_i]*A[max_i])
    minor=pet_mult[max_i]*B[max_i]
    flux=flux_pet[max_i]

    eff_radius_1=eff_radius[max_i]
    B_A_1=B[max_i]/A[max_i]#elipticity                                                                                                                         \
    PA_1=PA[max_i]
    back_1=back[max_i]


    if len(x_pos)==1:#This means there was only one aperture detected therefore one stellar bulge                                                              \
                                                                                                                                                                
        n_bulges=1
        return sex_pet_r, minor, flux, x_max, y_max, n_bulges, eff_radius_1, B_A_1, PA_1, back_1
    else:
        import heapq
        two_largest=heapq.nlargest(2, flux_pet)
        if two_largest[1] > 0.1*two_largest[0]:
            #then it's 1/10th of the brightness and is legit                                                                                                   \
                                                                                                                                                                
            n_bulges=2
            max_i_sec=flux_pet.index(two_largest[1])
            x_max_2=x_pos[max_i_sec]
            y_max_2=y_pos[max_i_sec]
            flux_max_2=flux_pet[max_i_sec]
            eff_radius_2=eff_radius[max_i_sec]
            B_A_2=B[max_i_sec]/A[max_i_sec]
            PA_2=PA[max_i_sec] #was -90-PA[max_i]+180                                                                                                          \
                                                                                                                                                                
            return sex_pet_r, minor, flux, x_max, y_max, n_bulges, eff_radius_1, B_A_1, PA_1, back_1, x_max_2, y_max_2, flux_max_2, eff_radius_2, B_A_2, PA_2
        else:
            n_bulges=1
            return sex_pet_r, minor, flux, x_max, y_max, n_bulges, eff_radius_1, B_A_1, PA_1, back_1

'''Optionally plots the source extractor fit and captures the background'''
def plot_sex(name,xpos,ypos, prefix):
    pic=prefix+'imaging/aps_'+name+'.fits'
    im=pyfits.open(pic)
    #optional plotting of the Source Extractor output:                                                                                                      
    '''                                                                                                                                                        \
                                                                                                                                                                
    plt.clf()                                                                                                                                                  \
                                                                                                                                                                
    im=pyfits.open(pic)                                                                                                                                        \
                                                                                                                                                                
    plt.title('Sex Apertures')                                                                                                                                 \
                                                                                                                                                                
    plt.imshow(im[0].data,norm=matplotlib.colors.LogNorm())                                                                                                    \
                                                                                                                                                                
    plt.colorbar()                                                                                                                                             \
                                                                                                                                                                
    plt.scatter(xpos,ypos, color='black')                                                                                                                      \
                                                                                                                                                                
    plt.savefig('sdss/sex_aps_'+name+'.pdf')                                                                                                                   \
                                                                                                                                                                
    '''
    return im[0].data#this is the bg image                                                                                                                     \
                                                                                                                                                                

'''Writes the galfit feedme file'''
def write_galfit_feedme(name,prefix,xcen,ycen,xcen2,ycen2, mag, mag_zpt, num_bulges, length_gal, r_1, r_2, mag_2, B_A_1, PA_1, B_A_2, PA_2, background):

    if num_bulges==2:
        try:
            f = open(prefix+'imaging/galfit.feedme_'+str(name),'w')
        except IOError:
            print('File not found for writing')
            print(prefix+'imaging/galfit.feedme_'+str(name))
            STOP
        f.write('==============================================================================='+'\n')
        f.write('# IMAGE and GALFIT CONTROL PARAMETERS'+'\n')
        f.write('A) '+prefix+'imaging/out_convolved_'+str(name)+'.fits            # Input data image (FITS file)'+'\n')
        f.write('B) '+prefix+'imaging/out_'+str(name)+'.fits       # Output data image block'+'\n')
        f.write('C) '+prefix+'imaging/out_sigma_convolved_'+str(name)+'.fits                # Sigma image name (made from data i$'+'\n')
        f.write('D) none        # Input PSF image and (optional) diffusion kernel'+'\n')
        f.write('E) none                   # PSF fine sampling factor relative to data'+'\n')
        f.write('F) none                # Bad pixel mask (FITS image or ASCII coord list)'+'\n')
        f.write('G) none                # File with parameter constraints (ASCII file)'+'\n')
        f.write('H) 1    '+str(length_gal)+'   1    '+str(length_gal)+'   # Image region to fit (xmin xmax ymin ymax)'+'\n')
        f.write('I) '+str(length_gal)+' '+ str(length_gal)+'          # Size of the convolution box (x y)'+'\n')
        f.write('J) '+str(mag_zpt)+' # Magnitude photometric zeropoint'+'\n')
        f.write('K) 0.5  0.5        # Plate scale (dx dy)    [arcsec per pixel]'+'\n')
        f.write('O) regular             # Display type (regular, curses, both)'+'\n')
        f.write('P) 0                   # Choose: 0=optimize, 1=model, 2=imgblock, 3=subcomps'+'\n')
        f.write('# Object number: 1 '+'\n')
        f.write(' 0) sersic                 #  object type'+'\n')
        f.write(' 1) '+str(xcen)+' '+str(ycen)+'  1 1  #  position x, y'+'\n')#these positions need to be automated                                             
        f.write(' 3) '+str(mag)+'    1          #  Integrated magnitude'+'\n')#'+str(mag)+'                                                                    \
                                                                                                                                                                
        f.write(' 4) '+str(r_1)+'      1          #  R_e (half-light radius)   [pix]'+'\n')
        f.write(' 5) 2      1          #  Sersic index n (de Vaucouleurs n=4)'+'\n')
        f.write(' 6) 0.0000      0          #     -----'+'\n')
        f.write(' 7) 0.0000      0          #     -----'+'\n')
        f.write(' 8) 0.0000      0          #     -----'+'\n')
        f.write(' 9) '+str(B_A_1)+'      1          #  axis ratio (b/a)'+'\n')
        f.write('10) '+str(PA_1+90)+'    1          #  position angle (PA) [deg: Up=0, Left=90]'+'\n')
        f.write(' Z) 0                      #  output option (0 = resid., 1 = Dont subtract)'+'\n')
        f.write('# Object number: 2 '+'\n')
        f.write(' 0) sersic                 #  object type'+'\n')
        f.write(' 1) '+str(xcen2)+' '+str(ycen2)+'  1 1  #  position x, y'+'\n')
        f.write(' 3) '+str(mag_2)+'    1          #  Integrated magnitude'+'\n')#'+str(mag)+'                                                                  \
                                                                                                                                                                
        f.write(' 4) '+str(r_2)+'      1          #  R_e (half-light radius)   [pix]'+'\n')
        f.write(' 5) 4      1          #  Sersic index n (de Vaucouleurs n=4)'+'\n')
        f.write(' 6) 0.0000      0          #     -----'+'\n')
        f.write(' 7) 0.0000      0          #     -----'+'\n')
        f.write(' 8) 0.0000      0          #     -----'+'\n')
        f.write(' 9) '+str(B_A_2)+'      1          #  axis ratio (b/a)'+'\n')
        f.write('10) '+str(PA_2+90)+'    1          #  position angle (PA) [deg: Up=0, Left=90]'+'\n')
        f.write(' Z) 0                      #  output option (0 = resid., 1 = Dont subtract)'+'\n')
        f.write('# Object number: 4'+'\n')
        f.write(' 0) sky                    #  object type'+'\n')
        f.write(' 1) '+str(background)+'      1          #  sky background at center of fitting region [ADU$'+'\n')
        #was 100.3920      1                                                                                                                                   \
                                                                                                                                                                
        f.write(' 2) 0.0000      0          #  dsky/dx (sky gradient in x)'+'\n')
        f.write(' 3) 0.0000      0          #  dsky/dy (sky gradient in y)'+'\n')
        f.write(' Z) 0                      #  output option (0 = resid., 1 = Dont subtract)'+'\n')
        f.write('================================================================================'+'\n')
        f.close()
        return
    if num_bulges==1:
        '''I need to make a code to write out the GALFIT.feedme file'''
        f = open(prefix+'imaging/galfit.feedme_'+str(name), 'w')
        f.write('==============================================================================='+'\n')

        f.write('# IMAGE and GALFIT CONTROL PARAMETERS'+'\n')

        f.write('A) '+prefix+'imaging/out_convolved_'+str(name)+'.fits            # Input data image (FITS file)'+'\n')
        f.write('B) '+prefix+'imaging/out_'+str(name)+'.fits       # Output data image block'+'\n')
        f.write('C) '+prefix+'imaging/out_sigma_convolved_'+str(name)+'.fits                # Sigma image name (made from data i$'+'\n')
        f.write('D) none   #        # Input PSF image and (optional) diffusion kernel'+'\n')
        f.write('E) none                   # PSF fine sampling factor relative to data'+'\n')
        f.write('F) none                # Bad pixel mask (FITS image or ASCII coord list)'+'\n')
        f.write('G) none                # File with parameter constraints (ASCII file)'+'\n')
        f.write('H) 1    '+str(length_gal)+'   1    '+str(length_gal)+'   # Image region to fit (xmin xmax ymin ymax)'+'\n')
        f.write('I) '+str(length_gal)+' '+ str(length_gal)+'          # Size of the convolution box (x y)'+'\n')
        f.write('J) '+str(mag_zpt)+' # Magnitude photometric zeropoint'+'\n')

        f.write('K) 0.5  0.5        # Plate scale (dx dy)    [arcsec per pixel]'+'\n')
        f.write('O) regular             # Display type (regular, curses, both)'+'\n')
        f.write('P) 0                   # Choose: 0=optimize, 1=model, 2=imgblock, 3=subcomps'+'\n')

        #first bulge                                                                                                                                           \
                                                                                                                                                                
        f.write('# Object number: 1 '+'\n')
        f.write(' 0) sersic                 #  object type'+'\n')
        f.write(' 1) '+str(xcen)+' '+str(ycen)+'  1 1  #  position x, y'+'\n')#these positions need to be automated                                            \
                                                                                                                                                                
        f.write(' 3) '+str(mag)+'     1          #  Integrated magnitude'+'\n')#'+str(mag)+'                                                                   \
                                                                                                                                                                
        f.write(' 4) '+str(r_1)+'      1          #  R_e (half-light radius)   [pix]'+'\n')
        f.write(' 5) 2      1          #  Sersic index n (de Vaucouleurs n=4)'+'\n')
        f.write(' 6) 0.0000      0          #     -----'+'\n')
        f.write(' 7) 0.0000      0          #     -----'+'\n')
        f.write(' 8) 0.0000      0          #     -----'+'\n')
        f.write(' 9) '+str(B_A_1)+'     1          #  axis ratio (b/a)'+'\n')
        f.write('10) '+str(PA_1+90)+'    1          #  position angle (PA) [deg: Up=0, Left=90]'+'\n')
        f.write(' Z) 0                      #  output option (0 = resid., 1 = Dont subtract)'+'\n')

        '''file.write('# Object number: 2 '+'\n')                                                                                                              \
                                                                                                                                                                
        file.write(' 0) sersic                 #  object type'+'\n')                                                                                           \
                                                                                                                                                                
        file.write(' 1) '+str(xcen)+' '+str(ycen)+'  1 1  #  position x, y'+'\n')#these positions need to be automated                                         \
                                    \                                                                                                                           
                                                                                                                                                               \
                                                                                                                                                                
        file.write(' 3) '+str(mag)+'     1          #  Integrated magnitude'+'\n')#'+str(mag)+'                                                                \
                                    \                                                                                                                           
                                                                                                                                                               \
                                                                                                                                                                
        file.write(' 4) '+str(r_1)+'      1          #  R_e (half-light radius)   [pix]'+'\n')                                                                 \
                                                                                                                                                                
        file.write(' 5) 1      1          #  Sersic index n (de Vaucouleurs n=4)'+'\n')                                                                        \
                                                                                                                                                                
        file.write(' 6) 0.0000      0          #     -----'+'\n')                                                                                              \
                                                                                                                                                                
        file.write(' 7) 0.0000      0          #     -----'+'\n')                                                                                              \
                                                                                                                                                                
        file.write(' 8) 0.0000      0          #     -----'+'\n')                                                                                              \
        file.write(' 9) '+str(B_A_1)+'     1          #  axis ratio (b/a)'+'\n')                                                                               \
                                                                                                                                                                
        file.write('10) '+str(PA_1+90)+'    1          #  position angle (PA) [deg: Up=0, Left=90]'+'\n')                                                      \
                                                                                                                                                                
        file.write(' Z) 0                      #  output option (0 = resid., 1 = Dont subtract)'+'\n')'''
        f.write('# Object number: 3'+'\n')
        f.write(' 0) sky                    #  object type'+'\n')
        f.write(' 1) '+str(background)+'      1          #  sky background at center of fitting region [ADU$'+'\n')
        f.write(' 2) 0.0000      0          #  dsky/dx (sky gradient in x)'+'\n')
        f.write(' 3) 0.0000      0          #  dsky/dy (sky gradient in y)'+'\n')
        f.write(' Z) 0                      #  output option (0 = resid., 1 = Dont subtract)'+'\n')

        f.write('================================================================================'+'\n')



        f.close()
        return
'''                                                                                                                                                            \
                                                                                                                                                                
Calls galfit with the feedme file you created                                                                                                                  \
                                                                                                                                                                
'''

def run_galfit(name, prefix):
    command =["/Users/rnevin/Applications/galfit",prefix+"imaging/galfit.feedme_"+str(name)]
   
    #command =["galfit",prefix+"imaging/galfit.feedme_"+str(name)]
    # removed the ./ from before galfit                                                                                                                         
    with open(os.devnull, "w") as fnull:
        result = subprocess.call(command, stdout = fnull, stderr = fnull)

    return
'''Now open the galfit result and get the effective radius                                                                                                                                                                                                                                                         
I've kept a lot of unnecessary stuff in here in case you want to                                                                                               \
                                                                                                                                                                
use additional pieces of the output; this code is generally                                                                                                    \
                                                                                                                          
very useful for running galfit inline and extracting the output info'''
def galfit_params(name,num_bulges, prefix):

    '''First, figure out how many bulges'''


    output=prefix+'imaging/out_'+str(name)+'.fits'
    out=pyfits.open(output)

    '''                                                                                                                                                        \
                                                                                                                                                                
    fig=plt.figure()                                                                                                                                           \
                                                                                                                                                                
    ax1=fig.add_subplot(211)                                                                                                                                   \
                                                                                                                                                                
    im1=ax1.imshow(out[1].data, norm=matplotlib.colors.LogNorm())                                                                                              \
                                                                                                                                                                
    plt.colorbar(im1,label='Magnitudes')                                                                                                                       \
                                                                                                                                                                
    ax2=fig.add_subplot(212)                                                                                                                                   \
                                                                                                                                                                
    im2=ax2.imshow(out[2].data,norm=matplotlib.colors.LogNorm())                                                                                               \
                                                                                                                                                                
    plt.colorbar(im2,label='Magnitudes')                                                                                                                       \
                                                                                                                                                                
    plt.savefig('imaging/side_by_side_galfit_input_'+str(name)+'.pdf')                                                                                         \
                                                                                                                                                                
    '''
    if out[2].header['COMP_2']=='sky':
        try:
            mag_1=float(out[2].header['1_MAG'][:7])

            sep=0
            flux_ratio=0
            x_1=float(out[2].header['1_XC'][:7])
            x_2=x_1
            y_1=float(out[2].header['1_YC'][:7])
            y_2=y_1
            PA1=float(out[2].header['1_PA'][:7])
            PA2=PA1
    #        AR1=float(out[2].header['1_AR'][:7])                                                                                                              \
                                                                                                                                                                

            AR1=float(out[2].header['1_AR'][:7])
            sersic = float(out[2].header['1_N'][:7])
        except ValueError or KeyError:
            try:
                sersic=float(out[2].header['1_N'][1:7])
            except ValueError or KeyError:
                    return 0,0,999, 0,out[2].header

    else:
        try:
            mag_1=float(out[2].header['1_MAG'][:7])
            mag_2=float(out[2].header['2_MAG'][:7])
            #m = -2.5 log_10 (F/ t) + (mag zpt = 26.563)                                                                          
            #t*10^((m-zpt)/(-2.5)) = F                                                                                             
            #F/F = 10^(m-m/-2.5)                                                                                                                          
            #The flux ratio:                                                                                                                         
            flux_ratio=10**((mag_1-mag_2)/-2.5)

            if mag_1 < mag_2:
                #this means that #1 is brighter                                                                                                                                                                                                               
                x_1=float(out[2].header['1_XC'][:7])
                x_2=float(out[2].header['2_XC'][:7])
                y_1=float(out[2].header['1_YC'][:7])
                y_2=float(out[2].header['2_YC'][:7])
                PA1=float(out[2].header['1_PA'][:7])
                PA2=float(out[2].header['2_PA'][:7])
                AR1=float(out[2].header['1_AR'][:7])
                sersic=float(out[2].header['1_N'][:7])
            else:
                x_1=float(out[2].header['2_XC'][:7])
                x_2=float(out[2].header['1_XC'][:7])
                y_1=float(out[2].header['2_YC'][:7])
                y_2=float(out[2].header['1_YC'][:7])
                PA1=float(out[2].header['2_PA'][:7])
                PA2=float(out[2].header['1_PA'][:7])
                AR1=float(out[2].header['2_AR'][:7])
                sersic=float(out[2].header['2_N'][:7])

            #Also, the separation in physical space between the bulges:                          
            sep=np.sqrt(abs(x_1-x_2)**2+abs(y_1-y_2)**2)
        except ValueError or KeyError:

            try:
                mag_1=float(out[2].header['1_MAG'][:7])
                sep=0
                flux_ratio=0
                flux_ratio=0
                x_1=float(out[2].header['1_XC'][:7])
                x_2=x_1
                y_1=float(out[2].header['1_YC'][:7])
                y_2=y_1
                PA1=float(out[2].header['1_PA'][:7])
                PA2=PA1
                AR1=float(out[2].header['1_AR'][:7])
                sersic=float(out[2].header['1_N'][:7])
            except ValueError or KeyError:
                try:
                    mag_1=float(out[2].header['2_MAG'][:7])
                    sep=0
                    flux_ratio=0
                    flux_ratio=0
                    x_1=float(out[2].header['2_XC'][:7])
                    x_2=x_1
                    y_1=float(out[2].header['2_YC'][:7])
                    y_2=y_1
                    PA1=float(out[2].header['2_PA'][:7])
                    PA2=PA1
                    AR1=float(out[2].header['2_AR'][:7])
                    sersic=float(out[2].header['2_N'][:7])
                except ValueError or KeyError:
                    return 0,0,999, 0, out[2].header

    galfit_input=out[1].data
    galfit_model=out[2].data
    galfit_resid=out[3].data
    chi2=(out[2].header['CHI2NU'])
    #print('Sersic', sersic)                                                                                                                                   \
                                                                                                                                                                
    out.close()

    try:
        return sep, flux_ratio, sersic, AR1
    except UnboundLocalError:
        try:
            return sep, flux_ratio, sersic, 0, out[2].header
        except UnboundLocalError:
            return 0, 0, 0, 0, out[2].header



'''Uses the Petrosian radius measured with Source Extractor and the semimajor axis and the petrosian flux                                                      \
                                                                                                                                                                
to calculate the surface brightness at the petrosian radius which we then will use to construct the segmentation                                               \
                                                                                                                                                                
maps in a later step'''
def input_petrosian_radius(input_r, input_b, flux_w_i):
    Area=(input_b*input_r*np.pi) #area in pixels of an ellipse                                                                                     
    limiting_sb=0.2*(flux_w_i/Area)#was 0.2*(flux_w_i/Area)                                                                                         
    return limiting_sb#returns the surface brightness at the petrosian radius                                                                                 
def img_assy_shape(img,mangaid):
    gal_zoom=img

    '''                                                                                                                                                        \
                                                                                                                                                                
    Now we need to do 8-connectivity:                                                                                                                          \
                                                                                                                                                                
    Basically from the center out, is something really 8-connected?                                                                                            \
                                                                                                                                                                
    '''

    half_bit=int(np.shape(gal_zoom)[0]/2)



    labeled = skimage.measure.label(gal_zoom, background=False, connectivity=2)
    label = labeled[half_bit, half_bit] # known pixel location                                                                                                 \
                                                                                                                                                                

    rp = skimage.measure.regionprops(labeled)
    props = rp[label - 1] # background is labeled 0, not in rp                                                                                                 \
                                                                                                                                                                

    props.bbox # (min_row, min_col, max_row, max_col)                                                                                                          \
                                                                                                                                                                
    gal_zoom=props.image





    gal_zoom=gal_zoom.astype(np.float32)

    A=np.sum(abs(gal_zoom-np.rot90(np.rot90(gal_zoom))))/np.sum(abs(gal_zoom))

    '''                                                                                                                                                        \
                                                                                                                                                                
    plt.clf()                                                                                                                                                  \
                                                                                                                                                                
    plt.imshow(abs(gal_zoom-np.rot90(np.rot90(gal_zoom))))                                                                                                     \
                                                                                                                                                                
    plt.title(A)                                                                                                                                               \
                                                                                                                                                                
    plt.savefig('imaging/SA_'+str(mangaid)+'.pdf')                                                                                                             \
                                                                                                                                                                
    '''
    return A





