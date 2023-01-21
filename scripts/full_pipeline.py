#imorts 
import astropy.units as u
import numpy as np
#import gala.potential as gp
#import gala.dynamics as gd
from astropy.coordinates import SkyCoord, Distance
#import popsims
import matplotlib.pyplot as plt
#from popsims.plot_style import  plot_style
from astropy.coordinates import SkyCoord
import astropy.coordinates as astro_coord
from popsims import sample_from_powerlaw
import astropy.units as u
from tqdm import tqdm
import popsims
import pandas as pd
from easyshapey import Box
from scipy.interpolate import interp1d, griddata
from findthegap.gapper import Gapper
import torch
import itertools
#paths
import numba 
#import jax #use jax.jit instead of numba --> might change for 
#plot_style()

path_isochrone='../data/isochrones/'
path_data='../data/images/'
path_streamdata='../data/stream/'
path_pipeline='../data/pipeline/'
path_pandas= '../data/pandas/'
path_plot='../figures/'
isochrone_path=path_isochrone


mag_keys=['gmag', 'imag', 'F062mag', 'F087mag']

#read strem
def read_stream_file(N_pal5, gap_center, box, rgc, mhalo, vhalo,    distance_to_hit =0.5):
    
    #filename='pal5_rgc{}_mhalo{:.2e}_vhalo{:.0f}'.format(rgc, mhalo, vhalo)
 
    #filename='pal5_rgc{}_mhalo{:.2e}_vhalo{:.0f}_distance_to_hit{}'.format(rgc, mhalo, vhalo, distance_to_hit )
    filename='no_self_grav_pal5_rgc{}_mhalo{:.2e}_vhalo{:.0f}_distance_to_hit{}'.format(rgc, mhalo, vhalo, distance_to_hit )
    st=(np.load(path_streamdata+'/{}.npy'.format(filename), allow_pickle=True).flatten()[0])['stream']
    
    x0=st.y.value
    y0=st.x.value

    center=np.nanmedian(np.array(rgc.split('_')).astype(float))

    #deptermine optimal rotation angle by fitting a line to the half stream
    mask= (x0-np.nanmedian(x0))>0.
    line = np.polyfit(x0[mask], y0[mask], 1)
    angle=np.arctan(line[0])
    x0, y0=rotate(x0, y0, -angle, c=(np.nanmedian(x0), np.nanmedian(y0)))

    xshift= np.nanmedian(x0)-center
    yshift=np.nanmedian(y0)-center

    x=x0-xshift
    y=y0-yshift

    xshift=gap_center[0]
    yshift=gap_center[1]

    x=x-xshift
    y=y-yshift
    
    choose=np.random.choice(np.arange(len(x)), N_pal5)
    
    return box.select(np.array([x[choose], y[choose]]))


def count_pal5_stars_old(mag_limit, dmod):
    import scipy.interpolate as interp
    dmod_pal5=16.85
    cfht_mini = np.loadtxt(isochrone_path+'/Isochrone_CFHT_PARSEC_withDust_newZ.txt', skiprows = 8, usecols = [2])
    cfht_gmag = np.loadtxt(isochrone_path+'/Isochrone_CFHT_PARSEC_withDust_newZ.txt', skiprows = 8, usecols = [24])
    cfht_imag = np.loadtxt(isochrone_path+'/Isochrone_CFHT_PARSEC_withDust_newZ.txt', skiprows = 8, usecols = [25])
    wfirst_mini = np.loadtxt(isochrone_path+'/Isochrone_WFIRST_PARSEC_withDust_newZ.txt', skiprows = 8, usecols = [2])
    wfirst_mags = np.loadtxt(isochrone_path+'/Isochrone_WFIRST_PARSEC_withDust_newZ.txt', skiprows = 8, usecols = [23,24,25,26,27,28,29])

    sample_lowmasses=sample_from_powerlaw(-0.5, xmin=np.min(cfht_mini), xmax=np.max(cfht_mini), nsample=int(1e6))
    inter_gmags = interp.interp1d(cfht_mini, cfht_gmag, bounds_error=False,  fill_value=np.nan)
    inter_imags = interp.interp1d(cfht_mini, cfht_imag, bounds_error=False,  fill_value=np.nan)
    inter_Rmags = interp.interp1d(wfirst_mini, wfirst_mags[:,0], bounds_error=False,  fill_value=np.nan)
    inter_Zmags = interp.interp1d(wfirst_mini, wfirst_mags[:,1], bounds_error=False,  fill_value=np.nan)
    inter_Ymags = interp.interp1d(wfirst_mini, wfirst_mags[:,2], bounds_error=False,  fill_value=np.nan)
    inter_Jmags = interp.interp1d(wfirst_mini, wfirst_mags[:,3], bounds_error=False,  fill_value=np.nan)
    inter_Hmags = interp.interp1d(wfirst_mini, wfirst_mags[:,4], bounds_error=False,  fill_value=np.nan)
    inter_Fmags = interp.interp1d(wfirst_mini, wfirst_mags[:,5], bounds_error=False,  fill_value=np.nan)
    inter_Wmags = interp.interp1d(wfirst_mini, wfirst_mags[:,6], bounds_error=False,  fill_value=np.nan)

    if np.sum(sample_lowmasses < np.min(cfht_mini)) > 0 or np.sum(sample_lowmasses > np.max(cfht_mini)) > 0:
        print("outside of interpolation range based on the CFHT-parsec SSP file. Need to resolve this.")
    else:
        #CFHT magnitudes
        sample_gmags = inter_gmags(sample_lowmasses) +  dmod_pal5  #distance modulus Pal5
        sample_imags = inter_imags(sample_lowmasses) +  dmod_pal5 #distance modulus Pal5
        #WFIRST magnitudes
        sample_Rmags = inter_Rmags(sample_lowmasses) +  dmod_pal5  #distance modulus Pal5
        sample_Zmags = inter_Zmags(sample_lowmasses)+  dmod_pal5 #distance modulus Pal5
        sample_Ymags = inter_Ymags(sample_lowmasses)  +  dmod_pal5
        sample_Jmags = inter_Jmags(sample_lowmasses)+  dmod_pal5  #distance modulus Pal5
        sample_Hmags = inter_Hmags(sample_lowmasses) +  dmod_pal5 #distance modulus Pal5
        sample_Fmags = inter_Fmags(sample_lowmasses)+  dmod_pal5 #distance modulus Pal5
        sample_Wmags = inter_Wmags(sample_lowmasses) +  dmod_pal5 #distance modulus Pal5

        #renormalize the luminosity function by computing a normalization factor
        num_20_23= len(sample_gmags[np.logical_and(sample_gmags>=20, sample_gmags<=23)])
        #print ('number of stars between 20 and 23 G mag {}'.format( num_20_23))

        norm= 3000/num_20_23
        #print (norm)

        #compute the difference between distance moduli and offset stars
        dist_mod_And = dmod-dmod_pal5
        #offset_Zmags= sample_Zmags+  dist_mod_And
        nstars_wfirst = (len(np.where((sample_Zmags < (mag_limit-dist_mod_And)  ))[0]))*norm

        #fig, ax=plt.subplots()
        #plt.scatter( sample_lowmasses, sample_gmags)
        #ax.set(ylabel=k, xlabel='Mass ')
        #j
        return nstars_wfirst

def read_cmd_file(df, rgc, d_galaxy, mag_limit):
    d_m31= 770*u.kpc
    dmod_m31=5*np.log10(d_m31.to(u.pc).value/10.0)

    
    dmod_galaxy=5*np.log10(d_galaxy.to(u.pc).value/10.0)
    dmod_diff= dmod_galaxy-dmod_m31
    kpc_conversion = np.pi * d_galaxy / 180.
    
    #put to the desired distance modulus 
    mw_df= df.query("galaxy == 'MW'").reset_index(drop=True)
    m31_df= df.query("galaxy =='M31'").reset_index(drop=True)
    
    for k in ['appF062mag', 'appF087mag', 'appgmag', 'appimag']:
         m31_df[k]=  m31_df[k].values+dmod_diff
            
    #appply magnitude cut
    df_final=pd.concat([m31_df, mw_df]).reset_index(drop=True)
    df_final=(df_final[df_final.appF087mag < mag_limit]).reset_index(drop=True)
    
    
    #assign RA, DEC, xki based on the data
    from astropy.io.votable import parse_single_table
    data=parse_single_table(path_pandas+'M31_{}kpc_new.vot'.format(rgc)).to_table().to_pandas()
    
    
    for k in ['RA', 'Dec','xki', 'eta']:
        df_final[k]=np.random.choice(data[k].values, len(df_final), replace=True)
              
    s=SkyCoord(ra=df_final.RA, dec=df_final.Dec,frame = 'icrs', unit = (u.hourangle, u.deg))
    
    center=np.nanmedian(np.array(rgc.split('_')).astype(float))
              
    shift_x=np.nanmedian(kpc_conversion.value*(s.ra.to(u.degree).value))-center
    shift_y=np.nanmedian(kpc_conversion.value*(s.dec.to(u.degree).value))-center
    
    df_final['x_coord']=kpc_conversion.value*(s.ra.to(u.degree).value)-shift_x
    df_final['y_coord']=kpc_conversion.value*(s.dec.to(u.degree).value)-shift_y
    
    return df_final

def boostrap_density_estimate(gapper_base, bw, grid_data, data, bounds, nboostrap, max_eigenvalue=True):
    ##purpose: with a gapper object, 
    #we can estimate the hessian and the maximum eigevalue by boostrapping
    #PiHPi_boots=[]
    #maxeigval_PiHPi_boots =[]
    #mineigval_PiHPi_boots =[]
    #loop over all bootstraps
    
    #@jax.jit
    def run_over_bootsraps():
        mineigval_PiHPi_boots =[]
        maxeigval_PiHPi_boots =[]
        PiHPi_boots=[]
        for i in range(nboostrap):
            boot_indx = np.random.choice(np.arange(data.shape[0]), data.shape[0], 
                                     replace=True) ## Sample with replacement:bootstrap
            gapper_ = Gapper(data[boot_indx], bw, bounds)
            PiHPis_grid = []
            eigval_PiHPi = [] 

            for pt in grid_data:
                _pihpi = gapper_.get_PiHPi(pt) 
                #_pihpi_eigval, _ =  jax.numpy.linalg.eigh(jax.numpy.array(_pihpi)) #np.linalg.eigh(_pihpi)
                #print (_pihpi_eigval)
                #print (np.array(jax.numpy.asarray(_pihpi_eigval)))
                #print (type(_pihpi_eigval))
                _pihpi_eigval, _= np.linalg.eigh(_pihpi)
                PiHPis_grid.append(_pihpi)
                eigval_PiHPi.append(_pihpi_eigval)
                #print (eigval_PiHPi)

            PiHPis_grid, eigval_PiHPi = np.array(PiHPis_grid), np.array(eigval_PiHPi)
        
            #option for using minium or maximum eigenvalue
            #if max_eigenvalue:
            max_eigval_PiHPi_k = np.nanmax(eigval_PiHPi, axis=1)
            #rescale everything between 0 and 1
            max_eigval_PiHPi_k = (max_eigval_PiHPi_k-np.nanmin(max_eigval_PiHPi_k))/(np.nanmax(max_eigval_PiHPi_k)-np.nanmin(max_eigval_PiHPi_k))

            #if not max_eigenvalue:
            min_eigval_PiHPi_k = np.nanmin(eigval_PiHPi, axis=1)
            min_eigval_PiHPi_k = (min_eigval_PiHPi_k-np.nanmin(min_eigval_PiHPi_k))/(np.nanmax(min_eigval_PiHPi_k)-np.nanmin(min_eigval_PiHPi_k))


            maxeigval_PiHPi_boots.append(max_eigval_PiHPi_k)
            mineigval_PiHPi_boots.append(min_eigval_PiHPi_k)

            PiHPi_boots.append(PiHPis_grid)
        #print(f'Run {i} finished')
        return np.array(maxeigval_PiHPi_boots), np.array(mineigval_PiHPi_boots), np.array(PiHPi_boots)
    return run_over_bootsraps()


def find_gaps(bw, data, nboostrap=5):
    n= 50
    
    bounds = np.array([[np.min(data[:,d]),np.max(data[:,d])] for d in range(data.shape[1])])

    #scale of y to 
    x_to_y= np.ptp(data[:,0])/ np.ptp(data[:,1])
    
    gridding_size = [ n,int( n/(x_to_y))]

    #print (gridding_size)

    
    grid_linspace = [ np.linspace(bounds[d][0], bounds[d][1], gridding_size[d]) for d in range(2) ]
    #could use a rectangular grid instead

    #print (grid_linspace)


    meshgrid = np.meshgrid(*grid_linspace, indexing='ij')

    meshgrid_ravel = [ xi.ravel().reshape(-1,1) for xi in meshgrid]
    grid_data = np.hstack(meshgrid_ravel)
    #print (grid_data)
    #print (data)
    
    gapper_base = Gapper(data, bw, bounds)
    #print (gapper_base)
    grid_density = gapper_base.kde.score_samples(torch.tensor(grid_data))
    #density matrix 
    density_matr = grid_density.reshape((gridding_size[0], gridding_size[1]))

    #compute piHpi matrix by bootstraping
    maxeigval_PiHPi_boots,  mineigval_PiHPi_boots, PiHPi_boots = boostrap_density_estimate(gapper_base, bw,\
                                                                   grid_data, data, bounds, nboostrap,\
                                                                   max_eigenvalue=True)
    
    #visualize and take the median
    maxeigval_PiHPi_boots = np.array(maxeigval_PiHPi_boots)
    mineigval_PiHPi_boots = np.array(mineigval_PiHPi_boots)
    
    #median
    med_maxeigval_pihpi = np.median(maxeigval_PiHPi_boots, axis=0)
    med_maxeigval_pihpi_resh = med_maxeigval_pihpi.reshape((gridding_size[0], gridding_size[1]))

    med_mineigval_pihpi = np.median(mineigval_PiHPi_boots, axis=0)
    med_mineigval_pihpi_resh = med_mineigval_pihpi.reshape((gridding_size[0], gridding_size[1]))

    return  {'density': np.array(density_matr), \
              'max_eigen':np.array(med_maxeigval_pihpi_resh),
              'min_eigen':np.array(med_mineigval_pihpi_resh),
              'meshgrid':np.array(meshgrid), 
              'PiHPi': np.array(PiHPi_boots),
              'data': np.array(data)}

def count_pal5_stars(mag_limit, dmod):
    dmod_pal5=16.85
    def read_pandas_isochrones():
        from astropy.io import ascii
        return ascii.read(path_isochrone+'/cfht_pre2014_isochrones_pal5.txt').to_pandas()

    def read_roman_isochrones():
        from astropy.io import ascii
        return ascii.read(path_isochrone+'/roman_isochrones_vega_pal5.txt').to_pandas()
    
    nsample=1e6
    masses= sample_from_powerlaw(-.5, xmin=0.1, xmax=120, nsample=nsample)
    cfht=read_pandas_isochrones()
    roman= read_roman_isochrones()
    comb= pd.concat([cfht, roman]).reset_index()
    
    
    isos={}
    for k in mag_keys:
        x= np.log10(comb.Mini.values)
        y=comb[k].values
        nans=np.logical_or(np.isnan(x), np.isnan(y))
        isos['mass']= masses
        #f=griddata(x[~nans], y[~nans], np.log10(masses) , fill_value=np.nan, method='linear', rescale=True)
        sort=np.argsort(x[~nans])
        f=interp1d(x[~nans][sort], y[~nans][sort], fill_value =np.nan, bounds_error=False)(np.log10(masses))
        isos.update({k: f+dmod_pal5})
    
    df=pd.DataFrame(isos)
    
    num_20_23= len(df.gmag.values[np.logical_and(df.gmag.values>=20, df.gmag.values<=23)])
    norm= (3000/num_20_23)
    
    #compute the difference between distance moduli and offset stars
    dist_mod_And = dmod- dmod_pal5
    
    return len(df.F087mag.values[df.F087mag.values<(mag_limit-dist_mod_And)])*norm

def make_box(center, xextent, yextent):
    b=Box()
    x_min, x_max =center[0]- xextent/2, center[0]+ xextent/2
    y_min, y_max =center[-1]- yextent/2, center[-1]+ yextent/2
    v1= (x_min, y_min)
    v2=(x_min, y_max)
    v4= (x_max, y_min)
    v3=(x_max,y_max)

    b.vertices=[v1, v2, v3, v4, v1]
    return b

def rotate(x, y, ang, c=(0,0)):
    """
    Angle must be in radians
    """
    
    #rotation matrix
    r=[[np.cos(ang), -np.sin(ang)],
       [np.sin(ang), np.cos(ang)]]
    
    i=np.identity(2)
    
    mat=np.matrix([[r[0][0], r[0][1], np.dot(i-r, c)[0]],
                   [r[1][0], r[1][1], np.dot(i-r, c)[1]],
                   [0., 0., 1.]])

    z=np.ones_like(x)
    
    rotated=np.array(np.dot(mat, np.array([x, y, z])))
    
    return rotated[0], rotated[1] 

def diagnostic_plots_one_stream(resx):
    fig, (ax, ax1, ax2)=plt.subplots(ncols=3, figsize=(14, 4))

    meshgrid= resx['meshgrid']
    meshgrid_ravel = [ xi.ravel().reshape(-1,1) for xi in  meshgrid]
    grid_data = np.hstack(meshgrid_ravel)

    #max
    ax.contour(meshgrid[0], meshgrid[1],  resx['density'],  20, \
                      cmap='spring', alpha=.5)

    cax = fig.add_axes([0.08, 0., .3, 0.04])
    p= ax.contourf(meshgrid[0], meshgrid[1], resx['max_eigen'],\
                            20, cmap='afmhot', vmin=0., vmax=1)
    plt.colorbar(p, ax=ax, cax=cax, orientation='horizontal')
    cax.set(xlabel=r'Max Eigenvalue of $\Pi H \Pi$')
    

    #min

    cax = fig.add_axes([0.4, 0., .3, 0.04])
    ax1.contour(meshgrid[0], meshgrid[1],  resx['density'],  20, \
                      cmap='spring', alpha=.5)
    p= ax1.contourf(meshgrid[0], meshgrid[1], resx['min_eigen'],\
                            20, cmap='cubehelix', vmin=0., vmax=1)
    plt.colorbar(p,  ax=ax1, cax=cax, orientation='horizontal')
    cax.set(xlabel=r'Min Eigenvalue of $\Pi H \Pi$')


    gap_mask = resx['max_eigen'] > np.percentile(resx['max_eigen'], 95)
    stream_mask = resx['min_eigen'] < np.percentile(resx['min_eigen'], 5)
    ax2.scatter( resx['data'][:,0],  resx['data'][:,1], s=1, color='k')
    ax2.scatter(grid_data[:,0][gap_mask.flatten().astype(bool)], \
                grid_data[:,1][gap_mask.flatten().astype(bool)], s=50,\
                marker='+', c='#0074D9')
    ax2.scatter(grid_data[:,0][stream_mask.flatten().astype(bool)], \
               grid_data[:,1][stream_mask.flatten().astype(bool)], s=50,\
                marker='+', c='#FF851B')

    for a in [ax, ax1, ax2]:
        a.set(xlabel='x (kpc) ', ylabel=' y (kpc) ', )
        #a.axis('equal')
    plt.tight_layout()
    plt.savefig(path_plot+'//stream_cutout_gap_rescaled.jpeg', bbox_inches='tight')
    #plt.show()


def run_all(rgc, mag_limit, gap_center, box_size, box_center, distance_to_hit=0.5):
    mhalo=5e6
    vhalo=-50
    fname=path_isochrone+'simulated_df_at_M31_normalized_extended_rgc{}.csv'.format(rgc)
    MASTER_DF=pd.read_csv(fname)
    #metallicity cut
    MASTER_DF=MASTER_DF.query('MH < -1').reset_index(drop=True)
    print ('finished loading file --------------------------------------')    
    #hg
    #@numba.jit
    def run_process(d, bw):
     try: 
        d_galaxy=d*u.kpc
        kpc_conversion = np.pi * d_galaxy / 180.
        roman_fov= 0.52*u.degree*(kpc_conversion /u.degree)

        center=np.nanmedian(np.array(rgc.split('_')).astype(float))
        b=make_box( (center, center), roman_fov.value, roman_fov.value)
        
        dmod_galaxy=5*np.log10(d_galaxy.to(u.pc).value/10.0)
        N_pal5=int(count_pal5_stars_old(mag_limit, dmod_galaxy))
        vls=read_stream_file(N_pal5,gap_center, b, rgc, mhalo, vhalo,  distance_to_hit=distance_to_hit)

        bck=read_cmd_file(MASTER_DF, rgc, d_galaxy, mag_limit)
        s=b.select(bck[['x_coord', 'y_coord']])
        img= [np.concatenate([vls[0], s.x.values]), np.concatenate([vls[1], s.y.values])]
        #print ('finished making a cut --------------------------------------')
        #print (img)
        #fig, ax=plt.subplots()
        #ax.scatter(img[0], img[1], s=.1, c='k')
        #plt.savefig('image_test.jpeg')
        #plt.show()
		#print ('number of stars/sq. degree {}'.format(len(img[0])/0.52))
		#print ('---------------------------------------')
        center=np.nanmedian(np.array(rgc.split('_')).astype(float))
        b=make_box(box_center, 5, 2)
        s=b.select(np.array(img)).T
        print (s)
        print (len(s[:,0]))
        print (len(img[0]))
        res=find_gaps(bw, s, nboostrap=5)
        print (np.nanmin(s, axis=0), np.nanmax(s, axis=0))

        return {'bw{:.1f}dmod_galaxy{:.2f}'.format(bw,  dmod_galaxy):  res}
     except:
         return {}
    
    from concurrent.futures import ThreadPoolExecutor, wait , ALL_COMPLETED
    from  functools import partial
    
    #ds=np.arange(3000, 5000, 50)
    #ds = np.arange(5000, 10000, 100)
    ds=np.arange(500, 10000, 100)
    #bws=np.arange(0.1,1., 0.1)
    #bws=[0.1, 0.8, 0.9, 1.]
    bws=np.arange(0.0, 2, 0.1)
    #ds=[770]
    #bws=[0.7]
    iterables=list(np.array([(x, y) for x, y in np.array(list(itertools.product(ds, bws)))]).T)
    method=partial(run_process)
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures=list(executor.map( method, *iterables, timeout=None, chunksize=10))

    results=[x for x in futures]
    #results={}
    #for d in ds:
    #    for bw in bws
    # results.update(run_process(d, bw))
    #        except:
    #            print ('failed at dmod {} bw {}'.format(d, bw))
    #results= 
    #print (res.keys())
    #diagnostic_plots_one_stream(results)
    #choose a random nunber between o and 10,000 to represent the iteration
    iteration=f'{int(np.random.uniform(0, 10000)):05}'
    #creative filename
    fname= 'pipeline_rgc{}_mhalo{:.2e}_maglimit{}_run{}'.format(rgc, mhalo, mag_limit, iteration)
    #print (fname)
    #print (results)
    #ran into issues with pickle, it kept requiring jax
    #df= pd.DataFrame(results).applymap(np.array)
    #df.to_pickle(path_pipeline+'/{}.pkl'.format(fname))
    np.save(path_pipeline+'/{}.npy'.format(fname),results)
    #import json
    #with open(path_pipeline+'/{}.json', 'w+') as fp:
    #    json.dump(results, fp)

def make_mock_images_bckgd_only(mag_limit,rgc):
    
    mhalo=5e6
    vhalo=-50
    
    fname=path_isochrone+'simulated_df_at_M31_normalized_extended_rgc{}.csv'.format(rgc)
    
    MASTER_DF=pd.read_csv(fname)

    MASTER_DF=MASTER_DF.query('MH < -1').reset_index(drop=True)

    def run_process(d):
        try:
            d_galaxy=d*u.kpc
            kpc_conversion = np.pi * d_galaxy / 180.
            roman_fov= 0.52*u.degree*(kpc_conversion /u.degree)

            center=np.nanmedian(np.array(rgc.split('_')).astype(float))
            b=make_box( (center, center), roman_fov.value, roman_fov.value)

            dmod_galaxy=5*np.log10(d_galaxy.to(u.pc).value/10.0)

            bck=read_cmd_file(MASTER_DF, rgc, d_galaxy, mag_limit)

            img= [s.x.values,  s.y.values]

            center=np.nanmedian(np.array(rgc.split('_')).astype(float))
        
            b=make_box(center,roman_fov, roman_fov)
        
            s=b.select(np.array(img)).T
            print ('finished rgc {} d{}'.format(rgc, d))
            res={'rgc': rgc,
            'mag_limit': mag_limit,
            'data': s}
        
            return {'dmod_galaxy{:.2f}'.format(dmod_galaxy):  res}
        except:
             return {}

    from concurrent.futures import ThreadPoolExecutor, wait , ALL_COMPLETED
    from  functools import partial
    ds=np.arange(500, 10000, 10)
    #iterables=list([ds])
    #method=partial(run_process)
    #with ThreadPoolExecutor(max_workers=10) as executor:
    #    futures=list(executor.map( method, *iterables, timeout=None, chunksize=10))

    results=[run_process(x) for x in ds]
    
    iteration=f'{int(np.random.uniform(0, 10000)):05}'
    fname= 'backgrounds_rgc{}_mhalo{:.2e}_maglimit{}_run{}'.format(rgc, mhalo, mag_limit, iteration)
    
    path_bckgd='../data/backgrounds/'

    np.save(path_bckgd+'/{}.npy'.format(fname),results)


if __name__ =='__main__':
    #box_size=(5, 2)
    #for i in range(0, 50):
    #for mag_limit in [27.15, 28.69]:
    #        run_all('10_20', mag_limit, (3, 0.5), box_size, (15, 15), distance_to_hit=0.5)
    #        run_all('30_40', mag_limit, (3, 0), box_size, (35, 35.2), distance_to_hit=0.7)
    #        run_all('50_60', mag_limit, (-4., 0.5), box_size,  (55, 54.7), distance_to_hit=0.8)
    #run mulithread

    for m in [27.15, 28.69]:
        make_mock_images_bckgd_only(m,'10_20')
        make_mock_images_bckgd_only(m,'30_40')
        make_mock_images_bckgd_only(m,'50_60')
