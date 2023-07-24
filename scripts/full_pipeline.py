import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord, Distance
import matplotlib.pyplot as plt
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
import numba
from astropy.io.votable import parse_single_table

path_isochrone='../data/isochrones/'
path_data='../data/images/'
path_streamdata='../data/stream/'
path_pipeline='../data/pipeline/'
path_pandas= '../data/pandas/'
path_plot='../figures/'
isochrone_path=path_isochrone
import jax
import glob

mag_keys=['gmag', 'imag', 'F062mag', 'F087mag']
#avoid reading files each time

mhalo=5e6

use_cols=['appF087mag', 'galaxy', 'MH', 'x_coord', 'y_coord']
rgcs=['30_40']
def make_master_files():
    master_dfs={}
    for rgc in rgcs: 
        #['10_20','30_40', '50_60']:
        fname=path_isochrone+'simulated_df_at_M31_normalized_extended_rgc{}.csv'.format(rgc)
        df_final=pd.read_csv(fname, usecols=use_cols).query('MH<-1').reset_index(drop=True)
        #add positions
        data=parse_single_table(path_pandas+'M31_{}kpc_new.vot'.format(rgc)).to_table().to_pandas()
        #print (len(data))
        #print (data.columns)
        #data=data[data.FeH_phot<-1]
        for k in ['RA', 'Dec']:
            df_final[k]=np.random.choice(data[k].values, len(df_final), replace=True)
        
        print (df_final)
        xs= df_final.x_coord.values
        ys= df_final.y_coord.values
        df_final['x_coord']=np.random.uniform(xs.min(), xs.max(), len(xs))
        df_final['y_coord']= np.random.uniform(ys.min(), ys.max(), len(ys))
        master_dfs[rgc]=df_final
        del df_final
        del data
        print ('finished rgc', rgc)
        #del s
    return master_dfs

master_dfs=make_master_files()

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

def read_cmd_file(rgc, d_galaxy, mag_limit):
    df=master_dfs[rgc]
    d_m31= 770*u.kpc
    dmod_m31=5*np.log10(d_m31.to(u.pc).value/10.0)

    dmod_galaxy=5*np.log10(d_galaxy.to(u.pc).value/10.0)
    dmod_diff= dmod_galaxy-dmod_m31
    kpc_conversion = np.pi * d_galaxy / 180.

    #put to the desired distance modulus
    mw_df= df.query("galaxy == 'MW'").reset_index(drop=True)
    m31_df= df.query("galaxy =='M31'").reset_index(drop=True)

    m31_df['appF087mag']=  m31_df['appF087mag'].values+dmod_diff

    #appply magnitude cut
    df_final=pd.concat([m31_df, mw_df]).reset_index(drop=True)
    df_final=(df_final[df_final.appF087mag < mag_limit]).reset_index(drop=True)
    
    s=SkyCoord(ra=df_final.RA, dec=df_final.Dec,frame = 'icrs', unit = (u.hourangle, u.deg))
    
    center=np.nanmedian(np.array(rgc.split('_')).astype(float))

    shift_x=np.nanmedian(kpc_conversion.value*(s.ra.to(u.degree).value))-center
    shift_y=np.nanmedian(kpc_conversion.value*(s.dec.to(u.degree).value))-center

    df_final['x_coord']=kpc_conversion.value*(s.ra.to(u.degree).value)-shift_x
    df_final['y_coord']=kpc_conversion.value*(s.dec.to(u.degree).value)-shift_y
    
    return df_final

def read_cmd_file_no_rescale(rgc, d_galaxy, mag_limit):
    #do not rescale the distances between stars 
    df=master_dfs[rgc]
    d_m31= 770*u.kpc
    dmod_m31=5*np.log10(d_m31.to(u.pc).value/10.0)

    dmod_galaxy=5*np.log10(d_galaxy.to(u.pc).value/10.0)
    dmod_diff= dmod_galaxy-dmod_m31
    #kpc_conversion = np.pi * d_galaxy / 180.
    kpc_conversion=np.pi*d_m31/180

    #put to the desired distance modulus
    mw_df= df.query("galaxy == 'MW'").reset_index(drop=True)
    m31_df= df.query("galaxy =='M31'").reset_index(drop=True)

    m31_df['appF087mag']=  m31_df['appF087mag'].values+dmod_diff

    #appply magnitude cut
    df_final=pd.concat([m31_df, mw_df]).reset_index(drop=True)
    df_final=(df_final[df_final.appF087mag < mag_limit]).reset_index(drop=True)

    #s=SkyCoord(ra=df_final.RA, dec=df_final.Dec,frame = 'icrs', unit = (u.hourangle, u.deg))

    #center=np.nanmedian(np.array(rgc.split('_')).astype(float))

    #shift_x=np.nanmedian(kpc_conversion.value*(s.ra.to(u.degree).value))-center
    #shift_y=np.nanmedian(kpc_conversion.value*(s.dec.to(u.degree).value))-center

    #df_final['x_coord']=kpc_conversion.value*(s.ra.to(u.degree).value)-shift_x
    #df_final['y_coord']=kpc_conversion.value*(s.dec.to(u.degree).value)-shift_y

    return df_final

def read_cmd_file_old(df, rgc, d_galaxy, mag_limit):
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

    print (data)

    
    for k in ['RA', 'Dec','xki', 'eta']:
        df_final[k]=np.random.choice(data[k].values, len(df_final), replace=True)
              
    s=SkyCoord(ra=df_final.RA, dec=df_final.Dec,frame = 'icrs', unit = (u.hourangle, u.deg))
    
    center=np.nanmedian(np.array(rgc.split('_')).astype(float))
              
    shift_x=np.nanmedian(kpc_conversion.value*(s.ra.to(u.degree).value))-center
    shift_y=np.nanmedian(kpc_conversion.value*(s.dec.to(u.degree).value))-center
    
    #df_final['x_coord']=kpc_conversion.value*(s.ra.to(u.degree).value)-shift_x
    #df_final['y_coord']=kpc_conversion.value*(s.dec.to(u.degree).value)-shift_y

    xs=kpc_conversion.value*(s.ra.to(u.degree).value)-shift_x
    ys=kpc_conversion.value*(s.dec.to(u.degree).value)-shift_y
    
    df_final['x_coord']=np.random.uniform(xs.min(), xs.max(), len(xs))
    df_final['y_coord']=np.random.uniform(ys.min(), ys.max(), len(ys))
    
    return df_final

def count_pal5_stars(mag_limit, dmod):
    dmod_pal5=16.85
    def read_pandas_isochrones():
        from astropy.io import ascii
        return ascii.read(path_isochrone+'/cfht_pre2014_isochrones_pal5.txt').to_pandas()

    def read_roman_isochrones():
        from astropy.io import ascii
        return ascii.read(path_isochrone+'/roman_isochrones_vega_pal5_3.7.txt').to_pandas()
    
    nsample=1e6
    masses= sample_from_powerlaw(-.5, xmin=0.1, xmax=10, nsample=nsample)
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

def get_density(data, gridding_size, grid_data, bw, bounds, nboot=2):
    @jax.jit
    def run_over_boots(): 
        mineigval_PiHPi_boots =[]
        maxeigval_PiHPi_boots =[]
        dens_boots=[]
    
        for _ in range(nboot):
            boot_indx = np.random.choice(np.arange(data.shape[0]), data.shape[0], 
                                     replace=True) ## Sample with replacement:bootstrap
            gapp=Gapper(data[boot_indx], bw, bounds=bounds)
        
            dens=gapp.get_density(torch.tensor(grid_data),  requires_grad=False).flatten()
    
            #PiHPis_grid = []
            eigval_PiHPi = [] 

            for pt in grid_data:
                _pihpi = gapp.get_PiHPi(pt) 
                _pihpi_eigval, _= np.linalg.eigh(_pihpi)
                #PiHPis_grid.append(_pihpi)
                eigval_PiHPi.append(_pihpi_eigval)
           

            #PiHPis_grid, eigval_PiHPi = np.array(PiHPis_grid), np.array(eigval_PiHPi)
            eigval_PiHPi=np.array(eigval_PiHPi)

            #option for using minium or maximum eigenvalue
            #if max_eigenvalue:
            max_eigval_PiHPi_k = np.nanmax(eigval_PiHPi, axis=1)
            #rescale everything between 0 and 1
            max_eigval_PiHPi_k = (max_eigval_PiHPi_k-np.nanmin(max_eigval_PiHPi_k))/(np.nanmax(max_eigval_PiHPi_k)-np.nanmin(max_eigval_PiHPi_k))

            #if not max_eigenvalue:
            min_eigval_PiHPi_k = np.nanmin(eigval_PiHPi, axis=1)
            min_eigval_PiHPi_k = (min_eigval_PiHPi_k-np.nanmin(min_eigval_PiHPi_k))/(np.nanmax(min_eigval_PiHPi_k)-np.nanmin(min_eigval_PiHPi_k))

        
            maxeigval_PiHPi_boots.append(np.array(max_eigval_PiHPi_k))
            mineigval_PiHPi_boots.append(np.array(min_eigval_PiHPi_k))
            dens_boots.append(np.array(dens))
        #print (dens_boots)
        #ghj
        return np.array(dens_boots), np.array(maxeigval_PiHPi_boots), np.array(mineigval_PiHPi_boots)
    
    dens_boots, maxeigval_PiHPi_boots, mineigval_PiHPi_boots = run_over_boots()
        
    maxeigval_PiHPi_boots = np.array(maxeigval_PiHPi_boots)
    mineigval_PiHPi_boots = np.array(mineigval_PiHPi_boots)
    
    #median
    med_maxeigval_pihpi = np.median(maxeigval_PiHPi_boots, axis=0).reshape((gridding_size[0], gridding_size[1]))
    med_mineigval_pihpi = np.median(mineigval_PiHPi_boots, axis=0).reshape((gridding_size[0], gridding_size[1]))
    med_dens=np.nanmedian([np.array(x) for x in dens_boots], axis=0).reshape((gridding_size[0], gridding_size[1]))
    
    return {'density':med_dens,
            'grid_data':grid_data,
            'max_eigen':med_maxeigval_pihpi,
            'min_eigen':med_mineigval_pihpi}

def make_an_image(d, rgc, mag_limit,  gap_center, box_size, box_center, distance_to_hit=0.5, vhalo=-50):
    mhalo=5e6
    #fname=path_isochrone+'simulated_df_at_M31_normalized_extended_rgc{}.csv'.format(rgc)
    #MASTER_DF=pd.read_csv(fname)#pd.read_hdf(fname, key='data')
    #cut out metal-poor stars
    #MASTER_DF=(MASTER_DF[MASTER_DF.MH<-1]).reset_index(drop=True)
    d_galaxy=d*u.kpc
    kpc_conversion = np.pi * d_galaxy / 180.
    roman_fov= 0.52*u.degree*(kpc_conversion /u.degree)

    center=np.nanmedian(np.array(rgc.split('_')).astype(float))
    b=make_box( (center, center), roman_fov.value, roman_fov.value)

    dmod_galaxy=5*np.log10(d_galaxy.to(u.pc).value/10.0)
    N_pal5=int(count_pal5_stars(mag_limit, dmod_galaxy))
    vls=read_stream_file(N_pal5,gap_center, b, rgc, mhalo, vhalo,  distance_to_hit=distance_to_hit)

    #bck=read_cmd_file(rgc, d_galaxy, mag_limit)
    bck=read_cmd_file_no_rescale(rgc, d_galaxy, mag_limit)# <<<<<<<<<< remember to comment this out 
    s=b.select(bck[['x_coord', 'y_coord']])
    img= [np.concatenate([vls[0], s.x.values]), np.concatenate([vls[1], s.y.values])]
    
    return img

def make_grid(center, xsize, ysize):
    bounds = [[center-xsize/2, center+xsize/2], [center-ysize/2, center+ysize/2]]
    #then add data around it
    print (bounds)
    gridding_size = np.array([50, 20]).astype(int)
    grid_linspace = [ np.linspace(bounds[d][0], bounds[d][1], gridding_size[d]) for d in range(2) ]
    meshgrid = np.meshgrid(*grid_linspace, indexing='ij')
    meshgrid_ravel = [ xi.ravel().reshape(-1,1) for xi in meshgrid]
    grid_data = np.hstack(meshgrid_ravel)
    #print (len(grid_data[:,0]), len(meshgrid[0]))
    return meshgrid, grid_data, bounds, gridding_size

def draw_data_round_grid(data, bw, bounds):
    #draw data within 2*bw
    n=2
    
    xmax=bounds[0][1]+n*bw
    xmin=bounds[0][0]-n*bw

    print ('inputbounds', bounds)
    print ('xmaxmin', xmax, xmin)
    
    if xmax > data[0].max():
        xmax=data[0].max()
    if xmin < data[0].min():
        xmin=data[0].min()
    #no worries about y-axis
    boolean=np.logical_and.reduce([data[0] < xmax,
                                   data[0] > xmin,
                                   data[1] >  bounds[1][0]-n*bw,
                                   data[1] <  bounds[1][1]+n*bw])
    #print (bounds[0][0]-n*bw, bounds[0][1]+n*bw) 
    #print (boolean)
    #print (data[0].min(), data[0].max())
    #print (xmax, xmin)
    #print (bounds[0])
    #kl
    return [data[0][boolean], data[1][boolean]]   

def run_image(bw, rgc, mag_limit, distance):
    #centering keywords (keywords to get center)
    cent_values={'30_40': ((2.5, 0.1), (35.0, 35.2)), '10_20': ((2.5, 0.55), (15., 15)), '50_60': ((-4, .35),  (55, 55))}
    distances_to_hit={'30_40': 0.7, '10_20':0.5, '50_60': 0.8}
    vhalos={'10_20': -50, '30_40': 70, '50_60': -50}
    box_size=(15, 6)  #original box size, should this be the FOV of Roman? Obviously can't go beyond that >>
    img= make_an_image(distance, rgc, mag_limit,  cent_values[rgc][0], box_size, cent_values[rgc][1],  distance_to_hit=distances_to_hit[rgc], vhalo=vhalos[rgc])
    meshgrid, grid_data, bounds, gridding_size = make_grid(np.array(rgc.split('_')).astype(float).mean(),  5, 2)
    dt=draw_data_round_grid(img, bw, bounds)
    res=get_density(np.array(dt).T, gridding_size, grid_data, bw, bounds, nboot=5)
    res['meshgrid']=meshgrid
    res['data']=dt

    #fig, ax=plt.subplots(figsize=(10*1.5, 2*1.5), nrows=2)
    #ax[0].scatter(img[0], img[1], s=0.1)
    #x[0].scatter(grid_data[:,0], grid_data[:,1], s=10, marker='*', color='y')
    #ax[0].scatter(dt[0], dt[1], s=0.1)

    #c=ax[1].contourf(meshgrid[0], meshgrid[1], res['max_eigen'], alpha=0.5, cmap='cubehelix')
    #_ =ax[1].contour(meshgrid[0], meshgrid[1], res['density'], alpha=0.5, cmap='coolwarm')
    #ax[1].scatter(dt[0], dt[1], s=0.1)
    #plt.show()
    dmod_galaxy=5*np.log10(distance*1000/10.0)
    return {'bw{:.1f}dmod_galaxy{:.2f}'.format(bw,  dmod_galaxy):  res}

def unperturbed_stream(N_pal5, gap_center, box, rgc):

    filename='orgininalno_self_grav_pal5_rgc{}'.format(rgc)
    f='../data/stream/{}.npy'.format(filename)
    filenames=np.array(glob.glob(f))[0]
    print (filenames)
    read_f= lambda x: np.load(x, allow_pickle=True)
    st0=(read_f(filenames).flatten()[0])['stream']
    center=np.nanmedian(np.array(rgc.split('_')).astype(float))

    xu, yu=st0.y.value, st0.x.value
    #mask= ((x0-np.nanmedian(x0))**2+(y0-np.nanmedian(y0))**2)**0.5<5
    mask= (xu-np.nanmedian(xu))>0.
    line = np.polyfit(xu[mask], yu[mask], 1)
    angle=np.arctan(line[0])
    xu, yu=rotate(xu, yu, -angle, c=(np.nanmedian(xu), np.nanmedian(yu)))

    x0, y0= xu, yu
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

def make_an_image_intact(d, rgc, mag_limit,  gap_center, box_size, box_center):
    d_galaxy=d*u.kpc
    kpc_conversion = np.pi * d_galaxy / 180.
    roman_fov= 0.52*u.degree*(kpc_conversion /u.degree)

    center=np.nanmedian(np.array(rgc.split('_')).astype(float))
    b=make_box( (center, center), roman_fov.value, roman_fov.value)

    dmod_galaxy=5*np.log10(d_galaxy.to(u.pc).value/10.0)
    N_pal5=int(count_pal5_stars(mag_limit, dmod_galaxy))
    #vls=read_stream_file(N_pal5,gap_center, b, rgc, mhalo, vhalo,  distance_to_hit=distance_to_hit)
    vls=unperturbed_stream(N_pal5, gap_center, b, rgc)
    #print (vls)
    #jkl

    #bck=read_cmd_file(rgc, d_galaxy, mag_limit)
    bck=read_cmd_file_no_rescale(rgc, d_galaxy, mag_limit)# <<<<<<<<<< remember to comment this out 
    s=b.select(bck[['x_coord', 'y_coord']])
    img= [np.concatenate([vls[0], s.x.values]), np.concatenate([vls[1], s.y.values])]

    return img


def run_intact_image(bw, rgc, mag_limit, distance):
    cent_values={'30_40': ((2.5, 0.1), (35.0, 35.2)), '10_20': ((2.5, 0.55), (15., 15)), '50_60': ((-4, .35),  (55, 55))}
    distances_to_hit={'30_40': 0.7, '10_20':0.5, '50_60': 0.8}
    vhalos={'10_20': -50, '30_40': 70, '50_60': -50}
    box_size=(15, 6)  #original box size, should this be the FOV of Roman? Obviously can't go beyond that >>
    #img= make_an_image(distance, rgc, mag_limit,  cent_values[rgc][0], box_size, cent_values[rgc][1],  distance_to_hit=distances_to_hit[rgc], vhalo=vhalos[rgc])
    img=make_an_image_intact(distance, rgc, mag_limit,   cent_values[rgc][0], box_size,  cent_values[rgc][1])
    meshgrid, grid_data, bounds, gridding_size = make_grid(np.array(rgc.split('_')).astype(float).mean(),  5, 2)
    dt=draw_data_round_grid(img, bw, bounds)
    res=get_density(np.array(dt).T, gridding_size, grid_data, bw, bounds, nboot=5)
    res['meshgrid']=meshgrid
    res['data']=dt

    #fig, ax=plt.subplots(figsize=(10*1.5, 2*1.5), nrows=2)
    #ax[0].scatter(img[0], img[1], s=0.1)
    #x[0].scatter(grid_data[:,0], grid_data[:,1], s=10, marker='*', color='y')
    #ax[0].scatter(dt[0], dt[1], s=0.1)

    #c=ax[1].contourf(meshgrid[0], meshgrid[1], res['max_eigen'], alpha=0.5, cmap='cubehelix')
    #_ =ax[1].contour(meshgrid[0], meshgrid[1], res['density'], alpha=0.5, cmap='coolwarm')
    #ax[1].scatter(dt[0], dt[1], s=0.1)
    #plt.show()
    dmod_galaxy=5*np.log10(distance*1000/10.0)
    return {'bw{:.1f}dmod_galaxy{:.2f}'.format(bw,  dmod_galaxy):  res}

def run_offcentered_image(bw, rgc, mag_limit, distance):
    cent_values={'30_40': ((3.5, 0.1), (33.0, 35.2)), '10_20': ((1.5, 0.55), (15., 15)), '50_60': ((-3, .35),  (55, 55))}
    distances_to_hit={'30_40': 0.7, '10_20':0.5, '50_60': 0.8}
    vhalos={'10_20': -50, '30_40': 70, '50_60': -50}
    box_size=(30, 6)  #original box size, should this be the FOV of Roman? Obviously can't go beyond that >>
    img= make_an_image(distance, rgc, mag_limit,  cent_values[rgc][0], box_size, cent_values[rgc][1],  distance_to_hit=distances_to_hit[rgc], vhalo=vhalos[rgc])
    #img=make_an_image_intact(distance, rgc, mag_limit,   cent_values[rgc][0], box_size,  cent_values[rgc][1])
    meshgrid, grid_data, bounds, gridding_size = make_grid(np.array(rgc.split('_')).astype(float).mean(),  5, 2)
    dt=draw_data_round_grid(img, bw, bounds)
    res=get_density(np.array(dt).T, gridding_size, grid_data, bw, bounds, nboot=5)
    res['meshgrid']=meshgrid
    res['data']=dt

    #fig, ax=plt.subplots(figsize=(10*1.5, 2*1.5), nrows=2)
    #ax[0].scatter(img[0], img[1], s=0.1)
    #x[0].scatter(grid_data[:,0], grid_data[:,1], s=10, marker='*', color='y')
    #ax[0].scatter(dt[0], dt[1], s=0.1)

    #c=ax[1].contourf(meshgrid[0], meshgrid[1], res['max_eigen'], alpha=0.5, cmap='cubehelix')
    #_ =ax[1].contour(meshgrid[0], meshgrid[1], res['density'], alpha=0.5, cmap='coolwarm')
    #ax[1].scatter(dt[0], dt[1], s=0.1)
    #plt.show()
    dmod_galaxy=5*np.log10(distance*1000/10.0)
    return {'bw{:.1f}dmod_galaxy{:.2f}'.format(bw,  dmod_galaxy):  res}

def run_stuff(rgc, mag_limit):

    from concurrent.futures import ThreadPoolExecutor, wait , ALL_COMPLETED
    from  functools import partial

    def run_process(d, bw):
         return run_image(bw, rgc, mag_limit, d)
    ds= np.arange(500, 10000, 100)
    bws=np.arange(0.1, 2, 0.1)
    #bws=[0.7, 0.8]
    iterables=list(np.array([(x, y) for x, y in np.array(list(itertools.product(ds, bws)))]).T)
    method=partial(run_process)
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures=list(executor.map( method, *iterables, timeout=None, chunksize=10))

    results=[x for x in futures]
    #something off with the parallilsm? try for loop 
    #results=[]
    #for d in ds:
    #    for bw in bws:
    #        results.append( run_image(bw, rgc, mag_limit,d))
    mhalo=5e6
    iteration=f'{int(np.random.uniform(0, 10000)):05}'
    #creative filename
    fname= 'pipeline_rgc{}_mhalo{:.2e}_maglimit{}_run{}'.format(rgc, mhalo, mag_limit, iteration)
    np.save(path_pipeline+'/{}.npy'.format(fname),results)

if __name__ =='__main__':
    #vals=run_image(0.7, '10_20', 28.69, 800)
    vals= [ run_offcentered_image(bw, '30_40', 27.15, 1000) for bw in np.arange(0.5, 1, 0.1)]
    np.save('{}.npy'.format('appendix_offcenter'),vals)
    #np.save('test.npy', vals)
    jk
    for i in range(0, 10):
        for m in [27.15, 28.69]: 
            for rgc in rgcs:
                run_stuff(rgc, m)

