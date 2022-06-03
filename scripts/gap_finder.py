#imports 
import astropy.units as u
import numpy as np
import popsims #custom libray for plotting aesthetics
import matplotlib.pyplot as plt
#%matplotlib notebook
from sklearn.preprocessing import MinMaxScaler


#import HSS
import seaborn as sns
import matplotlib as mpl


import pandas as pd
sns.set_style("dark")
mpl.rc('xtick', labelsize=16) 
mpl.rc('ytick', labelsize=16) 
font = {'axes.titlesize'      : 'large',   # fontsize of the axes title
        'axes.labelsize'      : 'large', # fontsize of the x any y labels
        'size'   : 20}


from findthegap.gapper import Gapper
import torch
#paths
path_plot = '/users/caganze/research/stellarstreams/figures/paper/'
#path_data = '/users/caganze/research/stellarstreams/data/rotating/'
path_data = '/users/caganze/research/stellarstreams/data/stream/'
isochrone_path='/users/caganze/research/stellarstreams/data/isochrones/'



def get_cutout_m31(rgc, mhalo, mag_limit):
    filename=path_data+'/gaps_at_M31{} mlimit {}Mhalo={:.2e}_cutout'.format(rgc, mag_limit, mhalo) 
    return pd.read_csv(filename).values
                       
    
def get_cutout_distance(mhalo, mag_limit, dmod):
    filename=path_data+'/gaps_at_OTHER{:.2f}Mhalo={:.2e}_maglimit{}_cutout.txt'.format(dmod, mhalo, mag_limit)
    return pd.read_csv(filename).values


def boostrap_density_estimate(gapper_base, bw, grid_data, data, bounds, nboostrap, max_eigenvalue=True):
    ##purpose: with a gapper object, 
    #we can estimate the hessian and the maximum eigevalue by boostrapping
    PiHPi_boots=[]
    maxeigval_PiHPi_boots =[]

    mineigval_PiHPi_boots =[]
    
    #loop over all bootstraps
    for i in range(nboostrap):
        boot_indx = np.random.choice(np.arange(data.shape[0]), data.shape[0], 
                                     replace=True) ## Sample with replacement:bootstrap

        gapper_ = Gapper(data[boot_indx], bw, bounds)
        PiHPis_grid = []
        eigval_PiHPi = [] 

        for pt in grid_data:
            _pihpi = gapper_.get_PiHPi(pt) 
            _pihpi_eigval, _pihpi_eigvec = np.linalg.eigh(_pihpi)

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
        print(f'Run {i} finished')
        
    return np.array(maxeigval_PiHPi_boots), np.array(mineigval_PiHPi_boots), np.array(PiHPi_boots)

def detect_gap_by_boostrap(bws, data, xlims, ylims, rescale=False, max_eigenvalue=True, nboostrap=5):
    
    """
    Purpose detect gap by bootstrapping and using an median over several bandwidths
    
    """
    #min_bw_x= 0.07
    #min_bw_y= 0.07
    
    #Boundaries for the Gapper (if none are provided, this is the default mode)
    bounds = np.array([[np.min(data[:,d]),np.max(data[:,d])] for d in range(data.shape[1])])

    #scale of y to 
    x_to_y= np.ptp(data[:,0])/ np.ptp(data[:,1])

    n= 100

    gridding_size = [ n, np.int(0.1*n/x_to_y)]

    
    grid_linspace = [ np.linspace(bounds[d][0], bounds[d][1], gridding_size[d]) for d in range(2) ]
    #could use a rectangular grid instead


    meshgrid = np.meshgrid(*grid_linspace, indexing='ij')

    meshgrid_ravel = [ xi.ravel().reshape(-1,1) for xi in meshgrid]
    grid_data = np.hstack(meshgrid_ravel)
    

    res=dict(zip(bws, [None for x in bws]))
    
    density_matr=None

    for bwx in bws: #option for doing this over multiple bandwidths
        #option for rescaling 
        
        if rescale:
            #new rescale
            scale=1
            target_u = 0.1## <= "target_unit", arbitraty value but the bandwidth you'll use? or twice the bandwidth?
            ## see below, apparently twice the bw.. not sure i understand why... but the plots look best doing this...

            exp_width_y = 0.2  ## fake stream width is expected .2
            exp_width_x = 0.7  ## Fake gap width is expected .7 
            ## As we discussed, if you don't want to span accross search, just say you make assumption / present protocols for
            ## specific gap sizes to some extend? We can think how to combine runs with multiple / various gap sizes later...

            ## Convoluted way of doing the rescaling but that's how i understand it:
            ## Get "how many expected_width" you can put in each dimension in the data-range
            Ny = (np.max(data[:,1]) - np.min(data[:,1]))/exp_width_y
            Nx = (np.max(data[:,0]) - np.min(data[:,0]))/exp_width_x

            print(Ny, Nx)

            ## Your new data-space should range from (0, (number of expected widths) * target_unit)
            ## So (0, b_x) / ( 0, b_y) respectively
            b_y = Ny * target_u
            b_x = Nx * target_u

            ## This does the actual rescaling to (0, b) in each dimension
            data_resc_y = ((data[:,1] - np.min(data[:,1]))*b_y)/(np.max(data[:,1]) - np.min(data[:,1]))
            data_resc_x = ((data[:,0] - np.min(data[:,0]))*b_x)/(np.max(data[:,0]) - np.min(data[:,0]))
            data_resc = np.stack([data_resc_x, data_resc_y],axis=1)


            bounds = np.array([[np.min(data_resc[:,d]),np.max(data_resc[:,d])] for d in range(data_resc.shape[1])])

            ## Do we still want to grid more finely on x than y with the rescaling? Maybe we don't need anymore? idk
            gridding_size = [int(Nx)*10, int(Ny)*10]
            grid_linspace = [ np.linspace(bounds[d][0], bounds[d][1], gridding_size[d]) for d in range(2) ]

            meshgrid = np.meshgrid(*grid_linspace, indexing='ij')

            meshgrid_ravel = [ xi.ravel().reshape(-1,1) for xi in meshgrid]
            grid_data = np.hstack(meshgrid_ravel)

            bw = target_u  ### <= so. I need to wrap my mind around this but, need to work on bw < target_u. 

            ## target_u / 2. also looked alrightish?
            gapper_base= Gapper(data_resc, bw, bounds)
            grid_density = gapper_base.kde.score_samples(torch.tensor(grid_data))
            density_matr = grid_density.reshape((gridding_size[0], gridding_size[1]))
            
            #old rescale using min max 
            #fit the grid using
            #minmaxscal = MinMaxScaler().fit(data)
            #get transformed data and grid
            #data_resc = minmaxscal.transform(data)
            #grid_data_resc = minmaxscal.transform(grid_data)
            #compute new bounds
            #bounds = np.array([[np.min(data_resc[:,d]),np.max(data_resc[:,d])] for d in range(data_resc.shape[1])])
            
            #estimate density a
            #rescale the bandwidth by the change from previous data to rescaled data
            #scale=np.ptp(data_resc[:,0])/ np.ptp(data[:,0])
            #gapper_base = Gapper(data_resc, bw*scale, bounds)
            #grid_density = gapper_base.kde.score_samples(torch.tensor(grid_data_resc))
            #density_matr = grid_density.reshape((gridding_size[0], gridding_size[1]))
            #estimate hessian by bootstrapping
            data= data_resc
            maxeigval_PiHPi_boots,  mineigval_PiHPi_boots, PiHPi_boots = boostrap_density_estimate(gapper_base, bw,\
                                                                           grid_data, data_resc, bounds, \
                                                                           nboostrap,\
                                                                           max_eigenvalue=max_eigenvalue)
        if not rescale:
            #compute density along the grid 
            scale=1.
            gapper_base = Gapper(data, bw, bounds)
            grid_density = gapper_base.kde.score_samples(torch.tensor(grid_data))
            #density matrix 
            density_matr = grid_density.reshape((gridding_size[0], gridding_size[1]))

            #compute piHpi matrix by bootstraping
            maxeigval_PiHPi_boots,  mineigval_PiHPi_boots, PiHPi_boots = boostrap_density_estimate(gapper_base, bw,\
                                                                           grid_data, data, bounds, nboostrap,\
                                                                           max_eigenvalue=max_eigenvalue)
     
        #visualize and take the median
        maxeigval_PiHPi_boots = np.array(maxeigval_PiHPi_boots)
        print(maxeigval_PiHPi_boots.shape)

        mineigval_PiHPi_boots = np.array(mineigval_PiHPi_boots)
        print(mineigval_PiHPi_boots.shape)

        #median
        med_maxeigval_pihpi = np.median(maxeigval_PiHPi_boots, axis=0)
        med_maxeigval_pihpi_resh = med_maxeigval_pihpi.reshape((gridding_size[0], gridding_size[1]))

        med_mineigval_pihpi = np.median(mineigval_PiHPi_boots, axis=0)
        med_mineigval_pihpi_resh = med_mineigval_pihpi.reshape((gridding_size[0], gridding_size[1]))



        res[bwx]= {'density':density_matr, \
                  'max_eigen':med_maxeigval_pihpi_resh,
                  'min_eigen':med_mineigval_pihpi_resh,
                  'meshgrid':meshgrid, 
                  'PiHPi': PiHPi_boots,
                  'bw_rescale': scale,
                  'data': data}

    return res

def visualize_gap_finder(all_res):

    data=all_res['data']
    gap_mask= all_res['gap_mask']
    stream_mask= all_res['stream_mask']
    grid_data=all_res['grid_data']
    grid= all_res['grid']
    pol= all_res['pol']
    file_prefix=all_res['file_prefix']
    gap_bw= all_res['gap_bw']
    stream_bw= all_res['stream_bw']
    stream_density_along_grid= all_res['stream_density_along_grid']
    stream_density_out= all_res['stream_density_out']
    gap_stars= all_res['gap_stars']
    gap_loc= all_res['gap_loc']
    gap_size= all_res['gap_size']
    stream_mask= all_res['stream_mask']
    gap_mask = all_res['gap_mask']
    res= all_res['gappy_res']
    
    #visualize gap finder
    fig, (ax, ax1)=plt.subplots(figsize=(12, 6), nrows=2, sharex=True)

    meshgrid= res[gap_bw]['meshgrid']

    ax.contour(meshgrid[0], meshgrid[1],  res[gap_bw]['density'],  20, \
                      cmap='spring', alpha=.5)

    cax = fig.add_axes([1.01, 0.6, .02, 0.3])
    p= ax.contourf(meshgrid[0], meshgrid[1], res[gap_bw]['max_eigen'],\
                            20, cmap='afmhot',  extend='min')
    plt.colorbar(p, label=r'Max Eigenvalue of $\Pi H \Pi$', ax=ax, cax=cax)
    ax.set(xlabel='x (kpc)', ylabel='y (kpc)')
    plt.minorticks_on()

    ax1.scatter(data[:,0], data[:,1], s=1, c='k')
    ax1.scatter(grid_data[:,0][gap_mask.flatten().astype(bool)], \
                grid_data[:,1][gap_mask.flatten().astype(bool)], s=10, marker='+', c='#FFDC00')
    ax1.set(xlabel='x (kpc)', ylabel='y (kpc)')
    ax.tick_params(labelbottom=True)
    plt.tight_layout()
    plt.savefig(path_plot+file_prefix+'stream_cutout_gap.jpeg', rasterized=True,\
                bbox_inches='tight')
    
    #visualize stream finder
    fig, (ax, ax1)=plt.subplots(figsize=(12, 6), nrows=2, sharex=True)

    meshgrid= res[gap_bw]['meshgrid']

    ax.contour(meshgrid[0], meshgrid[1],  res[stream_bw]['density'],  20, \
                      cmap='spring', alpha=.5)

    cax = fig.add_axes([1.01, 0.6, .02, 0.3])
    p= ax.contourf(meshgrid[0], meshgrid[1], res[stream_bw]['max_eigen'],\
                            20, cmap='afmhot',  extend='min')
    plt.colorbar(p, label=r'Max Eigenvalue of $\Pi H \Pi$', ax=ax, cax=cax)
    ax.set(xlabel='x (kpc)', ylabel='y (kpc)')
    plt.minorticks_on()

    ax1.scatter(data[:,0], data[:,1], s=1, c='k')
    ax1.scatter(grid_data[:,0][stream_mask.flatten().astype(bool)], \
                grid_data[:,1][stream_mask.flatten().astype(bool)], s=10, marker='+', c='#FFDC00')

    ax1.set(xlabel='x (kpc)', ylabel='y (kpc)')
    ax.tick_params(labelbottom=True)
    plt.tight_layout()
    plt.savefig(path_plot+ file_prefix+'stream_cutout_stream.jpeg', rasterized=True, \
                bbox_inches='tight')
    
    
    #visualize polynomila fit
    fig, (ax, ax1)=plt.subplots(figsize=(12, 6), nrows=2, sharex=True)

    ax.scatter(data[:,0], data[:,1], s=1, c='k')
    #ax.plot(grid_data[:,0], pol(grid_data[:,0]), linewidth=3, c='b')
    ax.plot(grid_data[:,0], pol(grid_data[:,0])-stream_bw, linewidth=3, c='#0074D9')
    ax.plot(grid_data[:,0], pol(grid_data[:,0])+stream_bw, linewidth=3, c='#0074D9')

    #ax.plot(grid_data[:,0], pol(grid_data[:,0])+0.5, linewidth=3, c='r')
    ax.plot(grid_data[:,0], pol(grid_data[:,0])+0.5-stream_bw, linewidth=3, c='#FF4136')
    ax.plot(grid_data[:,0], pol(grid_data[:,0])+0.5+stream_bw, linewidth=3, c='#FF4136')

    ax.set(xlabel='x (kpc)', ylabel='y (kpc)')

    ax.tick_params(labelbottom=True)

    plt.tight_layout()

    ax1.plot(grid, stream_density_along_grid, label='Stream', color='#0074D9' )
    ax1.fill_between(grid, stream_density_along_grid-np.sqrt(stream_density_along_grid),  
                  stream_density_along_grid+np.sqrt(stream_density_along_grid), alpha=0.5, color='#0074D9')
    ax1.plot(grid, stream_density_out , alpha=1, label='Background' , color='#FF4136')
    ax1.fill_between(grid, stream_density_out-np.sqrt(stream_density_out),
                    stream_density_out+np.sqrt(stream_density_out),
                    alpha=0.5, color='#FF4136')
    ax1.set(ylabel=' # of stars', xlabel='x (kpc)')
    ax1.axvline(gap_loc[0], c='k', linestyle='--', label='Gap')
    plt.legend()
    plt.tight_layout()
    plt.savefig(path_plot+file_prefix+'stream_cutout_fit.jpeg', rasterized=True)
    
    #plt.show()
    return 

def run_gap_diagnostics(polynomial_degree=2, gap_threshold=99, stream_threshold=1, \
                        galaxy='M31', mag_limit=27.15, mhalo=5e6, rgc='50_60', 
                        dmod=None, gap_scale=0.1, visualize=True, rescale=False, nsteps_grid=5):
    #file prefix to save data
    file_prefix='gaps_at_{}{}_mlimit_{}Mhalo={:.2e}_dmod{:.2f}'.format(galaxy, rgc, mag_limit, mhalo, dmod)
    if rescale: file_prefix='gaps_rescaled_at_{}{}_mlimit_{}Mhalo={:.2e}_dmod{:.2f}'.format(galaxy, rgc, mag_limit, mhalo, dmod)

    #print ('bandwidth ..............', bw)

    
    #first read in the data 
    data=get_cutout_m31(rgc, mhalo, mag_limit)
    
    if galaxy=='OTHER':
        data=get_cutout_distance(mhalo, mag_limit, dmod)

    #take 10 % of vertical extent and 10% of horizontal extend  
    d= 10**(dmod/5+1)*u.pc.to(u.Mpc) 
    gap_bw= 0.6#*(d/0.77) #gap bw should be increas
    stream_bw= 0.15#*(d/0.77) #gap bw should be increasing with distance
    res_gap= detect_gap_by_boostrap([gap_bw],data, [data[:,0].min(), \
                                                                data[:,0].max()],\
                                [data[:,1].min(), data[:,1].max()], rescale=rescale)

    res_stream=  detect_gap_by_boostrap([stream_bw], data, [data[:,0].min(), \
                                                                data[:,0].max()],\
                                [data[:,1].min(), data[:,1].max()], max_eigenvalue=False, rescale=rescale)

    res= {**res_gap, **res_stream}
    
    #select threshold above 
    print (res)
    gap_mask = res[gap_bw]['max_eigen'] > np.percentile(res[gap_bw]['max_eigen'], gap_threshold) 

    meshgrid_ravel = [ xi.ravel().reshape(-1,1) for xi in  res[gap_bw]['meshgrid']]
    grid_data = np.hstack(meshgrid_ravel)
    
    #compute gap location
    gap_loc=(np.nanmean(grid_data[:,0][gap_mask.flatten().astype(bool)]), \
            np.nanmean(grid_data[:,1][gap_mask.flatten().astype(bool)]))
    gap_size=np.nanstd(data[:,0][gap_mask.flatten().astype(int)])
    
    #select stars in the stream
    stream_mask= (res[stream_bw]['max_eigen'] < np.percentile(res[stream_bw]['max_eigen'], stream_threshold) ).flatten()
    
    #fit poynomial  to the stream stars, make sure to add 
    x=grid_data[:,0][stream_mask.flatten().astype(bool)]
    y=grid_data[:,1][stream_mask.flatten().astype(bool)]
    x[-1]=gap_loc[0]
    y[-1]=gap_loc[1]
    pol= np.poly1d(np.polyfit(x, y, polynomial_degree))
    
    
    #compute the density along the polynomial fit 
    stream_density_along_grid =[]
    stream_density_out =[]
    xdiff= gap_size/2
    grid=np.arange(data[:,0].min(), data[:,0].max(), xdiff)
    for g in grid:
        mask= np.logical_and.reduce([data[:,1]> pol(g)-stream_bw,\
                                    data[:,1]< pol(g)+stream_bw, 
                                    data[:,0] >= g,
                                    data[:,0] <g+xdiff])

        mask2= np.logical_and.reduce([data[:,1]> pol(g)+0.5-stream_bw,\
                                    data[:,1]< pol(g)+0.5+stream_bw, 
                                    data[:,0] >= g,
                                    data[:,0] <g+xdiff])

        stream_density_along_grid.append(len(data[:,1][mask]))

        stream_density_out.append(len(data[:,1][mask2]))
        
    #figure out the number of stars in the gap
    gap_stars= len(data[:,0][np.logical_and.reduce([data[:,1]> pol(gap_loc[0])-stream_bw,\
                                    data[:,1]< pol(gap_loc[0])+stream_bw, 
                                    data[:,0] >= gap_loc[0],
                                    data[:,0] <gap_loc[0]+xdiff])])
    
    #visualization --> important, have to make sure I'm fiting the right thing 
    analysis_result={'data': data,
                   'gap_mask': gap_mask,
                   'stream_mask': stream_mask,
                    'grid_data': grid_data,
                    'grid': grid,
                    'pol': pol,
                    'file_prefix': file_prefix,
                    'gap_bw': gap_bw,
                     'stream_bw': stream_bw,
                     'stream_density_along_grid': stream_density_along_grid,
                     'stream_density_out': stream_density_out,
                     'gap_stars': gap_stars,
                     'gap_loc': gap_loc,
                     'gap_size': gap_size,
                     'stream_mask': stream_mask,
                     'gap_mask': gap_mask,
                     'gappy_res': res}
                     
    #visualize
    #visualize_gap_finder(analysis_result)

    #save 
    filename= file_prefix+'analysis_results' 
    np.save(path_data+'/{}.npy'.format(filename), analysis_result)


def run_gap_diagnostics_kat( galaxy='M31', mag_limit=27.15, mhalo=5e6, rgc='50_60', 
                        dmod=None, gap_scale=0.1, rescale=False, nsteps_grid=5):
    
    #file prefix to save data
    file_prefix='gaps_at_{}{}_mlimit_{}Mhalo={:.2e}_dmod{:.2f}'.format(galaxy, rgc, mag_limit, mhalo, dmod)
    if rescale: file_prefix='gaps_rescaled_at_{}{}_mlimit_{}Mhalo={:.2e}_dmod{:.2f}'.format(galaxy, rgc, mag_limit, mhalo, dmod)

    print (file_prefix)

    
    #first read in the data 
    data=get_cutout_m31(rgc, mhalo, mag_limit)
    
    if galaxy=='OTHER':
        data=get_cutout_distance(mhalo, mag_limit, dmod)

    #take 10 % of vertical extent and 10% of horizontal extend  
    #d= 10**(dmod/5+1)*u.pc.to(u.Mpc) 
    #gap_bw= 0.6*(d/0.77) #gap bw should be increas
    #stream_bw= 0.15*(d/0.77) #gap bw should be increasing with distance

    bws= np.linspace(0.01, 1.5, 15)

    print (data[:,0].min(), data[:,0].max())
    res={}
    for bw in bws:
            res_gap= detect_gap_by_boostrap([bw],data, [data[:,0].min(), \
                                                data[:,0].max()],\
                                        [data[:,1].min(), data[:,1].max()], \
                                        rescale=True, nboostrap=5)
            res.update({bw: res_gap})
                     
    filename= file_prefix+'analysis_results_morebandwidths' 
    np.save(path_data+'/{}.npy'.format(filename), res)

if __name__ =='__main__':
        #ds=([0.5, 0.77, 1.0, 1.5, 2.0, 2.5 ])*u.Mpc
        ds=([0.5, 0.6, 0.77, 0.8, 0.9, 1.0, 1.3, 1.5, 1.6, 1.7, 2., 2.5, 3., 3.5, 4., 4.5, 5.])*u.Mpc
        #manually compute bandwidths
        rgc='50_60'
        for mag_limit in [28.54, 27.15]:
                for dx in ds:
                    dmod = 5*np.log10(dx.to(u.pc)/(10*u.pc)).value
                    mhalo=5e6
                    galaxy='OTHER'
                    run_gap_diagnostics_kat( galaxy=galaxy, mag_limit=mag_limit, \
                        mhalo=mhalo, rgc=rgc, dmod=dmod, rescale=True)
                    #diag_results=run_gap_diagnostics(\
                    #        polynomial_degree=2, rescale=True,\
                    #         gap_threshold=97, stream_threshold=1, \
                    #        galaxy=galaxy, mag_limit=mag_limit, gap_scale=None, \
                    #        mhalo=mhalo, rgc=rgc,  dmod=dmod )