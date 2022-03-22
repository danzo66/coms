"""
1D arrays:
 'r',          --- spherical R in AU (1D grid)
 'theta'       ---- Spherical theta (radian)
 'sigma_dust',  --- large and small dust surface density  (g cm^-2)
 'sigma_g',     ---- gas surface density (g cm^-2)

2D arrays:
 'tgas',       ---- gas temperature (K, 2D array)
 'td1',        ---- large dust temperature (K)
 'td2',         ---- small dust temperature (K)
 'n_H2',        --- H2 number density (cm^-3) = rho_gas/2.8/mp
 'X_CO',        --- CO gas abundance (n_CO/n_H2) in thermo-chemical models
 'rho_d1',      ---- large dust density(g/cm^3)
 'rho_d2',      ---- small dust density(g/cm^3)
 'flux_Lyalpha' ---- Ly-alpha line flux (1210-1220A) (erg/cm^2/s)
 'flux_UV_cont' ---- UV flux (910-2000A) (erg/cm^2/s)
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.integrate as sp
from matplotlib.colors import ListedColormap
from scipy import stats as st



"""SETS THE MOLECULE AND THE DISC FOR THE CODE TO RUN"""
args = ["CH3CN", "HD_163296", 'x2.0x1.5_loglog']



def loaddata(molecule, disc, version, ifprint=True):
    '''
    loads the interpolated data
    takes in *args, after specifying the parameters within this variable
    ifprint is just whether or not you want to print the disc name etc.
    '''
    
    if ifprint == True:
        toprint = [molecule, disc, version]
        for i, p in enumerate(toprint):
            p = p.split('_')
            toprint[i]= p
        print(str(toprint[0][0]) + ' in ' + str(toprint[1][0]) + ' ' + str(toprint[1][1]))    
        
    if version == 'original':
        if ifprint == True:
            print(str('Original Data'))

        filename = "C:/Users/danip/OneDrive/Documents/Uni/Year 4/MPhys Project/MAPS discs/MAPS_working_dir/disc_" + str(disc) + "/" + str(disc) + "_disk_structure.npz"
        file = np.load(filename, allow_pickle=True)
        
        #creates meshgrid of original spherical coords
        #each entry in R_sph and TH_sph give the coordinates for one of the points            
        r = file['r']
        theta = file['theta']
        
        R_sph, TH_sph = np.meshgrid(r, theta)
        
        #converts spherical to cylindrical
        RR = R_sph*np.sin(TH_sph) 
        RR = np.transpose(RR)

        ZZ = R_sph*np.cos(TH_sph)
        ZZ = np.transpose(ZZ)
        
        data = [RR,                         #0      mesh grid of radial coords
                ZZ,                         #1      mesh grid of z coords
                r,                          #2
                
                file['td1'],                #3      temperature of dust size 1
                file['td2'],                #4      temperature of dust size 2
                file['tgas'],               #5
                
                file['rho_d1'],             #6      density of dust size 1
                file['rho_d2'],             #7      density of dust size 2
                
                file['sigma_dust'][0],      #8      dust 1 surface density
                file['sigma_dust'][1],      #9      dust 1 surface density
                file['sigma_g'],            #10      gas surface density
                
                file['n_H2'],               #11      number density of H_2
                file['X_CO'],               #12     CO abundance
            
                file['flux_Lyalpha'],       #13     Lyman alpha flux
                file['flux_UV_cont'],       #14     UV flux
                ]
        data.append(data[-1] + data[-2])    #total flux
        
    else:
        if ifprint == True:
            print(str(toprint[2][0]) + ' ' + str(toprint[2][1]) + ' resolution')
        
        filename = 'C:/Users/danip/OneDrive/Documents/Uni/Year 4/MPhys Project/MAPS discs/MAPS_working_dir/disc_' + str(disc) + '/interpolateddata/intdata_' + str(version) + '.npz'
        file = np.load(filename, allow_pickle=True)
        
        #dictionary of each array
        data = [file['RR_in'],              #0      mesh grid of radial coords
                file['ZZ_in'],              #1      mesh grid of z coords
                0,               #2      1d r coords
                
                file['td1_in'],             #3      temperature of dust size 1
                file['td2_in'],             #4      temperature of dust size 2
                file['tgas_in'],            #5      gas temperature
                
                file['rho_d1_in'],          #6      density of dust size 1
                file['rho_d2_in'],          #7      density of dust size 2
                
                0,     #8      dust 1 surface density
                0,     #9      dust 2 surface density
                0,         #10     gas surface density, didnt like being interpolated but also is not needed so who cares
                
                file['n_H2_in'],            #11     number density of H_2
                file['X_CO_in'],            #12     CO abundance
                
                file['flux_Lyalpha_in'],    #13     Lyman alpha flux
                file['flux_UV_cont_in'],    #14     UV flux
                ]
        data.append(data[-1] + data[-2])    #total flux
        
    return data


data = loaddata(*args,)


"""CONSTANTS, MOLECULE PARAMETERS, DISC PARAMETERS"""
global CONSTS, MOLECULES, LUMINOSITIES
CONSTS = {'K_B' : 1.380649e-16,                  #Boltzman const in ergs
          'INVERSEMOLE' : 6.02214076e-23,        #inverse of avogadros number
          'L_SOL' : 9e33,                        #Solar luminosity
          'N_SITES': 1.5e15                      #number density of surface sites per grain, cm-2
          }

LUMINOSITIES = {"GM_Aur":       (6.5630E-03 *CONSTS['L_SOL']),
                "HD_163296":    (2.1411     *CONSTS['L_SOL']),
                "MWC_480":      (5.6090E-01 *CONSTS['L_SOL']),
                }

# 0: binding energies K         https://arxiv.org/ftp/arxiv/papers/1701/1701.06492.pdf
# 1: mass                       https://pubchem.ncbi.nlm.nih.gov/compound/methanol
# 2: sticking coefficient       Bisschop et al 2006, cant find for more than CO so all 0.9 for now
# 3: gas abundance              https://www.aanda.org/articles/aa/pdf/2014/03/aa22446-13.pdf Table 1
    #gas abundance here is when everythinh desorped 
    #for where non thermal desorption, use a draction of the abundance here --> walsh 2014
# 4: ice abundance              
#                       0       1       2       3          4
MOLECULES = {"CO" :     [1300,  28.01,  0.9,    data[-3],  1.3e-5],    
             "CH3OH" :  [5000,  32.042, 0.9,    2.2e-10,   8.3e-7],
             "CH3CHO" : [5400,  44.05,  0.9,    7e-12,     3e-11],
             "CH4" :    [960,   16.043, 0.9,    6.5e-8,    2.5e-5],
             "CH3CN" :  [4680,  41.05,  0.9,    8.7e-13,   3.7e-9],
             "HCO" :    [2400,  29.01,  0.9,    3.3e-10,   1.7e-13],
             "C2H6":    [1600,  30.069, 0.9,    1.5e-10,   7.8e8],
             "HCOOCH3": [4100,  60.05,  0.9,    1e-15,     1e-15],  #binding energy from walsh 2014
             "Glycine": [300,   75.07,  0.9,    1e-16,     1e-10]
             }




def plot_data(RR, ZZ, R, td1, td2, t_gas, rho_d1, rho_d2, sigma_dust1, sigma_dust2, sigma_g, n_H2, X_CO, flux_Lyalpha, flux_UV_cont, flux_UV_tot, molecule, disc, version):
    """
    plots the parameters in cylindrical coords
    takes in *data, *args
    this doesn't work for the 1d plots because they are all zero at the minute
    """
    
    #sets up figure and colour map
    fig = plt.figure(figsize=(20,15))
    fig.tight_layout(pad=10.0)
    cmap = 'Spectral_r'
    

    label = [r'$T_{d_1}$(K)', r'$T_{d_2}$(K)', r'$T_{gas}$(K)',
             r'log($\rho_{d_1}(g/cm^3)$)', r'log($\rho_{d_2}(g/cm^3)$)', r'log($n_{H_2}(cm^{-3}$))', r'log($X_{CO}$)', 
             r'$\Sigma_{gas}$', r'$\Sigma_{d_1}$', r'$\Sigma_{d_2}$']
        
    
    #plots each subfigure
    for i, parameter in enumerate([td1, td2, t_gas, rho_d1, rho_d2, n_H2, X_CO, sigma_g, sigma_dust1, sigma_dust2]):   
        
        zmax = np.max(ZZ)
        zmax = 100
        
        #temperature plots
        if i < 3:
            ax = fig.add_subplot(3,4,(1+i))
            
            #sets contour levels to an appropriate amount
            if i == 0:
                clevels = np.linspace(1, 200, 20)
                llevels = [20, 50, 100]
                plt.ylabel("z (AU)")
            elif i == 1:
                clevels = np.linspace(1, 200, 20)
                llevels = [10, 20, 50,]
            elif i == 2:
                clevels = np.linspace(1, 200, 20)   
                llevels = [10, 20, 30,]           
            
            plt.contourf(RR, ZZ, parameter, levels=clevels, cmap=cmap)
            cbar = plt.colorbar()
        
            plt.contour(RR,ZZ,parameter,levels=llevels,colors='w',linewidths=1)
            
            ax.set_ylim([0,zmax])
            ax.set_aspect('equal')
            
            plt.title(label[i])
            plt.xlabel("r (AU)")

            
        #logarithmic density plots
        elif i < 7:
            parameter = np.log10(parameter)

            ax = fig.add_subplot(3,4,(2+i))
            
            #sets contour levels for each subfigure
            dmax = math.ceil(np.nanmax(parameter))
            dmin = math.floor(np.nanmin(parameter))
            
            if i == 3:
                clevels = np.linspace(-30, dmax, 20)
                llevels = np.linspace(dmin, dmax, 4, endpoint=False)
                plt.ylabel("z (AU)")
            elif i == 4:
                clevels = np.linspace(-30, dmax,20)
                llevels = np.linspace(dmin, dmax, 4, endpoint=False)
            elif i == 5:
                clevels = np.linspace(-3, dmax, 20)
                llevels = np.linspace(dmin, dmax, 4, endpoint=False)
            elif i == 6:
                clevels = np.linspace(-30, dmax, 20)
                llevels = np.linspace(dmin, dmax, 4, endpoint=False)
                
            plt.contourf(RR,ZZ,parameter,levels=clevels, cmap=cmap)
            cbar = plt.colorbar()
        
            plt.contour(RR,ZZ,parameter,levels=llevels,colors='w', linewidths=1)
            
            ax.set_ylim([0,zmax])
            ax.set_aspect('equal')
            
            plt.title(label[i])
            plt.xlabel("r (AU)")


        '''
        #surface density    
        elif i > 6:
            ax = fig.add_subplot(3,4,(2+i))
            plt.plot(R, parameter)
            ax.set_yscale("log")
            
            plt.title(label[i])
            plt.xlabel("r (AU)")
            if i == 7:
                plt.ylabel(r"$(g/cm^2)$")
        '''


def photodissociation(RR, ZZ, R, td1, td2, t_gas, rho_d1, rho_d2, sigma_dust1, sigma_dust2, sigma_g, n_H2, X_CO, flux_Lyalpha, flux_UV_cont, flux_UV_tot, molecule, disc, version, ifplot=False):
    """
    determines the visual extinction of the disc
    by considering F*, the flux from the star if there were no dust
    and the UV flux at a given point
    
    Takes in *data, *args
    ifplot is just whether you want to plot the different steps to get to the visual extinction
    outputs the visual extinction
    """    
    
    """determines the Flux without stuff in the way"""
    
    L_star_uv = LUMINOSITIES[disc] #luminosity

    #for a number of these, there are ij variables
    #and not ij variables
    #this is for testing purposes
    #ij is the individual value, and is used in the for loop
    #these are added to arrays which aren't used going forwards
    #but just to check the shape and see if its good or not

    flux_star_uv = np.zeros(RR.shape)   #uv flux without any dust
    exptau_uv = np.zeros(RR.shape)      #exopnential of the optical depth
    tau_uv = np.zeros(RR.shape)         #optical depth
    A_v = np.zeros(RR.shape)           #uv extinction

    #iterates over all rows in the r meshgrid
    #then all the bits in that row
    #then finds the radius of that coordinate with the same z coordinate 
    #then changes the above array entry to the flux at that point
    for i, row in enumerate(RR):
        for j, r in enumerate(row):
            radius_AU = math.sqrt(r**2+ZZ[i,j]**2)
            radius_m = radius_AU*(1.496e13) #converts from AU to m
            
            flux_star_uv[i,j] = L_star_uv/(4*math.pi*radius_m**2) #flux with nothing in the way
            
            #(1,430) should be 57411.4/215.371
            exp_minustau_uv_ij = flux_UV_tot[i,j]/flux_star_uv[i,j] #exponential of optical depth for each grid point
            
            #gets rid of problematic values like nan or negative
            if np.isnan(exp_minustau_uv_ij) == True:
                #exp_minustau_uv_ij = 0.0000000001
                tau_uv[i,j] = float("nan")
            
            #from regions where theres much emission measured flux is higher than the star flux
            elif exp_minustau_uv_ij <= 0:
                exp_minustau_uv_ij = 10**(-16)
            
            else:
                exptau_uv[i,j] = exp_minustau_uv_ij

            tau_uv_ij = -math.log(exp_minustau_uv_ij) #actual optical depth
            tau_uv[i,j] = tau_uv_ij
            
            A_v[i,j] = tau_uv_ij/3.02 #change to A_v

    
    if ifplot == True:
        fig = plt.figure(figsize=(20,15))
        fig.tight_layout(pad=10.0)
        cmap = 'Spectral_r'
        
        plots = [[flux_UV_tot, np.logspace(-4, 2.6, 1000, endpoint=True, base=10), "Log UV Flux Dust"],
                 [flux_star_uv, np.linspace(0, 1e4, 1000, endpoint=True), "UV Flux Star"],
                 [tau_uv, np.linspace(0, 20, 1000, endpoint=True), "Optical Depth"],
                 [A_v, np.linspace(0, 10, 1000, endpoint=True), "Visual Extinction"],
                 ]
        
        for i, parameter in enumerate(plots):
            clevels = parameter[1]
            ax = fig.add_subplot(2,2,i+1)
            
            #photodissociation at extinction 3
            plt.contour(RR, ZZ, A_v, levels=3, colors='darkviolet', linewidths=2)
        
            plt.contourf(RR,ZZ, parameter[0], levels=clevels, cmap=cmap)
            plt.colorbar()
            ax.set_aspect("equal")
            ax.set_xlim(0,200)
            ax.set_ylim(0,125)
            plt.title(parameter[2])
    
    return A_v    




def freezeoutrate(RR, ZZ, R, td1, td2, t_gas, rho_d1, rho_d2, sigma_dust1, sigma_dust2, sigma_g, n_H2, X_CO, flux_Lyalpha, flux_UV_cont, flux_UV_tot, molecule, disc, version):
    """
    determines the freeze out rate of the molecule as in van't Hoff 2016
    takes in *data, *args
    outputs freezeout rate
    """
    
    """MOLECULE PARAMETERS"""
    molecule = MOLECULES[molecule]
    #mass, g
    m_mol = molecule[1]*CONSTS['INVERSEMOLE']
    #sticking parameter
    S_mol = molecule[2]

    
    #MEAN THERMAL VELOCITY OF MOLECULES
    vel = np.sqrt((8 * CONSTS['K_B'] * t_gas)/(math.pi * m_mol))  
    
    #FINDING THE REACTION CROSS SECTION  
    #min and max grain radii, cm
    #https://arxiv.org/pdf/1610.06788.pdf
    amin = 0.005e-4
    amax2 = 1e-4
    amax1 = 1e-1
    
    #CONSTANT FOR CROSS SECTION gcm-3
    #particle density gcm-3
    #https://www.aanda.org/articles/aa/pdf/2005/43/aa2249-04.pdf
    
    rho_dust = rho_d1 + rho_d2
    rho_bulk = 3.5    
    
    rho_integrand = lambda a : a**(-3.5) * rho_bulk * (((4*math.pi)/3) * a**3)  
    rho_integral = sp.quad(rho_integrand, amin, amax1)
    
    
    # constant of proportionality between integral and rho
    Const = rho_dust / rho_integral[0]                                         
    
    #CROSS SECTION PER UNIT VOLUME cm2cm-3
    xs_integrand = lambda a : a**(-3.5) * (math.pi * a**2)
    xs_integral = sp.quad(xs_integrand, amin, amax1)
    xsection = Const * xs_integral[0]
    
    
    #n_dust = (rho_dust * rho_bulk) / ((4/3) * math.pi * a_avg**3)   #assuming spherical
    #xsection = 4 * math.pi * a_avg**2
    
    n_dust = 1      #no its not its just so i dont have to delete everything
    
    #FREEZE OUT RATE! s-1
    k_f_mol = vel * n_dust * xsection * S_mol
    
    return k_f_mol




def mol_abundance(RR, ZZ, R, td1, td2, t_gas, rho_d1, rho_d2, sigma_dust1, sigma_dust2, sigma_g, n_H2, X_CO, flux_Lyalpha, flux_UV_cont, flux_UV_tot, 
                 molecule, disc, version, x_gas,
                 ifplot = False):    
    '''
    determines the phase ratio of the molecule throughout the disc
    by equating the freeze out and desorption rates
    
    takes in *data, *args, x_gas
    if the molecule is CO, this will be the observational data array X_CO
    otherwise, it is a constant initial abundance from Walsh et al. 2014
    x_ice is constant initial abundance for all molecules
    ifplot is just whether you want to plot the phase ratio and snowline
    
    outputs x_ice and phaseratio
    '''
    
    """MOLECULAR PARAMETERS"""
    mol_para = MOLECULES[molecule]
    #Binding energy
    E_b = mol_para[0]
    E_b_ergs = E_b * CONSTS["K_B"]
    #mass, g
    m_mol = mol_para[1] * CONSTS['INVERSEMOLE']
    

    """VIBRATIONAL FREQUENCY"""
    nu0 = math.sqrt((2 * CONSTS['N_SITES'] * E_b_ergs) / (math.pi**2 * m_mol))
    
    
    """CALLS FREEZE OUT RATE FUNCTION"""
    k_f = freezeoutrate(*data, *args)

    
    """DESORPTION RATE"""
    k_d_d1 = nu0 * np.exp(-E_b / td1)
    k_d_d2 = nu0 * np.exp(-E_b / td2)
    k_d = (k_d_d1 + k_d_d2)
        

    """TOTAL AND ICE ABUNDANCES
    assumes constant abundance through time"""    
    n_ice = MOLECULES[molecule][4] * n_H2
    x_ice = n_ice / n_H2
    
    
    """PHASE RATIO"""
    phaseratio = k_f / k_d
    
    #for plotting
    expkd = np.log(k_d)    
    prlog = np.log(phaseratio)


    """PLOTTING"""
    if ifplot == True:
        fig = plt.figure(figsize=(20,10))
        
        ax = fig.add_subplot(1,1,1)          
        
        cmapT = "Spectral_r"
        clevelsT = np.linspace(-300, 30, 125, endpoint=True)
        clevelspr = np.linspace(-60, +60, 125, endpoint=True)

        
        plt.contourf(RR, ZZ, prlog, levels=clevelspr, cmap=cmapT)
        plt.colorbar()
        plt.contour(RR, ZZ, phaseratio, levels=[1], colors='black', linewidths=2,)
        plt.title("log($k_d$) for " + str(molecule))
        plt.annotate(xy= [5, 120], text="Black line = molecule snow surface")
    
        ax.set_aspect("equal")
        ax.set_xlim(0,200)
        ax.set_ylim(0,125)
        

    return x_ice, phaseratio




def co_snowsurface(RR, ZZ, R, td1, td2, t_gas, rho_d1, rho_d2, sigma_dust1, sigma_dust2, sigma_g, n_H2, X_CO, flux_Lyalpha, flux_UV_cont, flux_UV_tot, 
                 molecule, disc, version,
                 ifplot = False):    
    """
    same as mol_abundances() but for CO
    this is needed for the lower bound of the regions, regardless of molecule
    
    this is a separate function because i was doing it differently before
    and actually calculating the ice abundance rather than just using a number
    that did not work and this is the relic of it...
    """
    
    """CO PARAMETERS"""
    mol_para = MOLECULES["CO"]
    #Binding energy
    E_b = mol_para[0] #Kelvin
    E_b_ergs = E_b * CONSTS["K_B"]  #ergs
    #mass, g
    m_mol = mol_para[1] * CONSTS['INVERSEMOLE']
    

    """VIBRATIONAL FREQUENCY"""
    nu0 = math.sqrt((2 * CONSTS['N_SITES'] * E_b_ergs) / (math.pi**2 * m_mol))
    
    
    """FREEZE OUT RATE"""
    k_f = freezeoutrate(*data, "CO", args[1], args[2])

    
    """DESORPTION RATE"""
    k_d_d1 = nu0 * np.exp(-E_b / td1)
    k_d_d2 = nu0 * np.exp(-E_b / td2)
    k_d = (k_d_d1 + k_d_d2)

    """PHASE RATIO
    when phaseratio = 1, that is snow surface"""
    phaseratio = k_f / k_d
    expkd = np.log(k_d)


    """PLOTTING CO BINDING ENERGY OVER DUST TEMP"""
    if ifplot == True:
        fig = plt.figure(figsize=(20,10))
        
        ax = fig.add_subplot(1,1,1)          
        
        cmapT = "Spectral_r"
        clevelsT = np.linspace(-80, 40, 125, endpoint=True)
        
        plt.contourf(RR, ZZ, expkd, levels=clevelsT, cmap=cmapT)
        plt.colorbar()
        plt.contour(RR, ZZ, phaseratio, levels=[1], colors='black', linewidths=2,)
        plt.title("log($k_d$) for CO")
        plt.annotate(xy= [5, 120], text="Black line = CO snow surface")
    
        ax.set_aspect("equal")
        ax.set_xlim(0,200)
        ax.set_ylim(0,125)
    

    return phaseratio




def get_contour_verts(cn):
    """
    gets the coordinates for a contour line cn
    https://stackoverflow.com/questions/18304722/python-find-contour-lines-from-matplotlib-pyplot-contour
    """
    contours = []
    # for each contour line
    for cc in cn.collections:
        paths = []
        # for each separate section of the contour line
        for pp in cc.get_paths():
            xy = []
            # for each segment of that section
            for vv in pp.iter_segments():
                xy.append(vv[0])
            paths.append(np.vstack(xy))
        contours.append(paths)

    return contours




def desorptiontemp(RR, ZZ, R, td1, td2, t_gas, rho_d1, rho_d2, sigma_dust1, sigma_dust2, sigma_g, n_H2, X_CO, flux_Lyalpha, flux_UV_cont, flux_UV_tot, molecule, disc, version):
    """
    finds the desorption temperature of the molecule
    this function isn't actually used in the model anymore, 
    but it gets printed out and plotted
    
    this function uses a linearly interpolated set of data, 
    because otherwise the temp comes out too large
    as most of the grid points in the inner disc
    so they raise the average
    
    takes in *data, *args
    
    detailed in van't Hoff 2016
    """
    
    #loading linearly spaced data so that we can average it
    datalinlin = loaddata(molecule, disc, 'x3.0_linlin', ifprint=False)
    
    """MOLECULE PARAMETERS"""
    mol = MOLECULES[molecule]
    #Binding energy
    E_b = mol[0]
    E_b_ergs = E_b * CONSTS['K_B']
    #mass, g
    m_mol = mol[1]*CONSTS['INVERSEMOLE']
    
    #CHARACTERISTIC  VIBRATIONAL FREQUENCY OF ADSORBED SPECIES IN POTENTIAL WELL (Allen + Robinson 1977)
    #frequency, s-1
    nu0 = math.sqrt((2 * CONSTS['N_SITES'] * E_b_ergs) / (math.pi**2 * m_mol))
            
        
    """ABUNDANCES"""
    #sets gas abundances
    if molecule == "CO":
        x_gas = X_CO
        n_ice = n_H2 * MOLECULES[molecule][4]        
        x_ice = n_ice/n_H2
        
    else: 
        n_gas = n_H2 * MOLECULES[molecule][3]
        n_ice = n_H2 * MOLECULES[molecule][4]
        
        x_gas = n_gas/n_H2
        x_ice = n_ice/n_H2
       
    
    """MOLECULE RATES"""
    k_f_mol = freezeoutrate(*datalinlin, *args)
    
    #list of values for desorption temp
    T_des_sum = []
    
    #gotta do this by element cos of negatives and zeroes
    for i, row in enumerate(x_ice):
        for j, x in enumerate(row):
            
            if x == 0:
                continue
            else:
                logarithm = (k_f_mol[i,j] * x_gas[i,j]) / (nu0 * x_ice[i,j])
            
            #deals with not good values
            if logarithm <= 0:
                continue
            elif np.isnan(logarithm) == True:
                continue
                
            #finds desorption temp for point, adds to list
            else:
                T_des_ij = -E_b / np.log(logarithm)
                T_des_sum.append(T_des_ij)
               
    T_des = sum(T_des_sum) / len(T_des_sum)
        
    return T_des




def snowline(RR, ZZ, R, td1, td2, t_gas, rho_d1, rho_d2, sigma_dust1, sigma_dust2, sigma_g, n_H2, X_CO, flux_Lyalpha, flux_UV_cont, flux_UV_tot, molecule, disc, version,
              phaseratio):
    '''finds snowline by plotting snowsurface contour
    getting the contour vertices
    finding which of the contours is the longest (to get rid of any extra ones)
    it finds the lowest r value and sets that as the snowline
    then creates a line for the x and y coords so it can be plotted
    
    takes in *data, *args, phaseratio
    '''
    
    fig = plt.figure()
    surfacecntr = plt.contour(RR, ZZ, phaseratio, levels=[1], colors='white', linewidths=2,)  
    
    surfaceverts = get_contour_verts(surfacecntr)[0]
    surfaceverts = np.array(surfaceverts, dtype=object)
    
    plt.close(fig)
    
    lens = np.zeros(len(surfaceverts))
    for i, cntr in enumerate(surfaceverts):
        lens[i] = len(cntr[:,1])
        
    longest = np.max(lens)
    bestsurfaceindex = np.where(lens==longest)[0]
    bestsurface = surfaceverts[bestsurfaceindex][0]
    
    r_snow = np.min(bestsurface[:,0])
    ym = r_snow * 0.5 + 20
    
    snow_y = np.linspace(0, ym, 10)
    snow_x = r_snow * np.ones(np.shape(snow_y))
    
    return r_snow, snow_x, snow_y, bestsurface




def abundances(RR, ZZ, R, td1, td2, t_gas, rho_d1, rho_d2, sigma_dust1, sigma_dust2, sigma_g, n_H2, X_CO, flux_Lyalpha, flux_UV_cont, flux_UV_tot, molecule, disc, version, plotbin=True, plotfrac=True, save=False):
    """
    Actually does the thing of placing the molecules in the disc
    
    takes in *data, *args, 
    
    PARAMETERS calls all the functions
    THRESHOLDS sets the threshold values
    Then the threshold values are applied in a for loop per grid point
    And the results are then plotted plotbin = True
    the fractional abundances are then set for the baseline, layer and reservoir
    these can also be plotted plotfrac = True
    
    can also save A_v = 3 and CO snow surface contours if save = True
    which are used in interpolationfunction.py to see where the gridpoints are in relation to the COMs
    """        

    
    """PARAMETERS"""
        
    #VISUAL EXTINCTION
    A_v = photodissociation(*data, *args, ifplot=False) 
    
    #CO SNOW SURFACE
    snowsurface_co = co_snowsurface(*data, *args)


    #MOLECULE SNOW SURFACE
    #sets gas abundance
    if molecule == "CO":
        x_gas = X_CO
    else:
        x_gas = MOLECULES[molecule][3] 

    x_ice = MOLECULES[molecule][4] 
    mol_ab_func = mol_abundance(*data, *args, x_gas, ifplot=True)
    snowsurface_mol = mol_ab_func[1]


    #SNOW LINE
    #calls function
    snowlinefunc = snowline(*data, *args, snowsurface_mol)
    #number value for the snowline
    r_snow = snowlinefunc[0]
    #x and y values to get vertical line
    snow_x = snowlinefunc[1]
    snow_y = snowlinefunc[2]
    #snowsurface for co but without all the wiggles
    snowsurface_n = snowlinefunc[3]
    
    
    #DESORPTION TEMP
    #loading linearly spaced data so that we can average it
    datalinlin = loaddata(molecule, disc, 'x3.0_linlin', ifprint=False)
    
    T_des = desorptiontemp(*datalinlin, *args)
    
    
    """THRESHOLDS"""
    T_destruction = 1000    #K
    extinc_cutoff = 3
    
    print("Desorption temperature of " + str(molecule) + ' = ' + '{0:1.3f}'.format(T_des) + "K")
    print('Snowline of ' + str(molecule) + ' = ' + '{0:1.3f}'.format(r_snow) + 'AU')

     
    """FINDING COM RESERVOIR"""
    reservoir = np.ones(np.shape(RR))
    layer = np.ones(np.shape(RR))
    baseline = np.zeros(np.shape(RR))
    
    
    for i, row in enumerate(reservoir):
        for j, point in enumerate(row):
            
            #then goes thru and sets everything equal to the baseline
            #this is divided by the gas abundance so that it gets back to the baseline
            #when the abundance is calculated                        
            
            #DESTRUCTION
            if t_gas[i,j] > T_destruction:
                reservoir[i,j] = 0
                layer[i,j] = 0
                baseline[i,j] = 1
            
            #COM RESERVOIR
            #R beyond snowline
            if RR[i,j] > r_snow:
                reservoir[i,j] = 0
                layer[i,j] = 1
                baseline[i,j] = 0
                
            #MOLECULAR LAYER
            #stick with the large dust
            #CHANGE TO CO SNOWLINE
            if snowsurface_co[i,j] > 1:
                reservoir[i,j] = 0
                layer[i,j] = 0
                baseline[i,j] = 1
                
            #EXTINCTION
            if A_v[i,j] < extinc_cutoff:
                reservoir[i,j] = 0
                layer[i,j] = 0
                baseline[i,j] = 1
            
            #sorts out the NaNs
            if np.isnan(n_H2[i,j]) == True:
                reservoir[i,j] = float("nan")
                layer[i,j] = float("nan")
                baseline[i,j] = float("nan")
                
                

            
    zooms = {'full':  [[0,600],[0,300]],
             'close': [[0,200],[0,100]],
             'closerish': [[0,150],[0,75]],
             'closerisher': [[0,120],[0,60]],
             'closer': [[0,100],[0,50]],
             'closest': [[0,10],[0,4]],
             'closermid': [[150,200],[20,50]]}
    
    zoom = zooms['closerisher']    
    
    """PLOTTING BINARY ABUNDANCES"""
    if plotbin == True:
        fig = plt.figure(figsize=(20,10))
        fig.tight_layout(pad=10.0)
        
        cmapres = ListedColormap([[0,0,0,0], "darkmagenta"])
        clevelsres = np.linspace(0, 1, 3, endpoint=True)
        
        cmaplay = ListedColormap([[0,0,0,0], "mediumvioletred"])
        clevelslay = np.linspace(0, 1, 3, endpoint=True)
        
        cmapT = "Spectral_r"
        clevelsT = np.linspace(0, 300, 125, endpoint=True)
        
        ax = fig.add_subplot(1,1,1)
        
        #snow line
        rs, = plt.plot(snow_x, snow_y, "k--", linewidth=3)
        
        #photodissociation at extinction 3
        av = plt.contour(RR, ZZ, A_v, levels=3, colors='blue', linewidths=3,)
        hav,_ = av.legend_elements()

        
        #CO snow line, deals with snowsurface being different sizes
        if molecule == "CO":
            ssco = plt.contour(RR, ZZ, snowsurface_co, levels=[1], colors='white', linewidths=3,)  
        else:
            ssco = plt.contour(RR, ZZ, snowsurface_co, levels=[1], colors='white', linewidths=3,)  
            ssmol = plt.contour(RR, ZZ, snowsurface_mol, levels=[1], colors='darkgreen', linewidths=3)    
            ssartists, sslabels = ssmol.legend_elements()
    
        #thermodesorption
        tdes = plt.contour(RR, ZZ, td1, levels=[T_des], colors='gold', linewidths=3,)
        
        
        #colourplots
        plt.contourf(RR,ZZ, t_gas, levels=clevelsT, cmap=cmapT)
        plt.colorbar()
        la = plt.contourf(RR,ZZ, layer, levels=clevelslay, cmap=cmaplay)
        re = plt.contourf(RR,ZZ, reservoir, levels=clevelsres, cmap=cmapres)

        #legend
        cs = [av, ssco, tdes]
        csl = ['Thermally desorbed molecular reservoir',
               'Non-thermally desorbed molecular layer',
               'Photodissociation, $A_V = 3$',
               'CO snow surface',
               str(molecule) + ' desorption temperature, $T_D = $' + '{0:1.1f}'.format(T_des) + 'K',
               'Snowline, $R_S = $' + '{0:1.1f}'.format(r_snow) + 'AU'
               ]
        
        artistsre, labelsre = re.legend_elements()
        are = artistsre[1]
        
        artistsla, labelsla = la.legend_elements()
        ala = artistsla[1]

        h = [are, ala]
        
        for i, c in enumerate(cs):
            artists, labels = c.legend_elements() 
            h.append(artists[0])
            
        h.append(rs)
        
        if molecule != 'CO':
            h.append(ssartists[0])
            csl.append(str(molecule) + ' snow surface')
                
        plt.legend(handles = h, labels = csl,
                   loc='upper left', fontsize='x-large', facecolor='lightsteelblue', edgecolor='navy')
        
        #other stuff
        ax.set_aspect("equal")
        ax.set_xlim(zoom[0])
        ax.set_ylim(zoom[1])
        
        
        plt.xlabel('R (AU)', fontsize='x-large')
        plt.ylabel('Z (AU)', fontsize='x-large')

        disctitle = disc.split("_")
        restitle = version.split("_")
        plt.title("Abundance of " + str(molecule) + " in " + str(disctitle[0]) + ' ' + str(disctitle[1]) + ' ' + str(restitle[0]) + ' ' + str(restitle[1]) + ' resolution', fontsize='xx-large')
        plt.title("Abundance of " + str(molecule) + " in " + str(disctitle[0]) + ' ' + str(disctitle[1]), fontsize='xx-large')

        
    #saving the snowsurface and A_V contour lines to look at interpolation
    if save == True:
        
        ss_line = snowline(*data, *args, snowsurface_co)[3]
        av_line = snowline(*data, *args, A_v)[3]

        np.savez("C:/Users/danip/OneDrive/Documents/Uni/Year 4/MPhys Project/MAPS discs/MAPS_working_dir/disc_" + str(disc) + "/interpolateddata/lines.npz", 
                 ss = ss_line,
                 av = av_line,) 
            
        

    """DEFINING FRACTIONAL ABUNDANCES"""    
    #defining the overall abundance multiplier
    res_frac_ab = x_ice * reservoir * 10
    lay_frac_ab = x_gas * layer * 10
    base_frac_ab = 1e-16 * baseline
    
    abundance = res_frac_ab + lay_frac_ab + base_frac_ab


    """PLOTTING FRACTIONAL ABUNDANCES"""
    if plotfrac == True:
        fig = plt.figure(figsize=(20,15))
        fig.tight_layout(pad=10.0)
        
        cmapres = ListedColormap([[0,0,0,0], "darkmagenta"])
        clevelsres = np.linspace(0, 1, 3, endpoint=True)
        
        cmaplay = ListedColormap([[0,0,0,0], "mediumvioletred"])
        clevelslay = np.linspace(0, 1, 3, endpoint=True)
        
        cmapT = "Spectral_r"
        clevelsT = np.linspace(0, 1e-12, 125, endpoint=True)
        
        ax = fig.add_subplot(1,1,1)
        
        #snow line
        plt.plot(snow_x, snow_y, "k--")
        
        #photodissociation at extinction 3
        plt.contour(RR, ZZ, A_v, levels=3, colors='blue', linewidths=2,)
        
        #CO snow line, deals with snowsurface being different sizes
        if molecule == "CO":
            plt.contour(RR, ZZ, snowsurface_co, levels=[1], colors='black', linewidths=2,)  
            #plt.contour(RR, ZZ, td1, levels=[20], colors='darkgrey', linewidths=2,)    
        else:
            plt.contour(RR, ZZ, snowsurface_co, levels=[1], colors='white', linewidths=2,)  
            #plt.contour(RR, ZZ, td1, levels=[20], colors='darkgrey', linewidths=2,)    
            plt.contour(RR, ZZ, snowsurface_mol, levels=[1], colors='darkgreen', linewidths=2)    
    
    
        #thermodesorption
        #stick with the large grains
        #plt.contour(RR, ZZ, td1, levels=[T_des], colors='yellow', linewidths=2,)
        #plt.contour(RR, ZZ, td2, levels=[T_desorption], colors='darkgreen', linewidths=2)
        
        
        plt.contourf(RR,ZZ, abundance, levels=clevelsT, cmap=cmapT)
        plt.colorbar()

        
        plt.annotate(xy=[5,64], text=' Black dashed line: snow line radius \n Blue: $A_V=3$, above this molecules are photo dissociated \n White: CO snow surface, above this assume molecules are in gas phase \n Grey: T_d1 = 20K, rough estimate for CO snow surface \n Dark green: snow line of molecule (already in white if CO) \n Yellow: desorption temperature in large grains \n \n Pink: Molecular Layer \n Purple = Molecular Reservoir', size='large')

        
        ax.set_aspect("equal")
        ax.set_xlim(0,200)
        ax.set_ylim(0,125)
        

        plt.title("Abundance of " + str(molecule))



    
    return abundance 
            
    


def writetofile(RR, ZZ, R, td1, td2, t_gas, rho_d1, rho_d2, sigma_dust1, sigma_dust2, sigma_g, n_H2, X_CO, flux_Lyalpha, flux_UV_cont, flux_UV_tot, molecule, disc, version, plotbin = True, plotfrac = False):
    """
    writes the data out to the format needed by line survey code
    takes in *data, *args
    also kwargs for if you want to plot when abundances() is called 
    """   
    
    """GETTING TO CORRECT FORMAT"""
    abundance = abundances(*data, *args, plotbin = plotbin, plotfrac = plotfrac, save=True)
    
    #COORDINATES NEEDED FOR BOTH FILES
    Rs = []
    Zs = []
    
    #ABUNDANCE FILE
    As = []
    
    #STRUCTURE FILE
    Structure = np.array([[0,1,2,3,4,5,6,7,8,9]])
    
    #counts for R and Z values
    # will only be added to if countR = True
    #ie if the values have been added
    n_r = 0
    
    rows = np.shape(RR)[0]
    cols = np.shape(RR)[1]
    
    print(rows, cols)
    
    #iterates over each column first: for each R value
    for col in range(0, np.shape(RR)[1]):
        
        #resets the thing to say if a thing has been added
        countR = False

        #iterates over each row : each Z value for a fixed R
        for row, a in enumerate(abundance[:,col]):

            #makes sure no nans
            if RR[row,col] > 200:
                break
            
            elif np.isnan(a) == False:
                
                
                #COORDINATES
                r = RR[row,col]
                z = ZZ[row,col]
                
                Rs.append(r)
                Zs.append(z)
                
                
                #ABUNDANCES
                As.append(a)
                
                
                #STRUCTURE
                rho = n_H2[row,col]
                temp = t_gas[row,col]
                dusttemp = td1[row,col]
                h2 = 0.9
                h = 0
                hplus = 0
                he = 0.1
                heplus = 0
                xray = 0
                uvflux = flux_UV_tot[row,col]
                
                structure = np.array([[rho, temp, dusttemp, h2, h, hplus, he, heplus, xray, uvflux]])
                
                for i, s in enumerate(structure[0]):
                    if np.isnan(s) == True:
                        structure[0,i] = 0
                
                Structure = np.concatenate([Structure, structure], axis=0)
                
                countR = True                
                
            else: 
                continue
            
        if countR == True:
            n_r +=1
          
    n_z = 0
    rmax = Rs[-1]
    for i, r in enumerate(Rs):
       if r == rmax:
           n_z += 1
    
    n_tot = len(Rs)
        
    Rs = np.array([Rs])        
    Zs = np.array([Zs])  
    Coords = np.transpose(np.concatenate([Rs, Zs], axis=0))
      
    As = np.transpose(np.array([As]))
    
    Structure = Structure[1:]

    
    abundancetab = np.concatenate([Coords, As], axis=1)
    structuretab = np.concatenate([Coords, Structure], axis=1)

    
    print("N_R = " + str(n_r) + ", N_Z = " + str(n_z) + ", N = " + str(n_tot))

    
    """OUTPUTTING ABUNDANCE FILE"""
    filename_ab = 'C:/Users/danip/OneDrive/Documents/Uni/Year 4/MPhys Project/MAPS discs/MAPS_working_dir/disc_' + str(disc) +'/outputfiles/' + str(version) + '/' + str(disc) + '_' + str(molecule) + '_abundances_' + str(version) + '.txt'

    txt = "# Disk chemistry output: parametric structure for " + str(molecule) + ' in ' + str(disc) +" \n# \n# Gnuplot format, Resolution = " + str(version) + ", N_R = " + str(n_r) + ", N_Z = " + str(n_z) + ", N = " + str(n_tot) + "\n# \n# Radius [au] # Height [au] # Abundance \n# \n"

    for row in abundancetab:
        str_row = ''
        for i, val in enumerate(row):
            #val = round(val, 3)
            
            if i + 1 == len(row):
                str_row = str_row + '{0:1.3e}'.format(val) + '\n'
            else:
                str_row = str_row + '{0:1.3e}'.format(val) + '    '
        
        #str_row = '{row[0]}'.format('e') + "    " + str(row[1]) + "    " + str(row[2]) + "\n"
        txt = txt + str_row
        
    f = open(filename_ab, 'w')
    f.write(txt)
    
    
    """OUTPUTTING STRUCTURE FILE"""
    filename_str = 'C:/Users/danip/OneDrive/Documents/Uni/Year 4/MPhys Project/MAPS discs/MAPS_working_dir/disc_' + str(disc) +'/outputfiles/' + str(version) + '/' + str(disc) + '_structure_' + str(version) + '.txt'
    
    txt = "# Model structure: Qi et al. (2011) model for " + str(disc) + " \n# \n# Gnuplot format \n# \n# # Radius [au] # Height [au] # Density [cm-3] # Gas Temp [K] # Dust Temp [K] # H2 # H # H+ # He # He+ #X-Ray Flxu #UV Flux \n# \n"

    for row in structuretab:
        str_row = ''
        for i, val in enumerate(row):
            #val = round(val, 3)
            
            if i + 1 == len(row):
                str_row = str_row + '{0:1.3e}'.format(val) + '\n'
            else:
                str_row = str_row + '{0:1.3e}'.format(val) + '    '
        
        #str_row = '{row[0]}'.format('e') + "    " + str(row[1]) + "    " + str(row[2]) + "\n"
        txt = txt + str_row
        
    f = open(filename_str, 'w')
    f.write(txt)

    
    return abundancetab, structuretab,




#plot_data(*data, *args)

ab = abundances(*data, *args, plotbin = True, plotfrac = False, save=True)

#tab = writetofile(*data, *args, plotbin = True, plotfrac = False)
