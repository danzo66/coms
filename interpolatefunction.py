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
from scipy import interpolate
from scipy.interpolate import griddata
from matplotlib import gridspec


DISCS = ['AS_209',
         'GM_Aur',
         'HD_163296',
         'IM_Lup',
         'MWC_480']

disc = DISCS[2]


def loaddata(disc, lines=True):
    '''
    loads spherical data
    if lines = True
    then it also loads the snow surface and visual extinction
    so you can see where in the plot is important
    '''
    
    FILENAME = "C:/Users/danip/OneDrive/Documents/Uni/Year 4/MPhys Project/MAPS discs/MAPS_working_dir/disc_" + str(disc) + "/" + str(disc) + "_disk_structure.npz"
    
    #loads original spherical coordinate data
    file = np.load(FILENAME, allow_pickle=True)
    
    #dictionary of each array
    data = [file['r'],                  #0      mesh grid of radial coords
            file['theta'],              #1      mesh grid of z coords
            
            file['sigma_dust'],         #2      dust 1 surface density
            file['sigma_g'],            #3      gas surface density
            
            file['td1'],                #4      temperature of dust size 1
            file['td2'],                #5      temperature of dust size 2
            file['tgas'],               #6      gas temperature
            
            file['rho_d1'],             #7      density of dust size 1
            file['rho_d2'],             #8      density of dust size 2
            
            file['n_H2'],               #9      number density of H_2
            file['X_CO'],               #10     CO abundance
            
            file['flux_Lyalpha'],       #11     Lyman alpha flux
            file['flux_UV_cont'],       #12     UV flux
            ]
    
    
    #list of 2d arrays for names for saving
    arr2d = [[data[6], "tgas_in",],[data[4], "td1_in"],[data[5],"td2_in"],
             [data[9],"n_H2_in"],[data[10], "X_CO_in"],
             [data[7], "rho_d1_in"], [data[8], "rho_d2_in"], 
             [data[11], "flux_Lyalpha_in"], [data[12], "flux_UV_cont_in"]]
    
    #list of 1d arrays
    arr1d = [[data[2],"sigma_dust_in"], [data[1], "sigma_g_in"]]
    
    if lines == True:
        #snowsurface and extinction so we can see what it is like where the COMs actually are
        filelines = np.load("C:/Users/danip/OneDrive/Documents/Uni/Year 4/MPhys Project/MAPS discs/MAPS_working_dir/disc_" + str(disc) + "/interpolateddata/lines.npz", allow_pickle=True)
        lines = [filelines["ss"],
                 filelines["av"],]
        
        return data, arr2d, arr1d, lines
    
    else:
        return data, arr2d, arr1d


d = loaddata(disc, lines=True)
data = d[0]
arr1d = d[1]
arr2d = d[2]


def interpolate2d(parameter, r_sph, theta_sph, n_R, n_Z, logorlin, baser, basez, rmaxint):
    """
    interpolates the parameter to n_R by n_Z many points
    
    loglin is either 'loglog' or 'linlin' or 'loglin'
    and defines whether interpolation is linear or logarithmic (R first, Z second)
    
    baser basez are the logarithmic bases
    
    rmaxint takes either 'trunc' (truncated to 200AU)
    or 'full' (interpolated over the whole disc)
    """
        
    #creates meshgrid of original spherical coords
    #each entry in R_sph and TH_sph give the coordinates for one of the points            
    R_sph, TH_sph = np.meshgrid(r_sph, theta_sph)
    
    #converts spherical to cylindrical
    R = R_sph*np.sin(TH_sph) 
    Z = R_sph*np.cos(TH_sph) 
    
    """creating the array of original cylindrical coordinates, and the values at that point"""
    
    # RZV is the array of R (0), Z (1) and the value at that coord (2)
    RZV = np.array([[0,0,0]])   #array of coordinates to be added to
    
    #arrays of the column and row indexes
    index_i = np.arange(0,len(theta_sph),1) # = 0 --> 81
    index_j = np.arange(0,len(r_sph),1) # = 0 --> 303
    
    #iterates over the row (indexes, then over the column indexes
    #and then creates a little array of that specific coordinate, with the value at that coordinate
    #then it gets added to the big array RZV
    #I DON'T KNOW WHICH IS COLUMN AND ROW BUT I DON'T THINK IT MATTERS
    #ITS JUST IN AN ORDER AND THE ORDER IS IRRELEVANT
    for i in index_i:
        for j in index_j:
            rzv = np.array([[R[i,j], Z[i,j], parameter[i,j]]])
            RZV = np.concatenate((RZV,rzv),axis=0)    
            
    RZV = RZV[1:] # this just gets rid of the [0,0] at the start
            
    """creating the new coordinates that we want to interpolate to"""       
    
    #finds the highest in R and Z, i.e. the highest and lowest we want the new grid to go to
    
    if rmaxint == 'full':
        R_max = np.max(R)
    elif rmaxint == 'trunc':
        R_max = 200
        
    Z_max = np.max(Z)
    
    if logorlin == "linlin":
        #creates 1d linear arrays of new r and z
        r_in = np.linspace(0, R_max, n_R)
        z_in = np.linspace(0, Z_max, n_Z)
        
        #creates meshgrid of new r and z
        RR_in, ZZ_in = np.meshgrid(r_in, z_in)
        
        """interpolating"""
        
        para_in_l = griddata(RZV[:,0:2], RZV[:,2], (RR_in, ZZ_in), method='linear')
        
    elif logorlin == "loglin":
        #creates 1d logarithmic arrays of new r and linear of z
        stop_r = math.log(R_max, baser)
        r_in = np.logspace(-5, stop_r, n_R, base=baser, endpoint = True)
        z_in = np.linspace(0, Z_max, n_Z)
        
        #creates meshgrid of new r and z
        RR_in, ZZ_in = np.meshgrid(r_in, z_in)
        
        """interpolating"""
        
        para_in_l = griddata(RZV[:,0:2], RZV[:,2], (RR_in, ZZ_in), method='linear')
        
    elif logorlin == "loglog":
        #creates 1d logarithmic arrays of new r and z
        #remember than logspace is silly with its starts and stops
        stop_r = math.log(R_max, baser)
        stop_z = math.log(Z_max, basez)
        
        if rmaxint == 'full':
            n_R = n_R
        elif rmaxint == 'trunc':
            n_R = int(n_R/3)      
                
        r_in = np.logspace(-5, stop_r, n_R, base=baser, endpoint = True)
        z_in = np.logspace(-5, stop_z, n_Z, base=basez, endpoint = True)
        
        #creates meshgrid of new r and z
        RR_in, ZZ_in = np.meshgrid(r_in, z_in)
        
        """interpolating"""
        
        para_in_l = griddata(RZV[:,0:2], RZV[:,2], (RR_in, ZZ_in), method='linear')
        
    
        
    return RR_in, ZZ_in, para_in_l




def plot2d(r_sph, theta_sph, parameter, lines, plots, baser, basez, z,):
    """
    plots the data and interpolated grid points for a given parameter
    
    lines is either the list of contour lines, if you want them
    or lines = False if you do not
    
    plots is a list of all the things you want to plot size by side
    
    z is how close in you want to be because spyder won't let me zoom for some reason
    (line 271 for dict keys)
    """
    
    parameter = np.transpose(parameter)
    
    #creates meshgrid of original spherical coords            
    R_sph, TH_sph = np.meshgrid(r_sph, theta_sph)
    
    #converts spherical to cylindrical
    R = R_sph*np.sin(TH_sph) 
    Z = R_sph*np.cos(TH_sph) 
    
    fig = plt.figure(figsize=(20,10))
    fig.tight_layout()

    cmap = "Spectral_r"
    
    clevels = np.linspace(1, math.log10(3000), 100)   #colour levels
    llevels = [math.log(20),math.log(50)]             #contour levels
   
    
    #for titles
    keys = list(INTERPOLATIONSIZES.keys())
    vals = list(INTERPOLATIONSIZES.values())
    
    lp = len(plots) 
    wr = []
    for i in range(0, lp, 1): wr.append(1)
    
    spec = gridspec.GridSpec(ncols=lp, nrows=3,
                     width_ratios=wr, wspace=0.25,
                     hspace=0.1, height_ratios=[1, 1,1])

    #plotting of the interpolations
    for i, row in enumerate(plots):
        
        #this if deals with interpolations
        if i > 0:
            #gotta actually interpolate it first
            r_res = row[0][0]
            z_res = row[0][1]
            inter = interpolate2d(parameter, r_sph, theta_sph, r_res, z_res, row[1], row[2], row[3], row[4])
        
            RR = inter[0]
            ZZ = inter[1]  
            para = inter[2]
            ylab = ""
            
        #this if deals with the original data
        elif i == 0:
            RR = R
            ZZ = Z
            para = parameter
            ylab = "Gridpoints\nZ (AU)"
            
            RR_sph = R
            ZZ_sph = Z
            
        
        #this is mostly just for aesthetic reasons so theyre all nice and lined up
        
        zooms = {'full':  [[0,600],[0,300]],
                 'close': [[0,200],[0,100]],
                 'closerish': [[0,150],[0,75]],
                 'closer': [[0,100],[0,50]],
                 'closest': [[0,10],[0,4]],
                 'closermid': [[150,200],[20,50]]}
        
        zoom = zooms[z]
        
        para = np.log10(para)
        
        #this if statement does nothing i just want to hide things
        if cmap == "Spectral_r":
            ax = fig.add_subplot(spec[i])
            
            cf = plt.contourf(RR, ZZ, para, levels=clevels, cmap=cmap)
                        
            plt.contour(RR,ZZ,para,levels=llevels,colors='w',linewidths=1)
            
            ax.set_xlim(zoom[0])
            ax.set_ylim(zoom[1])
            ax.set_aspect('equal')
            
            res = keys[vals.index(row[0])] 
            title = str(res) + " " + str(row[1])
            
            plt.title(title, fontsize='xx-large')
            
            if i == 0:
                plt.ylabel("Full disc \nZ (AU)", fontsize='x-large') 
            
            if lines != False:
                plt.plot(lines[0][:,0], lines[0][:,1], color='black', zorder=1000000, linewidth=2) 
                plt.plot(lines[1][:,0], lines[1][:,1], color='blue', zorder=1000000, linewidth=2) 
        
        
        #close ups
            ax3 = fig.add_subplot(spec[i+lp])
            
            cf3 = plt.contourf(RR, ZZ, para, levels=clevels, cmap=cmap)
                        
            plt.contour(RR,ZZ,para,levels=llevels,colors='w',linewidths=1)
            
            ax3.set_xlim(zooms['closest'][0])
            ax3.set_ylim(zooms['closest'][1])
            ax3.set_aspect('equal')
            
            
            if i == 0:
                plt.ylabel("Inner disc \nZ (AU)", fontsize='x-large') 
            
            if lines != False:
                plt.plot(lines[0][:,0], lines[0][:,1], color='black', zorder=1000000, linewidth=2) 
                plt.plot(lines[1][:,0], lines[1][:,1], color='blue', zorder=1000000, linewidth=2) 

            
        #plots interpolated grid points
            
            ax2 = fig.add_subplot(spec[i+2*lp])
            ax2.set_aspect('equal')
            plt.scatter(RR, ZZ, s=1, zorder=10000)
            plt.xlabel("R (AU)", fontsize='x-large')
            plt.ylabel(ylab, fontsize='x-large')
            ax2.grid(zorder=0)
            ax2.set_xlim(zoom[0])
            ax2.set_ylim(zoom[1])
            
            #makes the line of where the coords are
            thmax = np.max(ZZ_sph)
            rsphmaxindex = np.where(ZZ_sph==thmax)
            rsphmax = RR_sph[rsphmaxindex[0][0]][rsphmaxindex[1][0]]
            
            line = np.array([[0,rsphmax],[0, thmax]])
            plt.plot(line[0],line[1], color='r', zorder=1000000, linewidth=2)
            
            #snowsurface and extinction so we can see the important bits
            if lines != False:
                plt.plot(lines[0][:,0], lines[0][:,1], color='black', zorder=1000000, linewidth=2) 
                plt.plot(lines[1][:,0], lines[1][:,1], color='blue', zorder=1000000, linewidth=2) 
                
        
    return RR, ZZ, para, line, ZZ_sph




def interpolate1d(r_sph, theta_sph, parameter, n_R, loglin, baser,):
    """
    interpolates 1d arrays for surface density
    theres a bug in here somewhere i think...
    """
    
    R = r_sph*np.sin(theta_sph[0])
    
    R_max = np.max(R)
    R_min = np.min(R)
    
    if loglin == "linlin":
        #creates 1d linear arrays of new r and z
        r_in = np.linspace(R_min, R_max, n_R)
    if loglin == "loglin" or "loglog":
        #creates 1d logarithmic arrays of new r and linear of z
        #deals with the funny start and stop
        stop_r = math.log(R_max, baser)
        start_r = math.log(R_min, baser)
        r_in = np.logspace(start_r, stop_r, n_R, base=baser, endpoint = True)
    
    #linear refers to the kind of interpolation
    #NOT the spacing of the points
    interp = interpolate.interp1d(R, parameter, kind = "linear")

    para_in = interp(r_in)
        
    return R, r_in, para_in




def plot1d(r_sph, theta_sph, parameter, baser):
    '''
    plots the data in spherical, cylindrical and interpolated cylindrical coordinates
    '''
    
    inter = interpolate1d(r_sph, theta_sph, parameter, 152, "log", baser)
    R = inter[0]
    r_in = inter[1]
    para_in = inter[2]
    
    fig = plt.figure()
    
    #original spherical coords
    ax1 = fig.add_subplot(1,3,1)
    plt.scatter(r_sph, parameter)
    ax1.set_yscale("log")    
    plt.xlabel("r (sph) (AU)")
    plt.ylabel(r"$(g/cm^2)$")
    
    #cylindrical coords
    ax2 = fig.add_subplot(1,3,2)
    plt.scatter(R, parameter)
    ax2.set_yscale("log")    
    plt.xlabel("r (AU)")
    plt.ylabel(r"$(g/cm^2)$")
    
    #
    ax3 = fig.add_subplot(1,3,3)
    plt.scatter(r_in, para_in)
    ax3.set_yscale("log")    
    plt.xlabel("r (AU)")
    plt.ylabel(r"$(g/cm^2)$")




def save(r_sph, theta_sph, newsize, loglin, arr2d, arr1d, baser, basez, rmaxint):
    """
    interpolates each array and saves them all to an npz file
    not working for 1d arrays but no 1d arrays are needed
    
    takes in r_sph, theta_sph, arr2d, arr1d
    
    newsize is one of the keys from INTERPOLATIONSIZES
    specifies the no. of R and Z grid points
    
    loglin is either 'loglog' or 'linlin' or 'loglin'
    and defines whether interpolation is linear or logarithmic (R first, Z second)
    
    baser basez are the logarithmic bases
    
    rmaxint takes either 'trunc' (truncated to 200AU)
    or 'full' (interpolated over the whole disc)
    """
        
    n_R = INTERPOLATIONSIZES[newsize][0]
    n_Z = INTERPOLATIONSIZES[newsize][1]
    print(newsize)

    #interpolates and saves 2d arrays
    for i, arr in enumerate(arr2d):
        arr[0] = np.transpose(arr[0])
        print(arr[1])
        #saves RR and ZZ the first time it runs
        if i == 0:
            inter = interpolate2d(arr[0], r_sph, theta_sph, n_R, n_Z, loglin, baser, basez, rmaxint)
            RR_in = inter[0]
            ZZ_in = inter[1]
            globals()[arr[1]] = inter[2]    #sets the name 
        else:
            globals()[arr[1]] = interpolate2d(arr[0], r_sph, theta_sph, n_R, n_Z, loglin, baser, basez, rmaxint)[2]

        
    #interpolates and saves 1d arrays    
    for arr in arr1d:
        print(arr[1])
        size = np.shape(arr[0])
        #checks sigma dust and does each column separately
        if len(size) == 2:
            d1 = arr[0][:,0]
            #inter = interpolate1d(r_sph, theta_sph, d1, n_R, loglin, baser)
            #R_in = inter[1]
            #sigd1 = inter[2]
            
            #d2 = arr[0][:,1]
            #sigd2 = interpolate1d(r_sph, theta_sph, d2, n_R, loglin, baser)[2]
        
        #for sigma gas
        #else:
         #   sigma_g_in = interpolate1d(r_sph, theta_sph, arr[0], n_R, loglin)[2]

    filename = "C:/Users/danip/OneDrive/Documents/Uni/Year 4/MPhys Project/MAPS discs/MAPS_working_dir/disc_" + str(DISC) + "/interpolateddata/intdata_" + str(newsize) + "_" + str(loglin) + "full.npz"
    np.savez(filename, 
             RR_in=RR_in,
             ZZ_in=ZZ_in,
             #R_in=R_in,
             tgas_in=tgas_in, 
             td1_in=td1_in,
             td2_in=td2_in,
             n_H2_in=n_H2_in,
             X_CO_in=X_CO_in,
             rho_d1_in=rho_d1_in, 
             rho_d2_in=rho_d2_in, 
             flux_Lyalpha_in=flux_Lyalpha_in, 
             flux_UV_cont_in=flux_UV_cont_in,
             #sigma_dust1_in=sigd1,
             #sigma_dust2_in=sigd2,
             #sigma_g_in=sigma_g_in,
             )
            
    return tgas_in


#sizes for interpolation
global INTERPOLATIONSIZES
INTERPOLATIONSIZES = {"Original":   [304, 82],
                      "x1.0":       [304, 82],
                      "x1.25":       [380, 102],
                      "x1.5":       [456, 123],
                      "x2.0":       [608, 164],     #loglog N_R = 212, N_Z = 154, N = 26499
                      "x2.5":       [760, 205],     #loglog N_R = 264, N_Z = 193, N = 41243
                      "x3.0":       [912, 246],     #loglog N_R = 317, N_Z = 232, N = 59469
                      "x3.5":       [1064, 287],
                      "x4.0":       [1216, 328],
                      "x5.0":       [1520, 410],
                      "x10.0":      [3040, 820],
                      
                      "x2.0x0.5":   [608, 41],
                      "x2.0x0.8":   [608, 60],
                      "x2.0x1.0":   [608, 82],
                      "x2.0x1.5":   [608, 123],
                      "x2.0x4.0":   [608, 328],
                      'x2.0x10.0':  [608, 820]}



#test plots of various different interpolation choices
#the first one is always the original for comparisons, putting somethng else in there doesnt do anything
#0: size    #1: log or lin      #2,3: baser, basez,     #4: where you want to interpolate to
plots = [[INTERPOLATIONSIZES["Original"], "data", 2,2,'full'], 
          [INTERPOLATIONSIZES["x2.0"], "loglin", 10,2,'full'],
          [INTERPOLATIONSIZES["x2.0x1.5"], "loglog", 5,2,'full'],
          ]

baser = 5
basez = 2


interpolated = plot2d(data[0], data[1], data[6], d[3], plots, baser, basez, 'full')

#save(data[0], data[1], 'x2.0x1.5', "loglog", arr2d, arr1d, baser, basez, 0.3, 'full')     
