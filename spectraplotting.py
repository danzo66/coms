import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.integrate as sp
from matplotlib.colors import ListedColormap

#0: K=2 freq (GHz),     1: Disc integrated flux mJykms-1   2: error
transitions_ilee = {'AS_209':      [[220.7302611, 4.1,0]],
      'GM_Aur':      [[220.7302611, 9.3, 0.4]],
      'HD_163296':   [[220.7302611, 12.5, 0.8], [220.7090174, 8.2, 0.8]],
      'IM_Lup':      [[220.7302611, 3.5, 0]],
      'MWC_480':     [[220.7302611, 26.1, 1.3]],
         }

#currently x2.0_loglin is working very well
args = ['CH3CN', 'HD_163296', 'x2.0_loglin']


def loadfiles(molecule, disc, version):
    '''
    reads the .out files and makes them into arrays
    '''
    

    filename = str(molecule) + '_test_' + str(version) + '_x10ab'
    #filename = str(molecule) + '_test_' + str(version)
    '''LINE PROFILES'''
    filename_p = 'C:/Users/danip/OneDrive/Documents/Uni/Year 4/MPhys Project/MAPS discs/MAPS_working_dir/disc_' + str(disc) +'/spectrafiles/' + str(filename) + '_profiles.out'
    file_p = open(filename_p, 'r')
    
    datap = np.array([[0,1,2,3]],)

    for i, line in enumerate(file_p):
        if i > 0:
            line = line.split(" ")

            row = []
            for d in line:
                if d != '' and d != '\n':
                    row.append(float(d))
            
            row = np.array([row])
            if np.shape(row) == (1,4):
                row[0][3] = row[0][3] * 1e3
                datap = np.concatenate((datap,row), axis=0)
            
    datap = datap[1:]
    
    
    '''LINE INTENSITIES'''
    filename_i = 'C:/Users/danip/OneDrive/Documents/Uni/Year 4/MPhys Project/MAPS discs/MAPS_working_dir/disc_' + str(disc) +'/spectrafiles/' + str(filename) + '_intensities.out'
    file_i = open(filename_i, 'r')
    
    datai = np.array([[0,1,2,]],)
    
    for i, line in enumerate(file_i):
        if i > 0:
            line = line.split(" ")

            row = []
            for d in line:
                if d != '' and d != '\n':
                    row.append(float(d))

            
            row = np.array([row])
            if np.shape(row) == (1,3):
                row[0][2] = row[0][2] * 1e3
                datai = np.concatenate((datai,row), axis=0)

    datai = datai[1:]
    
    
    '''CONTINUUM'''
    filename_c = 'C:/Users/danip/OneDrive/Documents/Uni/Year 4/MPhys Project/MAPS discs/MAPS_working_dir/disc_' + str(disc) +'/spectrafiles/' + str(filename) + '_continuum.out'
    file_c = open(filename_c, 'r')
    
    datac = np.array([[0,1,]],)
        
    for i, line in enumerate(file_c):
        if i > 0:
            line = line.split(" ")

            row = []
            for d in line:
                if d != '' and d != '\n':
                    row.append(float(d))
            
            row = np.array([row])
            if np.shape(row) == (1,2):
                row[0][1] = row[0][1] * 1e3
                datac = np.concatenate((datac,row), axis=0)

    datac = datac[1:]
    
    
    
    return datap, datai, datac


data = loadfiles(*args)


def plot(molecule, disc, version, profiles, intensities, continuum, transitions):
    '''
    plots the spectrum 
    takes *args, *data, transitions_ilee
    '''
    
    '''PLOTTING'''
    #getting data, into correct form
    freq = profiles[:,1]
    intensity = profiles[:,3]
    
    #plots
    fig = plt.figure(constrained_layout=True)
    plt.plot(freq, intensity, color='indigo')
    disctitle = disc.split("_")
    verstitle = version.split("_")
    plt.title(str(molecule) + ' in ' + str(disctitle[0]) + ' ' + str(disctitle[1]) + ', ' + str(verstitle[0]) + ' ' + str(verstitle[1]), fontsize='xx-large')
    plt.title(str(molecule) + ' in ' + str(disctitle[0]) + ' ' + str(disctitle[1]), fontsize='x-large')
    plt.xlabel('Frequency (GHz)', fontsize='large')
    plt.ylabel('Flux (mJy)', fontsize='large')
    
    ann = ''
    
    #goes thru the intensities
    # if the frequency is equal to the K=2 transition in Ilee et al. it prints both values
    # determines transition number and flux i found
    for i, trans in enumerate(intensities):

        trans_num = trans[0]
        trans_freq = trans[1]
        flux = trans[2]
        
        pmax = np.max(intensity)
        fmin = np.min(freq)
        
        for j, t_ilee in enumerate(transitions[disc]):
            if round(trans_freq, 3) == round(t_ilee[0], 3):
                
                
                frtxt = '\n' + str(int(trans_num)) + '   ' + '{0:1.3f}'.format(trans_freq) + ' GHz \n'
                iltxt = 'Integrated flux, Ilee et al. 2021 = ' + str(t_ilee[1]) + ' $\pm$ ' + str(t_ilee[2]) + ' mJy km/s'
                
                if flux < 0.1:
                    fltxt = 'Integrated flux = ' + '{0:1.3e}'.format(flux) + ' mJy km/s\n'
                else:
                    fltxt = 'Integrated flux = ' + '{0:1.3f}'.format(flux) + ' mJy km/s\n'
                    
                trtxt = frtxt + fltxt +iltxt
                
                ann = ann + trtxt + '\n'
    
    plt.text(0.94*fmin, -1.05*pmax, s=ann, fontsize='x-large', fontfamily='CALIBRI')
    
    print(ann)
    

    
plot(*args, *data, transitions_ilee)
