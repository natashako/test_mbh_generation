#!/home/nataliak/anaconda3/bin/python3

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import MBH_generate as mbhb
import argparse

import h5py

year_sec = 365.25*24*3600


def parser():
    """Parses command line arguments"""
    parser = argparse.ArgumentParser(prog='test_MBHgen.py',description='Test the generation of the MBHBs with the LDC provided phenomD')
    parser.add_argument('-z', '--seed', type=int, default=123, help='the random seed')
    return parser.parse_args()



# Assign attributes to record to the hdf5 file
def assign_attr(values,subgr):

        subgr.attrs['Redshift'] = values[0]  # redshift of coalescence
        subgr.attrs['Mass1'] = values[1] #  intrinsic mass of primary (solar masses), redshifted
        subgr.attrs['Mass2'] = values[2] # intrinsic mass of secondary (solar masses), redshifted

        subgr.attrs['Spin1'] = values[3]
        subgr.attrs['Spin2'] = values[4]

        if (subgr.attrs['Spin1'] >= 0.):
            subgr.attrs['PolarAngleOfSpin1'] = 0.0
        else:
            subgr.attrs['PolarAngleOfSpin1'] = np.pi
            subgr.attrs['Spin1'] = np.absolute(subgr.attrs['Spin1'])

        if (subgr.attrs['Spin2'] >= 0.):
            subgr.attrs['PolarAngleOfSpin2'] = 0.0
        else:
            subgr.attrs['PolarAngleOfSpin2'] = np.pi
            subgr.attrs['Spin2'] = np.absolute(subgr.attrs['Spin1'])

        subgr.attrs['Spin1'] = values[3]*np.cos(subgr.attrs['PolarAngleOfSpin1']) # spin magnitude of primary;
        subgr.attrs['Spin2'] = values[4]*np.cos(subgr.attrs['PolarAngleOfSpin2']) # spin magnitude of secondary;

        subgr.attrs['CoalescenceTime'] = values[9]  #  coalescence time in seconds 

        subgr.attrs['EclipticLatitude'] = 0.5*np.pi - values[8] # $\theta$ sky location, uniform in $\cos \theta$;
        subgr.attrs['EclipticLongitude'] = values[7] #  $\phi$ sky location, uniform in $[0,2\pi]$;
        subgr.attrs['InitialPolarAngleL'] = values[6] #  inclination $\iota$ defined as the angle between $L$ and the line of sight, uniform in $\cos \iota$; 
        subgr.attrs['Cadence'] = values[10]
        subgr.attrs['ObservationTime'] = values[11]


def main(args):

    # Some set of parameters
    dt = 10.0
    Tobs = year_sec
    Tc = 5.0*year_sec/6.0
    N = int(np.floor(Tobs/dt))
    tvec = np.arange(N)*dt
    theta = []

   
  
    theta.append(1.0) # [0] red shift
    z = theta[0]
    theta.append(0.32e6*(1.+z)) # [1] red shifted mass one
    theta.append(1.e6*(1.+z))   # [2] red shifted mass two
    theta.append(0.81) # [3] spin magnitude of the first 
    theta.append(0.73) # [4] spin magnitude of the second

    theta.append(0.0) # [5] psi -- polarisation angle; this one is absent in phenomD and is either 0.0 or pi

    theta.append(1.1) # [6] inclination angle with respect to barycenter
    theta.append(4.647) # [7] lambda -- sky position, longitude (barycenter)
    theta.append(-0.07) # [8] beta -- sky position, latitude (barycenter)

    theta.append(Tc) # [9] Tc -- coalescence time 
    theta.append(dt)  # [10] cadence
    theta.append(Tobs)  # [11] Tobs -- observation time, two months 

    # Read the value of the seed from the command line
    seed = args.seed
    np.random.seed(seed)

    fh5 = h5py.File('data_MBHB_phenomD_'+str(seed)+'.hdf5')
    gr = fh5.create_group('MBHB')

    size_of_data=2

    i = 1

    # Number of the waveforms to write in the file
    while i<size_of_data:
     
        nm = "MBHB-"+str(i)
        subgr = gr.create_group(nm)

        # PhenomD waveform
        mbh_ph = mbhb.MBHB_phenomD(theta)
        freq, Af, Ef, Tf = mbh_ph.TDI_signal(tvec)
    
        tm, At = mbhb.ComputeTD(Af, dt)
        tm, Et = mbhb.ComputeTD(Ef, dt)


        # Compare to td waveforms        
        mbh_td = mbhb.MBHB_TD(theta)
        A,E = mbh_td.TDI_signal(tvec)
       
        # Transform time domain waveform to the frequency domain
        fr, Afr = mbhb.ComputeFD(A,dt)
        plt.figure(0)
        plt.loglog(fr,np.abs(Afr),label='time domain wf')
        plt.loglog(freq,np.abs(Af), label = 'frequency domain wf')
        plt.legend()
        plt.savefig('/data/public_html/nataliak/MBH_phenom/Af'+str(i)+'.png')

        plt.figure(1)
        Ntc = int(np.floor(Tc/dt))
        plt.plot(tvec[Ntc-2000:Ntc-500],A[Ntc-2000:Ntc-500])
        plt.plot(tm[Ntc-2000:Ntc-500],At[Ntc-2000:Ntc-500])
      
        plt.savefig('/data/public_html/nataliak/MBH_phenom/At_newtest'+str(i)+'.png')

        tdis = np.array([At,Et])
        dset = subgr.create_dataset(nm,data=tdis,dtype='f')
        assign_attr(theta,subgr)
        i = i+1


if __name__=='__main__':
    args = parser()
    main(args)
           

