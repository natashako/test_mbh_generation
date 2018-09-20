import numpy as np
import h5py

import LISAConstants as LC
import LISAParameters as LP
import Cosmology
import tdi as tdi
import GenerateFD_SignalTDIs as FD

from LISAhdf5 import LISAhdf5,ParsUnits,Str

import scipy.signal as sg
import scipy.interpolate as ip
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from scipy.interpolate import UnivariateSpline
from scipy import interpolate



"""
Compute the time evolution of TDI x and + polarisations corresponding to an inspiral part of MBHB
Code from Antoine and Stas based on https://arxiv.org/pdf/1001.5380.pdf
"""
class MBHB_TD:

    def __init__(self, theta):

        self.z  = theta[0]     #   red shift
        self.m1 = theta[1]     #   chirp mass 
        self.m2 = theta[2]      #   mass ratio
        self.chi1 = theta[3]   #   spin magnitude [-1, 1] of the primary
        self.chi2 = theta[4]   #   spin magnitude [-1, 1] of the secondary
        self.psi = theta[5]    #   polarization angle
        self.incl = theta[6]  #   inclination angle (L.n) wrt to barycenter 
        self.lam = theta[7]    #   sky position, longitude (barycenter)
        self.bet = theta[8]    #   bet -- sky position, latitude (barycenter) 
        self.Tc = theta[9]     #
        self.dt = theta[10]    #
        self.Tobs = theta[11]  #
      


    def TDI_signal(self, tvec):

        ### Getting parameters: TODO: FIX THIS, HARDCODED
        self.z = 1.   #!!!! HARDCODED TODO
        distance   = (Cosmology.DL(self.z, 0.0)[0])*1.e6 # in pc 

        #?| Defone the constants that we are going to be using
        P_SI     = LC.pc
        MTSUN_SI = LC.MTsun
        C_SI     = LC.clight
        AU_SI    = LC.ua
        year     = LC.year
        arm      = LP.lisaL

        #?| Additional parameters
        #phic = 2.12 ## FIXME FIXME FIXME
        #phic = 1.2
        #phic = 2.12
        phic = 0.0
        Rmin = 6.0             # minimum distance (in M) used to terminate the waveform
        M    = self.m1 + self.m2;
        Mt   = M*MTSUN_SI;
        M2   = M*M;
        eta  = self.m1*self.m2/M2;

        DLt = distance*P_SI/C_SI # distance in sec
        if (self.m1 < self.m2):
          tmp = self.m1
          self.m1=self.m2
          self.m2=tmp

        tc = self.Tc
        if (self.Tobs < self.Tc):
          tc = self.Tobs;

        # The N-th sample where the model is non-zero

        N = (int)(tc/self.dt) 
        tm = np.arange(N)*self.dt


        # Coefficients
        fac  = eta/(5.0*Mt)
      
        p0   = -2.0                          # -2 fac^(5/8)
        p10  = -(3715./4032.+55./48.*eta) # - (3715/4032 + 55/48 eta) fac^(3/8)
        p15  = 0.375                         # (3/8) fac^(1/4)
        p150 = p15*(4.*np.pi);                   # 4 pi (3/8) fac^(1/4)
        p20  = -1.0;                             # - fac^(1/8)
        p200 = p20*(9275495./7225344.+284875./129024.*eta+1855./1024.*eta*eta); # - (9275495/7225344 + 284875/129024 eta + 1855/1024 eta^2) fac^(1/8)

        beta  = (self.chi1/12.0)*(113.0*self.m1*self.m1/M2 + 75.0*eta) + (self.chi2/12.0)*(113.0*self.m2*self.m2/M2 + 75.0*eta);
        sigma = (237.0/24.0)*eta*self.chi1*self.chi2;

        f10 = 743./2688.+11./32.*eta;
        f15 = -3.*(4.*np.pi-beta)/40.;
        f20 = 1855099./14450688.+56975./258048.*eta + 371./2048.*eta*eta - 3./64.*sigma;

        #Orbital motion
        AU       = AU_SI/C_SI
        Phi_LISA = 2.0*np.pi*tm/year
        R        = np.zeros((len(tm), 3))
        R[:, 0]  = AU*np.cos(Phi_LISA)
        R[:, 1]  = AU*np.sin(Phi_LISA)
        thS      = 0.5*np.pi - self.bet
        phS      = self.lam

        n  = np.array([np.cos(phS)*np.sin(thS), np.sin(phS)*np.sin(thS), np.cos(thS)])
        nR = np.dot(R, n)

        tmk   = tm + nR
        L     = arm/C_SI
        tmk2L = tmk - 2.0*L

        N     = len(tvec)-1
        Mom   = np.zeros(N)
        phase = np.zeros(N)
        ampl  = np.zeros(N)

        ampl0 = 2.0*Mt*eta/DLt
        #print "Amplitude:", ampl0, Mt, eta, 1.0/DLt
        tau   = fac*(tc-tmk)
        tau   = np.clip(tau, 0.0001, 1.*year)
        Mom   = (np.power(tau,(-3./8.)) + f10*np.power(tau,(-5./8.)) + f15*np.power(tau,(-3./4.)) + f20*np.power(tau,(-7./8.)))/8.
        phase_tk = (p0*np.power(tau,(0.625)) + p10*np.power(tau,(0.375)) + (p150 - p15*beta)*np.power(tau,(0.25)) +
                (p200 + 15.0*sigma/32.)*np.power(tau,(0.125)))/eta
        #tau0  = fac*(tc-tm)
        #phase = (p0*tau0**(0.625) + p10*tau0**(0.375) + (p150 - p15*beta)*tau0**(0.25) +
        #            (p200 + 15.0*sigma/32.)*tau0**(0.125))/eta

        tau2L = fac*(tc-tmk2L)
        tau2L = np.clip(tau2L, 0.0001, 1.*year)
        phase_tk2L = (p0*np.power(tau2L,(0.625)) + p10*np.power(tau2L,(0.375)) + (p150 - p15*beta)*np.power(tau2L,(0.25)) +
                (p200 + 15.0*sigma/32.)*np.power(tau2L,(0.125)))/eta
        ampl = ampl0*np.power(Mom,(2./3.))

        # let's find where we need to terminate the waveform
        Mom_max = np.power(7.0,(-1.5))
        Mom_max6 = np.power(6.0,(-1.5))

        if np.max(Mom) < Mom_max:
            Mom_max = 0.98*Mom
            #Mom_max = Mom_max6
        #i_max = np.argwhere(Mom > Mom_max)[0][0]
        #i_max6 = np.argwhere(Mom > Mom_max6)[0][0]


        taper = 0.5*(1.0 + np.tanh( 57.0*( np.power((Mom_max),(2./3.)) - np.power(Mom,(2./3.))) ))
        test1 = (np.power((Mom_max),(2./3.)))
        test2 =  np.power(Mom,(2./3.))

        ampl = ampl*taper

        om = Mom/Mt

        # Antenna pattern
        th_d = self.bet + 0.5*np.pi
        lam_d = self.lam + np.pi

        ### experiment
        Nt = (int)(tc/self.dt) # + 1
        tm_f = tvec[:Nt]


        tck = interpolate.splrep(tm, phase_tk, s=0)
        ph_tk = interpolate.splev(tm_f, tck, der=0)

        tck = interpolate.splrep(tm, phase_tk2L, s=0)
        ph_tk2L = interpolate.splev(tm_f, tck, der=0)

        tck = interpolate.splrep(tm, ampl, s=0)
        ampl_f = interpolate.splev(tm_f, tck, der=0)

        tck = interpolate.splrep(tm, om, s=0)
        om_f = interpolate.splev(tm_f, tck, der=0)


        ampl = ampl_f
        om = om_f
        del_phi = 0.5*(ph_tk - ph_tk2L)
        phi_p = 0.5*(ph_tk + ph_tk2L)

        Phi_LISA = 2.0*np.pi*tm/year


        #del_phi = 0.5*(phase_tk - phase_tk2L)
        #phi_p = 0.5*(phase_tk + phase_tk2L)

        Om_A = 0.0 # for A and Om = -0.5*np.pi for E
        Om_E = -0.5*np.pi # for E

        Fp_A = (1.0/32.0)*( 6.0*np.sin(2.0*th_d) *(3.0*np.sin(Phi_LISA + lam_d + Om_A) - np.sin(3.0*Phi_LISA - lam_d + Om_A) ) \
                    -18.0*np.sqrt(3.0)*np.sin(th_d)*np.sin(th_d)*np.sin(2.0*Phi_LISA+Om_A) - \
                    np.sqrt(3.0)*(1.0+np.cos(th_d)*np.cos(th_d))*(np.sin(4.0*Phi_LISA-2.0*lam_d+Om_A) + 9.0*np.sin(2.0*lam_d+Om_A))  )

        Fc_A = (1.0/16.0)*( np.sqrt(3.0)*np.cos(th_d)*(np.cos(4.0*Phi_LISA - 2.0*lam_d + Om_A)) -\
                    9.0*np.cos(2.0*lam_d + Om_A) + 6.0*np.sin(th_d)*( np.cos(3.0*Phi_LISA-lam_d + Om_A) + \
                    3.0*np.cos(Phi_LISA +lam_d + Om_A) )  )

        Fp_E = (1.0/32.0)*( 6.0*np.sin(2.0*th_d) *(3.0*np.sin(Phi_LISA + lam_d + Om_E) - np.sin(3.0*Phi_LISA - lam_d + Om_E) ) \
                    -18.0*np.sqrt(3.0)*np.sin(th_d)*np.sin(th_d)*np.sin(2.0*Phi_LISA+Om_E) - \
                    np.sqrt(3.0)*(1.0+np.cos(th_d)*np.cos(th_d))*(np.sin(4.0*Phi_LISA-2.0*lam_d+Om_E) + 9.0*np.sin(2.0*lam_d+Om_E))  )

        Fc_E = (1.0/16.0)*( np.sqrt(3.0)*np.cos(th_d)*(np.cos(4.0*Phi_LISA - 2.0*lam_d + Om_E)) -\
                    9.0*np.cos(2.0*lam_d + Om_E) + 6.0*np.sin(th_d)*( np.cos(3.0*Phi_LISA-lam_d + Om_E) + \
                    3.0*np.cos(Phi_LISA +lam_d + Om_E) )  )

        cpsi = np.cos(2.0*self.psi)
        spsi = np.sin(2.0*self.psi)
        hp0  = -(1.0+np.cos(self.incl)*np.cos(self.incl))*ampl # multiply by sin\Phi
        hc0  = 2.0*np.cos(self.incl)*ampl

       
        print('[om] = ',om.shape)
        print('[hp0] = ',hp0.shape)
        print('[Fp_A] = ',Fp_A.shape)
        print('[cpsi] = ',cpsi.shape)
        print('[Fc_A] = ',Fc_A.shape)
        print('[spsi] = ',spsi.shape)
        print('[phi_p] = ',phi_p.shape)


        h_A = - 2.0*L*om*np.sin(del_phi)*( hp0*(Fp_A*cpsi - Fc_A*spsi)*np.cos(phi_p+phic) + \
                                hc0*(Fp_A*spsi + Fc_A*cpsi)*np.sin(phi_p+phic) )

        h_E = - 2.0*L*om*np.sin(del_phi)*( hp0*(Fp_E*cpsi - Fc_E*spsi)*np.cos(phi_p+phic) + \
                                hc0*(Fp_E*spsi + Fc_E*cpsi)*np.sin(phi_p+phic) )

        # Zero-pad
        A = np.pad(h_A, (0, len(tvec) - len(h_A)), 'constant')
        E = np.pad(h_E, (0, len(tvec) - len(h_E)), 'constant')

        return A, E


# Generate the signal in the frequency domain with the phenomD waveform and frequency domain antenna responce
class MBHB_phenomD:

    def __init__(self, theta):
        #print(theta)

        self.z  = theta[0]     #   red shift
        self.m1 = theta[1]     #   chirp mass 
        self.m2 = theta[2]      #   mass ratio
        self.chi1 = theta[3]   #   spin magnitude [-1, 1] of the primary
        self.chi2 = theta[4]   #   spin magnitude [-1, 1] of the secondary
        self.psi = theta[5]    #   polarization angle
        self.incl = theta[6]  #   inclination angle (L.n) wrt to barycenter 
        self.lam = theta[7]    #   sky position, longitude (barycenter)
        self.bet = theta[8]    #   bet -- sky position, latitude (barycenter) 
        self.Tc = theta[9]     #
        self.dt = theta[10]    #
        self.Tobs = theta[11]  #


  
    def TDI_signal(self, tvec):


        distance   = (Cosmology.DL(self.z, 0.0)[0])*1.e6 # in pc 

        values = [self.z,
                  self.m1,
                  self.m2,
                  self.chi1,
                  self.chi2,
                  0.0,
                  0.0,
                  self.Tc,
                  self.bet,
                  self.lam,
                  self.incl,
                  distance,
                  self.dt,
                  self.Tobs]


        params = self._fillMBHBparams(values)
        units = self._getMBHBunits()

        pr = ParsUnits(params,units)

        # Compute frequency domain response
        #freq, Xf, Yf, Zf = FD.ComputeMBHBXYZ_FD(pr)
        freq, Xf, Yf, Zf = FD.ComputeMBHBXYZ_FD_old(pr)

        Af, Ef, Tf = tdi.AET(Xf, Yf, Zf)
        return freq, Af, Ef, Tf

    def _getMBHBunits(self):
        return {'EclipticLatitude':                 'Radian',\
                'EclipticLongitude':                'Radian',\
                'PolarAngleOfSpin1':                'Radian',\
                'PolarAngleOfSpin2':                'Radian',\
                'Spin1':                            'MassSquared',\
                'Spin2':                            'MassSquared',\
                'Mass1':                            'SolarMass',\
                'Mass2':                            'SolarMass',\
                'CoalescenceTime':                  'Second',\
                'PhaseAtCoalescence':               'Radian',\
                'InitialPolarAngleL':               'Radian',\
                'InitialAzimuthalAngleL':           'Radian',\
                'Approximant':                      'ModelName',\
                'Cadence':                          'Seconds',\
                'Redshift':                         'dimensionless',\
                'Distance':                         'pc',\
                'ObservationDuration':              'Seconds'}

    # Read them from the catalog
    def _fillMBHBparams(self,values):

        param = {}

        param['Redshift'] = values[0] # redshift of coalescence
        param['Mass1'] = values[1] #  intrinsic mass of primary (solar masses);
        param['Mass2'] = values[2] # intrinsic mass of secondary (solar masses);
        param['Spin1'] = values[3] # spin magnitude of primary;
        param['Spin2'] = values[4] # spin magnitude of secondary;
        param['PolarAngleOfSpin1'] = values[5] #  angle between $s_1$ and $L$ ($L$ is the binary orbital angular momentum) in radians;
        param['PolarAngleOfSpin2'] = values[6] #  angle between $s_2$ and $L$ in radians;
        param['CoalescenceTime'] = values[7]
        param['EclipticLatitude'] = 0.5*np.pi - values[8] # beta
        param['EclipticLongitude'] = values[9] #  lambda
        param['InitialPolarAngleL'] = values[10] #  inclination 

        param['PhaseAtCoalescence'] = np.pi
        param['InitialAzimuthalAngleL'] = np.pi

        param['Distance'] = values[11]
        param['Cadence'] = values[12]
        param['Approximant'] = 'IMRPhenomD'
        param['ObservationDuration'] = values[13]

        return param
 
def normalise(x):
    y = (2*(x - x.min())/(x.max() - x.min())) - 1
    return y


# Generate noise
def generate_noise_AE(freq):

    psd_j = tdi.noisepsd_AE(freq[1:])
    # Gererate ramdom realisation of the noise
    n_real = np.random.normal(loc=0.0, scale=np.sqrt(psd_j/2))
    n_imag = np.random.normal(loc=0.0, scale=np.sqrt(psd_j/2))
    n_real = np.insert(n_real, 0, 0., axis=0)
    n_imag = np.insert(n_imag, 0, 0., axis=0)

    return n_real+1j*n_imag

# Inverse fft to have the noise in the time domain
def ComputeTD(Xf, dt):
    # Xf -- frequency series
    # dt -- cadence

    Xt= np.fft.irfft(Xf) # I am not sure if the normalisation is corretc and if we need to shift it by tc np.exp(1.0j*freq*tc)
    tm = np.arange(len(Xt))*dt

   ### TODO Check the normalization
    return tm, Xt*(1.0/dt)


 
def ComputeFD(Xt,dt):

    Xf = np.fft.rfft(Xt)
    df = 1.0/len(Xt)
    fvec = np.arange(len(Xf))*df
    
    ### TODO Check the normalization
    return (fvec, np.conjugate(Xf)*dt*np.sqrt(2.0))
            

# Inverse fft to have the noise in the time domain
#def ComputeTD_shift(Xf, dt, freq, tc):
    # Xf -- frequency series
    # dt -- cadence 
    # freq -- frequencies at which Xf is sampled 
    # tc -- time of coalescence

#    Xt= np.fft.irfft(Xf*np.exp(2.0j*np.pi*freq*tc)) # I am not sure if the normalisation is corretc and if we need to shift it by tc np.exp(1.0j*freq*tc)
#    tm = np.arange(len(Xt))*dt

#    return tm, Xt*(1.0/dt)













