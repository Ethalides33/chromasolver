
import matplotlib.pyplot as plt
import numpy as np
import scipy
import tmm
import pandas as pd
import math
from scipy.interpolate import interp1d
from scipy.constants import h, c, k, pi

VALID_OPT_CONST = {"n","k"}


class Simulation:
    def __init__(self,wavelengths, nkdata_path,thermo=True):
        self.wavelengths = wavelengths
        self.nkdata = pd.read_csv(nkdata_path,sep=";",decimal=",")
        self.power = pd.read_csv('SolarAM15.csv',header=0,sep=';')
        self.nkdata, self.power = self.interpolate_data(self.wavelengths,self.nkdata, self.power)
        self.thermo = thermo

    def interpolate_data(self,wavelengths,nkdata, power): #Also interpolate power data
        interpolated_df = pd.DataFrame(wavelengths,columns=["used_wls"])
        columns = nkdata.columns
        for index in range(0,len(columns),2):
            interpolator = interp1d(nkdata[columns[index]], nkdata[columns[index+1]], kind='linear', fill_value='extrapolate')
            interpolated_df[columns[index+1]] = interpolator(interpolated_df["used_wls"])
            interpolated_df[columns[index+1]].fillna(method="ffill",inplace=True)
        interpolated_df = interpolated_df.clip(lower=0)

        interpolated_power_df = pd.DataFrame(wavelengths,columns=["used_wls"])
        interpolator_power = interp1d(power["Wavelength (nm)"], power["Global tilt  mW*cm-2*nm-1 (1sun AM 1.5)"], kind='linear', fill_value='extrapolate')
        interpolated_power_df["Global tilt  mW*cm-2*nm-1 (1sun AM 1.5)"] = interpolator_power(interpolated_power_df["used_wls"])
        interpolated_power_df["Global tilt  mW*cm-2*nm-1 (1sun AM 1.5)"].fillna(method="ffill",inplace=True)

        return interpolated_df, interpolated_power_df
    
    def get_nk(self):
        return self.nkdata
    def get_power(self):
        return self.power
    def get_opt_const(self,material,wl):
        if material == "Air":
            return 1
        if material == "Glass":
            return 1.52
        # if material == "PET":
        #     return 1.56
        if material == "SiO2":
            return 1.43
        column_n = material+'_n'
        column_k = material+'_k'
        return self.nkdata.loc[self.nkdata["used_wls"] == wl][column_n].values[0] + self.nkdata.loc[self.nkdata["used_wls"] == wl][column_k].values[0]*1j
    
    def calculate_nk_particles(self,filling_factor,L=0.33333333,IR=False):
        e_matrix = (self.nkdata["PET_n"]+self.nkdata["PET_k"]*1j)**2
        def nk_part(n_film,k_film):
            n_film = np.array(n_film)
            k_film = np.array(k_film)
            complex_n = n_film+1j*k_film
            e_complex_vo2 = complex_n**2
            alpha_pol = (e_complex_vo2-e_matrix)/(e_matrix+L*(e_complex_vo2-e_matrix))
            e_composite = e_matrix*(1.0+(2.0/3.0)*filling_factor*alpha_pol)/(1.0-(1.0/3.0)*filling_factor*alpha_pol)
            n_complex_composite = np.sqrt(e_composite)
            n, k = np.real(n_complex_composite), np.imag(n_complex_composite)
            return n,k
        if IR:
            self.nkdata["vo2_composite_ins_n"],self.nkdata["vo2_composite_ins_k"] = nk_part(self.nkdata["vo2IR_ins_n"],self.nkdata["vo2IR_ins_k"])
            self.nkdata["vo2_composite_met_n"],self.nkdata["vo2_composite_met_k"] = nk_part(self.nkdata["vo2IR_met_n"],self.nkdata["vo2IR_met_k"])
        else:
            self.nkdata["vo2_composite_ins_n"],self.nkdata["vo2_composite_ins_k"] = nk_part(self.nkdata["VO2_ins_n"],self.nkdata["VO2_ins_k"])
            self.nkdata["vo2_composite_met_n"],self.nkdata["vo2_composite_met_k"] = nk_part(self.nkdata["VO2_met_n"],self.nkdata["VO2_met_k"])

    def sim_stack(self,stack,index_of_vo2_layer,coherent=False,THETA=0,POL="s"):
        """Returns (Total transmittance spectrum, total reflectance spectrum, absorbed % in VO2 spectrum,
            Total AM1.5 power absorbed in W/m^2, Tlum, Tsol,trans550)"""
        thicknesses = [mat[1] for mat in stack if mat[1] !=0] # in nanometers
        abs_in_vo2 = np.zeros(len(self.nkdata["used_wls"]))
        tot_trans = np.zeros(len(self.nkdata["used_wls"]))
        tot_refl = np.zeros(len(self.nkdata["used_wls"]))
        P_abs_tot = 0
        Tlum_num = 0
        Tlum_den = 0
        Tsol_num = 0
        Tsol_den = 0
        Tlum =0
        Tsol=0
        for index, wl in enumerate(self.nkdata["used_wls"]): 
            refr_indices = [self.get_opt_const(material[0], wl) for material in stack if material[1] != 0]
            if coherent:
                coh_tmm_data = tmm.coh_tmm(POL, refr_indices, thicknesses, THETA, wl)
                absorb_per_layer_data = tmm.absorp_in_each_layer(coh_tmm_data)
            else:
                coherence_list = []
                for m in stack:
                    if m[1] != 0:
                        if m[0] in ("Air","Glass","SiO2_fused","vo2_composite_met","vo2_composite_ins"):
                            coherence_list.append("i")
                        else:
                            coherence_list.append("c")
                coh_tmm_data = tmm.inc_tmm(POL, refr_indices, thicknesses,coherence_list, THETA, wl)
                absorb_per_layer_data = tmm.inc_absorp_in_each_layer(coh_tmm_data)
                
            abs_in_vo2[index] = absorb_per_layer_data[index_of_vo2_layer]
            tot_trans[index] = absorb_per_layer_data[-1]        
            tot_refl[index] = absorb_per_layer_data[0]
            if wl == 550:
                trans550 = absorb_per_layer_data[-1]
            else:
                trans550 = -1
            if wl <=4000 and self.thermo:    
                P_abs_tot += 10*absorb_per_layer_data[index_of_vo2_layer]*float(self.power[self.power['used_wls']==wl]['Global tilt  mW*cm-2*nm-1 (1sun AM 1.5)'].values)
                # /!\ Wavelength step must be 1 nm.
                Tlum_num += absorb_per_layer_data[-1]*float(self.nkdata[self.nkdata['used_wls']==wl]['Eye_sensitivity'].values)
                Tlum_den += float(self.nkdata[self.nkdata['used_wls']==wl]['Eye_sensitivity'].values)  

                Tsol_num += absorb_per_layer_data[-1]*float(self.power[self.power['used_wls']==wl]['Global tilt  mW*cm-2*nm-1 (1sun AM 1.5)'].values)
                Tsol_den += float(self.power[self.power['used_wls']==wl]['Global tilt  mW*cm-2*nm-1 (1sun AM 1.5)'].values)
        if self.thermo:
            Tlum = Tlum_num/Tlum_den
            Tsol = Tsol_num/Tsol_den
            
        return (100*tot_trans,100*tot_refl,100*abs_in_vo2,P_abs_tot,100*Tlum,100*Tsol,100*trans550)
    
    def calculate_average_emissivity_all(self, stack, indexVO2, temperature):

        def plancks_law(wavelength, temperature):
            return ((8 * pi * h * c) / (wavelength**5)) * 1 / (np.exp(h * c / (wavelength * k * temperature)) - 1)
        
        res = self.sim_stack(stack,indexVO2)
        emissivities = (100 - res[0] - res[1])/100
        wavelengths = self.wavelengths*1e-9 # en m        

        # Calculate the integrals for numerator and denominator
        numerator_integral = np.trapz(emissivities[wavelengths<2e-5] * plancks_law(wavelengths[wavelengths<2e-5], temperature), wavelengths[wavelengths<2e-5])
        denominator_integral = np.trapz(plancks_law(wavelengths[wavelengths<2e-5], temperature),wavelengths[wavelengths<2e-5])

        # Calculate average emissivity
        avg_emissivity = numerator_integral / denominator_integral

        return avg_emissivity

    def get_emis_fromRT(self,T,R,temperature):
        
        def plancks_law(wavelength, temperature):
            return ((8 * pi * h * c) / (wavelength**5)) * 1 / (np.exp(h * c / (wavelength * k * temperature)) - 1)
        
        emissivities = (100 - T - R)/100
        wavelengths = self.wavelengths*1e-9 # en m        

        # Calculate the integrals for numerator and denominator
        numerator_integral = np.trapz(emissivities[wavelengths<2e-5] * plancks_law(wavelengths[wavelengths<2e-5], temperature), wavelengths[wavelengths<2e-5])
        denominator_integral = np.trapz(plancks_law(wavelengths[wavelengths<2e-5], temperature),wavelengths[wavelengths<2e-5])

        # Calculate average emissivity
        avg_emissivity = numerator_integral / denominator_integral

        return avg_emissivity