# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 12:27:52 2025

@author: colonellj
"""
from pathlib import Path
import threshold_based_event_plots as tb

def main():
    # make plots of threshold based events, e.g. for stability measurments
    # note that the file paths, etc are specific, alter the bin_list creation
    # to match your file organization
    # also: remember that the data should be preprocessed through CatGT!

    bin_parent = Path(r'F:\JHU_data\NXT\sign_corrected\catgt_240523_NXT015_6_YT112_pm_jtipRef_g0')     
    rec_list = ['240523_NXT015_6_YT112_pm_jtipRef_ex0',
                '240523_NXT015_6_YT112_pm_jtipRef_ex1',
                '240523_NXT015_6_YT112_pm_jtipRef_ex2'
                ]

    
    # one value per day, first element = 0
    # 2nd = estiamted drift between second and first recording
    # 3rd = estimated drift between theird and fourth recording ...etc
    # only affects plotting in plotMult raster plots
    drift_list=[0,0,0]
    day_list = [0,1,2]
    
    
    prb_ind = 0
    sh_list = [0,1,2,3]
    b_recalc = True    # set to false if the spikes have already been found (e.g. to re-plot data)
    
    analysis_time_sec = 60  # readData extracts spikes in the last analysis_time_sec of the recording
    excl_chan = [] # 127 is ref; add known noisy channels    
    threshold = -80
  
 
     
    for sh_ind in sh_list: 
        bin_list = list()   
        for ind1 in range(len(rec_list)):            
            # build bin name for NXT example      
            # change thi scode to match your file organization!
            bin_name = f'{rec_list[ind1]}_g0_tcat.imec{prb_ind}.ap.bin'
            bin_list.append(bin_parent.joinpath(bin_name))
        
        print(bin_list)
        
        if b_recalc:
            for binFullPath in bin_list:               
                tb.readData(binFullPath, sh_ind, analysis_time_sec, excl_chan, threshold)
                

        # shank specific plots
        tb.plotMult(bin_list,drift_list,sh_ind)
        tb.plotRateVsZ(bin_list, day_list, sh_ind)  
    
    # summary over probe
    tb.plotMultPDF(bin_list, sh_list)
    tb.plotSpikeRate(bin_list, day_list, sh_list)
    
        
if __name__ == "__main__":
        main()   