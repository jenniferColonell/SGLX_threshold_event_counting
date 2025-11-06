# SGLX_threshold_event_counting
Simple code to find spiking events in SGLX data and estimate rms noise from the median absolute deviation.

Should only be run on filtered, background subtracted data, usually from CatGT.

Requires SGLX utilties to read metadata and binary. These are included in the repo.

tb_plots_NXT_ex.py is example code to call threshold_based_event_plots on multiple recordings, for example, recordings from multiple days in a chronic implant. The plots are useful for manually assessing drift and activity changes over time.

Functions in threshold_based_event_plots:

readData thresholds the binary, merges spikes that are sufficiently close in time and space to count as events. The algorithm is adapated from JRClust, by James Jun.

The output of readData is an npy filee of spike properties for a selected shank, named '{binary_name}_dd_sh{shank index}.npy'. The file is saved in the directory with the binary. The plotting routines read these files to make the plots.

The columns in the drfit data array are

The columns in the drfit data array are:
    spike times(sec) 
    spike z position (from center of mass), in um 
    amplitude of negative going peak, uV
    spike x position (from center of mass), in um 
    peak channel (of the channels on the shank -- not remapped to original channel in binary)
    
Plot types:
    plotOne: drift raster plot for a single shank, single recording
    plotMult: plot drift rasters for single shank, multiple recordings (typically, multiple days)
              useful for by eye drift assessment
    plotMultPDF: plot amplitude prob density function across all shanks, multiple recordings
                 (figure 2A of Steinmetz, et al. NP 2.0 paper)    
    plotSpikeRate: plot total spike rate across all shanks
                 (figure 2C of Steinmetz, et al. NP 2.0 paper) 
    plotRateVsZ: plot relative spike rate vs. z position for a single shank, multiple recordings
                 (figure 2B of Steinmetz, et al. NP 2.0 paper) 

The rms estimate is combined with the spike rate in a dataframe. An estimate of in vivo noise can be made by thresholding channels for low spike rate (<1 Hz an OK choice for standard probes).