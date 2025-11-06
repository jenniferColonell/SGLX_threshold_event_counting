# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 18:07:21 2025

Should only be run on filtered, background subtracted data (e.g. from CatGT).
Threshold the binary, merge spikes that are sufficiently close in time and space
to count as events. Algorithm adapated from JRClust.

Requires SGLX utilties to read metadata and binary.

readData creates an npy file of spike properties for a selected shank, named
'{binary_name}_dd_sh{shank index}.npy'. The file is saved in the directory with the 
Binary. The plotting routines read these files to make the plots.

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

              
"""
import numpy as np
from readSGLX import readMeta, SampRate, makeMemMapRaw, ChannelCountsIM, ChanGainsIM
from SGLXMetaToCoords import MetaToCoords
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle
import matplotlib.cm as cm
import time
import pandas as pd
import seaborn as sns

def getRMSDataPath(binFullPath):
    parent_path = binFullPath.parent
    bin_name = binFullPath.stem
    rms_name = f'{bin_name}_est_rms.npy'
    rms_path = parent_path.joinpath(rms_name)
    return rms_path

def getFrRMSdfPath(binFullPath):
    parent_path = binFullPath.parent
    bin_name = binFullPath.stem
    rms_name = f'{bin_name}_fr_rms.csv'
    rms_path = parent_path.joinpath(rms_name)
    return rms_path

def getDriftDataPath(binFullPath, sh_index):
    parent_path = binFullPath.parent
    bin_name = binFullPath.stem
    dd_name = f'{bin_name}_dd_sh{sh_index}.npy'
    dd_path = parent_path.joinpath(dd_name)
    return dd_path
    
def findPeaks(samplesIn, thresh, xc, zc, excl_chan, fs=30000 ):
    # find peaks with negative amp > threshold, in a batch of 
    # samples from all channels
    # range of points for calculateing a full waveform
    n_chan, n_samp = samplesIn.shape
    start_time = np.floor(fs*1.0/1000).astype(int)
    end_time = n_samp - np.floor(fs*1.0/1000).astype(int)
    
    
    # threshold should be in bits
    exceeds_thresh = samplesIn < -np.abs(thresh)  # boolean (n_chan, n_samp)
    
    # find local minima by taking only points whose neighbor points are
    # more positive.
    (peak_chan, peak_ind) = np.where(exceeds_thresh)  
    
    # remove spikes that occur on excluded channels
    for ch in excl_chan:
        rem_ind = np.where(peak_chan==ch)[0]
        if rem_ind.size > 0:
            peak_ind = np.delete(peak_ind, rem_ind)
            peak_chan = np.delete(peak_chan, rem_ind)
    
    too_early_ind = np.where(peak_ind < start_time)[0]
    if too_early_ind.size > 0:
        peak_ind = np.delete(peak_ind, too_early_ind)
        peak_chan = np.delete(peak_chan, too_early_ind)
    
    too_late_ind = np.where(peak_ind > end_time)[0]
    if too_late_ind.size > 0:
        peak_ind = np.delete(peak_ind, too_late_ind)
        peak_chan = np.delete(peak_chan, too_late_ind)
        
    peak_center = samplesIn[peak_chan,peak_ind]  # amplitudes of the peaks
    # compare each peak to neighbors in time, earlier and later, keep only 
    # those that are local minima
    loc_min = ((samplesIn[peak_chan, peak_ind-1] > peak_center) & \
                (samplesIn[peak_chan, peak_ind+1] > peak_center))
    peak_ind = peak_ind[loc_min]
    peak_chan = peak_chan[loc_min]   
    
    # valid peaks have 3 points in a row below threshold
    pass_inarow = ((samplesIn[peak_chan, peak_ind-1] < thresh) & \
                (samplesIn[peak_chan, peak_ind+1] < thresh))    
    peak_ind = peak_ind[pass_inarow]
    peak_chan = peak_chan[pass_inarow]   
    peak_sig = samplesIn[peak_chan,peak_ind]
    #print(f'before merging: {peak_ind.size}')
    peak_ind, peak_chan, peak_sig = mergePeaks(peak_ind, peak_chan, peak_sig, xc, zc, fs)   
    #print(f'after merging: {peak_ind.size}')
    xz = spike_pos(samplesIn, peak_ind, peak_chan, xc, zc)
    return peak_ind, peak_chan, peak_sig, xz
    
def mergePeaks(peak_ind, peak_chan, peak_sig, xc, zc, fs=30000):
    # spikes are detected on multiple sites
    # assume that spikes detected within a time threshold
    # and physical radius belong to the same spikeing 'event.'
    # Merge these into one event, calculate peak channel,
    # x and z center of mass based on the negative-going signal
    
    nLim = np.floor((1/1000)*fs).astype(int) # merge spikes within +/- 1 ms
    neigh_radius_um = 60
    near_sites = calc_neighbor_sites(xc,zc,neigh_radius_um)
    
    # sort all spikes in time
    sort_order = np.argsort(peak_ind)
    peak_ind = peak_ind[sort_order]
    peak_chan = peak_chan[sort_order]
    peak_sig = peak_sig[sort_order]
    

    chan_set = np.unique(peak_chan)
    num_chan = chan_set.size
    
    # remove spikes that are within 1 ms in each channels spike train 
    for i_chan in chan_set:        
        curr_ind = np.where(peak_chan==i_chan)[0]
        curr_times = peak_ind[curr_ind]
        curr_amp = peak_sig[curr_ind]
        spikes_to_check = np.where(np.diff(curr_times) < nLim)[0]
        amp_early = curr_amp[spikes_to_check]
        amp_late = curr_amp[spikes_to_check + 1]
        keep_early = (amp_early < amp_late).astype(int)   # looking for the more negative spike
        ind_to_remove = curr_ind[spikes_to_check + keep_early]
        peak_ind = np.delete(peak_ind, ind_to_remove)
        peak_chan = np.delete(peak_chan, ind_to_remove)
        peak_sig = np.delete(peak_sig, ind_to_remove)
        
     
    
    for i_chan in chan_set:    
        neigh_chan = near_sites[i_chan]        
        # remove current channel
        neigh_chan = neigh_chan[neigh_chan != i_chan]
        
        for j_chan in neigh_chan:
                        
            i_ind = np.where(peak_chan==i_chan)[0]
            j_ind = np.where(peak_chan==j_chan)[0]
         
            orig_ind = np.concatenate((i_ind,j_ind))
            ij_labels = np.concatenate((np.zeros((i_ind.size,),dtype=int), np.ones((j_ind.size,),dtype=int)))
            ij_times = np.concatenate((peak_ind[i_ind],peak_ind[j_ind]))
            ij_amps = np.concatenate((peak_sig[i_ind],peak_sig[j_ind]))
            
            order = np.argsort(ij_times)
            # reorder everything
            orig_ind = orig_ind[order]
            ij_labels = ij_labels[order]
            ij_times = ij_times[order]
            ij_amps = ij_amps[order]
            
            spikes_to_check = np.where(np.diff(ij_times) < nLim)[0]
            amp_early = ij_amps[spikes_to_check]
            amp_late = ij_amps[spikes_to_check + 1]
            keep_early = (amp_early < amp_late).astype(int)   # looking for the more negative spike
            relative_ind_to_remove = spikes_to_check + keep_early
            ind_to_remove = orig_ind[relative_ind_to_remove]
            peak_ind = np.delete(peak_ind, ind_to_remove)
            peak_chan = np.delete(peak_chan, ind_to_remove)
            peak_sig = np.delete(peak_sig, ind_to_remove)            
            
            
    return peak_ind, peak_chan, peak_sig

def spike_pos(samplesIn, peak_ind, peak_chan, xc, zc):
    xz = np.zeros((peak_ind.size, 2))
    near_sites = calc_neighbor_sites(xc,zc,neigh_radius_um=60)
    chan_set = np.unique(peak_chan)
    nSamp = 15  #before and after peak, 31 total ~  1 msec
    
    # calculate these channel-wise because those use the same set of 
    # neighbor channels
    for i_chan in chan_set:
        i_ind = np.where(peak_chan==i_chan)[0]
        cn = near_sites[i_chan]       
        for ci in i_ind:
            # get section of data
            ct = peak_ind[ci]           
            curr_dat = samplesIn[cn, ct-nSamp:ct+nSamp]           
            amps = np.abs(np.squeeze(np.min(curr_dat,axis=1)))
            norm = np.sum(amps)
            cm_x = np.sum(np.multiply(xc[cn],amps))/norm
            cm_z = np.sum(np.multiply(zc[cn],amps))/norm
            xz[ci] = [cm_x,cm_z]
    
    return xz
    
def calc_neighbor_sites(xc,zc,neigh_radius_um):
    # return a list of arrays of site indicies within site_radius_um
    n_site = xc.size
    near_sites = list()
    rad_sq = neigh_radius_um*neigh_radius_um
    for i in range(n_site):
        dist = np.square(xc-xc[i]) + np.square(zc-zc[i])
        neigh = np.where(dist < rad_sq)[0]
        near_sites.append(neigh)
    return near_sites

def plotOne(drift_data):
    plt.subplots(figsize=(6,2))
    n_spike = drift_data.shape[0]
    skip_step = np.floor(n_spike/50000) + 1
    plot_spikes = np.arange(0,n_spike,skip_step).astype(int)
    pd = drift_data[plot_spikes]
    
    c_lim = np.asarray([np.quantile(pd[:,2],0.1),np.quantile(pd[:,2],0.9)])
    even_divisor = 50
    c_lim = even_divisor*np.floor(c_lim/even_divisor)
    plt.scatter(pd[:,0],pd[:,1],s=0.1,c=pd[:,2],cmap='plasma',vmin=c_lim[0],vmax=c_lim[1])
    c = plt.colorbar()
    
def plotMult(bin_list, drift_list, sh_ind):
    # build a large sampled array from the n_sets, to look for 'obvious' drift
    fig, ax = plt.subplots(figsize=(6,2))
    max_spike = 50000  # total spikes in the output plot
    even_divisor = 10  # color and z limits will be integer multiples of this value
    n_set = len(bin_list)
    total_spike = 0
    set_dur = np.zeros((n_set,))
    dd_list = list()
    for n, curr_bin in enumerate(bin_list):
        curr_path = getDriftDataPath(curr_bin, sh_ind)
        dd_list.append(np.load(curr_path))
        n_spike, n_meas = dd_list[n].shape
        total_spike = total_spike + n_spike
        set_dur[n] = np.max(dd_list[n][:,0])
    skip_step = np.floor(total_spike/max_spike)+1
    # loop over the arrays, buid sampled array that covers all the datasets
    samp_spike = np.zeros((max_spike,n_meas))
    n_samp = 0
    boundaries=np.zeros((n_set,))
    for n, dd in enumerate(dd_list):
        curr_nspike = dd.shape[0]
        curr_ind = np.arange(0,curr_nspike,skip_step).astype(int)
        curr_samp = dd[curr_ind,:]
        boundaries[n] = np.sum(set_dur[0:n])
        curr_samp[:,0] = curr_samp[:,0] + boundaries[n]
        curr_samp[:,1] = curr_samp[:,1] + np.sum(drift_list[0:n+1])
        n_curr = curr_ind.size
        samp_spike[n_samp:n_samp+n_curr,:] = curr_samp
        n_samp = n_samp + n_curr
        
    c_lim = np.asarray([np.quantile(samp_spike[:,2],0.05),np.quantile(samp_spike[:,2],0.95)])
    c_lim = even_divisor*np.floor(c_lim/even_divisor)
    plt.scatter(samp_spike[:,0],samp_spike[:,1],s=0.1,c=samp_spike[:,2],cmap='plasma',vmin=c_lim[0],vmax=c_lim[1])
    c = plt.colorbar()
    
    min_z = even_divisor*np.floor(np.min(samp_spike[:,1])/even_divisor)
    max_z = even_divisor*np.floor(np.max(samp_spike[:,1])/even_divisor)
    print(min_z, max_z)
    ax.set_ylim([min_z-15,max_z+15])
    #ax.set_ylim((max_z-700,max_z+50))
    plt.vlines(boundaries[1:],ymin=min_z+15,ymax=max_z-15, color='black', linestyles='solid')
    plt.show()
    
def readData(binFullPath, selected_sh, time_sec = 300, excl_chan = [127], thresh = -80):
    # Read in metadata; returns a dictionary with string for values
    meta = readMeta(binFullPath)
    parent_path = binFullPath.parent
    quirogaDenom = 0.6745

    # plan to detect peaks in the last 5 minutes of recording
    sRate = SampRate(meta)
    n_ap, n_lf, n_sync = ChannelCountsIM(meta)
    APChan0_to_uV = ChanGainsIM(meta)[2]
    thresh_bits = thresh/APChan0_to_uV
    maxInt = int(meta['imMaxInt'])
    x_coord, z_coord, sh_ind, connected, n_chan_tot = \
        MetaToCoords(binFullPath.with_suffix('.meta') , -1)
    rawData = makeMemMapRaw(binFullPath, meta)
    nChan, nFileSamp = rawData.shape
    
    
    start_samp = np.floor(nFileSamp - time_sec*sRate).astype(int)
    if start_samp < 0:
        start_samp = 0
        
    # check for file of rms estimates. If absent, calc for whole probe
    # (just saves reassembling from a file per shank)
    rmsFullPath = getRMSDataPath(binFullPath)
    if not(rmsFullPath.is_file()):
        rms_est = np.zeros((nChan,))
        for i in range(nChan):
            curr_samp = rawData[i,start_samp:]        
            rms_est[i] = estMedian(curr_samp, maxInt)
        rms_est = rms_est*APChan0_to_uV/quirogaDenom
        np.save(rmsFullPath,rms_est)
    
    batch_samp = np.floor(2*sRate).astype(int)
    n_batch = np.floor((nFileSamp - start_samp)/batch_samp).astype(int)
    sel_sh_ind = np.where(sh_ind == selected_sh)[0]   
    # translate excluded channels for this shank
    ex_sh = list()
    for ch in excl_chan:
        if np.sum(sel_sh_ind == ch) > 0:
            ex_sh.append(np.where(sel_sh_ind == ch)[0][0])
   
    x_sh = x_coord[sel_sh_ind]
    z_sh = z_coord[sel_sh_ind]
      
    for j in range(n_batch):
        st = start_samp + j*batch_samp
        cb = rawData[sel_sh_ind, st:st+batch_samp]
        peak_ind, peak_chan, peak_sig, xz = findPeaks(cb, thresh_bits, x_sh, z_sh, ex_sh, fs=sRate )
        # add offset to peak_ind, concatenate onto set
        offset = st - start_samp
        peak_ind = (peak_ind + offset)/sRate  # convert the times to sec
        peak_sig = abs(peak_sig * APChan0_to_uV)
        if j == 0:
            all_spikes = np.vstack((peak_ind, xz[:,1].T, peak_sig, xz[:,0].T,peak_chan)).T
        else:
            curr_spikes = np.vstack((peak_ind, xz[:,1].T, peak_sig, xz[:,0].T,peak_chan)).T
            all_spikes = np.concatenate((all_spikes,curr_spikes))    
        
        
    dd_path = getDriftDataPath(binFullPath, selected_sh)
    np.save(parent_path.joinpath(dd_path), all_spikes)
    
    
def calc_pdf(bin_path, sh_list):
    # calculate prob density function across all shanks    
    for j, sh_ind in enumerate(sh_list):       
        curr_path = getDriftDataPath(bin_path, sh_ind)
        curr_dat = np.load(curr_path)
        if j == 0:
            # calculate bin width and bin edgeds
            amp_sort = np.sort(curr_dat[:,2])
            bin_width = np.unique(np.diff(amp_sort))[1]
            bin_edges = (0.5*bin_width) + np.arange(0,1000,bin_width)
            sum_hist = np.histogram(curr_dat[:,2], bin_edges)[0]
        else:
            sum_hist = sum_hist + np.histogram(curr_dat[:,2], bin_edges)[0]
    # convert to pdf by normalizing
    npts = len(sum_hist)
    pdf = np.zeros((npts,2))
    pdf[:,0] = bin_edges[0:npts]
    pdf[:,1] = sum_hist/(np.sum(sum_hist)*bin_width)
    #plt.plot(pdf[:,0],pdf[:,1])
    return pdf

def calcRate(bin_path, sh_list):
    # calculate total spike rate for the probe
    for j, sh_ind in enumerate(sh_list):
        curr_path = getDriftDataPath(bin_path, sh_ind)
        curr_dat = np.load(curr_path)
        if j == 0:
            # calculate bin width and bin edge
            min_time = np.min(curr_dat[:,0])
            max_time = np.max(curr_dat[:,0])
            total_count = curr_dat.shape[0]
        else:
            min_time = np.min([min_time, np.min(curr_dat[:,0])])
            max_time = np.max([max_time, np.max(curr_dat[:,0])])
            total_count = total_count + curr_dat.shape[0]
    # convert rate
    spike_rate = total_count/(max_time - min_time)

    return spike_rate 
    
def plotMultPDF(bin_list, sh_list):
    fig, ax = plt.subplots(figsize=(4,2))
    n_pdf = len(bin_list)
    
    cmap = mpl.colormaps['winter']
    colors = cmap(np.linspace(0, 1, n_pdf))

    for j in range(n_pdf):
        print(repr(j))
        curr_pdf = calc_pdf(bin_list[j], sh_list)        
        ax.plot(curr_pdf[:,0], curr_pdf[:,1], color=colors[j] )
    ax.set_xlim(0,500)
    plt.xlabel('spike amplitude (uV)')
    plt.show()   
    
def plotSpikeRate(bin_list, day_list, sh_list):
    
    n_meas = len(bin_list)
    spike_rates = np.zeros((n_meas,))

    for j in range(n_meas):
        spike_rates[j] = calcRate(bin_list[j],sh_list)
    trend_x = np.asarray([min(day_list),max(day_list)])
    trend_fit = np.poly1d(np.polyfit(np.asarray(day_list),spike_rates,1))
    trend_y = trend_fit(trend_x)
    
    fig, ax = plt.subplots(figsize=(4,2))
    plt.scatter(day_list, spike_rates, s=3)
    plt.plot(trend_x,trend_y,marker=None, linewidth=1, linestyle='dashed')
    plt.xlabel('Days since implantation')
    plt.ylabel('spike rate (Hz)')
    plt.show()  
    
def calcFiringVsZ(binFullPath, selected_sh, bin_width, z_edges):
   # plot the relative firing rate vs. channel
   curr_path = getDriftDataPath(binFullPath, selected_sh)
   curr_dat = np.load(curr_path)     

   if z_edges is None:
       min_z = bin_width*np.floor(np.min(curr_dat[:,1])/bin_width)
       max_z = bin_width*np.ceil(np.max(curr_dat[:,1])/bin_width)
       z_edges = np.arange(min_z,max_z,bin_width)
       
   z_hist = (np.histogram(curr_dat[:,1], z_edges)[0]).astype('float64')
   z_rel = z_hist/np.sum(z_hist)
      
   return z_rel, z_edges
   
def plotRateVsZ(bin_list, day_list, selected_sh):
    
    bin_width = 15
    n_meas = len(bin_list)
    
    # call for first recording, returns z_edges
    rel0, z_edges =  calcFiringVsZ(bin_list[0], selected_sh, bin_width, None)   
    n_bin = len(z_edges)-1     
    bin_center = z_edges[0:n_bin] + bin_width/2       
    
    rel_rates = np.zeros((n_meas*n_bin,))
    x_vals = np.zeros((n_meas*n_bin))
    z_vals = np.zeros((n_meas*n_bin))
    
    # set values from first recording, x_val already = 0
    rel_rates[0:n_bin] = rel0
    z_vals[0:n_bin] = bin_center
    
    
    for j in range(1, n_meas):
        x_vals[j*n_bin:(j+1)*n_bin] = j
        z_vals[j*n_bin:(j+1)*n_bin] = bin_center
        rel_rates[j*n_bin:(j+1)*n_bin] = calcFiringVsZ(bin_list[j], selected_sh, bin_width, z_edges)[0]
        # plt.plot(bin_center,rel_rates[j*n_bin:(j+1)*n_bin])
        # plt.show()
        
    
    
    
    even_divisor = 10  # z limits will be integer multiples of this value
    min_z = even_divisor*np.floor(np.min(z_vals)/even_divisor)
    max_z = even_divisor*np.floor(np.max(z_vals)/even_divisor)
    print(min_z, max_z)
    fig, ax = plt.subplots(figsize=(0.3*len(day_list),3))    
    ax.set_ylim([min_z-15,max_z+15])   
     
    xt_range = np.arange(n_meas)
    xt_labels = np.asarray(day_list).astype('str')
    ax.set_xlim([-0.5,n_meas-0.5])
        
    even_divisor_c = 0.01
    c_lim = np.asarray([0,np.max(rel_rates)])
    #c_lim = even_divisor_c*np.ceil(c_lim/even_divisor_c)
    c_range = c_lim[1]-c_lim[0]
    
    # these points get covered by the patches; coloring with rel_rates
    # creates teh correct colorbar
    plt.scatter(x_vals,z_vals, c=rel_rates, s=2, marker='s', cmap='binary',vmin=c_lim[0],vmax=c_lim[1] )
    
    plt.xticks(xt_range, xt_labels)
    c = plt.colorbar()
    plt.xlabel('Days since implantation')
    plt.ylabel('distance from tip (um)')
    # Add rectangles
    width = 0.75 # in 'recording day' units
    height = 15 # in 'um'
    cmap = mpl.colormaps['binary']
    
    for i in range(len(x_vals)):
        color_val = (rel_rates[i]-c_lim[0])/c_range
        ax.add_patch(Rectangle(            
            xy=(x_vals[i]-width/2, z_vals[i]-height/2) ,width=width, height=height,
             edgecolor='None', facecolor=cmap(color_val)))
    #ax.axis('equal')
    
    plt.show()  
    
def estMedian(curr_samp, maxInt):
    nSamp = curr_samp.shape
    hist_edges = np.arange(maxInt)
    curr_dev = np.abs(curr_samp)
    counts = np.histogram(curr_dev, hist_edges)[0]
    dev_median = np.median(curr_dev)
    # interpolate the median bin in the cumulative dist to estimate the true median
    # This is 'estimating the mean for grouped data', wehre the groups are the bins
    lower_bound = int(dev_median)
    if lower_bound > 0:
        yDist = nSamp/2 - sum(counts[0:lower_bound])
    med_est = lower_bound + yDist/counts[lower_bound]
    return med_est
    
    
def estAllRMS(binFullPath, time_sec = 300, thresh = -80):
    # for each channel, get the mean absolute deviation 
    quirogaDenom = 0.6745
    # Read in metadata; returns a dictionary with string for values
    meta = readMeta(binFullPath)    
    # plan to detect peaks in the last 5 minutes of recording
    sRate = SampRate(meta)
    n_ap, n_lf, n_sync = ChannelCountsIM(meta)
    maxInt = int(meta['imMaxInt'])
    APChan0_to_uV = ChanGainsIM(meta)[2]
    rawData = makeMemMapRaw(binFullPath, meta)
    nChan, nFileSamp = rawData.shape
    sampRange = np.floor(time_sec*sRate).astype(int)
    rms_est = np.zeros((nChan,))
    for i in range(nChan):
        curr_samp = rawData[i,0:sampRange]        
        rms_est[i] = estMedian(curr_samp, maxInt)
        
    rms_est = rms_est*(APChan0_to_uV/quirogaDenom)      
    np.save(getRMSDataPath(binFullPath), rms_est)
    return rms_est
        
        
def build_rms_table(binFullPath):
    # read in meta, rms est, and dd files, create csv of channel, position, firing rate, rms
    x_coord, z_coord, sh_ind, connected, n_chan_tot = \
        MetaToCoords(binFullPath.with_suffix('.meta') , -1)
    rms = np.load(getRMSDataPath(binFullPath))
    n_chan = x_coord.shape[0]
    sh_present = np.unique(sh_ind)
    FR_rms_data = np.zeros((n_chan,6))
    FR_rms_data[:,0] = np.arange(n_chan)
    FR_rms_data[:,1] = sh_ind
    FR_rms_data[:,2] = x_coord
    FR_rms_data[:,3] = z_coord
    FR_rms_data[:,5] = rms[0:n_chan]
    
    for sh in sh_present.astype(int):
        orig_ch_ind = np.where(sh_ind == sh)[0]
        dd_curr = np.load(getDriftDataPath(binFullPath, sh))
        pk_chan, counts = np.unique(dd_curr[:,4], return_counts=True)
        span = np.max(dd_curr[:,0]) - np.min(dd_curr[:,0])
        curr_FR = counts.astype(float)/span
        orig_pk_chan = orig_ch_ind[pk_chan.astype(int)]
        FR_rms_data[orig_pk_chan,4] = curr_FR
    # load data into a dataframe for plotting and csv
    column_names = ['chan','sh','x','z','firing_rate','est_rms']
    FR_rms_df = pd.DataFrame(data=FR_rms_data, columns=column_names)
    FR_rms_df.to_csv(getFrRMSdfPath(binFullPath))
        
def plotRMS(binFullPath, FR_th=0.1):
    # read dataframe of firing rate and rms data, swarm plot of noise vs. shank
    # for sites with firing rate < threshold
    rms_df = pd.read_csv(getFrRMSdfPath(binFullPath), index_col=0)
    rms_pass_th = rms_df[ rms_df["firing_rate"]<FR_th ]
    sns.swarmplot(rms_pass_th,x='sh',y='est_rms')
    
def main():
    # samples for calling the above functions
    # note that the file paths, etc are specific, alter to match your data
    

    bin_parent = Path(r'Z:\cagatay\MB112\orig')     
    rec_list = ['200526_MB112_g0',
                '200527_MB112_H0_R1_g0',
                '200531_MB112_H0_R0_g0'
                ]

    
    # one value per day, first element = 0
    # 2nd = estiamted drift between second and first recording
    # 3rd = estimated drift between theird and fourth recording ...etc
    # only affects plotting in plotMult raster plots
    drift_list=[0,15,15]
    day_list = [0,1,5]
    
    
    prb_ind = 0
    sh_list = [0,1,2,3]
    b_recalc = False
    
    analysis_time_sec = 300  # readData extracts spikes in the last analysis_time_sec of the recording
    excl_chan = [96] # 127 is ref; add known noisy channels    
    threshold = -80
  
    
     
    for sh_ind in sh_list: 
        
        bin_list = list()   
        for ind1 in range(len(rec_list)):            
            # build bin path, name for ML data
            bin_dir = f'{rec_list[ind1]}'
            bin_name = f'{rec_list[ind1]}_tcat.imec{prb_ind}.ap.bin'
            bin_list.append(bin_parent.joinpath(bin_dir,bin_name))
            
        print(bin_list)
        
        if b_recalc:
            for binFullPath in bin_list:               
                readData(binFullPath, sh_ind, analysis_time_sec, excl_chan, threshold)
                
                
        # # for testing plotting, read bin_list[0]
        #out_name = f'drift_data_sh{sh_ind}.npy'
        #drift_data = np.load(bin_list[0].parent.joinpath(out_name))
        # plotOne(drift_data)
            
        # load saved drift data and plot        
        plotMult(bin_list,drift_list,sh_ind)
        plotRateVsZ(bin_list, day_list, sh_ind)
       
        if sh_ind == sh_list[0]:
            for binFullPath in bin_list:
                # make df of FR and rms for noise est
                build_rms_table(binFullPath)
                plotRMS(binFullPath)
        
    #plotMultPDF(bin_list, sh_list)
    plotSpikeRate(bin_list, day_list, sh_list)    
    
    
    
    
if __name__ == "__main__":
        main()  