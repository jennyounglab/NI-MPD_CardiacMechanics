
from re import sub
import warnings
warnings.simplefilter("ignore")
import pandas as pd
from pandas import ExcelWriter
import numpy as np
import os
from scipy.optimize import curve_fit as fit
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter as _svft

def svft(x, window_length, polyorder, **kwargs):
    # Auto-adjust window_length if signal is too short
    if len(x) < window_length:
        window_length = len(x) if len(x) % 2 == 1 else len(x) - 1  # must be odd
        if window_length < polyorder + 2:
            raise ValueError(
                f"savgol_filter error: insufficient data points (len={len(x)}) "
                f"for polyorder={polyorder}. Minimum required: {polyorder + 2}"
            )
    return _svft(x, window_length=window_length, polyorder=polyorder, **kwargs)

from scipy.stats import pearsonr
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import sys

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, to_rgba

import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to avoid memory issues
import matplotlib.pyplot as plt
import traceback


def shifted_colormap(base_cmap, new_min_color, n_colors=256, shift_fraction=0.1):
    base = plt.get_cmap(base_cmap)
    base_colors = base(np.linspace(shift_fraction, 1, n_colors))  # Skip lightest colors
    new_min_color_rgba = to_rgba(new_min_color)  # Convert hex to RGBA
    new_colors = np.vstack((new_min_color_rgba, base_colors))
    return LinearSegmentedColormap.from_list(f"{base_cmap}_shifted", new_colors)

mpl.rcParams.update({'font.size': 7})  # Set all fonts globally to size 9 small plot change to 7

cover='r' #cover = sys.argv[2]  # if you want to recalculate, set cover as "r", otherwise, set cover as "o"
mu = 0.5 # poisson ratio
rb = 50  # bead size

################# ROOT is the folder you put this pipeline file and all data samples #############
root = './' # for server
allslices = sorted([f.name for f in os.scandir(root) if f.is_dir() and not f.name.startswith('.')])
print(allslices)

######################################################################
def _init_worker(_fileinput, _output, _slice_name, _cover):
    # make these available to ProcessFile as module-level globals in each worker
    global fileinput, output, slice_name, cover
    fileinput = _fileinput
    output = _output
    slice_name = _slice_name
    cover = _cover

def ProcessFile(filepath):
    if filepath.find('XY position')>-1:
        return 'not this file'

    if (filepath.find('txt')==-1):
        return 'no such txt file'
    ## Create output folder
    outputsp = output+'/'+filepath[:-4]
    if os.path.exists(outputsp)==False:
        os.makedirs(outputsp)

    if cover=='o':   # go to judge whether we already have this point calculated 
        if os.path.exists(outputsp+'/result.csv')==True:
            print(filepath+' already calculated')
            return 

    # ✅ Robust check for P [max] == -0.001 uN
    indentation_present = True  # Default assume indentation is present
    with open(fileinput + '/' + filepath, 'r') as f:
        for line in f:
            if 'P [max]' in line:
                try:
                    pmax_value = float(line.split('=')[1].split()[0])
                    if pmax_value == -0.001:
                        indentation_present = False
                except (IndexError, ValueError):
                    print(f"[WARN] Failed to parse P[max] in {filepath}, assuming indentation present.")
                break  # Stop scanning after P[max] is found

    if not indentation_present:
        print(f"[SKIP] No indentation detected in {filepath} (P[max] = -0.001).")
        trtm = 'unknown'
        if any(tag in slice_name for tag in ['Y1', 'Y2', 'Y3', 'Y4']):
            trtm = 'young'
        elif any(tag in slice_name for tag in ['A1', 'A2', 'A3']):
            trtm = 'aged'
        res = pd.DataFrame([[slice_name, trtm, filepath, np.nan, np.nan, np.nan, np.nan, np.nan,
                            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                            np.nan, np.nan, np.nan, False]],
                        columns=['Sample', 'Treatment', 'PointPath', 'X (um)', 'Y (um)', 'Z (um)',
                                    'X', 'Y', 'Energy_l (fJ)', 'Energy_r (fJ)', 'Energy_ad (fJ)',
                                    'Loss', 'Hysteresis (fJ)', 'E_ad/E_l', 'Elasticity_l (Pa)', 'Elasticity_r (Pa)',
                                    'Perr_l', 'IndentationPresent'])
        res.to_csv(outputsp + '/result.csv')
        return 'skipped'

    # otherwise, we do the calculation fo this point

    ############ find the lines where we want the information ################
    print(f"Processing: {filepath}")
    all = pd.read_csv(fileinput+'/'+filepath,sep="\t",skip_blank_lines=False,names=range(15), encoding='ISO-8859-1')
    head_pos = 0
    xyz_pos = 0
    scan_pos = 0
    for i in range(len(all)):
        if all.iloc[i][0]=='Scan (#)':
            scan_pos = i
        if all.iloc[i][0]=='X-position (um)':
            xyz_pos = i
        if all.iloc[i][0]=='Time (s)':
            head_pos=i
            break

    df = pd.read_csv(fileinput+'/'+filepath,sep="\t",skiprows=head_pos-1,header=0,on_bad_lines='skip')
    #print(df)
    print(f"Loaded data shape for {filepath}: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    heads = list(df.head())
    
    profile = pd.read_csv(fileinput+'/'+filepath,sep="\t",nrows=7,names=range(10))
    xpos = float(profile.iloc[xyz_pos][1])
    ypos = float(profile.iloc[xyz_pos+1][1])

    # ✅ Check for valid indentation
    # Always use Z position from profile (don't block on indentation check)
    zpos = float(profile.iloc[xyz_pos+2][1])

    xx_raw, yy_raw = int(profile.iloc[scan_pos][3]), int(profile.iloc[scan_pos][5])  # preserve originals
    xx, yy = xx_raw - 1, yy_raw - 1  # shift indices so X-1 = 0 µm
    print(filepath, xpos, ypos, zpos, xx, yy)

    print(filepath,xpos,ypos,zpos,xx,yy)        

    try:
        sub_df = df[[heads[0],heads[1],heads[2]]] # heads[0]:Time, [1]:Load, [2]:Indentation
        fig,ax = plt.subplots(2,1,figsize=(7,6))
        ax[0].plot(sub_df[heads[0]],sub_df[heads[1]],color='blue')
        ax[0].set_xlabel(heads[0])
        ax[0].set_ylabel(heads[1])
        ax2 = ax[0].twinx()
        ax2.plot(sub_df[heads[0]],sub_df[heads[2]],color='red')
        ax2.set_ylabel(heads[2],rotation=270,labelpad=15)
        ax[0].spines['left'].set_color('blue')
        ax[0].tick_params(axis='y', colors='blue')
        ax[0].yaxis.label.set_color('blue')
        ax2.spines['right'].set_color('red')
        ax2.tick_params(axis='y', colors='red')
        ax2.yaxis.label.set_color('red')
        
        ### Remove holding #######
        ts,force,ind=sub_df[heads[0]].to_numpy(),sub_df[heads[1]].to_numpy(),sub_df[heads[2]].to_numpy()
        force = force-force[:100].mean()
        ind = ind-ind[:100].mean()
        std = np.std(force[:100]-force[:100].mean())/force.max()

        w = int(101*np.sqrt(std/0.005)/2)*2+1
        force = svft(force,w,1,mode='interp')
        ind = svft(ind,w,1,mode='interp')

        fg = svft(np.diff(force),w,1)
        pk = np.where(fg==fg.max())[0]
        vl = np.where(fg==fg.min())[0]-int(w/2)
        
        ax[1].plot(ts[1:],fg,color='black')
        ax3 = ax[1].twinx()
        ax3.plot(ts,force,color='blue')
        ax3.scatter(ts[pk],force[pk])
        ax3.scatter(ts[vl],force[vl])
        ceiling = force[vl]
        sel = np.where(force<ceiling)[0]
        unsel = np.where(force>=ceiling)[0]
        lend, rstart = unsel[0],unsel[-1]

        #Define trimmed segments
        time_l,time_r = ts[:lend],ts[rstart:]
        force_l,force_r = force[:lend],force[rstart:]
        ind_l, ind_r = ind[:lend],ind[rstart:]
        ax3.plot(time_l,force_l,color='red',lw=3,alpha=0.3)
        ax3.plot(time_r,force_r,color='lime',lw=3,alpha=0.3)

        ax[1].set_xlabel(heads[0])
        ax[1].set_ylabel('Force gradient')
        ax3.set_ylabel(heads[1],rotation=270,labelpad=15)
        ax3.spines['right'].set_color('blue')
        ax3.tick_params(axis='y',colors='blue')
        ax3.yaxis.label.set_color('blue')    
        plt.tight_layout()
        plt.savefig(outputsp+'/curve_holding_removed.jpg',dpi=300)
        plt.close('all')  # ✅ free memory

        ###### correct slope ################
        ind_r = ind_r-(ind_r[0]-ind_l[-1])
        time_r = time_r - (time_r[0]-time_l[-1])

        force_ls, force_rs=svft(force_l,51,1),svft(force_r,51,1)
        slope = svft(np.gradient(force_ls,time_l),51,1)
        sloslope = svft(np.gradient(slope,time_l),51,1)
        start= np.where(sloslope>pow(10,-8))[0][0]
        dx = np.diff(time_l)

        ind_start = int(np.where(ind_l>3)[0][0]/2)+1 # cannot be 1
        sl_l = slope[:ind_start].mean()
        if sl_l>pow(10,-3): # meaning that his slope is too large, invalid
            sl_l = 0
        fluctuation = (force_l[:ind_start]-sl_l).std()
        fl = time_l*sl_l+force_l[0]
        fr = -(time_r-time_r[0])*sl_l

        new_r = force_rs-fr-fl[-1]
        corr = new_r[-50].mean()

        sl_r = corr/(time_r.max()-time_r.min())
        fr=  (time_r-time_r[0])*(-sl_l+sl_r)

        fig,ax = plt.subplots(1,2,figsize=(7,3))
        ax[0].plot(time_l,force_ls, color='blue',alpha=0.5)
        ax[0].plot(time_l, force_ls-fl,color='blue', linestyle='dashed')
        ax[0].plot(time_r,force_rs, color='orange',alpha=0.5,label='original')
        ax[0].plot(time_r, force_rs-fr-fl[-1],color='orange', linestyle='dashed',label='corrected')
        
        #find the offset time
        force_l, force_r = force_l-fl,force_r-fr-fl[-1]
        dforce = np.where(svft(force_l,51,1)>min(3*fluctuation,force_l.max()/40))[0]
        if len(dforce)>0:
            start = dforce[0]
        else:
            start = ind_start
        time_s = time_l[start]    
        ax[0].plot([time_s,time_s],[-force_l.max()*0.1,force_l.max()*1.1],color='black')
        of = force_l[start]
        x1 = ind_l[start]
        finish = np.where(force_r<of)[0][0]
        time_f = time_r[finish]
        ax[0].plot([time_l[0],time_r[-1]],[0,0],color='black')
        ax[0].plot([time_f,time_f],[-force_l.max()*0.1,force_l.max()*1.1],color='grey')
        ax[0].set_ylabel(heads[1])
        ax[0].legend(loc=0,frameon=False)

        ################## select range for fitting #######################
        y0 = 3*fluctuation
        bs_st =  np.where(force_l>y0)[0][0]
        temp = (force_l-y0)[1:]*(force_l-y0)[:-1] 
        try:
            x0 = ind_l[np.where(temp<0)[0][-1]]
        except:
            x0 = 0
        temp = (force_r-y0)[1:]*(force_r-y0)[:-1]
        try:
            xf = ind_r[np.where(temp<0)[0][0]]
        except:
            xf = ind_r[-1]

        ax[1].scatter(ind_l-x0,force_l-y0,color='blue',alpha=0.8,s=5,label='approach')
        ax[1].scatter(ind_r-xf,force_r-y0,color='orange',alpha=0.8,s=5,label='retraction')
        ax[1].set_ylim(pow(10,-4),2*force_l.max())
        ax[1].plot([0.5*pow(10,2),3*pow(10,3)],[0.0000002*pow(0.5*pow(10,2),1.5),0.0000002*pow(3*pow(10,3),1.5)],color='black',label='Hertz(1.5)')
        ax[1].set_xscale('log')
        ax[1].set_yscale('log')
        ax[1].set_xlabel(heads[2])
        ax[1].set_ylabel(heads[1])
        ax[1].legend(loc=2,frameon=False)
        plt.tight_layout()
        plt.savefig(outputsp+'/correction, log-log curve.jpg',dpi=300)
        plt.close('all')  # ✅ free memory

        ####################  FITTING DATA #################
        time_ll = time_l[start:]
        time_rr = time_r[:finish]
        force_ll, force_rr = force_l[start:],force_r[:finish]
        ind_ll ,ind_rr = ind_l[start:],ind_r[:finish]
        
        ####################### ENERGY ########################
        energy_l = np.abs(np.trapz(force_ll-of,x=ind_ll)) 
        energy_r = np.abs(np.trapz(force_rr-of, x=ind_rr))
        hysteresis = energy_l - energy_r
        loss = hysteresis / energy_l
        ind_rn = ind_r[finish:]
        force_rn = force_r[finish:]
        sel = np.where(force_rn<0)[0]
        if len(sel)>0:
            energy_a = np.abs(np.trapz(force_rn[sel],x=ind_rn[sel]))
        else:
            energy_a = 0
        print('Eng_a',energy_l,'fJ','Eng_r',energy_r,'fJ','Hysteresis',hysteresis,'fJ','WorkLoss',loss,'Adhesive energy:',energy_a, 'fJ', energy_a/energy_l)
        
        ################################# ELASTIC MODULUS ################
        plt.figure(figsize=(5,3))
        def simpleHertz(x,x0,y0,A):
            return A*pow((x-x0),1.5)+y0
        off = force_ll[:100].mean()
        x0,y0,A0 = ind_ll[0],force_ll[0],(force_ll[-1]-force_ll[0])/pow((ind_ll[-1]-ind_ll[0]),1.5)
        x0l,y0l=x0,y0
        sel = np.where((ind_ll[:int(len(ind_ll)*0.9)]<2500)&(ind_ll[:int(len(ind_ll)*0.9)]>500))[0]
        try:
            para,pcov = fit(simpleHertz,ind_ll[sel],force_ll[sel],p0 = [0,off,A0])
            x0l,y0l,_=para
            Ea = 0.75*para[2]*(1-mu**2)*np.sqrt(1/rb) *pow(10,10.5)
            perr  = np.sqrt(np.abs(np.diag(pcov)))/para
            perr_l = perr[-1]
            newforce = para[2]*pow(ind_ll-para[0],1.5)+para[1]
            plt.plot(ind_ll-para[0],force_ll-para[1], color='blue')
            plt.plot(ind_ll-para[0],newforce-para[1], color='navy', linestyle='--',label='approach, '+str((round(Ea)))+" Pa")
        except:
            Ea = np.nan
            perr_l = np.nan
        
        ### for retraction curve
        sel = np.where((ind_rr-ind_rr[-1])>100)[0]
        x0,y0,A0 = ind_rr[-1],force_rr[-1],(force_rr[0]-force_rr[-1])/pow((ind_rr[0]-ind_rr[-1]),1.5)
        try:
            para,pcov = fit(simpleHertz,ind_rr[sel],force_rr[sel],p0 = [x0,y0,A0])
            Er = round(0.75*para[2]*(1-mu**2)*np.sqrt(1/rb)*pow(10,10.5))    
            newforce = para[2]*pow(ind_rr-para[0],1.5)+para[1]
            plt.plot(ind_rr-para[0],force_rr-para[1], color='red')
            plt.plot(ind_rr-para[0],newforce-para[1], color='brown', linestyle='--',label='retraction, '+str((Er))+" Pa")
        except:
            Er = np.nan     

        plt.legend(loc=0,frameon=False,handlelength=0,labelcolor='linecolor')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(50,ind_ll.max()*1.2)    
        plt.ylim(off,force_ll.max()*1.2)    
        plt.xlabel(heads[2])
        plt.ylabel(heads[1])
        plt.tight_layout()
        plt.savefig(outputsp+'/HertzFitting.svg')
        plt.savefig(outputsp+'/HertzFitting.jpg',dpi=300)
        plt.close('all')  # ✅ free memory

        #################### plot indentation cycle
        plt.figure(figsize=(5,3))
        plt.scatter(ind_l,force_l,color='blue')
        plt.scatter(ind_r,force_r,color='orange')
        plt.scatter(sub_df[heads[2]],sub_df[heads[1]],color='grey',s=3,alpha=0.5)
        plt.plot([ind_rr.min(),ind_ll.max()],[0,0],color='black',linestyle='--')
        plt.xlabel(heads[2])
        plt.ylabel(heads[1])
        plt.tight_layout()
        plt.savefig(outputsp+'/app-ret cycle.jpg',dpi=300)
        plt.close('all')  # ✅ free memory       

        #### output the results to csv file #############################
        trtm = 'unknown'
        if any(tag in slice_name for tag in ['Y1', 'Y2', 'Y3', 'Y4']):
            trtm = 'young'
        elif any(tag in slice_name for tag in ['A1', 'A2', 'A3']):
            trtm = 'aged'
        res = pd.DataFrame([[slice_name,trtm,filepath,xpos,ypos,zpos, xx,
                             yy, energy_l,energy_r,energy_a,loss,hysteresis,energy_a/energy_l,
                             Ea,Er,perr_l,True]],
                   columns=['Sample','Treatment','PointPath','X (um)','Y (um)','Z (um)',
                            'X','Y','Energy_l (fJ)','Energy_r (fJ)','Energy_ad (fJ)','Loss','Hysteresis (fJ)','E_ad/E_l',
                            'Elasticity_l (Pa)','Elasticity_r (Pa)','Perr_l', 'IndentationPresent'])
        res.to_csv(outputsp+'/result.csv')
        return res

    except Exception as e:
        print(f"[ERROR] Failed to process {filepath}: {e}")
        traceback.print_exc()

def Generate_map(qtys):
    if os.path.exists(output+'/all samples.csv')==False:
        print('no all samples.csv')
        return 
    
    allres = pd.read_csv(output+'/all samples.csv')

    # shift to 1-based indices for plotting and Excel
    allres['Xi'] = allres['X'] + 1
    allres['Yi'] = allres['Y'] + 1

    # 1-based grid dimensions
    xmax1 = int(allres['X'].max()) + 1   # number of columns
    ymax1 = int(allres['Y'].max()) + 1   # number of rows

    # Derived columns / masks for the simplified (basic-only) pipeline
    allres['El_r/El_l'] = allres['Elasticity_r (Pa)'] / allres['Elasticity_l (Pa)']

    # Mask Elasticity_l where significance threshold fails
    allres.loc[allres['Perr_l'] > 0.05, 'Elasticity_l (Pa)'] = np.nan

    # Mask Z (um) for skipped points (no indentation detected)
    if 'IndentationPresent' in allres.columns:
        allres['Z (um)'] = allres['Z (um)'].where(allres['IndentationPresent'] == True, np.nan)

    # Also exclude extreme Z values
    allres.loc[allres['Z (um)'] > 5000, 'Z (um)'] = np.nan

    xmax, ymax = allres['X'].max(), allres['Y'].max()
    newres = allres

    # Default basic-only quantities if none provided
    if not qtys:
        qtys = ['Z (um)', 'Energy_l (fJ)', 'Loss', 'Hysteresis (fJ)', 'Energy_ad (fJ)',
                'Elasticity_l (Pa)', 'Elasticity_r (Pa)']

    # Colormaps (only keys we actually use now)
    cmps = {
        'Z (um)': 'copper',
        'Energy_l (fJ)': 'GnBu',
        'Loss': 'jet',
        'Hysteresis (fJ)': 'viridis',
        'Energy_ad (fJ)': 'Blues',
        'Elasticity_l (Pa)': 'hot_r',
        'Elasticity_r (Pa)': 'pink_r'}
    
    # Optional fixed ranges (leave empty dict to autoscale everything)
    fixed_ranges = {
        # 'Energy_l (fJ)': [15, 45],
        # 'Loss': [0.2, 0.9],
        # 'Energy_ad (fJ)': [0, 15],
        # 'Elasticity_l (Pa)': [100, 2400],
        # 'Elasticity_r (Pa)': [150, 4500],
    }

    # ---- fixed 2 x 3 grid ----
    n = len(qtys)
    cols = 3
    rows = max(1, int(np.ceil(n / cols)))
    # make the grid panels bigger by increasing figsize (tweak as you like)
    BG_MODE  = 'transparent'
    BG_GREY  = '#f2f2f2'   # light grey background for 'grey' mode
    fig, axes = plt.subplots(rows, cols, figsize=(6, 2 * rows))

    # Flatten axes
    axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    # Apply background mode
    if BG_MODE == 'transparent':
        # Make figure and axes transparent
        for ax in axes_flat:
            ax.set_facecolor('none')
        fig.patch.set_alpha(0)
    elif BG_MODE == 'grey':
        # Use grey background for both fig and axes
        for ax in axes_flat:
            ax.set_facecolor(BG_GREY)
        fig.patch.set_facecolor(BG_GREY)
    # else: 'white' => leave defaults

    if isinstance(axes, np.ndarray):
        axes_flat = axes.flatten()
    else:
        axes_flat = [axes]  # single axis case

    plt.suptitle(newres['Sample'].unique()[0] if 'Sample' in newres.columns else '')

    for i in range(n):
        ax = axes_flat[i]
        qty = qtys[i]

        # thinner axes borders and ticks
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        ax.tick_params(axis='both', which='both', width=0.5, length=2)

        if qty not in newres.columns:
            ax.set_visible(False)
            continue

        # Choose colormap
        cmap = cmps.get(qty, 'viridis')

        # Color scaling
        if qty in fixed_ranges:
            fixed_min, fixed_max = fixed_ranges[qty]
            data_min, data_max = np.nanmin(newres[qty]), np.nanmax(newres[qty])
            vmn, vmx = fixed_min, fixed_max
            cbar_min, cbar_max = max(fixed_min, data_min), min(fixed_max, data_max)
        else:
            vmn = np.nanpercentile(newres[qty], 10)
            vmx = np.nanpercentile(newres[qty], 90)
            cbar_min, cbar_max = vmn, vmx

        # Determine marker sizes and selection logic
        # ---- draw as a gapless grid using imshow ----
        # make a grid of shape (ymax1, xmax1), bottom row = Yi==1 (origin='lower')
        grid = np.full((ymax1, xmax1), np.nan)
        for _, r in newres[['Xi', 'Yi', qty]].dropna().iterrows():
            xi = int(r['Xi']) - 1
            yi = int(r['Yi']) - 1
            grid[yi, xi] = float(r[qty])

        # mask NaNs so they render as transparent/solid background (no seams)
        grid_ma = np.ma.masked_invalid(grid)

        # if you want NaNs to be transparent on transparent background:
        # (optional) set cmap "bad" color depending on BG_MODE
        # cmap = plt.get_cmap(cmap).copy()  # if your Matplotlib supports .copy()
        # if BG_MODE == 'transparent':
        #     cmap.set_bad(alpha=0)
        # elif BG_MODE == 'grey':
        #     cmap.set_bad(color=BG_GREY)

        im = ax.imshow(
            grid_ma,
            origin='lower',
            extent=(0.5, xmax1 + 0.5, 0.5, ymax1 + 0.5),  # align to your 1-based cell edges
            cmap=cmap,
            vmin=vmn,
            vmax=vmx,
            interpolation='nearest'  # <- critical: no smoothing, no gaps
        )

        # draw a box outline around non-significant cells (doesn't create gaps)
        nonsig = newres.loc[(newres['Perr_l'] > 0.05) & (~newres[qty].isna())]
        for _, r in nonsig[['Xi','Yi']].iterrows():
            ax.add_patch(plt.Rectangle((r['Xi']-0.5, r['Yi']-0.5), 1, 1,
                                    fill=False, edgecolor='black', linewidth=0.6))

        # keep this handle for the colorbar
        mappable = im


        # Axes/ticks formatting
        # 1-based grid frame: centers on integer cells
        ax.set_xlim(0.5, xmax1 + 0.5)
        ax.set_ylim(0.5, ymax1 + 0.5)
        ax.set_aspect('equal', adjustable='box')

        # 1-based ticks/labels
        ax.set_xticks(np.arange(1, xmax1 + 1, 1))
        ax.set_yticks(np.arange(1, ymax1 + 1, 1))
        ax.set_xlabel('X')  # font sizes set below
        ax.set_ylabel('Y')
        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.set_aspect('equal', adjustable='box')

        # Colorbar
        if mappable is not None:
            c = plt.colorbar(mappable, ax=ax, extend='neither', shrink=0.8)
            c.outline.set_linewidth(0.5)
            c.set_ticks(np.linspace(cbar_min, cbar_max, 5))
            c.ax.tick_params(labelsize=5)
            if BG_MODE == 'transparent':
                c.ax.set_facecolor('none')
            elif BG_MODE == 'grey':
                c.ax.set_facecolor(BG_GREY)

        # Titles + median
        ax.set_title(qty, fontweight='bold', loc='center', pad=10)
        try:
            med = np.nanmedian(newres[qty])
            ax.text(0.5, 1.01, str(round(med, 2)),
                    transform=ax.transAxes, ha='center', va='bottom', fontsize=7)
        except Exception:
            pass

    # Hide any unused axes (if grid > n)
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)
    
    plt.subplots_adjust(hspace=0.15, wspace=0.15)
    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.95], pad=0.3)

    # Safe filename (in case slice_name has odd chars)
    import re
    safe_slice_name = re.sub(r'[^\w\-_. ]', '_', slice_name)
    filename_base = os.path.join(output, f"{safe_slice_name}_maps")

    transparent_flag = (BG_MODE == 'transparent')

    plt.savefig(f"{filename_base}.tiff", dpi=600, transparent=transparent_flag)
    plt.savefig(f"{filename_base}.png", dpi=600, transparent=transparent_flag)
    plt.savefig(f"{filename_base}.pdf", transparent=transparent_flag)
    if BG_MODE == 'transparent':
        # choose: either skip JPG or export with white background
        plt.savefig(f"{filename_base}.jpg", dpi=600, facecolor='white')
    else:
        plt.savefig(f"{filename_base}.jpg", dpi=600)  # grey/white ok
    plt.close('all')

    # ---- Excel export (kept as-is, same qty list) ----
    increment = 100
    xi = np.arange(1, xmax1 + 1)
    yi = np.arange(1, ymax1 + 1)

    # Labels in microns: 1->100 µm, 2->200 µm, ...
    cols = [f"X-{int(x * increment)}" for x in xi]
    rows = [f"Y-{int(y * increment)}" for y in yi[::-1]]  # flip Y for top row highest

    with ExcelWriter(output+'/maps.xlsx') as writer:
        for qty in qtys:
            if qty not in newres.columns:
                continue
            grid = np.full((len(yi), len(xi)), np.nan)
            for r, y in enumerate(yi[::-1]):
                for c, x in enumerate(xi):
                    vals = newres.loc[(newres['Xi'] == x) & (newres['Yi'] == y), qty].to_numpy()
                    grid[r, c] = vals[0] if vals.size else np.nan

            df_map = pd.DataFrame(grid, columns=cols, index=rows)
            sheetname = qty.replace('/',' vs ')
            df_map.to_excel(writer, sheetname)
    return newres

def Calculate_correlation(newres,qtys):
    corrs =[]
    heads = list(newres)
    newres=newres.replace([np.inf],np.nan)
    newres = newres.dropna()
    
    if len(qtys)==0: #by default 
        qtys = ['Energy_l (fJ)', 'Energy_r (fJ)', 'Loss', 'Hysteresis (fJ)', 'Energy_ad (fJ)',
        'Elasticity_l (Pa)', 'Elasticity_r (Pa)', 'Perr_l'] 
    
    for i in range(len(qtys)):
        for j in range(len(qtys)):
            if j<=i:
                continue
            
            x = newres[qtys[i]]
            y = newres[qtys[j]]

            # Combine x and y, drop NaNs and infs
            xy = pd.concat([x, y], axis=1).dropna()
            xy = xy[~xy.isin([np.inf, -np.inf]).any(axis=1)]

            if len(xy) >= 2:
                corr, pval = pearsonr(xy.iloc[:, 0], xy.iloc[:, 1])
                corrs.append([qtys[i], qtys[j], corr, pval])

    corrs = sorted(corrs,key = lambda x:np.abs(x[2]),reverse=True)
    corrs = pd.DataFrame(corrs,columns=['Qty1','Qty2','Corr','Pval'])
    corrs.to_csv(output+'/correlations.csv')


###################################### MAIN MAIN MAIN MAIN #####################################################################################

import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import freeze_support
import multiprocessing
import pathlib

if __name__ == '__main__':

    # ---- configure which slices to run ----
    start_idx = 0
    end_idx   = len(allslices) - 1   # or set to 94 explicitly

    for no in range(start_idx, end_idx + 1):
        slice_name = allslices[no]
        print(f"\n========== Processing slice [{no}] -> {slice_name} ==========\n")

        # Rebind per-slice paths (globals used by functions)
        fileinput = os.path.join(root, slice_name)
        run_date = datetime.now().strftime('%Y%m%d')
        output    = os.path.join(fileinput, f'output_{run_date}')
        outputVsc = os.path.join(output, 'viscosity')
        os.makedirs(outputVsc, exist_ok=True)

        # Collect files recursively for this slice (include subfolders)
        files_txt = []
        for p in pathlib.Path(fileinput).rglob('*.txt'):
            name = p.name
            if name.startswith('._') or name.startswith('.DS_Store'):
                continue
            files_txt.append(str(p.relative_to(fileinput)))  # store relative paths
        files_txt = sorted(files_txt)

        if not files_txt:
            print(f"[WARN] No .txt files found under {fileinput}, skipping slice.")
            continue
    
        # Pre-create per-point output folders (avoid race conditions)
        for f in files_txt:
            outputsp = os.path.join(output, f[:-4])
            os.makedirs(outputsp, exist_ok=True)

        for n, file in enumerate(files_txt):
            print(n, str(file))
        print('all points no.', len(files_txt))

        # =========================
        # PARALLEL per-file processing (inside each slice)
        # =========================
        t0 = time.time()
        freeze_support()  # safe on Windows; no-op elsewhere
        core = multiprocessing.cpu_count()
        max_workers = min(len(files_txt), max(1, core - 1))
        print('Start Parallellization, core no.', str(core), 'using workers:', max_workers)

        allres = []
        error  = []

        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_init_worker,
            initargs=(fileinput, output, slice_name, cover)
        ) as executor:
            for f, res in zip(files_txt, executor.map(ProcessFile, files_txt)):
                if isinstance(res, str) and 'error' in res.lower():
                    print(f"[ERROR] Failed for file: {f} -> {res}")
                    error.append([f, res])
                else:
                    print(f"[OK] Processed {f}")
                if isinstance(res, pd.DataFrame):
                    allres.append(res)  # optional; we also read result.csv below

        print('Parallel processing took', time.time() - t0, 'sec')

        # ==========================================
        # Integrate per-point results for this slice
        # ==========================================
        allresfile = list(pathlib.Path(output).rglob('*result.csv'))
        print('result.csv found:', len(allresfile))

        for filename in allresfile:
            try:
                resdf = pd.read_csv(filename, index_col=0, encoding='latin1')
                allres.append(resdf)
            except Exception as e:
                print(f"[WARN] Failed to read {filename}: {e}")

        if len(allres) == 0:
            print("[FATAL] No successful outputs found for this slice. Skipping maps.")
            continue

        allres = pd.concat(allres, ignore_index=True)
        print(allres)
        allres.to_csv(os.path.join(output, 'all samples.csv'))

        # =========================
        # Generating per-slice maps & correlations
        ## All quantitys to be selected are listed in the column names of the "all samples"
        # =========================
        selected_qty = []  # default list inside Generate_map()
        newres = Generate_map(selected_qty) # return a new table without NA
        Calculate_correlation(newres, selected_qty)

    print("\nAll slices done.")
