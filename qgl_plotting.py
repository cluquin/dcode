#!/usr/bin/python3
import matplotlib as mpl
import matplotlib.cm as cm
mpl.rcParams['text.usetex']=True
mpl.rcParams['text.latex.unicode']=True
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
import matplotlib.mlab as mlab
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D
import scipy.fftpack as spf
from scipy import interpolate
import scipy.stats as stats
import peakdetect as pd
from math import log,sqrt
import numpy as np
import networkmeasures as nm
import measures
from os import makedirs, environ
import scipy.fftpack as spf
from scipy.optimize import curve_fit
from math import pi, sin, cos
import h5py

figurelist = []
# font = {'family':'normal', 'weight':'bold', 'size':16}
# mpl.rc('font',**font)

# def model(x, a, b):
#     return a*np.tanh(b*x**2)

# def model(x, a, b, c, d, e, f):
#     return (a*x+b*x**2+c*x**3)/(1+d*x+e*x**2+f*x**3)

# def model(x, a, b, c, d):
#     return (a*x+b*x**2)/(1+c*x+d*x**2)

# def model(x, a, b, c, d, e):
#     return (a + b*x + c*x**2)/(1+d*x+e*x**2)

def model(x, a, b, c, d):
    return (a + b*x + c*x**2)/(1+d*x+x**2)

# def model(x, a, b, c, d, e, f, g):
#     return (a + b*x + c*x**2 + d*x**3)/(1 + e*x + f*x**2 + g*x**3)

# def model(x, a, b):
#     return (a*x + b*x**2)/(1+x**2)

# def model(x, a, b):
#     return (a*x**b)/(1+x**b)

# def model(x, a, b, c):
#     return a*np.tanh(b*x**c)

def density_function(x):
    return 3.0*2**(-x)
def clustering_function(x):
    return 3.0*2**(-x)
def disparity_function(x):
    return 1.11/(x-1)
def vnentropy_function(x):
    return (1 - 2**(1 - (x - 1))/2)
def bondentropy_function(x):
    return (x/2.)-((x-1.)/2.)
def function_creator(ave1, ave2):
    def bond_entropy_fluctuation_function(x): 
        if x == 10:
            return ave1
        if x == 20:
            return ave2
    return bond_entropy_fluctuation_function

def IC_name(IC):
    return '-'.join(['{}{:0.3f}'.format(name,val) \
                for (name, val) in IC])

def sim_name(L, dt, t_span, IC, a=None, hamiltonian_type = None, V='X'):
    if a==None:
        return 'L{}_dt{}_t_span{}-{}_IC{}'.format(L, dt, t_span[0], t_span[1], IC_name(IC))
    else:
        if hamiltonian_type == None:
            return 'L{}_dt{}_t_span{}-{}_IC{}_a{}'.format(L, dt, t_span[0], t_span[1], IC_name(IC), a)
        else:
            return ('L{}_dt{}_t_span{}-{}_IC{}_a{}'+hamiltonian_type + '_V' + V).format(L, dt, t_span[0], t_span[1], IC_name(IC), a)

def sim_name2(L, dt, t_span, IC, a=None, hamiltonian_type = None, V='X'):
    sim_name = ('L{}_dt{}_t_span{}_{}_IC{}_a{}'+hamiltonian_type + '_V' + V).format(L, dt, t_span[0], t_span[1], IC_name(IC), a)
    sim_name = sim_name.replace('.','_')
    return sim_name

def sim_name_ic(IC):
    return 'IC{}'.format (IC_name(IC))
  
def meas_file(output_dir, L, dt, t_span, IC, a=None, hamiltonian_type = None, V = 'X', model_dir =
        environ['HOME']+'/Documents/qgl_exact/'): 
    # model_dir+output_dir+'/'+sim_name(L, dt, t_span, IC)+'.meas'
    if a==None:
        return model_dir+output_dir+'/'+sim_name(L, dt, t_span, IC)+'.meas'
    else:
        if hamiltonian_type == None:
            return model_dir+output_dir+'/'+sim_name(L, dt, t_span, IC, a = a)+'.meas'
        else:
            return model_dir+output_dir+'/'+sim_name(L, dt, t_span, IC, a = a, hamiltonian_type = hamiltonian_type, V = V)+'.meas'

def import_ham(L, model_dir =
        environ['HOME']+'/Documents/qgl_exact/'):
    hame_name = 'L'+str(self.L)+'_qgl_ham.mtx'
    ham_dir = model_dir + 'hamiltonians/'+self.ham_name
    return sio.mmread(ham_dr)

def import_data(output_dir, L, dt, t_span, IC, a=None, hamiltonian_type = None, V = 'X'):
    if a==None:
        mydata = measures.Measurements.read_in(0,meas_file(output_dir, L, dt, t_span, IC))
    else:
        if hamiltonian_type == None:
            mydata = measures.Measurements.read_in(0,meas_file(output_dir, L, dt, t_span, IC, a = a))
        else:
            mydata = measures.Measurements.read_in(0,meas_file(output_dir, L, dt, t_span, IC, a = a, hamiltonian_type = hamiltonian_type, V = V))
    mydata['L'] = L
    mydata['dt'] = dt
    mydata['t_span'] = t_span
    mydata['IC'] = IC
    mydata['a'] = a
    mydata['Nsteps'] = round(t_span[1]/dt) - round(t_span[0]/dt) 
    return mydata

def make_time_series(mydata,task,subtask):
    time_series = [mydata[task][i][subtask] for i in range(mydata['Nsteps']+1)]
    times = [mydata['t'][i] for i in range(mydata['Nsteps']+1)]
    return np.array(times), np.array(time_series)

def make_board(mydata,task,subtask):
    board = np.array([mydata[task][i][subtask] for i in range(mydata['Nsteps'])]).transpose()
    return board

def make_fft(mydata,task,subtask):
    times,time_series = make_time_series(mydata,task,subtask)
    dt = mydata['dt']    
    Nsteps = mydata['Nsteps']
    time_series = time_series - np.mean(time_series)
    if Nsteps%2 == 1:
        time_sereis = np.delete(time_series,-1)
        Nsteps = Nsteps - 1
    amps =  (2.0/Nsteps)*np.abs(spf.fft(time_series)[0:Nsteps/2])
    freqs = np.linspace(0.0,1.0/(2.0*dt),Nsteps/2)
    return freqs, amps


def colorbar_index(colorbar_label, ncolors, cmap, cb_tick_labels=None):
    if cb_tick_labels is None:
        cb_tick_labels = list(map(int, np.linspace(int(val_min),
            int(val_max), ncolors)))
    cmap = cmap_discretize(cmap, ncolors)
    mappable = cm.ScalarMappable(cmap=cmap)
    mappable.set_array([])
    mappable.set_clim(-0.5, ncolors+0.5)
    colorbar = plt.colorbar(mappable)
    colorbar.set_ticks(np.linspace(0, ncolors, ncolors))
    colorbar.set_ticklabels(cb_tick_labels)
    colorbar.set_label(colorbar_label, fontsize = 16)

def cmap_discretize(cmap, N):
    if type(cmap) == str:
        cmap = plt.get_cmap(cmap)
    colors_i = np.concatenate((np.linspace(0, 1., N), (0.,0.,0.,0.)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1., N+1)
    cdict = {}
    for ki,key in enumerate(('red','green','blue')):
        cdict[key] = [ (indices[i], colors_rgba[i-1,ki], colors_rgba[i,ki])
                       for i in range(N+1) ]
    # Return colormap object.
    return mcolors.LinearSegmentedColormap(cmap.name + "_%d"%N, cdict, 1024)

def plot_time_series(mydata,task,subtask, plots,fignum=1,ax=111,yax_label='yax_label',title='title',start=None,end=None, fitparams=(1,1)):
    start = mydata['t_span'][0] if start is None else start
    end = mydata['t_span'][1] if end is None else end
    if task == 'EC-Center':
        times = [mydata['t'][i] for i in range(mydata['Nsteps']+1)]
        time_series = mydata[task]
    elif task == 'deltaStau':
        deltas = deltaS(mydata)
        for tau in [5, 10, 20]:
            time_series = time_averagetau(deltas, tau)
    else:
        times, time_series = make_time_series(mydata,task,subtask)
    fig = plt.figure(fignum)
    figurelist.append(fignum)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.plot(times, time_series, '-', ms = 4)
    # plt.semilogx(times[1:], time_series[1:], '-', ms = 4)
    plt.xlabel(r'$\mathbf{t}$', fontsize = 16)
    plt.ylabel(r''+yax_label)
    plt.xlim([start,end])
    plt.title(r''+title)

    # if task == 'EC-Center':
    #     plt.plot(times, model(np.array(times), *fitparams), 'b')
    plt.tight_layout()
    plt.savefig(plots, format = 'pdf')
    plt.cla()
    plt.close()
    return

def plot_time_series_gradient(times, time_series, gradient_list, xlim, xlabel, ylabel, filename):
    for i in range(len(gradient_list)):
        plt.plot(np.array(times), time_series[i], '-', c = plt.cm.jet(gradient_list[i]/max(gradient_list)))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.jet,norm=plt.Normalize(vmin=0,vmax=1))
    sm._A =[]
    cbar = plt.colorbar(sm)
    cbar.set_label(r'$\mathbf{a}$', fontsize = 16)
    plt.tight_layout()
    plt.savefig(filename + '.pdf', format = 'pdf')
    plt.cla()
    plt.close()

def plot_time_series_ic_normalized(times, time_series, ic_list, L_list, xlim, ylim, xlabel, ylabel, filename):
    ic_list2 = [tuple(i) for i in ic_list] 
    ic_list2 = set(ic_list2)
    ic_list2 =  [list(i) for i in ic_list2]
    L_list2 = list(set(L_list))

    for IC in ic_list2:
        if IC[0][0][0] == 'E':
            if IC[0][0] == 'E9_10_3':
                pos = [i for i,x in enumerate(ic_list) if x in [[('E'+str((L//2)-1)+'_'+str((L//2))+'_3', 1.0)] for L in L_list2]]
            elif IC[0][0] == 'E9_10_4':
                pos = [i for i,x in enumerate(ic_list) if x in [[('E'+str((L//2)-1)+'_'+str((L//2))+'_4', 1.0)] for L in L_list2]]
            else:
                continue
        else:
            pos = [i for i,x in enumerate(ic_list) if x == IC]
        for i in pos:
            plt.plot(times/L_list[i], time_series[i], '-', label = r'$L = ' + str(L_list[i])+'$')
            plt.xlim(xlim)
            plt.ylim(ylim)

        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(loc = 4, ncol = 2)
        plt.tight_layout()
        plt.savefig(filename + IC_name(IC)[:-5] + '.pdf', format = 'pdf')
        plt.cla()
        plt.close()

    return

def plot_time_series_ic(times, time_series, ic_list, L_list, xlim, ylim, xlabel, ylabel, filename):
    ic_list2 = [tuple(i) for i in ic_list] 
    ic_list2 = set(ic_list2)
    ic_list2 =  [list(i) for i in ic_list2]
    L_list2 = list(set(L_list))

    for IC in ic_list2:
        if IC[0][0][0] == 'E':
            if IC[0][0] == 'E9_10_3':
                pos = [i for i,x in enumerate(ic_list) if x in [[('E'+str((L//2)-1)+'_'+str((L//2))+'_3', 1.0)] for L in L_list2]]
            elif IC[0][0] == 'E9_10_4':
                pos = [i for i,x in enumerate(ic_list) if x in [[('E'+str((L//2)-1)+'_'+str((L//2))+'_4', 1.0)] for L in L_list2]]
            else:
                continue
        else:
            pos = [i for i,x in enumerate(ic_list) if x == IC]
        for i in pos:
            plt.plot(np.array(times), time_series[i], '-', label = r'$L = ' + str(L_list[i])+'$')
            plt.xlim(xlim)
            plt.ylim(ylim)

        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(loc = 4, ncol = 2)
        plt.tight_layout()
        plt.savefig(filename + IC_name(IC)[:-5] + '.pdf', format = 'pdf')
        plt.cla()
        plt.close()

    return


def plot_phase_diagram(mydata,task,subtask_x,subtask_y,fignum=1,ax=111,yax_label='subtask_y',xax_label='subtask_x',title='title'):
    time_series_x = make_time_series(mydata,task,subtask_x)[1]
    time_series_y = make_time_series(mydata,task,subtask_y)[1]
    fig = plt.figure(fignum)
    fig.add_subplot(ax)
    plt.plot(time_series_x,time_series_y)
    plt.xlabel(xax_label)
    plt.ylabel(yax_label)
    plt.title(title)
    plt.tick_params(axis='both',labelsize=9)
    plt.locator_params(nbins=5)
    plt.tight_layout()
    return

def plot_fft(mydata,task,subtask,fignum=1,ax=111,yax_label='Intensity',title='FFT',start=0,end=5):
    freqs, amps = make_fft(mydata,task,subtask)
    amp_ave = np.mean(amps)
    fig = plt.figure(fignum)
    fig.add_subplot(ax)
    if amp_ave>1e-14:
        plt.semilogy(freqs,amps,'.')
    else:
        plt.plot(freqs,amps,'.')
    plt.xlabel('Frequency [1/dt]')
    plt.xlim([start, end])
    plt.ylabel(yax_label)
    plt.ylim(amp_ave/3., 10.*amps.max())
    plt.title(title)
    plt.fill_between(freqs,0,amps)
    plt.tight_layout()
    return

def plot_specgram(mydata,task,subtask,fignum=1,ax=111,yax_label='Frequency',title='title',NFFT=420):
    times, time_series = make_time_series(mydata,task,subtask)
    time_series = time_series - np.mean(time_series)
    fig = plt.figure(fignum)
    fig.add_subplot(ax)
    
    Pxx, freqs, bin = mlab.specgram(np.array(time_series), \
            window=mlab.window_none, \
            Fs=int(1/(mydata['dt'])), \
            NFFT=NFFT, noverlap=NFFT-1, pad_to=600)
    
    plt.pcolormesh(bin,freqs,Pxx,rasterized=True, cmap=plt.cm.jet)
    cbar=plt.colorbar()
    cbar.set_label('Intensity')
    plt.ylim([0.0,5])
    plt.ylabel('Frequency')
    plt.xlabel('Time')
    plt.title(title)
    plt.tight_layout()
    return

def plot_histograms(mydata, plots, bins = 20):
    #http://stackoverflow.com/questions/20531176/what-is-the-difference-between-np-histogram-and-plt-hist-why-dont-these-co
    deltas = deltaS(mydata)[1]
    fignum = len(figurelist)+1
    fig = plt.figure(fignum)
    figurelist.append(fignum)
    plt.hist(deltas, bins)
    plt.rc('text', usetex = True)
    plt.rc('font', family = 'serif')
    plt.xlabel(r'\textbf{Average} $\mathbf{\Delta S}$', fontsize = 16)
    plt.ylabel(r'\textbf{Counts}')
    plt.savefig(plots, format = 'pdf')
    plt.cla()
    plt.close()    
    return
    
def plot_2dhistograms(mydata, plots, bins = 20):
    vonneumannentropy = np.mean(np.array(mydata['SvN']), axis = 1)
    time, clusteringcoefficient = make_time_series(mydata,'MI','CC')
    time, networkdensity = make_time_series(mydata,'MI','ND')
    time, disparity = make_time_series(mydata,'MI','Y')
    networkmeasures = [clusteringcoefficient, networkdensity, disparity]
    networkmeasurenames = ['\\textbf{Clustering Coefficient}', '\\textbf{Network Density}', '\\textbf{Disparity}']

    for i in range(len(networkmeasures)):
        for j in range(len(networkmeasures)):
            if i != j:
                plt.hist2d(networkmeasures[i], networkmeasures[j], bins = bins)
                plt.rc('text', usetex = True)
                plt.rc('font', family = 'serif')
                plt.xlabel(r''+networkmeasurenames[i])
                plt.ylabel(r''+networkmeasurenames[j])
                plt.savefig(plots, format = 'pdf')
                plt.cla()
                plt.close()    
 
    plt.hist2d(vonneumannentropy, clusteringcoefficient, bins = bins)
    plt.rc('text', usetex = True)
    plt.rc('font', family = 'serif')
    plt.xlabel(r'\textbf{Average von Neumann Entropy} $\mathbf{S}$')
    plt.ylabel(r'\textbf{Clustering Coefficient} $\mathbf{C}$')
    plt.savefig(plots, format = 'pdf')
    plt.cla()
    plt.close()    

    plt.hist2d(vonneumannentropy, networkdensity, bins = bins)
    plt.rc('text', usetex = True)
    plt.rc('font', family = 'serif')
    plt.xlabel(r'\textbf{Average von Neumann Entropy} $\mathbf{S}$')
    plt.ylabel(r'\textbf{Network Density} $\mathbf{D}$')
    plt.savefig(plots, format = 'pdf')
    plt.cla()
    plt.close()

    plt.hist2d(vonneumannentropy, disparity, bins = bins)
    plt.rc('text', usetex = True)
    plt.rc('font', family = 'serif')
    plt.xlabel(r'\textbf{Average von Neumann Entropy} S')
    plt.ylabel(r'\textbf{Disparity} $\mathbf{Y}$')
    plt.savefig(plots, format = 'pdf')
    plt.cla()
    plt.close()    

def plot_2dhistogram(x, y, xf, yf, xname, yname, xlim, ylim, bins, plots):
    xbins = np.logspace(xlim[0],xlim[1], bins)
    ybins = np.logspace(ylim[0],ylim[1], bins)
    counts, _, _ = np.histogram2d(x,y, bins = (xbins, ybins))
    plt.pcolormesh(xbins, ybins, counts.transpose())
    # plt.hist2d(x, y, bins = [xbins, ybins])
    plt.plot(np.linspace(np.min(xbins), np.max(xbins), 100), yf(10)*np.ones(100), 'w--', zorder = 1)
    plt.plot(np.linspace(np.min(xbins), np.max(xbins), 100), yf(20)*np.ones(100), 'w-.', zorder = 1)
    plt.plot(xf(10)*np.ones(100), np.linspace(np.min(ybins), np.max(ybins), 100), 'w--', zorder = 1)
    plt.plot(xf(20)*np.ones(100), np.linspace(np.min(ybins), np.max(ybins), 100), 'w-.', zorder = 1)
    plt.xscale("log")
    plt.yscale("log")
    cbar = plt.colorbar()
    plt.rc('text', usetex = True)
    plt.rc('font', family = 'serif')
    cbar.set_label('Counts', fontsize = 16)
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.savefig(plots, format = 'pdf')
    plt.cla()
    plt.close()    

def plot_2dhistograms_data_set(vonneumannentropy, clusteringcoefficient,
                               networkdensity, disparity, measurenames, functions, measure_lim, bins = 20, filename = 'histograms.pdf'):
    measures = [vonneumannentropy, clusteringcoefficient, networkdensity, disparity]
    plots = PdfPages(filename)
    for i in range(len(measures)):
        for j in range(len(measures)):
            if i != j:
                plot_2dhistogram(measures[i], measures[j], functions[i], functions[j], measurenames[i], measurenames[j], measure_lim[i], measure_lim[j], bins, plots)
    plots.close()

def time_average(mydata,task,subtask,i):
    time_series = make_time_series(mydata,task,subtask)[1]
    avg = np.mean(time_series[i:])
    var = np.var(time_series[i:])
    return [avg, var]

def time_averagetau(deltas, tau):
    deltas = np.abs(deltas)[(len(deltas)%tau):]
    deltaStau = deltas.reshape(((len(deltas)-len(deltas)%tau)//tau, tau))
    averages = np.mean(deltaStau, axis = 1)
    standarddeviations = np.std(deltaStau, axis = 1)
    return averages, standarddeviations

def fluctuations(times, time_series, n):
    times = times[n:]
    standarddeviations = []
    for i in range(n, len(time_series)):
        standarddeviations.append(np.std(time_series[i-n:i]))
    return times, np.array(standarddeviations)

def deltaS(mydata, r = 1):
    time = [mydata['t'][i] for i in range(mydata['Nsteps']+1)]
    time_series = mydata['EC-Center']
    deltas = [] 
    for i in range(r,len(time_series)):
        deltas.append(time_series[i]-time_series[i-r])
    return np.array(time[r:]), np.array(deltas)

def plot3d(X, Y, Z, plots):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(X, Y)
    surf = ax.plot_surface(X, Y, Z)
    plt.savefig(plots, format = 'pdf')
    plt.cla()
    plt.close()#, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)

def time_averagetau_plots(mydata, fignum, plots):
    time, deltas = deltaS(mydata)
    for tau in [5, 10, 20]:
        time_ave = time_averagetau(time, tau)
        time_series = time_averagetau(deltas, tau)    
        fig = plt.figure(fignum)
        figurelist.append(fignum)
        plt.errorbar(time_ave[0], time_series[0], yerr = time_series[1], fmt = 'bo')
        plt.rc('text', usetex = True)
        plt.rc('font', family = 'serif')
        plt.xlabel(r'$\bar{\mathbf{t}}$', fontsize = 16)
        plt.ylabel(r'$\bar{\mathbf{\Delta S}}$', fontsize = 16)
        plt.tight_layout()
        plt.savefig(plots, format = 'pdf')
        plt.cla()
        plt.close()

def time_averagetau_plots_hamiltonian_type(times, time_series_list, ylim, hamiltonian_types, colorbar_label, colorbar_tick_labels, plots):
    c_dict = dict(zip(colorbar_tick_labels, range(len(colorbar_tick_labels))))
    cmap = cmap_discretize(plt.cm.jet, len(colorbar_tick_labels))
    for i in range(len(time_series_list)):
        plt.plot(times[0], time_series_list[i][0], c = cmap(c_dict[hamiltonian_types[i]]/len(colorbar_tick_labels)))
        plt.xscale("log")
        plt.yscale("log")
        plt.xlim([times[0][1], times[0][-1]])
        plt.ylim(ylim)

    plt.rc('text', usetex = True)
    plt.rc('font', family = 'serif')
    colorbar_index(colorbar_label, len(colorbar_tick_labels), cmap = plt.cm.jet, cb_tick_labels = colorbar_tick_labels)
    plt.xlabel(r'\textbf{Average Time}', fontsize = 16)
    plt.ylabel(r'\textbf{Average} $\mathbf{|\Delta S|}$', fontsize = 16)
    plt.tight_layout()
    plt.savefig(plots, format = 'pdf')
    plt.cla()
    plt.close()

def time_averagetau_plots_ic(times, time_series_list, plots):
    for time_series in time_series_list:
         plt.errorbar(times[0], time_series[0])#, yerr = time_series[1])#, basex = 10, basey = 10, yerr = time_series[1])#, fmt = 'bo')
         plt.xscale("log")
         plt.yscale("log")
         plt.xlim([times[0][0], times[0][-1]])
    plt.rc('text', usetex = True)
    plt.rc('font', family = 'serif')
    plt.xlabel(r'$\mathbf{t}$', fontsize = 16)
    plt.ylabel(r'$\mathbf{|\Delta S|}$', fontsize = 16)
    plt.tight_layout()
    plt.savefig(plots, format = 'pdf')
    plt.cla()
    plt.close()

def bond_entropy_fluctuation_equilibrium_plot(deltaS_equilibrium_list, hamiltonian_type_list, random_state_deltas_equilibrium_ave, random_state_deltas_equilibrium_std, filename):
    hamiltonian_type_list2 = list(set(hamiltonian_type_list))
    hamiltonian_type_list2 = sorted(hamiltonian_type_list2)
    x_dict = dict(zip(hamiltonian_type_list2, range(len(hamiltonian_type_list2))))
    plt.scatter([x_dict[elem] for elem in hamiltonian_type_list], deltaS_equilibrium_list)
    plt.xticks(range(len(hamiltonian_type_list2)),hamiltonian_type_list2)
    plt.yscale('log')
    plt.hlines(random_state_deltas_equilibrium_ave+random_state_deltas_equilibrium_std, np.min([x_dict[elem] for elem in hamiltonian_type_list]),np.max([x_dict[elem] for elem in hamiltonian_type_list]), linestyles = 'dashed')
    plt.hlines(random_state_deltas_equilibrium_ave-random_state_deltas_equilibrium_std, np.min([x_dict[elem] for elem in hamiltonian_type_list]),np.max([x_dict[elem] for elem in hamiltonian_type_list]), linestyles = 'dashed')
    # plt.axhline(random_state_deltas_equilibrium_ave+random_state_deltas_equilibrium_std, linestyle = '--', color = 'k')
    # plt.axhline(random_state_deltas_equilibrium_ave-random_state_deltas_equilibrium_std, linestyle = '--', color = 'k')
    plt.rc('text', usetex = True)
    plt.rc('font', family = 'serif')
    plt.xlabel(r'$\mathbf{R}$')
    plt.ylabel(r'$\mathbf{\bar{\Delta S}}$')
    plt.savefig(filename, format = 'pdf')
    plt.close()
    plt.cla()

def row(x, y, i):
    # return [1., x[i], x[i]**2, -x[i]*y[i]]#,-(x[i]**2)*y[i]]
    return [1., x[i], x[i]**2, x[i]**3, -x[i]*y[i], (-x[i]**2)*y[i],(-x[i]**3)*y[i]]

def initialize_params(time, time_series):
    matrix = []
    y = []
    for i in range(0, len(time_series), len(time_series)//100):
        matrix.append(row(time, time_series, i))
        y.append(time_series[i])
        # y.append(time_series[i]*(1+time[i]**2))
    matrix = np.array(matrix)
    parameters = np.linalg.lstsq(matrix, y)[0]
    return parameters

def fit_tanh(mydata):
    time = np.array([mydata['t'][i] for i in range(mydata['Nsteps']+1)])
    time_series = np.array(mydata['EC-Center'])
    try:
        # params = initialize_params(time, time_series)
        # popt, pcov = curve_fit(model, time, time_series, p0 = params, bounds = (0,10))   
        popt, pcov = curve_fit(model, time, time_series, bounds = ([0,0,0,0], [np.inf, np.inf, 1, np.inf]))   
        # popt, pcov = curve_fit(model, time, time_series, p0 = params)   
    except RuntimeError:
        popt, pcov = curve_fit(model, time, time_series, bounds = (0,10))   
    # popt, pcov = curve_fit(model, time, time_series)   
    return popt, pcov

def plot_board(mydata, task, subtask, plots, 
        fignum = 1, ax = 111, nticks = 8, 
               yax_label = 'Site Number ', title = '', label = '\\textbf{Bond Entropy} $\mathbf{S_{[1:i]}}$'):
    L = mydata['L']
    tmax = mydata['t_span'][1]
    tmin = mydata['t_span'][0]
    dt = mydata['dt']
    Nsteps = mydata['Nsteps']
    
    fig = plt.figure(fignum)
    figurelist.append(fignum)

    if task is 'EC':
        L = L-1
        ytick_lbls = range(L-1,-1,-1)
        ytick_locs = range(L)
        cmap = plt.cm.jet
        norm = None 
        board = np.array(mydata[task]).transpose()
        vmin = np.min(board)
        vmax = np.max(board)
    elif task is 'SvN':
        board = np.array(mydata['SvN']).transpose()
        cmap = plt.cm.jet
        norm = None 
        vmin = 0.0
        vmax = 1.0
    elif subtask in ['nexp', 'EV', 'EC', 'xexp']:
        cmap = plt.cm.jet
        norm = None 
        board = make_board(mydata,task,subtask)
        vmin = 0.0
        vmax = 1.0
        if subtask is 'xexp':
            vmin = -1.0
            vmax = 1.0
    elif subtask is 'yexp':
        cmap = plt.cm.jet
        norm = None 
        board = make_board(mydata,task,subtask)
        vmin = -1.0
        vmax = 1.0
    elif subtask == 'DIS':
        cmap = mpl.colors.ListedColormap([plt.cm.jet(0.), plt.cm.jet(1.)])
        bounds = [0,0.5,1]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        board = np.array(make_board(mydata,task,subtask)).transpose()
        vmin = 0.
        vmax = 1.

    plt.imshow(np.array(board).transpose(), 
               origin = 'lower',
               cmap = cmap,
               norm = norm,
               interpolation = 'none',
               aspect = 'auto',
               rasterized = True,
               extent = [1, L, tmin, tmax],
               vmin = vmin,
               vmax = vmax)

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.title(r''+title, fontsize = 16)
    plt.xlabel(r'\textbf{Site}', fontsize = 16)
    plt.ylabel(r'$\mathbf{t}$', fontsize = 16)
    cbar = plt.colorbar()
    cbar.set_label(r''+label, fontsize = 16)
    plt.tight_layout()
    plt.savefig(plots, format = 'pdf')
    plt.cla()
    plt.close()
    return

def theta_plots(output_dir,L,dt,t_span,IC_list,thlist):
   plot_tave_sweep(output_dir,L,dt,t_span,IC_list,'n','DEN',thlist,fignum=22,ax=111,yax_label=r'$\rho_{\infty}$',title=r'Equilibrium $\rho$')
   plot_tave_sweep(output_dir,L,dt,t_span,IC_list,'n','DIV',thlist,fignum=23,ax=111,yax_label=r'$\Delta_{\infty}$',title=r'Equilibrium $\Delta$')
   
   plot_tave_sweep(output_dir,L,dt,t_span,IC_list,'MI','CC',thlist,fignum=24,ax=111,yax_label=r'CC$_{\infty}$',title=r'$\mathcal{I}_{ij}$ Equilibrium CC')
   plot_tave_sweep(output_dir,L,dt,t_span,IC_list,'MI','ND',thlist,fignum=25,ax=111,yax_label=r'ND$_{\infth}$',title=r'$\mathcal{I}_{ij}$ Equilibrium ND')
   plot_tave_sweep(output_dir,L,dt,t_span,IC_list,'MI','Y',thlist,fignum=26,ax=111,yax_label=r'Y$_{\infty}$',title=r'$\mathcal{I}_{ij}$ Equilibrium Y')
   plot_tave_sweep(output_dir,L,dt,t_span,IC_list,'MI','HL',thlist,fignum=27,ax=111,yax_label=r'IHL$_{\infty}$',title=r'$\mathcal{I}_{ij}$ Equilibrium IHL')
   
   plot_tave_sweep(output_dir,L,dt,t_span,IC_list,'nn','CC',thlist,fignum=28,ax=111,yax_label=r'CC$_{\infty}$',title=r'g$_{ij}$ Equilibrium CC')
   plot_tave_sweep(output_dir,L,dt,t_span,IC_list,'nn','ND',thlist,fignum=29,ax=111,yax_label=r'ND$_{\infty}$',title=r'g$_{ij}$ Equilibrium ND')
   plot_tave_sweep(output_dir,L,dt,t_span,IC_list,'nn','Y',thlist,fignum=30,ax=111,yax_label=r'Y$_{\infty}$',title=r'g$_{ij}$ Equilibrium Y')
   plot_tave_sweep(output_dir,L,dt,t_span,IC_list,'nn','HL',thlist,fignum=31,ax=111,yax_label=r'IHL$_{\infty}$',title=r'g$_{ij}$ Equilibrium IHL')
   plt.tight_layout() 

def multipage(fname, figs=None, clf=True, dpi=300): 
    pp = PdfPages(fname) 
    if figs is None:
        figs = [plt.figure(fignum) for fignum in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp,format='pdf')
        
        if clf==True:
            fig.clf()
    pp.close()
    return

def make_tave_sweep(output_dir, L, dt, t_span, IC_list, task, subtask):
    avelist = []
    varlist = []
    for IC in IC_mydata:
        list = import_data(output_dir, L, dt, t_span, IC)
        ave,var = time_average(mydata,task,subtask)
        avelist.append(ave)
        varlist.append(var)
    return avelist, varlist

def plot_tave_sweep(output_dir, L, dt, t_span, IC_list, task, subtask, thlist, \
        fignum = 1, ax = 111, yax_label = 'Equilibrium val.', title = 'title'):
    
    avelist,varlist = make_tave_sweep(output_dir, L, dt, t_span, \
            IC_list, task, subtask) 
    fig = plt.figure(fignum)
    fig.add_subplot(ax)
    fmt = '-o'
    if subtask == 'DIV':
        varlist = [0]*len(thlist)
    plt.errorbar(thlist, avelist, yerr=varlist, fmt=fmt)
    plt.xlabel('Mixing Angle [rad]')
    plt.ylabel(yax_label)
    plt.title(title)
    plt.tight_layout()
    return

def scaling_plot(x, y, c, xname, yname, cname, plots):
    plt.scatter(x, y, c = c, s = 35)
    cbar = plt.colorbar()
    cbar.set_label(cname, fontsize = 16)     
    plt.rc('text', usetex = True)
    plt.rc('font', family = 'serif')
    plt.xlabel(xname, fontsize = 16)
    plt.ylabel(yname, fontsize = 16)
    plt.savefig(plots, format = 'pdf')
    plt.cla()
    plt.close()

def scaling_plots_hamiltonian_types(L_list, equilibrium_bond_entropy, ic_list, hamiltonian_types, xlabel, ylabel, ylim, colorbar_label, colorbar_tick_labels, filename = 'scaling-plots.pdf'):
    plots = PdfPages(filename)

    L_list2 = list(set(L_list))
    S_infinity_aves = []
    S_infinity_deviations = []
    ic_list2 = [tuple(i) for i in ic_list] 
    ic_list2 = set(ic_list2)
    ic_list2 =  [list(i) for i in ic_list2]
    hamiltonian_types2 = list(set(hamiltonian_types))
    hamiltonian_types2.sort()
    cmap = cmap_discretize(plt.cm.jet, len(colorbar_tick_labels))
    c_dict = dict(zip(hamiltonian_types2, range(len(hamiltonian_types2))))
    scaling_max = max(range(len(hamiltonian_types2)))

    for hamiltonian_type in hamiltonian_types2:
        pos = [i for i,x in enumerate(hamiltonian_types) if x == hamiltonian_type]
        ic_list3 = [ic_list[i] for i,x in enumerate(hamiltonian_types) if x == hamiltonian_type]
        for IC in ic_list2:
            if IC[0][0][0] == 'E':
                if IC[0][0] == 'E_11_12_3':
                    pos2 = [i for i,x in enumerate(ic_list3) if x in ['E_'+str((L//2)+1)+'_'+str((L//2)+2)+'_3' for L in L_list2]]
                elif IC[0][0] == 'E_11_12_4':
                    pos2 = [i for i,x in enumerate(ic_list3) if x in ['E_'+str((L//2)+1)+'_'+str((L//2)+2)+'_4' for L in L_list2]]
                else:
                    continue
            else:
                pos2 = [i for i,x in enumerate(ic_list3) if x == IC]
            plt.plot(np.array(L_list)[pos][pos2], np.array(equilibrium_bond_entropy)[pos][pos2], '-o', c = cmap(c_dict[hamiltonian_type]/scaling_max))

    plt.xlabel(xlabel, fontsize = 16)
    plt.ylabel(ylabel, fontsize = 16)
    plt.ylim(ylim)
    plt.rc('text', usetex = True)
    plt.rc('font', family = 'serif')
    colorbar_index(colorbar_label, len(colorbar_tick_labels), cmap = plt.cm.jet, cb_tick_labels = colorbar_tick_labels)
    plt.tight_layout()
    plt.savefig(plots, format = 'pdf')
    plt.cla()
    plt.close()
    plots.close()

def scaling_plots(L_list, fitparamlist, equilibrium_bond_entropy, fit_parameter_names, scaling_variables, scaling_variable_names, ic_list, filename = 'scaling-plots.pdf'):
    plots = PdfPages(filename)

    # for i in range(len(fitparamlist)):
    #     for j in range(len(scaling_variables)):
    #         scaling_plot(L_list, fitparamlist[i], scaling_variables[j], 
    #                      r'\textbf{System Size} $\mathbf{L}$', r'\textbf{Fit Parameter} '+ fit_parameter_names[i], 
    #                      scaling_variable_names[j], plots)

    # for j in range(len(scaling_variables)):
    #     # scaling_plot(L_list, np.array(fitparamlist[3])/np.array(fitparamlist[6]), scaling_variables[j], 
    #     #              r'\textbf{System Size} $\mathbf{L}$', r'\textbf{Equilibrium Bond Entropy} $\mathbf{S_{\infty}}$', 
    #     #              scaling_variable_names[j], plots)
    #     scaling_plot(L_list, np.array(fitparamlist[2]), scaling_variables[j], 
    #                  r'\textbf{System Size} $\mathbf{L}$', r'\textbf{Equilibrium Bond Entropy} $\mathbf{S_{\infty}}$', 
    #                  scaling_variable_names[j], plots)
    #     # plot3d(L_list, scaling_variables[j], np.array(fitparamlist[3])/np.array(fitparamlist[6]), plots)
    #     # plot3d(L_list, scaling_variables[j], np.array(fitparamlist[0]), plots)

    L_list2 = list(set(L_list))
    S_infinity_aves = []
    S_infinity_deviations = []

    ic_list2 = [tuple(i) for i in ic_list] 
    ic_list2 = set(ic_list2)
    ic_list2 =  [list(i) for i in ic_list2]

    # for L in L_list2:
    #     pos = [i for i,x in enumerate(L_list) if x == L]
    #     S_infinity_list = np.array(fitparamlist[2])[pos]/(L/2) #((np.array(fitparamlist[2])/np.array(fitparamlist[4]))/(L/2))[pos]
    #     S_infinity_aves.append(np.mean(S_infinity_list))
    #     S_infinity_deviations.append(np.std(S_infinity_list))

    for j in range(len(scaling_variables)):
        scaling_max = max(scaling_variables[j])
        for IC in ic_list2: 
            if IC[0][0][0] == 'E':
                if IC[0][0] == 'E_11_12_3':
                    pos = [i for i,x in enumerate(ic_list) if x in ['E_'+str((L//2)+1)+'_'+str((L//2)+2)+'_3' for L in L_list2]]
                if IC[0][0] == 'E_11_12_4':
                    pos = [i for i,x in enumerate(ic_list) if x in ['E_'+str((L//2)+1)+'_'+str((L//2)+2)+'_4' for L in L_list2]]
            else:
                pos = [i for i,x in enumerate(ic_list) if x == IC]
            plt.plot(np.array(L_list)[pos], np.array(equilibrium_bond_entropy)[pos]/(0.5*np.array(L_list)[pos]), '-', c = plt.cm.jet(np.array(scaling_variables[j])[pos][0]/scaling_max))
        plt.rc('text', usetex = True)
        plt.rc('font', family = 'serif')
        plt.xlabel(r'\textbf{System Size} $\mathbf{L}$', fontsize = 16)
        plt.ylabel(r'\textbf{Equilibrium Bond Entropy} $\mathbf{S_{\infty}/S_{\textrm{max}}}$', fontsize = 16)
        sm = plt.cm.ScalarMappable(cmap=plt.cm.jet,norm=plt.Normalize(vmin=0,vmax=1))
        sm._A =[]
        cbar = plt.colorbar(sm)
        cbar.set_label(scaling_variable_names[j], fontsize = 16)
        plt.tight_layout()
        plt.savefig(plots, format = 'pdf')
        plt.cla()
        plt.close()        

    plt.errorbar(L_list2, S_infinity_aves, yerr = S_infinity_deviations, fmt = 'bo')
    plt.rc('text', usetex = True)
    plt.rc('font', family = 'serif')
    plt.xlabel(r'\textbf{System Size} $\mathbf{L}$', fontsize = 16)
    plt.ylabel(r'\textbf{Equilibrium Bond Entropy} $\mathbf{S_{\infty}/S_{\textrm{max}}}$', fontsize = 16)
    plt.tight_layout()
    plt.savefig(plots, format = 'pdf')
    plt.cla()
    plt.close()        
    plots.close()

def scaling_plots_ic(ic_list, fitparamlist, L_list, plots_dir):
    ic_list2 = [tuple(i) for i in ic_list] 
    ic_list2 = set(ic_list2)
    ic_list2 =  [list(i) for i in ic_list2]
    L_list2 = list(set(L_list))
    for IC in ic_list2:
        if IC[0][0][0] == 'E':
            if IC[0][0] == 'E_11_12_3':
                pos = [i for i,x in enumerate(ic_list) if x in ['E_'+str((L//2)+1)+'_'+str((L//2)+2)+'_3' for L in L_list2]]
                S_infinity_list = np.array(fitparamlist[2])[pos]
                L_list2 = np.array(L_list)[pos]
                S_infinity_list = S_infinity_list/(L_list2/2)
                plt.plot(L_list2, S_infinity_list, 'bo')
                plt.rc('text', usetex = True)
                plt.rc('font', family = 'serif')
                plt.xlabel(r'\textbf{System Size} $\mathbf{L}$', fontsize = 16)
                plt.ylabel(r'\textbf{Equilibrium Bond Entropy} $\mathbf{S_{\infty}/S_{\textrm{max}}}$', fontsize = 16)
                plt.title(r'\textbf{Initial Condition} ' + '$\mathbf{E_{L/2,L/2 + 1}^3}$', fontsize = 16)
                plt.tight_layout()
                plt.savefig(plots_dir+'/'+sim_name_ic(IC)+'.pdf', format = 'pdf')
                plt.cla()
                plt.close()        
            if IC[0][0] == 'E_11_12_4':
                pos = [i for i,x in enumerate(ic_list) if x in ['E_'+str((L//2)+1)+'_'+str((L//2)+2)+'_4' for L in L_list2]]
                S_infinity_list = np.array(fitparamlist[2])[pos]
                L_list2 = np.array(L_list)[pos]
                S_infinity_list = S_infinity_list/(L_list2/2)
                plt.plot(L_list2, S_infinity_list, 'bo')
                plt.rc('text', usetex = True)
                plt.rc('font', family = 'serif')
                plt.xlabel(r'\textbf{System Size} $\mathbf{L}$', fontsize = 16)
                plt.ylabel(r'\textbf{Equilibrium Bond Entropy} $\mathbf{S_{\infty}/S_{\textrm{max}}}$', fontsize = 16)
                plt.title(r'\textbf{Initial Condition} ' + '$\mathbf{E_{L/2,L/2 + 1}^4}$', fontsize = 16)
                plt.tight_layout()
                plt.savefig(plots_dir+'/'+sim_name_ic(IC)+'.pdf', format = 'pdf')
                plt.cla()
                plt.close()        
        else:
            pos = [i for i,x in enumerate(ic_list) if x == IC]
            # S_infinity_list = np.array(fitparamlist[3])[pos]/np.array(fitparamlist[6])[pos]
            S_infinity_list = np.array(fitparamlist[2])[pos]
            L_list2 = np.array(L_list)[pos]
            S_infinity_list = S_infinity_list/(L_list2/2)
            plt.plot(L_list2, S_infinity_list, 'bo')
            plt.rc('text', usetex = True)
            plt.rc('font', family = 'serif')
            plt.xlabel(r'\textbf{System Size} $\mathbf{L}$', fontsize = 16)
            plt.ylabel(r'\textbf{Equilibrium Bond Entropy} $\mathbf{S_{\infty}/S_{\textrm{max}}}$', fontsize = 16)
            plt.title(r'\textbf{Initial Condition} ' + IC[0][0], fontsize = 16)
            plt.tight_layout()
            plt.savefig(plots_dir+'/'+sim_name_ic(IC)+'.pdf', format = 'pdf')
            plt.cla()
            plt.close()        

def temporal_average_plots(time_averages, time_average_names, temporal_average_variables, temporal_average_variable_names, temporal_average_tick_labels, functions, xlim, filename = 'temporal-averages.pdf'):
    temporalaverages = PdfPages(filename)
    for i in range(len(time_averages[0])):
        for j in range(len(time_averages[0])):
            if i < j:
                for k in range(len(temporal_average_variables)):
                    cmap = cmap_discretize(plt.cm.jet, len(temporal_average_tick_labels[k]))
                    c_dict = dict(zip(temporal_average_tick_labels[k], range(len(temporal_average_tick_labels[k]))))
                    plt.scatter(time_averages[:,i,0], time_averages[:,j,0], c = [cmap(c_dict[elem]/len(temporal_average_tick_labels[k])) for elem in temporal_average_variables[k]]
                                 , s = 35)
                    plt.xlim([0, np.max(time_averages[:,i,0])])
                    plt.ylim([0, np.max(time_averages[:,j,0])])

                    plt.plot(np.linspace(0, np.max(time_averages[:,i,0]), 100), functions[j](10)*np.ones(100), 'k--')
                    plt.plot(np.linspace(0, np.max(time_averages[:,i,0]), 100), functions[j](20)*np.ones(100), 'k-.')

                    plt.plot(functions[i](10)*np.ones(100), np.linspace(0, np.max(time_averages[:,j,0]), 100), 'k--')
                    plt.plot(functions[i](20)*np.ones(100), np.linspace(0, np.max(time_averages[:,j,0]), 100), 'k-.')

                    # cbar = plt.colorbar()
                    # cbar.set_label(temporal_average_variable_names[k], fontsize = 16)     
                    colorbar_index(temporal_average_variable_names[k], len(temporal_average_tick_labels[k]), cmap = cm.jet, cb_tick_labels = temporal_average_tick_labels[k])
                    plt.rc('text', usetex = True)
                    plt.rc('font', family = 'serif')
                    plt.xlabel(r''+time_average_names[i], fontsize = 16)
                    plt.ylabel(r''+time_average_names[j], fontsize = 16)
                    plt.savefig(temporalaverages, format = 'pdf')
                    plt.cla()
                    plt.close()


    for i in range(len(time_averages[0])):
        for j in range(len(time_averages[0])):
            if i < j:
                for k in range(len(temporal_average_variables)):
                    c_dict = dict(zip(temporal_average_tick_labels[k], range(len(temporal_average_tick_labels[k]))))
                    plt.scatter(time_averages[:,i,0], time_averages[:,j,0], c = [c_dict[elem] for elem in  temporal_average_variables[k]]
                                 , s = 35)

                    plt.plot(np.linspace(xlim[i][0], xlim[i][1], 100), functions[j](10)*np.ones(100), 'k--')
                    plt.plot(np.linspace(xlim[i][0], xlim[i][1], 100), functions[j](20)*np.ones(100), 'k-.')

                    plt.plot(functions[i](10)*np.ones(100), np.linspace(xlim[j][0], xlim[j][1], 100), 'k--')
                    plt.plot(functions[i](20)*np.ones(100), np.linspace(xlim[j][0], xlim[j][1], 100), 'k-.')

                    plt.xlim([xlim[i][0], xlim[i][1]])
                    plt.ylim([xlim[j][0], xlim[j][1]])

                    plt.xscale('log')
                    plt.yscale('log')
                    # cbar = plt.colorbar()
                    # cbar.set_label(temporal_average_variable_names[k], fontsize = 16)     
                    colorbar_index(temporal_average_variable_names[k], len(temporal_average_tick_labels[k]), cmap = cm.jet, cb_tick_labels = temporal_average_tick_labels[k])
                    plt.rc('text', usetex = True)
                    plt.rc('font', family = 'serif')
                    plt.xlabel(r''+time_average_names[i], fontsize = 16)
                    plt.ylabel(r''+time_average_names[j], fontsize = 16)
                    plt.savefig(temporalaverages, format = 'pdf')
                    plt.cla()
                    plt.close()


    # plt.tight_layout()
    temporalaverages.close()    

def time_plots(mydata, filename, fitparams, i, n, rewrite_fourier_data, make_plots):
    fourier_peaks = []
    if make_plots:
        plots = PdfPages(filename)
        plot_board(mydata, 'n', 'nexp', plots, ax=111, fignum=len(figurelist)+1, label = r'$\mathbf{\langle \hat{n}_i \rangle}$')
        plot_board(mydata, 'localObs', 'xexp', plots, ax=111, fignum=len(figurelist)+1, label = r'$\mathbf{\langle \hat{\sigma}^x_i \rangle}$')
        plot_board(mydata, 'localObs', 'yexp', plots, ax=111, fignum=len(figurelist)+1, label = r'$\mathbf{\langle \hat{\sigma}^y_i \rangle}$')
        plot_board(mydata, 'SvN', 'SvN', plots, ax=111, fignum=len(figurelist)+1, label = r'$\mathbf{S_i}$')
        # plot_board(mydata, 'EC', 'EC', plots, ax=111, fignum=len(figurelist)+1)
        plot_time_series(mydata, 'MI', 'CC', plots, fignum=len(figurelist)+1,ax=111, yax_label = '$\mathbf{C}$', title = '$\mathcal{I}$')
        plot_time_series(mydata, 'MI', 'ND', plots, fignum=len(figurelist)+1,ax=111, yax_label = '$\mathbf{D}$', title = '$\mathcal{I}$')
        plot_time_series(mydata, 'MI', 'Y', plots,fignum=len(figurelist)+1,ax=111, yax_label = '$\mathbf{Y}$', title = '$\mathcal{I}$')
        plot_time_series(mydata, 'EC-Center', 'EC-Center', plots, fignum=len(figurelist)+1,ax=111, yax_label= '$\mathbf{S_{[1:L/2]}}$', title ='',
                         fitparams = fitparams)
        time_averagetau_plots(mydata, len(figurelist)+1, plots)
        plot_histograms(mydata, plots, bins = 20)


        # plot_ft_2d(mydata, 'localObs', 'xexp', r'$\mathbf{F(\langle \sigma_i^x \rangle)}$', plots)
        plot_ft_2d(mydata, 'n', 'nexp', r'$\mathbf{F(\langle \hat{n}_i \rangle)}$', plots)
        plot_ft_2d(mydata, 'localObs', 'yexp', r'$\mathbf{F(\langle \hat{\sigma}_i^y \rangle)}$', plots)
        plot_ft_2d(mydata, 'SvN', 'SvN', r'$\mathbf{F(S_i)}$', plots)
    else:
        plots = None

    n = n
    fpks = plot_ft_1d(mydata, 'MI', 'CC', n, r'$\mathbf{f}$', '$|\mathbf{F(C)}|$', '', plots, rewrite_data = rewrite_fourier_data, make_plots = make_plots)
    fourier_peaks.append(fpks)
    fpks = plot_ft_1d(mydata, 'MI', 'ND', n, r'$\mathbf{f}$', '$|\mathbf{F(D)}|$', '', plots, rewrite_data = rewrite_fourier_data, make_plots = make_plots)
    fourier_peaks.append(fpks)
    fpks = plot_ft_1d(mydata, 'MI', 'Y', n, r'$\mathbf{f}$', '$|\mathbf{F(Y)}|$', '', plots, rewrite_data = rewrite_fourier_data, make_plots = make_plots)
    fourier_peaks.append(fpks)
    fpks = plot_ft_1d(mydata, 'n', 'nexp', n, r'$\mathbf{f}$', '$|\mathbf{F(\hat{n}_{L/2-1})}|$', '', plots, i = i, rewrite_data = rewrite_fourier_data, make_plots = make_plots)
    fourier_peaks.append(fpks)

    if make_plots:
        plots.close()
    # plot_board(mydata, 'n', 'DIS', ax=212, fignum=len(figurelist)+1, title='Discretized QGL')
    return fourier_peaks

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / N 

def autocorr(x, h=1):
    N = len(x)
    mu = np.mean(x)
    acorr = sum( (x[j] - mu) * (x[j+h] - mu) for j in range(N-h))
    denom = sum( (x[j] - mu)**2 for j in range(N) )
    if denom > 1e-14:
        acorr = acorr/denom
    else:
        print('auto correlation less than', 1e-14)
    return acorr

def red_noise(time_series, dt=1, h=1):
    a1 = autocorr(time_series, h=1)
    a2 = np.abs(autocorr(time_series, h=2))
    a = 0.5 * (a1 + np.sqrt(a2))

    def RN(f):
        rn = 1 - a**2
        rn = rn / (1 - 2*a*np.cos(2*pi*f/dt) + a**2)
        return rn
    return RN

def rfft_amps(x):
    if not len(x) % 2:
        amps = list(x[0:1]**2) + list(x[1:-1:2]**2 + x[2:-1:2]**2) + list(x[-1:]**2)
    else:
        amps = list(x[0:1]**2) + list(x[1::2]**2 + x[2::2]**2)
    return np.array(amps)

def rfft_freq(x):
    if not len(x) % 2:
        freqs  = list(x[0:1]) + list(x[1:-1:2]) + list(x[-1:])
    else:
        freqs  = list(x[0:1]) + list(x[1::2])
    return np.array(freqs)
        

def make_ft_1d(time_series, dt=1, h=1):
    # set nan's to 0
    time_series = np.nan_to_num(time_series)
    time_series = time_series - np.mean(time_series)
    # amps = np.abs(spf.rfft(time_series))**2
    fft = spf.rfft(time_series)
    amps = rfft_amps(fft)
    amps = amps/np.sum(amps)
    ws = rfft_freq(spf.rfftfreq(len(fft), d=dt))
    rn = red_noise(time_series, dt=dt, h=h)(ws)
    rn = rn * np.sum(amps)/np.sum(rn)
    return ws, amps, rn

def get_ft_peaks(freqs, FT, rn, win = 5, interval_factor=5.991):
    # smoothing spectrum with win-point moving average
    FT = running_mean(FT, win)
    freqs_trunc = freqs[0:len(FT)]
    # iterpolate to make a red noise function
    RNf = interpolate.interp1d(freqs, rn)
    # extract peaks
    mx, mn = pd.peakdetect(FT, freqs_trunc, lookahead=4,
            delta=np.std(FT))
    frq = np.array([m[0] for m in mx])
    amp = np.array([m[1] for m in mx])
    # keep peaks above 95% conf level of red noise
    pk_ind = RNf(frq) * interval_factor < amp
    fpks = frq[pk_ind]
    pks = amp[pk_ind]
    return fpks, pks, FT, freqs_trunc

def fourier_analysis(mydata, task, subtask, n, i=None):
    dt = mydata['dt']
    if i==None:
        if task == 'EC-Center':
            time_series = mydata['EC-Center'][n:]            
        else:
            time_series = make_time_series(mydata,task,subtask)[1][n:]
    else:
        time_series = make_time_series(mydata,task,subtask)[1][:, i][n:]
        
    ws, amps, rn = make_ft_1d(time_series, dt=dt, h=1)
    fpks, pks, amps, freqs_trunc = get_ft_peaks(ws, amps, rn)
    return fpks, pks, freqs_trunc, amps, ws, rn

def plot_ft_1d(mydata, task, subtask, n, xname, yname, title, plots, rewrite_data = False, make_plots = False, i=None):
    interval_factor = 5.991
    # fpks = params['fpks'][obs_key]
    # pks = params['pks'][obs_key]
    if (task+subtask +'_fourier'+str(n) in mydata) and (not rewrite_data):
        fpks = np.array(mydata[task + subtask +'_fourier'+str(n)]['fpks'])
        pks = np.array(mydata[task + subtask +'_fourier'+str(n)]['pks'])
        freqs_trunc = np.array(mydata[task + subtask +'_fourier'+str(n)]['freqs_trunc'])
        amps = np.array(mydata[task + subtask +'_fourier'+str(n)]['amps'])
        ws = np.array(mydata[task + subtask +'_fourier'+str(n)]['ws'])
        rn = np.array(mydata[task +subtask +'_fourier'+str(n)]['rn'])
    else:
        fpks, pks, freqs_trunc, amps, ws, rn = fourier_analysis(mydata, task, subtask, n, i = i)
        amps = np.nan_to_num(amps)
        mydata[task + subtask +'_fourier'+str(n)] = {}
        mydata[task + subtask +'_fourier'+str(n)]['fpks'] = list(fpks)
        mydata[task + subtask +'_fourier'+str(n)]['pks'] = list(pks)
        mydata[task + subtask +'_fourier'+str(n)]['freqs_trunc'] = list(freqs_trunc)
        mydata[task + subtask +'_fourier'+str(n)]['amps'] = list(amps)
        mydata[task + subtask +'_fourier'+str(n)]['ws'] = list(ws)
        mydata[task + subtask +'_fourier'+str(n)]['rn'] = list(rn)

        
    if make_plots:
        if len(amps) == 0:
            return
        if np.max(amps) == 0:
            plt.plot(freqs_trunc, amps, '-k', lw=0.9, zorder=1)
            plt.plot(ws, rn, 'b', lw=0.9, zorder=2)
            plt.plot(ws, rn*interval_factor, '--b', lw=0.9, zorder=2)
            plt.scatter(fpks, pks, c='r', marker='*', linewidth=0,
                        s=25, zorder=2)
        else:
            plt.semilogy(freqs_trunc, amps, '-k', lw=0.9, zorder=1)
            df = freqs_trunc[1]-freqs_trunc[0]
            plt.xlim([np.min(freqs_trunc), np.max(freqs_trunc)])
            # if len(pks) == 0:
            #     plt.xlim([np.min(freqs_trunc), np.max(freqs_trunc)])
            # elif len(pks) == 1:
            #     pos1 = [n for n,x in enumerate(freqs_trunc) if x == np.min(fpks)][0]
            #     pos2 = [n for n,x in enumerate(freqs_trunc) if np.abs(x-4*pks[0]) <= 0.5*df][0]
            #     plt.xlim([0.8*fpks[0], 4*pks[0]])
            #     plt.ylim([np.min(amps[pos1:pos2]), np.max(amps[pos1:pos2])])
            # else:
            #     plt.xlim([0.99*np.min(fpks), 1.01*np.max(fpks)])
            #     pos1 = [n for n,x in enumerate(freqs_trunc) if x == np.min(fpks)][0]
            #     pos2 = [n for n,x in enumerate(freqs_trunc) if x == np.max(fpks)][0]
            #     plt.ylim([np.min(amps[pos1:pos2]), np.max(amps[pos1:pos2])])
            plt.scatter(fpks, pks, c='r', marker='*', linewidth=0,
                    s=25, zorder=2)

        plt.xlabel(xname)
        plt.ylabel(yname)
        plt.title(title)
        plt.savefig(plots, format = 'pdf')
        plt.close()
        plt.cla()
    return fpks

def fourier_plot(freqs, amps, label):
    plt.plot(freqs, amps, label = label)
    plt.xlim([np.min(freqs), np.max(freqs)])
    plt.xlabel(r'$\mathbf{f}$')

def fourier_plot_semilogy(freqs, amps, label):
    plt.semilogy(freqs, amps, label = label)
    plt.xlim([np.min(freqs), np.max(freqs)])
    plt.xlabel(r'$\mathbf{f}$')

    
def plot_fourier_peaks(fourier_peak_list, L_list, L_list2, measure_list, xlabel,filename):
    empty_list = [[] for elem in measure_list]
    L_list2 = sorted(L_list2)
    x_dict = dict(zip(L_list2, range(len(L_list2))))
    x = []
    for alist in L_list:
        x.append([x_dict[L] for L in alist])
    if fourier_peak_list == empty_list:
        return
    else:
        fig = plt.figure(1)
        ax = fig.add_subplot(111)
        plt.rc('text', usetex = True)
        plt.rc('font', family = 'serif')
        ax.set_xlim([0.99*min(min(x)), 1.01*max(max(x))+0.1*len(measure_list)])
        for i in range(len(measure_list)):
            ax.plot(np.array(x[i]) + 0.1*i, fourier_peak_list[i], label = measure_list[i], marker = 'o',linestyle='None')

        handles, labels = ax.get_legend_handles_labels()
        lgd = ax.legend(handles, labels, loc=2, bbox_to_anchor=(1.01, 1))
        ax.set_xlabel(xlabel, fontsize = 16)
        ax.set_xticks(range(len(L_list2)))
        ax.set_xticklabels([str(L) for L in L_list2])
        ax.set_ylabel(r'\textbf{Peak Frequency} $\mathbf{f}$', fontsize = 16)
        fig.savefig(filename, format = 'pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.close()
        plt.cla()

def plot_fourier_peaksv2(fourier_peak_list, L_list, fourier_peak_count, L_list2, xlabel, measure_list, titles, filename):
    empty_list = [[] for elem in titles]
    plots = PdfPages(filename + '.pdf')
    plots2 = PdfPages(filename + '-counts.pdf')
    L_list2 = sorted(L_list2)
    x = range(len(L_list2))

    if fourier_peak_list == empty_list:
        return
    else:
        for i in range(len(titles)):
            fig = plt.figure(1)
            ax = fig.add_subplot(111)
            ax.set_xlim([min(x), max(x)])
            plt.rc('text', usetex = True)
            plt.rc('font', family = 'serif')
            ax.plot(np.array(L_list[i]), fourier_peak_list[i], marker = 'o', linestyle='None')
            ax.set_xlabel(xlabel, fontsize = 16)
            ax.set_ylabel(r'$\mathbf{f}$', fontsize = 16)
            ax.set_title(titles[i], fontsize = 16)
            fig.savefig(plots, format = 'pdf')
            # fig.savefig(filename + measure_file_strings[i] + '.pdf', format = 'pdf')
            plt.close()
            plt.cla()
        fig = plt.figure(1)
        ax = fig.add_subplot(111)
        ax.set_xlim([min(x), max(x)])
        plt.rc('text', usetex = True)
        plt.rc('font', family = 'serif')

        for i in range(len(measure_list)):
            ax.plot(np.array(x), fourier_peak_count[i], marker = 'o', linestyle='-', label = measure_list[i])
        ax.set_xlabel(xlabel, fontsize = 16)
        ax.set_xticks(range(len(L_list2)))
        ax.set_xticklabels([str(L) for L in L_list2])
        ax.set_ylabel(r'\textbf{Peak Count}', fontsize = 16)
        for elem in x[1:-1]:
            ax.axvline(elem, color = 'k', ls = '--')
        handles, labels = ax.get_legend_handles_labels()
        lgd = ax.legend(handles, labels, loc=2, bbox_to_anchor=(1.01, 1))
            # ax.set_title(titles[i], fontsize = 16)
        fig.savefig(plots2, format = 'pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')
            # fig.savefig(filename + measure_file_strings[i]+ '-counts' + '.pdf', format = 'pdf')
        plt.close()
        plt.cla()
    plots.close()
    plots2.close()

    # handles, labels = ax.get_legend_handles_labels()
    # lgd = ax.legend(handles, labels, loc=2, bbox_to_anchor=(1.01, 1))
    # ax.semilogy(freqs_trunc, amps, '-k', lw=0.9, zorder=1)
    # ax.semilogy(freqs, rn, 'b', lw=0.9, zorder=2)
    # ax.semilogy(freqs, rn*interval_factor, '--b', lw=0.9, zorder=2)
    # ax.scatter(fpks, pks, c='r', marker='*', linewidth=0,
    #         s=25, zorder=2)

    #ax.scatter(minfrq, minamp, c='g', marker='v', linewidth=0,
    #        s=25, zorder=2)

    # ax.yaxis.set_major_locator(mpl.ticker.LogLocator(numticks=4))
    # ax.set_ylim([np.min(amps[10::])/5, np.max(amps)*5])
    # ax.set_xlim([0,0.5])
    # ax.minorticks_off()
    # ax.get_xaxis().tick_bottom()   # remove unneeded ticks 
    # ax.get_yaxis().tick_left()

    #ax.xaxis.set_ticks_position('none')
    #ax.yaxis.set_ticks_position('none')
    #ax.grid('on')

    # fticks = False
    # if row == nrows-1:
    #     fticks = True
    #     ax.set_xlabel('Frequency')

    # plt.setp([ax.get_xticklabels()], visible=fticks)
    # ax.set_ylabel('$\mathcal{F}\left ('+obs_label(obs_key)+r'\right )$')
    # title = pt.make_U_name(params['mode'], params['S'], params['V'])
    # fig.suptitle(title)
    # plt.subplots_adjust(hspace=0.3, top=0.93)
def plot_bond_entropy_counts(bond_entropy_counts, hamiltonian_types, xlabel, filename):
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    plt.rc('text', usetex = True)
    plt.rc('font', family = 'serif')
    plots = PdfPages(filename)
    x = range(len(hamiltonian_types))
    ax.plot(np.array(x), bond_entropy_counts, marker = 'o', linestyle='-')
    ax.set_xlabel(xlabel, fontsize = 16)
    ax.set_xticks(range(len(hamiltonian_types)))
    ax.set_xticklabels([str(hamiltonian_type) for hamiltonian_type in hamiltonian_types])
    ax.set_ylabel(r'\textbf{Simulation Count}', fontsize = 16)
    for elem in x[1:-1]:
        ax.axvline(elem, color = 'k', ls = '--')
    fig.savefig(plots, format = 'pdf')
    plt.close()
    plt.cla()
    plots.close()



def make_ft_2d(board, dt=1, dx=1):
    # set nan's to 0
    T = len(board)
    end = int((T-1)/2)
    if T%2 == 0:
        end = int(T/2)


    L = len(board[0])
    endk = int((L-1)/2)
    if T%2 == 0:
        endk = int(L/2)
    board = np.nan_to_num(board)
    ny, nx = board.shape
    # NOTE: The slice at the end of amps is required to cut negative freqs from
    # axis=0 of #the rfft. This appears to be a but. Report it.
    fraw = np.fft.fft2(board)
    amps = (np.abs(fraw)**2).real
    iboard = np.fft.ifft2(fraw).real

    amps = amps[0:end+1,0:endk+1]
    amps = amps/np.sum(amps)
    ws = np.fft.rfftfreq(ny, d=dt)[0:end+1]
    ks = np.fft.rfftfreq(nx, d=dx)[0:endk+1]
    return ws, ks, amps, iboard, board

def plot_ft_2d(mydata, task, subtask, measurename, plots, rewrite_data = False):

    if (task+subtask +'_fourier2d' in mydata) and (not rewrite_data):
        ws = np.array(mydata[task + subtask +'_fourier2d']['ws'])
        ks = np.array(mydata[task + subtask +'_fourier2d']['ks'])
        amps = np.array(mydata[task + subtask +'_fourier2d']['amps'])
    else:
        if task is 'SvN':
            board = np.array(mydata['SvN']).transpose()
        else:
            board = make_board(mydata,task,subtask)
        dt = mydata['dt']
        ws, ks, amps, iboard, board = make_ft_2d(board, dt=dt, dx=1)
        amps = np.nan_to_num(amps)
        mydata[task + subtask +'_fourier2d'] = {}
        mydata[task + subtask +'_fourier2d']['ws'] = list(ws)
        mydata[task + subtask +'_fourier2d']['ks'] = list(ks)
        mydata[task + subtask +'_fourier2d']['amps'] = amps.tolist()

    kmin = min(ks)
    kmax = max(ks)
    wmin = min(ws)
    wmax = max(ws)

    if np.max(amps) == 0:
        plt.imshow(amps.transpose(), interpolation="none", origin='lower', aspect='auto', extent = [kmin, kmax, wmin, wmax])
    else:
        plt.imshow(amps.transpose(), interpolation="none", origin='lower', aspect='auto', norm=LogNorm(), extent = [kmin, kmax, wmin, wmax])
    plt.rc('text', usetex = True)
    plt.rc('font', family = 'serif')
    plt.xlabel(r'$\mathbf{k}$', fontsize = 16)
    plt.ylabel(r'$\mathbf{f}$', fontsize = 16 )
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=16)
    cbar.ax.locator_params(nbins=5)
    cbar.set_label(measurename, fontsize = 16)
    plt.savefig(plots, format = 'pdf')
    plt.close()
    plt.cla()

def principal_component_analysis(data):
    l = len(data[0])
    data -= data.mean(axis = 0)
    correlation_matrix = np.zeros((l,l))
    for i in range(l):
        for j in range(l):
            correlation_matrix[i, j] = stats.pearsonr(data[:, i], data[:, j])[0]
    eigs,eigvecs=np.linalg.eigh(correlation_matrix)
    return eigvecs

def init_files_single(params):
    if 'fname' in params:
        fname = params['fname']
    else:
        fname = make_file_name(params, iterate = False)
        params['fname'] = fname

def rule(x):
    if x==0:
        return 1
    if x==1:
        return 2
    if x==2:
        return 4
    if x==3:
        return 8
    if x==4:
        return 16
    else:
        return 0

def ruleNN(x):
    if x==0:
        return 1
    if x==1:
        return 2
    if x==2:
        return 4
    else:
        return 0

def rule_number(hamiltonian_type):
    if hamiltonian_type[0:2] == 'N_':
        rule_number = 0
        for elem in hamiltonian_type[2:]:
            rule_number += rule(int(elem))
    if hamiltonian_type[0:2] == 'NN':
        rule_number = 0
        for elem in hamiltonian_type[3:]:
            rule_number += ruleNN(int(elem))
    return rule_number

def main(output_dir, tasks, dt_list, t_span_list, IC_list, hamiltonian_types, V_list, gradients = [[None, None, None]], model_dir =
        environ['HOME']+'/Documents/qgl_exact/'):
    data_dir = model_dir+output_dir
    plots_dir = model_dir+output_dir+'/plots115_R1'
    makedirs(plots_dir, exist_ok=True)
    print('qgl_plotting.main')
    rewrite_fourier_data = False
    make_time_series_plots =  False
    make_histograms = True
    make_temporal_average_plots = True
    make_bond_entropy_fluctuation_plots = True
    make_bond_entropy_fluctuation_equilibrium_plots = True
    make_principal_components = True
    make_bond_entropy_time_series_plots = True
    make_bond_entropy_scaling_plots = True
    make_fourier_plots = True
    make_temporal_average_entropy_fluctuation_plots = True


    make_long_time_blinker_plot = False
    make_gradient_plot = False
    make_gradient_equilibrium_plots = False
    make_blinker_case_study_mps_plots = False
    
    fitparamlist = []
    fitparamsigmalist = []
    fit_parameter_list_len = 4
    fit_parameter_names = [r'$\mathbf{a}$', r'$\mathbf{b}$', r'$\mathbf{c}$', r'$\mathbf{d}$']
    filling_factor_list = []

    r_list = [1, 2, 3, 4, 5]

    for i in range(fit_parameter_list_len):
        fitparamlist.append([])
        fitparamsigmalist.append([])

    bondentropy = []
    
    bondentropy_histogram = []
    vonneumannentropy = [] 
    clusteringcoefficient = [] 
    networkdensity =  []
    disparity = [] 
    
    time_averages1 = []
    time_averages2 = []
    time_averages3 = []
    time_average_names1 = [ r'$\mathbf{C}$', 
                           r'$\mathbf{D}$', 
                           r'$\mathbf{Y}$',
                           r'$\mathbf{\bar{S}}$']
    time_average_names2 = [ r'$\mathbf{C}$', 
                           r'$\mathbf{D}$', 
                           r'$\mathbf{Y}$',
                           r'$L/2 - \mathbf{S_{[1:L/2]}}$']
    time_average_names3 = [ r'$\mathbf{C}$', 
                           r'$\mathbf{D}$', 
                           r'$\mathbf{Y}$',
                            r'$L/2 - \mathbf{S_{[1:L/2]}}$',
                            r'$\mathbf{\Delta S_1}$']

    fourier_measures = [r'$\mathbf{C}$', 
                        r'$\mathbf{D}$', 
                        r'$\mathbf{Y}$',
                        r'$\mathbf{\hat{n}_{L/2-1}}$']

    fourier_measure_names = [r'$\mathbf{C}$', 
                        r'$\mathbf{D}$', 
                        r'$\mathbf{Y}$',
                        r'$\mathbf{\hat{n}_{L/2-1}}$']

    fourier_measure_file_strings = ['clustering', 'density', 'disparity', 'number']

    Nfourier = 300
    Nequilibrium = 900
    L_list = []
    ic_list = []
    averagetime_list1 = []
    averagetime_list2 = []
    deltaS_list1 = []
    deltaS_list2 = []
    deltaS_equilibrium_list1 = []
    deltaS_equilibrium_list2 = []
    deltaS_equilibrium_random_list1 = []
    deltaS_equilibrium_random_list2 = []
    for r in r_list:
        deltaS_list1.append([])
        deltaS_list2.append([])
        deltaS_equilibrium_list1.append([])
        deltaS_equilibrium_list2.append([])
        deltaS_equilibrium_random_list1.append([])
        deltaS_equilibrium_random_list2.append([])
    fourier_peak_list = []
    equilibrium_bond_entropy = []
    equilibrium_delta_bond_entropy = []
    bond_entropy_standard_deviations = []
    gradient_list = []
    nexp_list = []
    hamiltonian_type_list = []
    hamiltonian_type_list_fluctuations = []
    hamiltonian_type_index = dict(zip(hamiltonian_types, [rule_number(hamiltonian_type) for hamiltonian_type in hamiltonian_types] ))
    initial_condition_index_list = []
    random_initial_condition_list = []
    random_initial_condition_list1 = []
    random_initial_condition_list2 = []

    for dt in dt_list:
        for t_span in t_span_list:
            for initial_condition in IC_list[::]:
                initial_condition_index_list.append(tuple(initial_condition[1:]))
                for gradient in gradients:
                    for hamiltonian_type in hamiltonian_types:
                        for V in V_list:
                            L = initial_condition[0]
                            L_list.append(L)
                            IC = initial_condition[1:]
                            ic_list.append(IC)
                            a = gradient[2]
                            gradient_list.append(a)
                            if hamiltonian_type == None and dt == 0.1:
                                hamiltonian_type_list.append(hamiltonian_type_index[hamiltonian_type])
                                a = None
                            else:
                                hamiltonian_type_list.append(hamiltonian_type_index[hamiltonian_type])
                            mydata = import_data(output_dir, L, dt, t_span, IC, a = a, hamiltonian_type = hamiltonian_type, V = V) 

                            if make_histograms or principal_component_analysis:
                                bondentropy_histogram += np.array(mydata['EC-Center'])[Nequilibrium:].tolist()
                                vonneumannentropy += np.mean(np.array(mydata['SvN']), axis = 1)[Nequilibrium:].tolist()
                                clusteringcoefficient += make_time_series(mydata,'MI','CC')[1][Nequilibrium:].tolist()
                                networkdensity += make_time_series(mydata,'MI','ND')[1][Nequilibrium:].tolist()
                                disparity += make_time_series(mydata,'MI','Y')[1][Nequilibrium:].tolist()
                                filling_factor = np.sum(make_time_series(mydata,'n','nexp')[1][0])/L
                                nexp = make_time_series(mydata,'n','nexp')[1][:, L//2 - 1]
                                nexp_list.append(nexp)
                                filling_factor_list.append(filling_factor)

                            if make_temporal_average_plots:
                                time_averages1.append([time_average(mydata,'MI','CC', Nequilibrium),
                                                      time_average(mydata,'MI','ND', Nequilibrium),
                                                      time_average(mydata,'MI','Y', Nequilibrium),
                                                      [np.mean(np.mean(np.array(mydata['SvN']), axis = 1)[Nequilibrium:]),
                                                       np.var(np.mean(np.array(mydata['SvN']), axis = 1)[Nequilibrium:])],
                                                  ])

                                time_averages2.append([time_average(mydata,'MI','CC', Nequilibrium),
                                                      time_average(mydata,'MI','ND', Nequilibrium),
                                                      time_average(mydata,'MI','Y', Nequilibrium),
                                                      [L/2.-np.mean(np.array(mydata['EC-Center'])[Nequilibrium:]),
                                                       np.var(np.array(mydata['EC-Center'])[Nequilibrium:])]
                                                  ])
                                bond_entropy_standard_deviations.append(np.std((np.array(mydata['EC-Center'])/(L/2.))[Nequilibrium:]))

                            if make_temporal_average_entropy_fluctuation_plots:
                                time, deltas = deltaS(mydata, r_list[0])
                                deltasave = np.mean(np.abs(deltas[Nequilibrium:]))
                                deltasvar = np.var(np.abs(deltas[Nequilibrium:]))
                                time_averages3.append([time_average(mydata,'MI','CC', Nequilibrium),
                                                      time_average(mydata,'MI','ND', Nequilibrium),
                                                      time_average(mydata,'MI','Y', Nequilibrium),
                                                      [L/2. - np.mean(np.array(mydata['EC-Center'])[Nequilibrium:]),
                                                       np.var(np.array(mydata['EC-Center'])[Nequilibrium:])],
                                                      [deltasave,
                                                       deltasvar]
                                                  ])
                                if L == 10:
                                    if IC[0][0][0:2] == 'r3':
                                        deltaS_equilibrium_random_list1[0].append(np.mean(np.abs(deltas[Nequilibrium:])))
                                if L == 20:
                                    if IC[0][0][0:2] == 'r3':
                                        deltaS_equilibrium_random_list2[0].append(np.mean(np.abs(deltas[Nequilibrium:])))
                                

                            if make_bond_entropy_fluctuation_plots:
                                if L == 10:
        
                                    for i in range(len(r_list)):
                                        time, deltas = deltaS(mydata, r_list[i])
                                        time_ave = time_averagetau(time, 20)
                                        time_series = time_averagetau(deltas, 20)
                                        averagetime_list1.append(time_ave)
                                        deltaS_list1[i].append(time_series)
                                        if make_bond_entropy_fluctuation_equilibrium_plots:
                                            deltaS_equilibrium_list1[i].append(np.mean(np.abs(deltas[Nequilibrium:])))
                                            if IC[0][0][0:2] == 'r3':
                                                deltaS_equilibrium_random_list1[i].append(np.mean(np.abs(deltas[Nequilibrium:])))
                                    if not IC[0][0][0:2] == 'r3':
                                        random_initial_condition_list1.append(True)
                                    else:
                                        random_initial_condition_list1.append(False)                                        

                                if L == 20:
                                    for i in range(len(r_list)):
                                        time, deltas = deltaS(mydata, r_list[i])
                                        time_ave = time_averagetau(time, 20)
                                        time_series = time_averagetau(deltas, 20)
                                        averagetime_list2.append(time_ave)
                                        deltaS_list2[i].append(time_series)
                                        if make_bond_entropy_fluctuation_equilibrium_plots:
                                            deltaS_equilibrium_list2[i].append(np.mean(np.abs(deltas[Nequilibrium:])))
                                            if IC[0][0][0:2] == 'r3':
                                                deltaS_equilibrium_random_list2[i].append(np.mean(np.abs(deltas[Nequilibrium:])))
                                    if not IC[0][0][0:2] == 'r3':
                                        random_initial_condition_list2.append(True)
                                    else:
                                        random_initial_condition_list2.append(False)                                        


                            if not IC[0][0][0:2] == 'r3':
                                random_initial_condition_list.append(True)
                            else:
                                random_initial_condition_list.append(False)
                                


                            #Time series plots
                            # fitparams = fit_tanh(mydata)
                            fitparams = (1,1)
                            time_series_plots_filename = plots_dir + '/' + sim_name2(L, dt, t_span, IC, a = a, hamiltonian_type = hamiltonian_type,  V = V)+'.pdf'
                            fourier_peaks = time_plots(mydata, time_series_plots_filename , fitparams, L//2 - 1, Nfourier, rewrite_fourier_data, make_time_series_plots)
                            fourier_peak_list.append(fourier_peaks)
                            
                            if make_blinker_case_study_mps_plots:
                                fpks,pks,freqs,amps,ws,rn = fourier_analysis(mydata, 'EC-Center', 'EC-Center', 300, i=None)
                                fourier_plot_semilogy(freqs, amps, r'\textbf{Trotter Exact}')
                                freq = fpks[0]
                                t = 1./freq
                                n_fluctuations = int(t/dt)

                                time_series = np.loadtxt('/media/dvargas/mypassport/qgol/qgolrun174/bondentropy/bondentropyb0.dat', delimiter = '\t')[300:]
                                ws, amps, rn = make_ft_1d(time_series, dt=dt, h=1)
                                fpks, pks, amps, freqs_trunc = get_ft_peaks(ws, amps, rn)
                                fourier_plot_semilogy(freqs, amps, r'\textbf{MPS} $\mathbf{\chi = 128}$')
                                # plt.xlim([0,0.4])
                                plt.ylabel(r'$\mathbf{F(S_{[1:L/2]})}$')
                                plt.legend(ncol=1, loc = 1)
                                plt.savefig(plots_dir+'/blinker-bondentropy-fourier_semilogy.pdf', format = 'pdf')
                                plt.close()
                                plt.cla()

                                times = make_time_series(mydata,'MI','CC')[0]
                                times,bondentropy_fluctuations = fluctuations(times, mydata['EC-Center'], n_fluctuations)
                                plt.plot(times,bondentropy_fluctuations, label = r'\textbf{Trotter Exact}')
                                times = make_time_series(mydata,'MI','CC')[0][1:]
                                time_series = np.loadtxt('/media/dvargas/mypassport/qgol/qgolrun174/bondentropy/bondentropyb0.dat', delimiter = '\t')
                                times,bondentropy_fluctuations = fluctuations(times, time_series, n_fluctuations)
                                plt.plot(times,bondentropy_fluctuations, label = r'\textbf{MPS} $\mathbf{\chi = 128}$')
                                plt.xlabel(r'$\mathbf{t}$')
                                plt.xlim([np.min(times),np.max(times)])
                                plt.ylabel(r'$\mathbf{\sigma_{S_{[1:L/2]}}}$')
                                plt.legend(ncol=1, loc = 1)
                                plt.savefig(plots_dir+'/blinker-bondentropy-fluctuations.pdf', format = 'pdf')
                                plt.close()
                                plt.cla()
                            
                            if make_bond_entropy_time_series_plots or make_long_time_blinker_plot or make_gradient_plot:
                                bondentropy.append(np.array(mydata['EC-Center']))
                            if make_bond_entropy_scaling_plots or make_gradient_equilibrium_plots:
                                equilibrium_bond_entropy.append(np.mean(np.array(mydata['EC-Center'])[Nequilibrium:]))

                            delta_bond = []
                            for i in range(len(np.array(mydata['EC-Center']))-1):
                                delta_bond.append(np.abs(np.array(mydata['EC-Center'])[i+1] - np.array(mydata['EC-Center'])[i]))
                            equilibrium_delta_bond_entropy.append(np.mean(np.array(delta_bond)))

                            if make_gradient_equilibrium_plots:
                                time_averages1.append([time_average(mydata,'MI','CC', Nequilibrium),
                                                      time_average(mydata,'MI','ND', Nequilibrium),
                                                      time_average(mydata,'MI','Y', Nequilibrium),
                                                      [np.mean(np.mean(np.array(mydata['SvN']), axis = 1)[Nequilibrium:]),
                                                       np.var(np.mean(np.array(mydata['SvN']), axis = 1)[Nequilibrium:])],
                                                  ])                                

                            if rewrite_fourier_data:
                                meas = measures.Measurements(tasks, data_dir+'/'+sim_name(L, dt, t_span, IC, a = a, hamiltonian_type = hamiltonian_type,  V = V) + '.meas')
                                meas.measures = mydata
                                meas.write_out()

    time_averages1 = np.array(time_averages1)
    time_averages2 = np.array(time_averages2)
    time_averages3 = np.array(time_averages3)
    fourier_peak_list = np.array(fourier_peak_list)
    hamiltonian_type_list2 = list(set(hamiltonian_type_list)) 
    hamiltonian_type_list2 = sorted(hamiltonian_type_list2)
    times = make_time_series(mydata,'MI','CC')[0]
    L_list2 = list(set(L_list))
    L_list2.sort()
    random_initial_condition_list = np.array(random_initial_condition_list)
    random_initial_condition_list1 = np.array(random_initial_condition_list1)
    random_initial_condition_list2 = np.array(random_initial_condition_list2)

    if make_bond_entropy_fluctuation_equilibrium_plots:
        random_state_deltas_equilibrium_ave1 = []
        random_state_deltas_equilibrium_std1 = []
        random_state_deltas_equilibrium_ave2 = []
        random_state_deltas_equilibrium_std2 = []
        for i in range(len(r_list)):
            random_state_deltas_equilibrium_ave1.append(np.mean(deltaS_equilibrium_random_list1[i]))
            random_state_deltas_equilibrium_std1.append(np.std(deltaS_equilibrium_random_list1[i]))
            random_state_deltas_equilibrium_ave2.append(np.mean(deltaS_equilibrium_random_list2[i]))
            random_state_deltas_equilibrium_std2.append(np.std(deltaS_equilibrium_random_list2[i]))


    if make_temporal_average_entropy_fluctuation_plots:
        bond_entropy_fluctuation_function = function_creator(np.mean(deltaS_equilibrium_random_list1[0]), np.mean(deltaS_equilibrium_random_list2[0])) 

    if make_long_time_blinker_plot:
        plt.semilogx(times, bondentropy[0])
        plt.xlabel(r'$\mathbf{t}$', fontsize = 16)
        plt.ylabel(r'$\mathbf{S_{[1:L/2]}}$', fontsize = 16)
        plt.savefig(plots_dir+'/blinker-long-time-evolution.pdf', format = 'pdf')
        plt.close()
        plt.cla()

    #Gradient Plot
    if make_gradient_plot:
        plot_time_series_gradient(times, bondentropy, gradient_list, [times[0], times[-1]], r'$\mathbf{t}$', r'$\mathbf{S_{[1:L/2]}}$', plots_dir + '/' + 'bondentropy')

    #Gradient Equilibrium Plots
    if make_gradient_equilibrium_plots:
        #Plot equilibrium bond entropy as a function of gradient
        for L in L_list2:
            pos = [n for n,x in enumerate(L_list) if x == L]
            plt.plot(np.array(gradient_list)[pos], np.array(equilibrium_bond_entropy)[pos]/(0.5*np.array(L_list)[pos]), 'o', label = '$\mathbf{L = '+str(L)+'}$')
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.xlabel(r'$\mathbf{a}$', fontsize = 16)
        plt.ylabel(r'$\mathbf{S_{\infty}/S_{\textrm{max}}}$', fontsize = 16)
        plt.legend(ncol=3,loc=0)
        plt.savefig(plots_dir+'/bondentropy-gradient.pdf', format = 'pdf')
        plt.close()
        plt.cla()

        plt.plot(gradient_list, time_averages1[:,0,0], 'o')
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.xlabel(r'$\mathbf{a}$', fontsize = 16)
        plt.ylabel(r'$\mathbf{C}$', fontsize = 16)
        plt.savefig(plots_dir+'/clustering-gradient.pdf', format = 'pdf')
        plt.close()
        plt.cla()

        plt.plot(gradient_list, time_averages1[:,1,0], 'o')
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.xlabel(r'$\mathbf{a}$', fontsize = 16)
        plt.ylabel(r'$\mathbf{D}$', fontsize = 16)
        plt.savefig(plots_dir+'/networkdensity-gradient.pdf', format = 'pdf')
        plt.close()
        plt.cla()

        plt.plot(gradient_list, time_averages1[:,2,0], 'o')
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.xlabel(r'$\mathbf{a}$', fontsize = 16)
        plt.ylabel(r'$\mathbf{Y}$', fontsize = 16)
        plt.savefig(plots_dir+'/disparity-gradient.pdf', format = 'pdf')
        plt.close()
        plt.cla()

        #Plot Fourier peaks as a function of gradient
        # fourier_peak_list2 = []
        # fourier_peak_count = []
        # gradient_list3 = []
        # for measure in fourier_measures:
        #     fourier_peak_list2.append([])
        # for measure in fourier_measures:
        #     gradient_list3.append([])
        # for measure in fourier_measures:
        #     fourier_peak_count.append([])

        # gradient_list2 = list(set(gradient_list))

        # for L in gradient_list2:
        #     pos = [n for n,x in enumerate(gradient_list) if x == L]
        #     fourier_peak_list_L = fourier_peak_list[pos]

        #     for i in range(len(fourier_measures)):
        #         count = 0
        #         for j in range(len(fourier_peak_list_L)):
        #             fourier_peak_list2[i] += list(fourier_peak_list_L[j][i])
        #             count += len(fourier_peak_list_L[j][i])
        #             gradient_list3[i] += len(fourier_peak_list_L[j][i])*[L]
        #         fourier_peak_count[i] += [count]

        # plot_fourier_peaksv2(fourier_peak_list2, gradient_list3, fourier_peak_count, gradient_list2, r'\textbf{Slope of Potential} $\mathbf{a}$', fourier_measures, fourier_measure_names, plots_dir+'/fourier-peaks-gradient')        

    if make_fourier_plots:
        #Plot Fourier peaks as a function of L
        fourier_peak_list2 = []
        fourier_peak_count = []
        L_list3 = []

        for measure in fourier_measures:
            fourier_peak_list2.append([])
        for measure in fourier_measures:
            fourier_peak_count.append([])
        for measure in fourier_measures:
            L_list3.append([])

        for L in L_list2:
            pos = [n for n,x in enumerate(L_list) if x == L]
            fourier_peak_list_L = fourier_peak_list[pos]

            for i in range(len(fourier_measures)):
                count = 0
                for j in range(len(fourier_peak_list_L)):
                    fourier_peak_list2[i] += list(fourier_peak_list_L[j][i])
                    count += len(fourier_peak_list_L[j][i])
                    L_list3[i] += len(fourier_peak_list_L[j][i])*[L]
                fourier_peak_count[i] += [count]
        
        if len(hamiltonian_types) == 1:
            plot_fourier_peaks(fourier_peak_list2, L_list3, L_list2, fourier_measures, r'$\mathbf{L}$', plots_dir + '/fourier-peaks'+'n'+str(Nfourier)+ '_R' + str(hamiltonian_type_list[0]) +'.pdf')
            plot_fourier_peaksv2(fourier_peak_list2, L_list3, fourier_peak_count, L_list2,  r'$\mathbf{L}$', fourier_measures, fourier_measure_names, plots_dir+'/fourier-peaks-Lv2'+'n'+str(Nfourier) + '_R' + str(hamiltonian_type_list[0]))        
        else:
            plot_fourier_peaks(fourier_peak_list2, L_list3, L_list2, fourier_measures, r'$\mathbf{L}$', plots_dir + '/fourier-peaks'+'n'+str(Nfourier) +'.pdf')
            plot_fourier_peaksv2(fourier_peak_list2, L_list3, fourier_peak_count, L_list2,  r'$\mathbf{L}$', fourier_measures, fourier_measure_names, plots_dir+'/fourier-peaks-Lv2'+'n'+str(Nfourier))   

        #Plot Fourier peaks as a function of Rule
        fourier_peak_list2 = []
        fourier_peak_count = []
        rule_list = []

        for measure in fourier_measures:
            fourier_peak_list2.append([])
        for measure in fourier_measures:
            fourier_peak_count.append([])
        for measure in fourier_measures:
            rule_list.append([])

        for k in range(len(hamiltonian_type_list2)):
            pos = [n for n,x in enumerate(hamiltonian_type_list) if x == hamiltonian_type_list2[k]]
            fourier_peak_list_L = fourier_peak_list[pos]
            for i in range(len(fourier_measures)):
                count = 0
                for j in range(len(fourier_peak_list_L)):
                    fourier_peak_list2[i] += list(fourier_peak_list_L[j][i])
                    count += len(fourier_peak_list_L[j][i])
                    rule_list[i] += len(fourier_peak_list_L[j][i])*[hamiltonian_type_list2[k]]
                fourier_peak_count[i] += [count]

        plot_fourier_peaks(fourier_peak_list2, rule_list, hamiltonian_type_list2, fourier_measures, r'$\mathbf{R}$', plots_dir + '/fourier-peaks-rule'+'n'+str(Nfourier)+'.pdf')
        plot_fourier_peaksv2(fourier_peak_list2, rule_list, fourier_peak_count, hamiltonian_type_list2,  r'$\mathbf{R}$', fourier_measures, fourier_measure_names, plots_dir+'/fourier-peaks-rulev2'+'n'+str(Nfourier))        

    #Bond entropy scaling plots
    if make_bond_entropy_scaling_plots:
        scaling_plot_tick_labels = hamiltonian_type_list2
        scaling_plots_hamiltonian_types(L_list, equilibrium_bond_entropy, ic_list, hamiltonian_type_list, r'$\mathbf{L}$', r'$\mathbf{S_{\infty}}$', [0,10.], r'$\mathbf{R}$', scaling_plot_tick_labels, filename = plots_dir + '/scaling-plots-hamiltonian_types.pdf')

    #Bond entropy time series plots
    if make_bond_entropy_time_series_plots:
        for hamiltonian_type in hamiltonian_type_list2:
            pos = [n for n,x in enumerate(hamiltonian_type_list) if x == hamiltonian_type]
            ic_list_time_series = [ic_list[n] for n,x in enumerate(hamiltonian_type_list) if x == hamiltonian_type]

            plot_time_series_ic(times, np.array(bondentropy)[pos], ic_list_time_series, np.array(L_list)[pos], [times[0], times[-1]], [0., 10.], r'$\mathbf{t}$', r'$\mathbf{S_{[1:L/2]}}$', plots_dir + '/' + 'bondentropy'+'_R'+ str(hamiltonian_type)+'_')

            plot_time_series_ic_normalized(times, np.array(bondentropy)[pos]/((0.5*np.array(L_list)[pos]).reshape((len(pos), 1))), ic_list_time_series, np.array(L_list)[pos], [times[0], times[-1]/max(L_list)], [0., 1.], r'$\mathbf{t/L}$', r'$\mathbf{S_{[1:L/2]}/S_{\textrm{max}}}$', plots_dir + '/' + 'bondentropy'+'_normalized'+'_R'+ str(hamiltonian_type)+'_')

    #Histograms
    if make_histograms:
        measurenames = [r'$\mathbf{\bar{S}}$', r'$\mathbf{C}$', r'$\mathbf{D}$', r'$\mathbf{Y}$']
        functions = [vnentropy_function, clustering_function,density_function,disparity_function]
        measure_lim = [[-2., 0.], [-6., 0.], [-6., 0.], [-2., 0.]]
        plot_2dhistograms_data_set(vonneumannentropy, clusteringcoefficient,
                                   networkdensity, disparity, measurenames, functions, measure_lim, bins = 20, filename = plots_dir+'/histograms1.pdf')
        measurenames = [r'$\mathbf{S_{[1:L/2]}}$', r'$\mathbf{C}$', r'$\mathbf{D}$', r'$\mathbf{Y}$']
        measure_lim = [[-2., 1.], [-6., 0.], [-6., 0.], [-2, 0.]]
        functions = [bondentropy_function, clustering_function,density_function,disparity_function]
        plot_2dhistograms_data_set(bondentropy_histogram, clusteringcoefficient,
                                   networkdensity, disparity, measurenames, functions, measure_lim, bins = 20, filename = plots_dir+'/histograms2.pdf')

    #Temporal Averages
    if make_temporal_average_plots:
        print('Temporal Averages')
        temporal_average_variables = [L_list, hamiltonian_type_list]
        temporal_average_variable_names = [r'$\mathbf{L}$', r'$\mathbf{R}$']
        hamiltonian_type_tick_labels = hamiltonian_type_list2
        temporal_average_tick_labels = [L_list2, hamiltonian_type_tick_labels]

        functions = [clustering_function,density_function,disparity_function, vnentropy_function]
        measure_lim = [[10**(-6), 1.], [10**(-6), 1.], [10**(-2), 1.], [10**(-2), 1.]]
        temporal_average_plots(time_averages1, time_average_names1, temporal_average_variables, temporal_average_variable_names, temporal_average_tick_labels, functions, measure_lim, filename = plots_dir+'/temporal-averages1.pdf')

        functions = [clustering_function,density_function,disparity_function, bondentropy_function]
        measure_lim = [[10**(-6), 1.], [10**(-6), 1.], [10**(-2), 1.], [10**(-1), 10.]]
        temporal_average_plots(time_averages2, time_average_names2, temporal_average_variables, temporal_average_variable_names, temporal_average_tick_labels, functions, measure_lim, filename = plots_dir+'/temporal-averages2.pdf')

    if make_temporal_average_entropy_fluctuation_plots:
        temporal_average_variables = [L_list, hamiltonian_type_list]
        temporal_average_variable_names = [r'$\mathbf{L}$', r'$\mathbf{R}$']
        hamiltonian_type_tick_labels = hamiltonian_type_list2
        temporal_average_tick_labels = [L_list2, hamiltonian_type_tick_labels]
        functions = [clustering_function,density_function,disparity_function, bondentropy_function, bond_entropy_fluctuation_function]
        measure_lim = [[10**(-6), 1.], [10**(-6), 1.], [10**(-2), 1.], [10**(-1), 10.], [10**(-5), 1.]]
        temporal_average_plots(time_averages3, time_average_names3, temporal_average_variables, temporal_average_variable_names, temporal_average_tick_labels, functions, measure_lim, filename = plots_dir+'/temporal-averages3.pdf')

    #Fluctuations of Bond Entropy
    if make_bond_entropy_fluctuation_plots:
        fluctuation_plot_tick_labels = hamiltonian_type_list2
        pos1 = [n for n,x in enumerate(L_list) if x == 10]
        pos2 = [n for n,x in enumerate(L_list) if x == 20]
        for i in range(len(r_list)):
            if 10 in L_list:
                time_averagetau_plots_hamiltonian_type(averagetime_list1[i], deltaS_list1[i], [10**(-5), 10**0], np.array(hamiltonian_type_list)[pos1], r'$\mathbf{R}$', fluctuation_plot_tick_labels, plots_dir + '/' + 'bondentropy-fluctuationstau20L10r'+str(r_list[i])+'.pdf')
            if 20 in L_list:
                time_averagetau_plots_hamiltonian_type(averagetime_list2[i], deltaS_list2[i], [10**(-5), 10**0], np.array(hamiltonian_type_list)[pos2], r'$\mathbf{R}$', fluctuation_plot_tick_labels, plots_dir + '/' + 'bondentropy-fluctuationstau20L20r'+str(r_list[i])+'.pdf')
    #Equilibrium Fluctuations of Bond Entropy
    if make_bond_entropy_fluctuation_equilibrium_plots:
        hamiltonian_type_list_fluctuations = np.array(hamiltonian_type_list)[random_initial_condition_list]
        L_list_fluctuations = (np.array(L_list)[random_initial_condition_list]).tolist()
        pos1 = [n for n,x in enumerate(L_list_fluctuations) if x == 10]
        pos2 = [n for n,x in enumerate(L_list_fluctuations) if x == 20]
        hamiltonian_type_list_fluctuations1 = (np.array(hamiltonian_type_list_fluctuations)[pos1]).tolist()
        hamiltonian_type_list_fluctuations2 = (np.array(hamiltonian_type_list_fluctuations)[pos2]).tolist()

        # pos1 = [n for n,x in enumerate(L_list_fluctuations) if x == 10]
        # pos2 = [n for n,x in enumerate(L_list_fluctuations) if x == 20]
        for i in range(len(r_list)):
            if 10 in L_list:
                bond_entropy_fluctuation_equilibrium_plot(np.array(deltaS_equilibrium_list1[i])[random_initial_condition_list1], np.array(hamiltonian_type_list_fluctuations1), random_state_deltas_equilibrium_ave1[i], random_state_deltas_equilibrium_std1[i], plots_dir+'/equilibrium_bond_entropy_fluctuationsL10r' + str(r_list[i]) + '.pdf')
            if 20 in L_list:
                bond_entropy_fluctuation_equilibrium_plot(np.array(deltaS_equilibrium_list2[i])[random_initial_condition_list2], np.array(hamiltonian_type_list_fluctuations2), random_state_deltas_equilibrium_ave2[i], random_state_deltas_equilibrium_std2[i], plots_dir+'/equilibrium_bond_entropy_fluctuationsL20r' + str(r_list[i]) + '.pdf')

        for i in range(len(r_list)):
            deltaS_count1 = []
            deltaS_count2 = []
            if 10 in L_list:
                deltaS_equilibrium_list1[i] = np.array(deltaS_equilibrium_list1[i])[random_initial_condition_list1]
            if 20 in L_list:
                deltaS_equilibrium_list2[i] = np.array(deltaS_equilibrium_list2[i])[random_initial_condition_list2]

            for hamiltonian_type in hamiltonian_type_list2:
                if 10 in L_list:
                    pos1 = [n for n,x in enumerate(hamiltonian_type_list_fluctuations1) if x == hamiltonian_type]
                    deltaS1 = deltaS_equilibrium_list1[i][pos1]
                    deltaS_count1.append(len(deltaS1[deltaS1 > random_state_deltas_equilibrium_ave1[i]]))
                if 20 in L_list:
                    pos2 = [n for n,x in enumerate(hamiltonian_type_list_fluctuations2) if x == hamiltonian_type]
                    deltaS2 = deltaS_equilibrium_list2[i][pos2]
                    deltaS_count2.append(len(deltaS2[deltaS2 > random_state_deltas_equilibrium_ave2[i]]))


            if 10 in L_list:
                plot_bond_entropy_counts(deltaS_count1, hamiltonian_type_list2, r'$\mathbf{R}$', plots_dir + '/bondentropy-counts1'+'r'+str(r_list[i])+'.pdf')
            if 20 in L_list:
                plot_bond_entropy_counts(deltaS_count2, hamiltonian_type_list2, r'$\mathbf{R}$', plots_dir + '/bondentropy-counts2'+'r'+str(r_list[i])+'.pdf')
            

    #Principal component analysis
    if make_principal_components:
        principal_components = principal_component_analysis(np.array([networkdensity, clusteringcoefficient, disparity]).transpose())
        np.savetxt(plots_dir+'/principal_components.dat', principal_components, delimiter = '\t')

#     # plt.plot(range(3, 11), np.array(equilibrium_bond_entropy)/(0.5*np.array(L_list)), 'o')
#     # plt.rc('text', usetex=True)
#     # plt.rc('font', family='serif')
#     # plt.xlabel(r'\textbf{Distance Between Blinkers}', fontsize = 16)
#     # plt.ylabel(r'\textbf{Equilibrium Bond Entropy} $\mathbf{S_{\infty}/S_{\textrm{max}}}$', fontsize = 16)
#     # plt.savefig(plots_dir+'/two-blinker-distance.pdf', format = 'pdf')
#     # plt.close()
#     # plt.cla()

#     # plt.plot(range(3, 11), time_averages[:,0,0], 'o')
#     # plt.rc('text', usetex=True)
#     # plt.rc('font', family='serif')
#     # plt.xlabel(r'\textbf{Distance Between Blinkers}', fontsize = 16)
#     # plt.ylabel(r'\textbf{Clustering Coefficient} $\mathbf{C}$', fontsize = 16)
#     # plt.savefig(plots_dir+'/two-blinker-distance-clustering.pdf', format = 'pdf')
#     # plt.close()
#     # plt.cla()
#     # plt.plot(range(3, 11), time_averages[:,1,0], 'o')
#     # plt.rc('text', usetex=True)
#     # plt.rc('font', family='serif')
#     # plt.xlabel(r'\textbf{Distance Between Blinkers}', fontsize = 16)
#     # plt.ylabel(r'\textbf{Network Density} $\mathbf{D}$', fontsize = 16)
#     # plt.savefig(plots_dir+'/two-blinker-distance-density.pdf', format = 'pdf')
#     # plt.close()
#     # plt.cla()
#     # plt.plot(range(3, 11), time_averages[:,2,0], 'o')
#     # plt.rc('text', usetex=True)
#     # plt.rc('font', family='serif')
#     # plt.xlabel(r'\textbf{Distance Between Blinkers}', fontsize = 16)
#     # plt.ylabel(r'\textbf{Disparity} $\mathbf{Y}$', fontsize = 16)
#     # plt.savefig(plots_dir+'/two-blinker-distance-disparity.pdf', format = 'pdf')
#     # plt.close()
#     # plt.cla()

    #Plot Fourier peaks as a function of IC
    # initial_condition_index = dict(zip(initial_condition_index_list, range(len(initial_condition_index_list))))
    # fourier_peak_list2 = []
    # fourier_peak_count = []
    # IC_list3 = []

    # for measure in fourier_measures:
    #     fourier_peak_list2.append([])
    # for measure in fourier_measures:
    #     fourier_peak_count.append([])
    # for measure in fourier_measures:
    #     IC_list3.append([])

    # IC_list2 = [tuple(i) for i in ic_list] 
    # IC_list2 = set(IC_list2)
    # IC_list2 =  [list(i) for i in IC_list2]

    # for k in range(len(IC_list2)):
    #     if IC_list2[k][0][0][0] == 'E':
    #         if IC_list2[k][0][0] == 'E9_10_3':
    #             pos = [i for i,x in enumerate(ic_list) if x in [[('E'+str((L//2)-1)+'_'+str((L//2))+'_3', 1.0)] for L in L_list2]]
    #         elif IC_list2[k][0][0] == 'E9_10_4':
    #             pos = [i for i,x in enumerate(ic_list) if x in [[('E'+str((L//2)-1)+'_'+str((L//2))+'_4', 1.0)] for L in L_list2]]
    #         else:
    #             continue
    #     else:
    #         pos = [n for n,x in enumerate(ic_list) if x == IC_list2[k]]

    #     fourier_peak_list_L = fourier_peak_list[pos]

    #     for i in range(len(fourier_measures)):
    #         count = 0
    #         for j in range(len(fourier_peak_list_L)):
    #             fourier_peak_list2[i] += list(fourier_peak_list_L[j][i])
    #             count += len(fourier_peak_list_L[j][i])
    #             IC_list3[i] += len(fourier_peak_list_L[j][i])*[initial_condition_index[tuple(IC_list2[k])]]
    #         fourier_peak_count[i] += [count]

    # plot_fourier_peaks(fourier_peak_list2, IC_list3, fourier_measures, r'\textbf{Initial Condition Index} $\mathbf{i}$', plots_dir + '/fourier-peaks-ic.pdf')
    # plot_fourier_peaksv2(fourier_peak_list2, IC_list3, fourier_peak_count, np.arange(10)+1,  r'\textbf{Initial Condition Index} $\mathbf{i}$', fourier_measures, fourier_measure_names, plots_dir+'/fourier-peaks-icv2')        
