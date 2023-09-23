import numpy as np
import matplotlib.pyplot as plt

def plot_trend(x,y,bins=20,ytype='median',
               fig=None,ax=None,
               ranges=None,auto_p=None,
               prop_kwargs=None,
               scatter_kwargs=None,
               plot_kwargs=None):
    """
    Make a plot to show the trend between x and y


    Parameters
    -----------------
    x : array_like[nsamples,]                                     
        The samples.                             
                                                           
    y : array_like[nsamples,]                            
        The samples.
                                                                     
    range: array_like[2, 2] or string ([x_min, x_max], [y_min, y_max]), 
        if not 'auto', the range is automatically determined according to quantile specified by auto_p, if 'auto'                                                        
        default: 'auto'

    auto_p: array_like[2, 2] or string
       Used to generate range if range == 'auto'
       x_min = np.percentile(x, auto_p[0][0])
       x_max = np.percentile(x, auto_p[0][1])
       y_min = np.percentile(y, auto_p[1][0])
       y_max = np.percentile(y, auto_p[1][1])
       default: ([1, 99], [1, 99])

    ytype: Character string or float
        The y value used to plot
        The available character string is "median" or 'mean'. If ytype is set as "median", the trend is shown by the median value of y as a function of x.
        if ytype is float, y_value = np.percentile(y, ytype)
        default: "median"
    
    ax : matplotlib.Axes
        A axes instance on which to add the line.
        
        
    plot_kwargs: function in ``matplotlib``


    prop_kwargs: dict (to be added)
        The extra property used to constrain the x, y, data

    scatter_kwargs: dict (to be added)
        ifscatter: whether to plot scatter
        uplim (%): The upper limit of the scatter
        bottomlim (%): The bottom limit of the scatter
        fkind: which ways to show the scatter, "errorbar" and "fbetween" are available
        plot_scatter_kwargs: function in ``matplotlib``
        
    """

    if ax is None:
        ax = plt.gca()

    if plot_kwargs is None: 
        plot_kwargs = {} 
        
    if scatter_kwargs is None:
        ifscatter = False;
    else:
        ifscatter = scatter_kwargs["ifscatter"]
        uplim = scatter_kwargs["uplim"]
        bottomlim = scatter_kwargs["bottomlim"]
        fkind = scatter_kwargs["fkind"]
        plot_scatter_kwargs = scatter_kwargs["plot_scatter_kwargs"]
        
        
    bad = np.isnan(x) + np.isinf(x) + np.isnan(y) + np.isinf(y)
    x = x[~bad]
    y = y[~bad]

    if ranges is None:
        xrange = [x.min(), x.max()]
        yrange = [y.min(), y.max()]
    elif ranges == 'auto':
        if auto_p is None:
            auto_p = ([1, 99], [1, 99])
        xrange = [np.percentile(x, auto_p[0][0]), np.percentile(x, auto_p[0][1])]
        yrange = [np.percentile(y, auto_p[1][0]), np.percentile(y, auto_p[1][1])]
    else:
        xrange = ranges[:][0]
        yrange = ranges[:][1]
    
    data_xrange = np.linspace(xrange[0],xrange[1],bins+1)
    #print(x.max(),data_xrange,data_xrange[len(data_xrange)-2])
    if(x.min()>=data_xrange[1] or x.max()<=data_xrange[len(data_xrange)-2]):
        raise ValueError("It looks like the range is so broad")
        
    xys = np.vstack((x,y)).T
    xyz_xsort = xys[np.argsort(xys[:,0])]

    loads = [] 
    xs = []; ys = []; indexs = 0; 
    for i in range(len(xyz_xsort[:,0])):
        if(indexs==len(data_xrange)-1): continue
        if(xyz_xsort[i,0]>=data_xrange[indexs] and xyz_xsort[i,0]<data_xrange[indexs+1]):
            if(i==len(xyz_xsort[:,0])-1):
                 #loads.append([len(xs),np.median(xs),np.median(ys),np.std(ys),np.quantile(ys,0.25,interpolation='lower'),np.quantile(ys,0.75,interpolation='higher'),np.sum(ys)])
                if(ytype=='median'): yvalue = np.median(ys)
                elif(ytype=='mean'): yvalue = np.mean(ys)
                else: yvalue = np.percentile(ys,float(ytype))
                if(ifscatter):
                    upvalue = np.percentile(ys,uplim)
                    btvalue = np.percentile(ys,bottomlim)
                    loads.append([np.median(xs),yvalue,upvalue,btvalue])
                else:
                    loads.append([np.median(xs),yvalue])
            else:
                if(xyz_xsort[i+1,0]>=data_xrange[indexs+1]):
                    if(ytype=='median'): yvalue = np.median(ys)
                    elif(ytype=='mean'): yvalue = np.mean(ys)
                    else: yvalue = np.percentile(ys,float(ytype))
                        
                    if(ifscatter):
                        upvalue = np.percentile(ys,uplim)
                        btvalue = np.percentile(ys,bottomlim)
                        loads.append([np.median(xs),yvalue,upvalue,btvalue])
                    else:
                        loads.append([np.median(xs),yvalue])
                    #print(yvalue,upvalue,btvalue,np.percentile(ys,50))
                    indexs = indexs + 1
                    xs = []; ys = []
                else:
                    if(xyz_xsort[i,1]>yrange[1] or xyz_xsort[i,1]<yrange[0]): continue
                    xs.append(xyz_xsort[i,0])
                    ys.append(xyz_xsort[i,1])

    #print(loads[0][:])
    loads = np.array(loads)
    ax.plot(loads[:,0],loads[:,1],**plot_kwargs)
    if(ifscatter):
        if(fkind=="errorbar"):
            ax.errorbar(loads[:,0],loads[:,1],yerr=(loads[:,2]-loads[:,3])/2.0,**plot_scatter_kwargs)
        if(fkind=="fbetween"):
            ax.fill_between(loads[:,0],loads[:,3],loads[:,2],**plot_scatter_kwargs)
