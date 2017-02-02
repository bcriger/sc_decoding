import matplotlib
from matplotlib import pyplot

a = [1, 0.001, 0.01]
b = [1, 0.001, 0.01]

#------ phys -------------------------------------------------------
c = [0.5,0.4,0.3,0.2,0.1,
     0.09,0.08,0.07,0.06,0.05,0.04,0.03,0.02,0.01,
     0.009, 0.008, 0.007]

#------ d=3 -------------------------------------------------------     
d = [0.503895736,0.488845114,0.437296018,0.305486858,0.121327236,
     0.100977517,0.085643154,0.067916091,0.050094814,0.037016891,0.024834389,0.014426965,0.006373848,0.001849439,
     0.001486361,0.001119088,0.000806149]
     
#------ d=5 -------------------------------------------------------
e = [0.503931452,0.494146125,0.4773549,0.366521675,0.125937484,
     0.099041153,0.075286465,0.055319806,0.039377368,0.024933987,0.013722085,0.006271616,0.002080434,0.000228532,
     0.000208547,0.000115984,0.000100416]
     
#------ d=7 -------------------------------------------------------
f = [0.499460,0.501170,0.491210,0.413390,0.134000,
     0.103810,0.076960,0.051280,0.031810,0.018597,0.008980,0.003313,0.000733,0.000046,
     0.000034,0.000024,0.000016]
     
     
pyplot.subplot(2,1,1)#,aspect='equal')
pyplot.plot(a, b, color='black', lw=3, ls='dashed', label='y=x', linewidth=0.5)

pyplot.subplot(2,1,1)#,aspect='equal')
pyplot.plot(c, d, color='#ADFF2F', lw=5, ls = 'None', marker='d',label='d=3')

pyplot.subplot(2,1,1)#,aspect='equal')
pyplot.plot(c, e, color='#FFA500', lw=5, ls = 'None', marker='d',label='d=5')

pyplot.subplot(2,1,1)#,aspect='equal')
pyplot.plot(c, f, color='c', lw=5, ls = 'None', marker='d',label='d=7')

axes = pyplot.gca()
axes.set_ylim([0.000001,1])
pyplot.yscale('log')
pyplot.xscale('log')
pyplot.title('Neural net - Data errors - iid')
pyplot.xlabel('Physical error rate')
pyplot.ylabel('Logical X error rate')

pyplot.legend(loc=2,prop={'size':12})

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 18}

matplotlib.rc('font', **font)
pyplot.savefig('nn_d_3_5_7_data_errors_iid.jpg')
pyplot.show()
