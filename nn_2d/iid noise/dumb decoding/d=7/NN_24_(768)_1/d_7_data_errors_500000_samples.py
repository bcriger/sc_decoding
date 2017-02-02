import matplotlib
from matplotlib import pyplot

a = [1, 0.001, 0.01]
b = [1, 0.001, 0.01]

c = [0.5,0.4,0.3,0.2,0.1,
     0.09,0.08,0.07,0.06,0.05,0.04,0.03,0.02,0.01,
     0.009, 0.008, 0.007]

d = [0.498130,0.500430,0.492830,0.414930,0.127040,
     0.097250,0.070180,0.046390,0.028480,0.016073,0.007630,0.002787,0.000597,0.000037,
     0.000027,0.000014,0.000009]

e = [0.499460,0.501170,0.491210,0.413390,0.134000,
     0.103810,0.076960,0.051280,0.031810,0.018597,0.008980,0.003313,0.000733,0.000046,
     0.000034,0.000024,0.000016]

pyplot.subplot(2,1,1)#,aspect='equal')
pyplot.plot(a, b, color='black', lw=2, ls='dashed', label='y=x', linewidth=0.5)

pyplot.subplot(2,1,1)#,aspect='equal')
pyplot.plot(c, d, color='blue', lw=5, ls = 'None', marker='+',label='mwpm')

pyplot.subplot(2,1,1)#,aspect='equal')
pyplot.plot(c, e, color='None', lw=5, ls = 'None', marker='o',label='neural net')

pyplot.yscale('log')
pyplot.xscale('log')
pyplot.title('Distance 7 - Data errors')
pyplot.xlabel('Physical error rate')
pyplot.ylabel('Logical X error rate')

pyplot.legend(loc=2,prop={'size':12})

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 18}

matplotlib.rc('font', **font)
pyplot.savefig('d_7_data_errors.jpg')
pyplot.show()
