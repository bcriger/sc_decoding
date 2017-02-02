import matplotlib
from matplotlib import pyplot

a = [1, 0.00001, 0.01]
b = [1, 0.00001, 0.01]

c = [0.5,0.4,0.3,0.2,0.1,
     0.09, 0.08,0.07,0.06,0.05,0.04,0.03,0.02,0.01,
     0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001,
     0.0009, 0.0008, 0.0007]

d = [0.496860,0.503590,0.500130,0.490240,0.376270,
     0.345780,0.309020,0.271990,0.224670,0.180840,0.132420,0.083600,0.041780,0.012290,
     0.009860,0.007880,0.006310,0.004360,0.003280,0.002080,0.001010,0.000350,0.000140,
     0.000100,0.000060,0.000070]

e = [0.504770,0.498410,0.498220,0.494380,0.364910,
     0.335450,0.296160,0.254580,0.210950,0.165700,0.116570,0.072800,0.035810,0.009820,
     0.008120,0.006240,0.005280,0.003770,0.002530,0.001710,0.000890,0.000340,0.000110,
     0.000080,0.000055,0.000045]

pyplot.subplot(2,1,1)#,aspect='equal')
pyplot.plot(a, b, color='black', lw=2, ls='dashed', label='y=x', linewidth=0.5)

pyplot.subplot(2,1,1)#,aspect='equal')
pyplot.plot(c, d, color='blue', lw=5, ls = 'None', marker='+',label='mwpm')

pyplot.subplot(2,1,1)#,aspect='equal')
pyplot.plot(c, e, color='None', lw=5, ls = 'None', marker='o',label='neural net')

pyplot.yscale('log')
pyplot.xscale('log')
pyplot.title('Distance 3 - Data +  meas errors')
pyplot.xlabel('Physical error rate')
pyplot.ylabel('Logical X error rate')

pyplot.legend(loc=2,prop={'size':12})

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 18}

matplotlib.rc('font', **font)
pyplot.savefig('d_3_data_meas_errors.jpg')
pyplot.show()
