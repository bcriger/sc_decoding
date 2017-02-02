import matplotlib
from matplotlib import pyplot

a = [1, 0.001, 0.01]
b = [1, 0.001, 0.01]

c = [0.5,0.4,0.3,0.2,0.1,
     0.09,0.08,0.07,0.06,0.05,0.04,0.03,0.02,0.01,
     0.009, 0.008, 0.007]

d = [0.472210417,0.458106189,0.407963446,0.287571174,0.11724704,
     0.097751711,0.082987552,0.06575919,0.048872989,0.036262628,0.024390482,0.01423901,0.006310741,0.001834761,
     0.001486361,0.001119088,0.000798167]

e = [0.503895736,0.488845114,0.437296018,0.305486858,0.121327236,
     0.100977517,0.085643154,0.067916091,0.050094814,0.037016891,0.024834389,0.014426965,0.006373848,0.001849439,
     0.001486361,0.001119088,0.000806149]

pyplot.subplot(2,1,1)#,aspect='equal')
pyplot.plot(a, b, color='black', lw=2, ls='dashed', label='y=x', linewidth=0.5)

pyplot.subplot(2,1,1)#,aspect='equal')
pyplot.plot(c, d, color='blue', lw=5, ls = 'None', marker='+',label='mwpm')

pyplot.subplot(2,1,1)#,aspect='equal')
pyplot.plot(c, e, color='None', lw=5, ls = 'None', marker='o',label='neural net')

pyplot.yscale('log')
pyplot.xscale('log')
pyplot.title('Distance 3 - Data errors')
pyplot.xlabel('Physical error rate')
pyplot.ylabel('Logical X error rate')

pyplot.legend(loc=2,prop={'size':12})

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 18}

matplotlib.rc('font', **font)
pyplot.savefig('d_3_data_errors.jpg')
pyplot.show()
