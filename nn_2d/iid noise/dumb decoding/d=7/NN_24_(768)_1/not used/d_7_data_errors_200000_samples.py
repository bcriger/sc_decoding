import matplotlib
from matplotlib import pyplot

a = [1, 0.001, 0.01]
b = [1, 0.001, 0.01]

c = [0.5,0.4,0.3,0.2,0.1,
     0.09,0.08,0.07,0.06,0.05,0.04,0.03,0.02,0.01,
     0.009, 0.008, 0.007, 0.006, 0.005, 0.004]

d = [0.505356782,0.500625782,0.494535384,0.41235413,0.127873967,
     0.096344683,0.069248241,0.047380341,0.029422149,0.016415869,0.007315289,0.002845833,0.000604113,5.21512E-05,
     4.15094E-05,1.82342E-05,1.33261E-05,5.92E-06,2.27891E-06,7.7899E-07]

e = [0.503941783,0.501276596,0.494535384,0.425714404,0.151914273,
     0.119255448,0.088042214,0.062551526,0.040755561,0.023485636,0.010929042,0.004467957,0.00106928,9.38722E-05,
     7.79874E-05,4.3762E-05,2.99836E-05,2.03E-05,7.97618E-06,4.34008E-06]

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
