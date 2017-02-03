import matplotlib
from matplotlib import pyplot

a = [1, 0.001, 0.01]
b = [1, 0.001, 0.01]

#------ phys -------------------------------------------------------
c = [0.5,0.4,0.3,0.2,0.1,
     0.09,0.08,0.07,0.06,0.05,0.04,0.03,0.02,0.01,
     0.009, 0.008, 0.007]

#------ d=3 -------------------------------------------------------     
d = [0.472210417,0.458106189,0.407963446,0.287571174,0.11724704,
     0.097751711,0.082987552,0.06575919,0.048872989,0.036262628,0.024390482,0.01423901,0.006310741,0.001834761,
     0.001486361,0.001119088,0.000798167]
     
#------ d=5 -------------------------------------------------------
e = [0.504032258,0.493997925,0.475737393,0.368622825,0.125836815,
     0.099259524,0.075286465,0.055110386,0.039228301,0.024770501,0.013632113,0.006207063,0.002068026,0.000224051,
     0.000206482,0.000113709,9.65538E-05]
     
#------ d=7 -------------------------------------------------------
f = [0.498130,0.500430,0.492830,0.414930,0.127040,
     0.097250,0.070180,0.046390,0.028480,0.016073,0.007630,0.002787,0.000597,0.000037,
     0.000027,0.000014,0.000009]
     
pyplot.subplot(2,1,1)#,aspect='equal')
pyplot.plot(a, b, color='black', lw=3, ls='dashed', label='y=x', linewidth=0.5)

pyplot.subplot(2,1,1)#,aspect='equal')
pyplot.plot(c, d, color='#ADFF2F', lw=5, ls = 'None', marker='d',label='d=3')

pyplot.subplot(2,1,1)#,aspect='equal')
pyplot.plot(c, e, color='#FFA500', lw=5, ls = 'None', marker='d',label='d=5')

pyplot.subplot(2,1,1)#,aspect='equal')
pyplot.plot(c, f, color='c', lw=5, ls = 'None', marker='d',label='d=7')

pyplot.yscale('log')
pyplot.xscale('log')
pyplot.title('Blossom - Data errors - iid')
pyplot.xlabel('Physical error rate')
pyplot.ylabel('Logical X error rate')

pyplot.legend(loc=2,prop={'size':12})

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 18}

matplotlib.rc('font', **font)
pyplot.savefig('blossom_d_3_5_7_data_errors_iid.jpg')
pyplot.show()
