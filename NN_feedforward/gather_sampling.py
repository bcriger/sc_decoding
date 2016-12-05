import numpy as np
from operator import itemgetter

distance = 7
physprob = [0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05]
samples = {}
inputs = 24
m = 0
m_new = 0
for probs in range(len(physprob)):
    with open('d='+str(distance)+'_p='+str(physprob[probs])+'_uniform_distr_samples.txt', 'r') as f:

        for line in f:
            if line == '':
                break

            data = line.split()
            if int(data[-1]) < 2:
                break
            m += 1
            x = ''
            for i in range(inputs):
                x += str(data[i])
            
            if x in samples.keys():
                v = samples[x]
                samples[x] = [v[0], v[1]+int(data[-1])] #[v[0], v[1]+long(data[-1])] 
            else:
                m_new += 1
                samples[x] = [int(data[-2]), int(data[-1])]#[int(data[-2]), long(data[-1])]
    print(physprob[probs], m, m_new)
				
sorted_samples = sorted(samples.items(), key=lambda e: e[1][1], reverse=True)

with open('d='+str(distance)+'_uniform_distr_samples.txt', 'w') as f:
    for key, value in sorted_samples:
        x = ''
        for i in range(inputs):
            x += key[i] + ' '

        x += ' '
        x += str(value[0]) + ' '
        x += str(value[1]) + '\n'
        f.write(x)