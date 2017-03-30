import decoding_2d as dc
# import cProfile as prf
from timeit import default_timer as timer

dx=12
dy=12
p=0.01
useBlossom = True
trials=1

print("Surface Code Simulation")

useBlossom = True
print("\n".join([
    "Starting simulation with blossom",
    "trials : {}".format(trials),
    "dx : {}".format(dx),
    "dy : {}".format(dy),
    "p : {}".format(p),
    "useBlossom : {}".format(useBlossom)
    ]))

sim = dc.Sim2D(dx, dy, p, useBlossom)
start = timer()
# prf.run("sim.run(trials, False, False, None, True)", "imran_blsm.prof")
sim.run(trials, False, False, None, True)
end = timer()
totaltime = (end-start)
print("Execution took : {0} ms".format(totaltime*1e3))
del sim



#useBlossom = True
#print("\n".join([
    #"Starting simulation with blossom",
    #"trials : {}".format(trials),
    #"dx : {}".format(dx),
    #"dy : {}".format(dy),
    #"p : {}".format(p),
    #"useBlossom : {}".format(useBlossom)
    #]))

#sim = dc.Sim2D(dx, dy, p, useBlossom)
#start = timer()
#sim.run(trials, False, False)
#end = timer()
#totaltime = (end-start)
#print("Execution took : {0} ms".format(totaltime*1e3))
#del sim

#useBlossom = False
#print("\n".join([
    #"\nStarting simulation without blossom",
    #"trials : {}".format(trials),
    #"dx : {}".format(dx),
    #"dy : {}".format(dy),
    #"p : {}".format(p),
    #"useBlossom : {}".format(useBlossom)
    #]))

#sim = dc.Sim2D(dx, dy, p, useBlossom)
##prf.run("sim.run(trials, False, False)", "imran_blsm.prof")
#start = timer()
#sim.run(trials, False, False)
#end = timer()
#totaltime = (end-start)
#print("Execution took : {0} ms".format(totaltime*1e3))
