import decoding_2d as dc
import cProfile as prf

dx=30
dy=30
p=0.05
useBlossom = True
# useBlossom = False
trials=10

print("Surface Code Simulation")

print("\n".join([
    "Starting simulation",
    "trials : {}".format(trials),
    "dx : {}".format(dx),
    "dy : {}".format(dy),
    "p : {}".format(p),
    "useBlossom : {}".format(useBlossom)
    ]))

sim = dc.Sim2D(dx, dy, p, useBlossom)
prf.run("sim.run(trials, False, False)", "imran_blsm.prof")
