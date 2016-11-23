import decoding_2d as dc

dx=5
dy=4
p=0.09
trials=1

print "Surface Code Simulation"

print "\n".join([
    "Starting simulation",
    "trials : {}".format(trials),
    "dx : {}".format(dx),
    "dy : {}".format(dy),
    "p : {}".format(p)
    ])

sim = dc.Sim2D(dx, dy, p)
sim.run(trials, True, False)
