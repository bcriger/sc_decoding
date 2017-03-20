''' blossom_wrapper: a wrapper for the disfigured blossom algorithm written
for this package.

As you will see if you read the code for blossom, the algorithm I use has had its
input heavily changed in order to mesh it best with the qec package.

Below I give functions that take in a standard format for graph data
and spit out the answer, without attempting to do anything crazy.

***Note that I still will use the special boundary vertex here. This can be
avoided by putting in a weight matrix with a disconnected first vertex. However,
I never check for a single ***
'''

from sys import version_info
if version_info[0] == 3:
    PY3 = True
    from importlib import reload
elif version_info[0] == 2:
    PY3 = False
else:
    raise EnvironmentError("sys.version_info refers to a version of "
        "Python neither 2 nor 3. This is not permitted. "
        "sys.version_info = {}".format(version_info))

if PY3:
    from . import blossom as Bloss
else:
    import blossom as Bloss

def insert_wm(wm, cutoff=None):
    '''
    Converts a weight matrix into a string of vertices to feed into a
    constantly updating version of blossom, inserts it, and returns result.

    Assumes that the first index corresponds to the boundary,
    and the last to the time boundary (if tbw not None). If tbw=None,
    it inserts everything then runs. if timestep != None, it runs every time
    it inserts timestep indices. If f_flag=True, it finalises every time
    with the next error as the timestep.

    Input:
    wm: a matrix in the form of a list of lists
    cutoff: maximum weight to feed the algorithm. If None, feeds all
        non-zero weights.
    '''

    num_vertices = wm.shape[0]

    b = Bloss.Bloss(tbw=10**6, max_vertices=num_vertices+500, vertex_buffer=20)

    # Run over weight matrix
    for index in range(1, len(wm)):

        # Pull list of weights to insert
        weight_list = wm[index][:index]
        # initialise the set that we will send to blossom
        # we always include the edge to the boundary

        # weight_set = [[0, weight_list[0]]]
        weight_set = []
        # Run over list of weights, turn them into the correct format
        # and insert them into weight_set, if they are sufficiently small
        # weight. We also assume zero weights correspond to 'no edge'
        for index2, weight in zip(range(1, index), weight_list[1:]):

            if weight == 0 or cutoff and weight > cutoff:
                continue  # Next set

            weight_set.append([index2, weight])

        # Insert into blossom
        b.add_vertex(weight_set)

    pairing = b.finish_safe()

    return pairing
