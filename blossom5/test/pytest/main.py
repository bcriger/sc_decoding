from cffi import FFI

ffi = FFI()

blossom = ffi.dlopen('../../libblossom.so')
print('Loaded lib {0}'.format(blossom))

# Describe the data type and function prototype to cffi.
ffi.cdef('''
int Init();
int Process(int node_num, int edge_num, int *edges, int *weights);
int PrintMatching();
int GetMatching(int * matching);
int Clean();
''')

node_num = 6;
edge_num = 9;
edgeList = [ i for i in range(2*edge_num) ]
weightList = [ i for i in range(edge_num) ]

edgeList[0] = 0;   edgeList[1] = 1;   weightList[0] = 3;
edgeList[2] = 0;   edgeList[3] = 3;   weightList[1] = 10;
edgeList[4] = 0;   edgeList[5] = 4;   weightList[2] = 7;
edgeList[6] = 1;   edgeList[7] = 2;   weightList[3] = -1;
edgeList[8] = 1;   edgeList[9] = 4;   weightList[4] = 5;
edgeList[10] = 1;  edgeList[11] = 5;  weightList[5] = 4;
edgeList[12] = 2;  edgeList[13] = 5;  weightList[6] = -7;
edgeList[14] = 3;  edgeList[15] = 4;  weightList[7] = 0;
edgeList[16] = 4;  edgeList[17] = 5;  weightList[8] = 4;

edges = ffi.new('int[]', edgeList)
weights = ffi.new('int[]', weightList)
matching = ffi.new('int[%d]' % (2*node_num) )

print('C Processing')
retVal = blossom.Init()
retVal = blossom.Process(node_num, edge_num, edges, weights)
retVal = blossom.PrintMatching()
nMatching = blossom.GetMatching(matching)
retVal = blossom.Clean()

for i in range(0,nMatching,2):
    print('{0} {1} '.format(matching[i], matching[i+1]) )
