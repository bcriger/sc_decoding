"""
We need to move all the common code into this file, so that we're not
repeating ourselves.
"""

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

cdef_str = """
            typedef struct {
                int uid;
                int vid;
                int weight;
            } Edge;

            int Init();
            int Process(int node_num, int edge_num, Edge *edges);
            int PrintMatching();
            int GetMatching(int * matching);
            int Clean();
            """

# blossom_path = "./blossom5/libblossom.so"
blossom_path = dir_path + "/blossom5/libblossom.so"