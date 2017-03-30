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


cdef_str_check_funcs = """
            int check0( float cmcq[4][4], float cmqc[4][4], int mcq_len, int sprt_len, int synd);
            int check1( float cmcq[4][4], float cmqc[4][4], int mcq_len, int sprt_len, int synd);
            int check2( float cmcq[4][4], float cmqc[4][4], int mcq_len, int sprt_len, int synd);
            int check3( float cmcq[4][4], float cmqc[4][4], int mcq_len, int sprt_len, int synd);
            int normalize( float cmcq[4][4], int mcq_len, int sprt_len);
            """

check_funcs_path = dir_path + "/check_funcs/libmyclib.so"
