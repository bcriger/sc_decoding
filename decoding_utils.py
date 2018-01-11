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
            int check_xx( float cmcq[4][4], float cmqc[4][4], int mcq_len, int sprt_len, int synd);
            int check_xxx( float cmcq[4][4], float cmqc[4][4], int mcq_len, int sprt_len, int synd);
            int check_xxxx( float cmcq[4][4], float cmqc[4][4], int mcq_len, int sprt_len, int synd);
            int check_zz( float cmcq[4][4], float cmqc[4][4], int mcq_len, int sprt_len, int synd);
            int check_zzz( float cmcq[4][4], float cmqc[4][4], int mcq_len, int sprt_len, int synd);
            int check_zzzz( float cmcq[4][4], float cmqc[4][4], int mcq_len, int sprt_len, int synd);
            int normalize( float cmcq[4][4], int mcq_len, int sprt_len);
            int pair_dist( int crd_0[2], int crd_1[2]);
            """

check_funcs_path = dir_path + "/check_funcs/libmyclib.so"
