#ifndef myclib_h
#define myclib_h

extern "C" int check0( float cmcq[4][4], float cmqc[4][4], int mcq_len, int sprt_len, int synd);
extern "C" int check1( float cmcq[4][4], float cmqc[4][4], int mcq_len, int sprt_len, int synd);
extern "C" int check2( float cmcq[4][4], float cmqc[4][4], int mcq_len, int sprt_len, int synd);
extern "C" int check3( float cmcq[4][4], float cmqc[4][4], int mcq_len, int sprt_len, int synd);
extern "C" int normalize( float cmcq[4][4], int mcq_len, int sprt_len);
extern "C" int pair_dist( int crd_0[2], int crd_1[2]);

// extern "C" int Process(int node_num, int edge_num, Edge *edges);
// extern "C" int PrintMatching();
// extern "C" int GetMatching(int * matching);
// extern "C" int Clean();

#endif
