#ifndef myclib_h
#define myclib_h

extern "C" int check_xx( float cmcq[4][4], float cmqc[4][4], int mcq_len, int sprt_len, int synd);
extern "C" int check_xxx( float cmcq[4][4], float cmqc[4][4], int mcq_len, int sprt_len, int synd);
extern "C" int check_xxxx( float cmcq[4][4], float cmqc[4][4], int mcq_len, int sprt_len, int synd);
extern "C" int check_zz( float cmcq[4][4], float cmqc[4][4], int mcq_len, int sprt_len, int synd);
extern "C" int check_zzz( float cmcq[4][4], float cmqc[4][4], int mcq_len, int sprt_len, int synd);
extern "C" int check_zzzz( float cmcq[4][4], float cmqc[4][4], int mcq_len, int sprt_len, int synd);

extern "C" int normalize( float cmcq[4][4], int mcq_len, int sprt_len);
extern "C" int pair_dist( int crd_0[2], int crd_1[2]);

// extern "C" int Process(int node_num, int edge_num, Edge *edges);
// extern "C" int PrintMatching();
// extern "C" int GetMatching(int * matching);
// extern "C" int Clean();

#endif
