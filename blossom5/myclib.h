#ifndef myclib_h
#define myclib_h

typedef struct {
    int uid;
    int vid;
    int weight;
} Edge;

extern "C" int Init();
extern "C" int Process(int node_num, int edge_num, Edge *edges);
extern "C" int PrintMatching();
extern "C" int GetMatching(int * matching);
extern "C" int Clean();

#endif
