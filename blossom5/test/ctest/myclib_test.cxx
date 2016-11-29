#include "myclib.h"
#include <iostream>
using namespace std;

int main()
{
    cout << "Testing lib blossom5" << endl;

    Init();

    int node_num = 6;
    int edge_num = 9;
    int *edges = new int[2*edge_num];
    int *weights = new int[edge_num];
    edges[0] = 0;   edges[1] = 1;   weights[0] = 3;
    edges[2] = 0;   edges[3] = 3;   weights[1] = 10;
    edges[4] = 0;   edges[5] = 4;   weights[2] = 7;
    edges[6] = 1;   edges[7] = 2;   weights[3] = -1;
    edges[8] = 1;   edges[9] = 4;   weights[4] = 5;
    edges[10] = 1;  edges[11] = 5;  weights[5] = 4;
    edges[12] = 2;  edges[13] = 5;  weights[6] = -7;
    edges[14] = 3;  edges[15] = 4;  weights[7] = 0;
    edges[16] = 4;  edges[17] = 5;  weights[8] = 4;
    Process(node_num, edge_num, edges, weights);
    PrintMatching();
    Clean();
    return 0;
}
