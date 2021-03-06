#include "myclib.h"
#include <iostream>
using namespace std;

int main()
{
    // cout << "Testing lib blossom5" << endl;
    // int node_num = 6;
    // int edge_num = 9;
        
    // run multiple times for timing purposes
    int repns = 1e6;
    for (int repn = 0; repn < repns; ++repn)
    {   
        FILE *graph_data;
        graph_data = fopen("graph_data.txt", "r");
        int edge_count = 0;
        int curr_uid, curr_vid, curr_weight;
        
        int node_num = 6;
        int edge_num = 9;
        
        Init();
        Edge* edges = new Edge[edge_num];
        // edges[0].uid = 0; edges[0].vid = 1;
        // edges[1].uid = 0; edges[1].vid = 2;
        // edges[2].uid = 0; edges[2].vid = 3;
        // edges[3].uid = 1; edges[3].vid = 2;
        // edges[4].uid = 1; edges[4].vid = 4;
        // edges[5].uid = 2; edges[5].vid = 5;
        // edges[6].uid = 3; edges[6].vid = 4;
        // edges[7].uid = 3; edges[7].vid = 5;
        // edges[8].uid = 4; edges[8].vid = 5;

        while( (fscanf(graph_data, "%d %d %d", &curr_uid, &curr_vid, &curr_weight)) != EOF )
        {
            edges[edge_count].uid = curr_uid;
            edges[edge_count].vid = curr_vid;
            edges[edge_count].weight = curr_weight;
            edge_count++;
        }
        
        Process(node_num, edge_num, edges);
        // if (repn==0)
        // {
             // PrintMatching();
        // }
        Clean();
        delete edges;
        fclose(graph_data);
    }
    
    return 0;
}
