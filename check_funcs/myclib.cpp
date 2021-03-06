#include "myclib.h"
#include <iostream>
#include <vector>
using namespace std;

extern "C" int check_xxxx( float cmcq[4][4], float cmqc[4][4], int mcq_len, int sprt_len, int synd)
{
    for(size_t i = 0; i < mcq_len; ++i)
        for(size_t j = 0; j < mcq_len; ++j)
            cmcq[i][j] = 0.0;

    vector<size_t> bs(sprt_len);
    for(size_t i = 0; i < sprt_len; ++i) 
        bs[i] = i;

    for(auto b: bs)
    {
        vector<size_t> lst;
        for(auto bb : bs)   
            if(bb != b) lst.push_back(bb);

        auto ix1 = cmqc[lst[0]][0] + cmqc[lst[0]][2];
        auto ix2 = cmqc[lst[1]][0] + cmqc[lst[1]][2];
        auto ix3 = cmqc[lst[2]][0] + cmqc[lst[2]][2];
        auto yz1 = cmqc[lst[0]][3] + cmqc[lst[0]][1];
        auto yz2 = cmqc[lst[1]][3] + cmqc[lst[1]][1];
        auto yz3 = cmqc[lst[2]][3] + cmqc[lst[2]][1];
        auto odd_2 = yz2 * ix3 + ix2 * yz3;
        auto even_2 = yz2 * yz3 + ix2 * ix3;
        auto odd_3 = yz1 * even_2 + ix1 * odd_2;
        auto even_3 = ix1 * even_2 + yz1 * odd_2;
        cmcq[b][0] += (synd == 1 ? odd_3 : even_3);
        cmcq[b][2] += (synd == 1 ? odd_3 : even_3);
        cmcq[b][1] += (synd == 1 ? even_3 : odd_3);
        cmcq[b][3] += (synd == 1 ? even_3 : odd_3);
    }

    normalize(cmcq, mcq_len, sprt_len);
    return 0;
}

extern "C" int check_zzzz( float cmcq[4][4], float cmqc[4][4], int mcq_len, int sprt_len, int synd)
{
    for(size_t i = 0; i < mcq_len; ++i)
        for(size_t j = 0; j < mcq_len; ++j)
            cmcq[i][j] = 0.0;

    vector<size_t> bs(sprt_len);
    for(size_t i = 0; i < sprt_len; ++i) 
        bs[i] = i;

    for(auto b: bs)
    {
        vector<size_t> lst;
        for(auto bb : bs)   
            if(bb != b) lst.push_back(bb);

        auto iz1 = cmqc[lst[0]][0] + cmqc[lst[0]][1];
        auto iz2 = cmqc[lst[1]][0] + cmqc[lst[1]][1];
        auto iz3 = cmqc[lst[2]][0] + cmqc[lst[2]][1];
        auto xy1 = cmqc[lst[0]][2] + cmqc[lst[0]][3];
        auto xy2 = cmqc[lst[1]][2] + cmqc[lst[1]][3];
        auto xy3 = cmqc[lst[2]][2] + cmqc[lst[2]][3];
        auto odd_2 = xy2 * iz3 + iz2 * xy3;
        auto even_2 = xy2 * xy3 + iz2 * iz3;
        auto odd_3 = xy1 * even_2 + iz1 * odd_2;
        auto even_3 = iz1 * even_2 + xy1 * odd_2;
        cmcq[b][0] += (synd == 1 ? odd_3 : even_3);
        cmcq[b][2] += (synd == 1 ? even_3 : odd_3);
        cmcq[b][1] += (synd == 1 ? odd_3 : even_3);
        cmcq[b][3] += (synd == 1 ? even_3 : odd_3);
    }

    normalize(cmcq,mcq_len,sprt_len);
    return 0;
}

extern "C" int check_xxx( float cmcq[4][4], float cmqc[4][4], int mcq_len, int sprt_len, int synd)
{
    for(size_t i = 0; i < mcq_len; ++i)
        for(size_t j = 0; j < mcq_len; ++j)
            cmcq[i][j] = 0.0;

    vector<size_t> bs(sprt_len);
    for(size_t i = 0; i < sprt_len; ++i) 
        bs[i] = i;

    for(auto b: bs)
    {
        vector<size_t> lst;
        for(auto bb : bs)   
            if(bb != b) lst.push_back(bb);

        auto ix1 = cmqc[lst[0]][0] + cmqc[lst[0]][2];
        auto ix2 = cmqc[lst[1]][0] + cmqc[lst[1]][2];
        auto yz1 = cmqc[lst[0]][3] + cmqc[lst[0]][1];
        auto yz2 = cmqc[lst[1]][3] + cmqc[lst[1]][1];
        auto odd_2 = yz1 * ix2 + ix1 * yz2;
        auto even_2 = yz1 * yz2 + ix1 * ix2;
        cmcq[b][0] += (synd == 1 ? odd_2 : even_2);
        cmcq[b][2] += (synd == 1 ? odd_2 : even_2);
        cmcq[b][1] += (synd == 1 ? even_2 : odd_2);
        cmcq[b][3] += (synd == 1 ? even_2 : odd_2);
    }

    normalize(cmcq, mcq_len, sprt_len);
    return 0;
}

extern "C" int check_zzz( float cmcq[4][4], float cmqc[4][4], int mcq_len, int sprt_len, int synd)
{
    for(size_t i = 0; i < mcq_len; ++i)
        for(size_t j = 0; j < mcq_len; ++j)
            cmcq[i][j] = 0.0;

    vector<size_t> bs(sprt_len);
    for(size_t i = 0; i < sprt_len; ++i) 
        bs[i] = i;

    for(auto b: bs)
    {
        vector<size_t> lst;
        for(auto bb : bs)   
            if(bb != b) lst.push_back(bb);

        auto iz1 = cmqc[lst[0]][0] + cmqc[lst[0]][1];
        auto iz2 = cmqc[lst[1]][0] + cmqc[lst[1]][1];
        auto xy1 = cmqc[lst[0]][2] + cmqc[lst[0]][3];
        auto xy2 = cmqc[lst[1]][2] + cmqc[lst[1]][3];
        auto odd_2 = xy1 * iz2 + iz1 * xy2;
        auto even_2 = xy1 * xy2 + iz1 * iz2;
        cmcq[b][0] += (synd == 1 ? odd_2 : even_2);
        cmcq[b][2] += (synd == 1 ? even_2 : odd_2);
        cmcq[b][1] += (synd == 1 ? odd_2 : even_2);
        cmcq[b][3] += (synd == 1 ? even_2 : odd_2);
    }

    normalize(cmcq,mcq_len,sprt_len);
    return 0;
}

extern "C" int check_xx( float cmcq[4][4], float cmqc[4][4], int mcq_len, int sprt_len, int synd)
{
    for(size_t i = 0; i < mcq_len; ++i)
        for(size_t j = 0; j < mcq_len; ++j)
            cmcq[i][j] = 0.0;

    vector<size_t> bs(sprt_len);
    for(size_t i = 0; i < sprt_len; ++i) 
        bs[i] = i;

    for(auto b: bs)
    {
        vector<size_t> lst;
        for(auto bb : bs)   
            if(bb != b) lst.push_back(bb);

        auto othr = lst[0];
        auto ix = cmqc[othr][0] + cmqc[othr][2];
        auto yz = cmqc[othr][3] + cmqc[othr][1];
        cmcq[b][0] += (synd == 0 ? ix : yz);
        cmcq[b][1] += (synd == 0 ? yz : ix);
        cmcq[b][2] += (synd == 0 ? ix : yz);
        cmcq[b][3] += (synd == 0 ? yz : ix);
    }

    normalize(cmcq, mcq_len, sprt_len);
    return 0;
}

extern "C" int check_zz( float cmcq[4][4], float cmqc[4][4], int mcq_len, int sprt_len, int synd)
{
    for(size_t i = 0; i < mcq_len; ++i)
        for(size_t j = 0; j < mcq_len; ++j)
            cmcq[i][j] = 0.0;

    vector<size_t> bs(sprt_len);
    for(size_t i = 0; i < sprt_len; ++i) 
        bs[i] = i;

    for(auto b: bs)
    {
        vector<size_t> lst;
        for(auto bb : bs)   
            if(bb != b) lst.push_back(bb);

        auto othr = lst[0];
        auto iz = cmqc[othr][0] + cmqc[othr][1];
        auto xy = cmqc[othr][2] + cmqc[othr][3];
        cmcq[b][0] += (synd == 0 ? iz : xy);
        cmcq[b][1] += (synd == 0 ? iz : xy);
        cmcq[b][2] += (synd == 0 ? xy : iz);
        cmcq[b][3] += (synd == 0 ? xy : iz);
    }

    normalize(cmcq, mcq_len, sprt_len);
    return 0;
}

extern "C" int normalize( float cmcq[4][4], int mcq_len, int sprt_len)
{
    for(auto b = 0; b < sprt_len; ++b)
    {
        float sum = 0.0;
        for(size_t i = 0; i < mcq_len; ++i)
            sum += cmcq[b][i];
        
        for(size_t i = 0; i < mcq_len; ++i)
            cmcq[b][i] = cmcq[b][i] / sum; 
    }
}

extern "C" int pair_dist(int crd_0[2], int crd_1[2]){
    /*
        Fast version of decoding_2d.pair_dist, I'm hoping to use this
        in matched_weights.all_pairs_multipath_sum
    */

    int d_x = crd_1[0] - crd_0[0]; // intdiv
    int d_y = crd_1[1] - crd_0[1]; // intdiv
    
    return (abs(d_x + d_y) + abs(d_x - d_y)) / 4; //intdiv
}

extern "C" int elem_i(int *mat, int r, int c, int n_cols){
    // printf("r * n_cols + c = %d \n", r * n_cols + c);
    return mat[r * n_cols + c];
}

extern "C" int elem_f(float *mat, int r, int c, int n_cols){
    return mat[r * n_cols + c];
}

extern "C" int calc_path_sum(double path_sum[], int sorted_prs[],
                            int length[], int graph_adj[],
                            int num_prs, int num_verts){
    
    /*
        Fast C version of matched_weights.py's `multipath_calc`, since
        direct array manipulation was taking most of the required time.

        The central idea is to begin with a |V|-by-|V| adjacency matrix
        and fill out the rest of the elements with weighted path sums. 
    */
    int l = 0, r = 0, c = 0, v = 0;

    //int num_prs = sizeof(sorted_prs) / sizeof(sorted_prs[0]);
    //for (size_t i = 0; i < 4 * num_verts; ++i)
    //    printf("%d \n", graph_adj[i]);

    for (int pr_dx = 0; pr_dx < num_prs; ++pr_dx)
    {
        // printf("get l\n");
        l = elem_i(sorted_prs, pr_dx, 0, 3);
        // printf("get r\n");
        r = elem_i(sorted_prs, pr_dx, 1, 3);
        // printf("get c\n");
        c = elem_i(sorted_prs, pr_dx, 2, 3);

        for (int nb_dx = 0; nb_dx < 4; ++nb_dx)
        {
            // printf("get v, c = %d \n", c);
            v = elem_i(graph_adj, c, nb_dx, 4);
            if (v != -1){ 
                // printf("get length elem, v = %d \n", v);
                if(elem_i(length, r, v, num_verts) == l - 1){
                // printf("get path_sum\n");
                path_sum[r * num_verts + c] += path_sum[r * num_verts + v] * path_sum[v * num_verts + c];
            }}
        }
    }
    
    return 0; //subroutine
}

// extern "C" int Clean()
// {
//  // cout << "Cleaned-up" << endl;
//  delete pm;
//  return 0;
// }
