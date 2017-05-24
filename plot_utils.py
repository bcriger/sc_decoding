from __future__ import division
import decoding_2d as dc
from glob import glob
import networkx as nx
import numpy as np
import pickle as pkl
import seaborn as sb

#-------------------------NetworkX Graph Drawing----------------------#
def nx_draw(g, edge_color=False):
    pos = {v: np.array(v) for v in g.nodes()}
    labels = {v: str(v) for v in g.nodes()}
    try:
        weights = [2 * e[2]['p_edge'] for e in g.edges(data=True)]
    except:
        weights = [2 for e in g.edges()]
    nx.draw(g, pos, width=weights)
    nx.draw_networkx_labels(g, pos, labels)

#---------------------------------------------------------------------#

def merge_pickles(pattern):
    """
    checks to see that a bunch of pickles are compatible, then merges.
    """
    f_list = glob(pattern)
    # set up default key-value pairs that should match across files
    test_keys = ['err_lo', 'n_points', 'errs', 'err_hi', 'dists']
    
    with open(f_list[0], 'rb') as phil:
        fl_dict = pkl.load(phil)
        merge_dict = fl_dict
        f_keys = fail_keys(fl_dict)

    for flnm in f_list[1:]:
        with open(flnm, 'rb') as phil:
            curr_dict = pkl.load(phil)
            
            #testing
            for key in test_keys:
                if np.any(curr_dict[key] != merge_dict[key]):
                    raise ValueError("dicts don't match. "
                        "File at {} has dict[{}]={}, "
                        "File at {} has dict[{}]={}.".format(
                            f_list[0], key, fl_dict[key],
                            flnm, key, curr_dict[key]))

            curr_f_keys = fail_keys(curr_dict)

            if set(curr_f_keys) != set(f_keys):
                raise ValueError("These files contain data on "
                    "different distances, keys are:\n{}\n{}".format(
                        curr_f_keys, f_keys))

            #if tests pass, actually merge
            for f_key in f_keys:
                merge_dict[f_key] = p_lst_merge(
                    merge_dict[f_key], curr_dict[f_key]
                    )

            merge_dict['n_trials'] += curr_dict['n_trials']

    #dumb
    out_flnm = pattern.replace('*', 'merge')
    out_flnm += '.pkl' if out_flnm[-4:] != '.pkl' else ''
    
    with open(out_flnm, 'wb') as phil:
        pkl.dump(merge_dict, phil)

def error_rate_plot(flnm, ltr='I'):
    """
    makes those fancy plots people like so much. 
    Always plots an error rate (i.e. plots 1 - p for ltr = 'I'). 
    """
    
    with open(flnm, 'rb') as phil:
        data_dct = pkl.load(phil)
    
    f_keys = fail_keys(data_dct)
    sb.set_palette(sb.palettes.color_palette('husl', len(f_keys)))

    fig = sb.plt.figure()
    for idx, key in enumerate(f_keys):
        log_ps = [d[ltr] / data_dct['n_trials'] for d in data_dct[key]]
        if ltr == 'I':
            log_ps = [1 - p for p in log_ps]
        sb.plt.plot(data_dct['errs'], log_ps, '.')
    sb.plt.ylim(0, 1)
    sb.plt.xlabel('physical error probability')
    sb.plt.ylabel(ltr + ' error rate')
    sb.plt.title('Threshold Plot')
    sb.plt.show()
    pass


#-----------------------convenience functions------------------------#

def p_lst_merge(p_lst_1, p_lst_2):
    """
    Takes two lists whose elements are dicts with keys
    ['I', 'X', 'Y', 'Z'] and returns a single list with dict elements,
    with the values corresponding to these keys summed.
    """
    new_lst = []
    for x in zip(p_lst_1, p_lst_2):
        new_lst.append({_: x[0][_] + x[1][_] for _ in x[0].keys()})
    return new_lst

fail_keys = lambda d: list(filter(lambda x: 'failures' in x, d.keys()))
