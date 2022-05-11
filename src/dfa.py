import networkx as nx
import os

from pyTWTL.twtl_to_dfa import twtl_to_dfa
from pyTWTL.lomap import Fsa


CONFIG_PATH = '../configs/default_static.yaml'


def create_dfa(twtl_cfg: dict, env_cfg: dict) -> tuple[Fsa, int, str]:

    n = env_cfg['width']
    def xy_to_region(x,y,z):
        # x is down, y is across
        # ignore 3rd dim
        return x * n + y

    region_coords = twtl_cfg['regions']
    custom_task = twtl_cfg['TWTL task']
    if custom_task == 'None':
        custom_task = None

    if custom_task != None:
        # Build the spec from regions in config

        for r,c in region_coords.items():
            region_num = xy_to_region(*c)
            custom_task = custom_task.replace(r, 'r' + str(region_num))
        phi = custom_task
    else:
        pickup = region_coords['pickup']
        delivery = region_coords['delivery']

        pick_up_reg = xy_to_region(*pickup)
        delivery_reg = xy_to_region(*delivery)
        pick_up_str  = str(pick_up_reg)
        delivery_str = str(delivery_reg)

        twtl_horizon = twtl_cfg['time horizon']
        tf1 = int((twtl_horizon-1)/2) # time bound
        tf2 = int(twtl_horizon) - tf1 - 1
        phi = '[H^1 r' + pick_up_str + ']^[0, ' +  str(tf1) + '] * [H^1 r' + delivery_str + ']^[0,' + str(tf2) + ']'

    kind = twtl_cfg['DFA kind']
    out = twtl_to_dfa(phi, kind=kind, norm=True)
    dfa = out[kind]
    bounds = out['bounds']
    dfa_horizon = bounds[-1]
    if custom_task == None and dfa_horizon != twtl_horizon:
        raise RuntimeError(f'Received unexpected time bound from DFA. DFA horizon: {dfa_horizon}, expected: {twtl_horizon}.')
    else:
        # dfa horizon and twtl horizon are the same. Good.
        pass        

    # add self edge to accepting state
    # All observation cases in accepting state should result in self edge
    input_set = dfa.alphabet    
    for s in dfa.final:
        dfa.g.add_edge(s,s, guard='(else)', input=input_set, label='(else)', weight=0)

    # String to print
    print_string = f'TWTL task: {phi}\n'
    if custom_task != None:
        for r,c in region_coords.items():
            rnum = xy_to_region(*c)
            print_string += f'{r} : {c} <---> Region {rnum}\n'
    else:
        print_string += 'Pick-up Location  : ' + str(pickup) + ' <---> Region ' + pick_up_str + '\n'
        print_string += 'Delivery Location : ' + str(delivery) + ' <---> Region ' + delivery_str + '\n'

    return dfa, dfa_horizon, print_string

def save_dfa(dfa: Fsa, path: str='../output/dfa.png') -> None:

    this_path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(this_path, path)
    A = nx.nx_agraph.to_agraph(dfa.g)
    A.draw(path, prog='dot')


def save_custom_dfa(dfa: Fsa) -> None:

    A = nx.nx_agraph.to_agraph(dfa.g)
    ns = list(A.iternodes())
    ns[0].attr['pos'] = '0,100'
    ns[1].attr['pos'] = '150,100'
    ns[2].attr['pos'] = '250,100'
    ns[3].attr['pos'] = '250,0'
    ns[4].attr['pos'] = '100,0'
    ns[4].attr['shape'] = 'doublecircle'
    for e in A.iteredges():
        if tuple(e) == ('4','4'):
            e.attr['pos'] = 'e,78,7 78,-7 60,-7 50,-4 48,0 50,4 60,7 70,7'
    A.draw('dfa.png', prog='neato', args='-n2 -Gsplines=polyline')
