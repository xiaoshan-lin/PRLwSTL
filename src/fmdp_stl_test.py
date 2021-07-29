from fmdp_stl import Fmdp
import create_environment as ce
import lomap
import synthesis as syn

EXPR1 = 'G[0,10](F[0,3](x<3)&(x>1))&((F[0,4]x>5)|(G[0,3]x>6))'
M = 4
N = 4
H = 1

disc = 1
TS, obs_mat, state_mat = ce.create_ts(M,N,H)	
path = '../data/ts_' + str(M) + 'x' + str(N) + 'x' + str(H) + '_1Ag_1.txt'
paths = [path]
bases = {(0,0,0): 'Base1'}
obs_mat = ce.update_obs_mat(obs_mat, state_mat, M, [], (0,0,0))
TS      = ce.update_adj_mat_3D(M, N, H, TS, obs_mat)
ce.create_input_file(TS, state_mat, obs_mat, paths[0], bases, disc, M, N, H, 0)
ts_file = paths
ts_dict = lomap.Ts(directed=True, multi=False) 
ts_dict.read_from_file(ts_file[0])
ts = syn.expand_duration_ts(ts_dict)
# make coordinate dict for tmdp_stl
# apply offset so pos is middle of state
state_to_pos = dict()
for s in ts.g.nodes():
    if s == 'Base1':
        state_to_pos[s] = tuple([c + 0.5 for c in (0,0,0)])
    else:
        num = int(s[1:])
        #TODO: 3rd dim?
        state_to_pos[s] = ((num // N) + 0.5, (num % N) + 0.5, 0.5)

fmdp = Fmdp(ts, EXPR1, state_to_pos)
