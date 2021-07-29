
import lomap
import networkx as nx

class Tmdp(lomap.Ts):
    def __init__(self, ts, tau):
        lomap.Ts.__init__(directed=True, multi=False)

        # Make a dictionary of ts edges and add state for null history that can transition to any state
        ts_edge_dict = {s:ts.g.edge[s].keys() for s in ts.g.edge.keys()}
        ts_edge_dict[None] = ts.g.edge.keys() + [None]

        # make list of tau mdp states where each state is represented by a tuple of mdp states
        tmdp_states = []
        for s in ts_edge_dict.keys():
            tmdp_states.extend(build_states([s], ts_edge_dict, tau))

        tmdp_states.remove((None,) * tau) # No state should end with a null

        # try and recreate process used in ts.read_from_file() except with tau mdp
        self.name = "Tau MDP"
        self.init = {((None,) * (tau-1)) + (ts.init.keys()[0],) :1}

        # create dict of dicts representing edges and attributes of each edge to construct the nx graph from
        # attributes are based on the mdp edge between the last (current) states in the tau mdp sequence
        edge_dict = {}
        for x1 in tmdp_states:
            edge_attrs = ts.g.edge[x1[-1]]
            # tmdp states are adjacent if they share the same (offset) history. "current" state transition is implied valid 
            # based on the set of names created
            if tau > 1:
                #TODO use next_mdp_states instead of conditional
                edge_dict[x1] = {x2:edge_attrs[x2[-1]] for x2 in tmdp_states if x1[1:] == x2[:-1]}
            else:
                # Case of tau = 1
                edge_dict[x1] = {(x2,):edge_attrs[x2] for x2 in ts_edge_dict[x1[0]]}

        self.g = nx.from_dict_of_dicts(edge_dict, create_using=nx.MultiDiGraph()) 

        # add node attributes based on last state in sequence
        for n in self.g.nodes():
            self.g.node[n] = ts.g.node[n[-1]]


	def build_states(self, past, ts_edge_dict, tau):
		if tau == 1:
			# One tau-MDP state per MDP state
			return [tuple(past)]
		next_states = ts_edge_dict[past[-1]]
		if len(next_states) == 0:
			# no next states. Maybe an obstacle?
			return []
		tmdp_states = [past + [ns] for ns in next_states]
		if len(tmdp_states[0]) == tau:
			# each tau-MDP state has 'tau' MDP states
			# make each state immutable
			return [tuple(s) for s in tmdp_states]
		
		# recurse for each state in states
		more_tmdp_states = []
		for x in tmdp_states:
			more_tmdp_states.extend(self.build_states(x, ts_edge_dict, tau))
		
		return more_tmdp_states



class TmdpStl:
    def __init__(self, stl_expr, ts_sig_dict = None):
        # stl_expr is format
        # F[t0,tn](x>2&x<3&y>2&y<3)
        self.expr = stl_expr
        self.ts_sig_dict = ts_sig_dict
        self.tau = None
        self.rdegree_lse_cache = {}  # cache dict

    def set_ts_sig_dict(self, ts_sig_dict):
        self.ts_sig_dict = ts_sig_dict

    def rdegree_lse(self, tmdp_s):

        if None in tmdp_s:
            raise Exception("State does not have complete history. Cannot compute robustness degree of incomplete state.")

        # check cache
        try:
            return self.rdegree_lse_cache[tmdp_s]
        except KeyError:
            # not in cache so keep going
            pass

        # Using log-sum-exp approximation, sign is positive for F (max) and negative for G (min)
        if self.expr[0] == 'F':
            sign = 1
        elif self.expr[0] == 'G':
            sign = -1
        else:
            # expression should start with either globally (G) or eventually (F)
            raise Exception("Invalid stl expression: " + str(self.expr))

        # Check if time step (t) is part of the lse sum. If not, return None signifying not to sum
        # tau = 
        # comma = self.expr.index(',')
        end = self.expr.index(']')
        split = end + 1
        # a = float(self.expr[2:comma])
        # b = float(self.expr[comma+1:end])
        # tau = len(tmdp_s)
        # if not (tau - 1) <= t <= b:
        #     return sign, None

        # TODO: check tmdp_s length vs horizon length
        # TODO: replace tau above with tau according to equation 9

        # remove K[.] from expr
        sub_expr = self.expr[split:]
        # Create signal as coordinate position of each state in the history
        if self.ts_sig_dict == None:
            raise Exception("State to signal mapping must be set with set_ts_sig_dict().")

        sig = [self.ts_sig_dict[x] for x in tmdp_s]
        rdeg = self.rdegree(sub_expr, sig)

        # add to cache dict
        self.rdegree_lse_cache[tmdp_s] = (sign,rdeg)

        return sign,rdeg

        # sig = [self.ts_sig_dict[x] for x in tmdp_s]
        # # compute the robustness degree of each ts state in tmdp state
        # ts_rdegs = [self.rdegree(sub_expr, s) for s in sig]

        # # return maximum robustness degree of ts states in tmdp state. This is effectively r(s_t^tau, phi).
        # return sign, max(ts_rdegs)

    def rdegree(self, expr, sig):
        if len(sig) == 0:
            raise Exception('rdegree entered with empty signal!')
        if expr[0] in 'FG':
            # parse
            comma = expr.index(',')
            end = expr.index(']')
            a = int(expr[2:comma])
            b = int(expr[comma+1:end])
            subStl = expr[end+1:]

            # sig needs to be broken down for each item of the max/min according to the horizon length of the interior
            # If we have, for example, (F[0,2]a)&(F[0,4]b) where a and b are predicates, we could be parsing the first term which
            # has a horizon length of 2 and needs a signal of length 3, but we are passed a signal of length 5 because of the second
            # term at the same level. This could also be the case if a is another expression containing F/G. We must get the horizon length
            # of the interior to find out
            hrz_intr = self.hrz(subStl)
            next_sig_len = int(hrz_intr + 1)
            # if next_sig_len < len(sig) - b then b+ elements of sig will be unused in the part of Phi
            # This also defines a minimum length for sig at this point
            if len(sig) < b + next_sig_len:
                raise Exception('Signal length too short in rdegree')

            # max/min can take a generator. Recurse for each time step.
            degs = (self.rdegree(subStl, sig[t:t+next_sig_len]) for t in range(a,b+1))

            # max for eventually, min for globally
            if expr[0] == 'F':
                return max(degs)
            elif expr[0] == 'G':
                return min(degs)


        if expr[0] == '(':
            # has format (phi) or (phi)&(phi) or (phi)|(phi)
            parts = self.sep_paren(expr)
            if len(parts) == 1:
                # format was (phi). Recurse.
                rdeg = self.rdegree(parts[0],sig)
                return rdeg
            # TODO: check that all ops are the same for len(parts > 2)
            op_idx = len(parts[0]) + 2
            op = expr[op_idx]
            if op == '&':
                # and: minimum of two predicates
                # j = expr.index('&')
                # subStl1 = expr[1:j]
                # # pad second sub expr with parentheses to allow for multiple & or |
                # subStl2 = '(' + expr[j+1:-1] + ')'
                return min(self.rdegree(sub,sig) for sub in parts)
            elif op == '|':
                # or: maximum of two predicates
                # j = expr.index('|')
                # subStl1 = expr[1:j]
                # subStl2 = '(' + expr[j+1:-1] + ')'
                # return max(self.rdegree(subStl1,sig),self.rdegree(subStl2,sig))
                return max(self.rdegree(sub,sig) for sub in parts)
            else:
                # # predicate: remove parentheses and recurse
                # return self.rdegree(expr[1:-1],sig)
                # invalid expression
                raise Exception('Invalid operator in STL expression: ' + op)
        elif expr[0] == '!':
            # negation, remove the ! and return the negative
            subStl = expr[1:]
            return -1 * self.rdegree(subStl,sig)
        else:
            # This should be a simple inequality 'x<f' where f is a float
            f = float(expr[2:])
            # also sig should be length 1
            if len(sig) != 1:
                raise Exception("STL parsing reached predicate with more than one state in history")
            else:
                sig = sig[0]
            if 'x<' in expr:
                return f - sig[0]
            elif 'x>' in expr:
                return sig[0] - f
            elif 'y<' in expr:
                return f - sig[1]
            elif 'y>' in expr:
                return sig[1] - f
            else:
                raise Exception('Invalid stl expression: ' + str(self.expr) + ' when evaluating: ' + str(expr))

    def get_tau(self):
        # just return it if it exists
        if self.tau != None:
            return self.tau
        # PHI = K[.]phi
        # we need hrz of phi, not PHI
        end = self.expr.index(']')
        phi = self.expr[end+1:]
        # TODO assuming time step of 1
        self.tau = int(self.hrz(phi) + 1)
        return self.tau
        
    def hrz(self, phi = None):
        """
        Recursively calculate the horizon length of expression phi. Phi is the expression passed to the constructor by default.
        Valid formats for phi include K[.]a, (K[.]a), (K[.]a)&(K[.]b)&(...), p, and (p) where p is a predicate, K is either G or F, 
            and a,b are valid expressions for phi by this same definition. Additionally & is replacible with |.
        """
        if phi == None:
            phi = self.expr
        if 'G' not in phi and 'F' not in phi:
            # simple predicate
            return 0
        # account for a top level conjunction/disjunction
        # Could have  phi as 'F[.]a' or '(F[.]a)' or '(F[.]a)&(F[.]b)
        if phi[0] == '(':
            # note this could be len 1
            sub_phis = self.sep_paren(phi)
            hrz = max(self.hrz(p) for p in sub_phis)
        else:
            # phi is F[.]a
            split = phi.index(']') + 1
            outer = phi[:split]
            inner = phi[split:]
            comma = outer.index(',')
            b = float(outer[comma+1:-1])
            hrz = b + self.hrz(inner)
        return hrz
    
    def sep_paren(self, phi):
        """
        Returns a list of phrases enclosed in top level parentheses
        """
        parts = []
        depth = 0
        start = 0
        for i,c in enumerate(phi):
            if c == '(':
                depth += 1
                if depth == 1:
                    start = i+1
            elif c == ')':
                depth -= 1
                if depth == 0:
                    parts.append(phi[start:i])
                elif depth < 0:
                    raise Exception("Mismatched parentheses in STL expression!")
        if depth != 0:
            raise Exception("Mismatched parentheses in STL expression!")
        return parts



