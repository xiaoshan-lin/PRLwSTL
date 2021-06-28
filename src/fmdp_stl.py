

class FmdpState:

    def __init__(mdp_state, flags):
        self.mdp_s = mdp_state
        self.flags = flags

class FmdpStl:

    def __init__(self, stl_expr, mdp_sig_dict = None):
        self.stl_expr = stl_expr
        # format
        # limited to depth of 2 {F,G}
        # Cannot be depth 1 i.e. G[0,2]x<3
        # P[t,t]p[t,t]q
        # P[t,t]p[t,t](q|q)
        # P[t,t]((p[t,t]q)&(p[t,t]q))

        # extract the big phi
        end = stl_expr.index(']') + 1
        self.big_phi = stl_expr[:end]
        self.small_phi = stl_expr[end:]

        self.sig_dict = mdp_sig_dict

    def set_mdp_sig_dict(mdp_sig_dict):
        self.sig_dict = mdp_sig_dict

    def sat(fmdp_s, phi = None, i=0):

        if phi == None:
            phi = self.small_phi

        if self.mdp_sig_dict == None:
            raise Exception("A dictionary mapping mdp states to a signal must be defined via set_mdp_sig_dict")

        mdp_s = fmdp_s[0]
        sig = self.sig_dict[mdp_s]

        if phi[0] in 'FG':
            end = phi.index(']') + 1
            predicate = phi[end:]
            satisfies = self.sat(fmdp_s, predicate, i=None)
            flags = fmdp_s[1]
            flag_i = flags[i]

            if phi[0] == 'F':
                if flag_i > 0 or satisfies:
                    return True
                elif flag_i == 0 and not satisfies:
                    return False
                else:
                    raise Exception("Something went wrong")
            elif phi[0] == 'G':
                if flag_i == 1 and satisfies:
                    return True
                elif flag_i < 1 or not satisfies:
                    return False
                else:
                    raise Exception("Something went wrong")
            else:
                raise Exception("Something went wrong")

        elif phi[0] == '(':
            


        # extract the small phis
        if phi_1_inner[0] == '(':
            parts = self.sep_paren(phi_1_inner)

        else:
            end = phi_1_inner.index(']') + 1
            phi_2 = [phi_1_inner[:end]]
            phi_2_inner = [phi_1_inner[end:]]

            

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
