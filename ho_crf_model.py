'''
@author: ahmed allam <ahmed.allam@yale.edu>

'''
import os
from copy import deepcopy
from collections import Counter 
import numpy
from utilities import ReaderWriter, create_directory, vectorized_logsumexp
 
class HOCRFModelRepresentation(object):
    def __init__(self, modelfeatures, states, L):
        """ modelfeatures: set of features defining the model
            states: set of states (i.e. tags)
            L: length of longest segment
        """ 
        self.modelfeatures = modelfeatures
        self.modelfeatures_codebook = modelfeatures
        self.Y_codebook = states
        self.L = L
        self.Z_codebook = self.get_pattern_set()
        self.patt_order = self.get_pattern_order()
        self.P_codebook = self.get_forward_states()
        self.S_codebook = self.get_backward_states()
        self.f_transition = self.get_forward_transition()
        self.b_transition = self.get_backward_transitions()
        
        self.pi_pky_z = self.map_pky_z()
        self.si_ysk_z = self.map_sky_z()
        
        # these subsequence relations are needed for the marginal computation
        self.z_pisj = self.map_z_pisj()
        self.z_cc_zk = self.map_z_subseq_zk()
        self.z_zk_pisj = self.map_zk_subseqkprime_pizsj()
        
        self.num_features = self.get_num_features()
        self.num_states = self.get_num_states()

    def represent_globalfeatures(self, seq_featuresum):
        modelfeatures_codebook = self.modelfeatures_codebook
        windx_fval = {}
        for y_patt, seg_features in seq_featuresum.items():
            for seg_featurename in seg_features: 
                fkey = y_patt + "&&" + seg_featurename
                if(fkey in modelfeatures_codebook):
                    windx_fval[modelfeatures_codebook[fkey]] = seg_features[seg_featurename]
        return(windx_fval)

        
    def represent_activefeatures(self, z_patts, seg_features):  
        modelfeatures = self.modelfeatures
        modelfeatures_codebook = self.modelfeatures_codebook 
        activefeatures = {}
#         print("segfeatures {}".format(seg_features))
#         print("z_patts {}".format(z_patts))
        for z_patt in z_patts:
            if(z_patt in modelfeatures):
                windx_fval = {}
                for seg_featurename in seg_features:
                    if(seg_featurename in modelfeatures[z_patt]):
#                         print("seg_featurename {}".format(seg_featurename))
#                         print("z_patt {}".format(z_patt))
                        fkey = z_patt + "&&" + seg_featurename
                        windx_fval[modelfeatures_codebook[fkey]] = seg_features[seg_featurename]     
                if(z_patt in modelfeatures[z_patt]):
                    fkey = z_patt + "&&" + z_patt
                    windx_fval[modelfeatures_codebook[fkey]] = 1
                    
                if(windx_fval):
                    activefeatures[z_patt] = windx_fval
#         print("activefeatures {}".format(activefeatures))         
        return(activefeatures)
    
    @property
    def modelfeatures_codebook(self):
        return(self._modelfeatures_codebook)
    @modelfeatures_codebook.setter
    def modelfeatures_codebook(self, modelfeatures):
        codebook = {}
        code = 0
        for y_patt, featuresum in modelfeatures.items():
            for feature in featuresum:
                fkey = y_patt + "&&" + feature
                codebook[fkey] = code
                code += 1
        self._modelfeatures_codebook = codebook

    @property
    def Y_codebook(self):
        return(self._Y_codebook)
    @Y_codebook.setter     
    def Y_codebook(self, states):
        self._Y_codebook = {s:i for (i, s) in enumerate(states)}
        
    def get_pattern_set(self):
        modelfeatures = self.modelfeatures
        Z_codebook = {y_patt:index for index, y_patt in enumerate(modelfeatures)}
        return(Z_codebook)
    
    def get_pattern_order(self):
        Z_codebook = self.Z_codebook
        patt_order = {}
        for y_patt in Z_codebook:
            elems = y_patt.split("|")
            l = len(elems)
            if(l in patt_order):
                patt_order[l].append(y_patt)
            else:
                patt_order[l] = [y_patt]
        return(patt_order)  
    
    def get_forward_states(self):
        Y_codebook = self.Y_codebook
        Z_codebook = self.Z_codebook
        P = {}
        for z_patt in Z_codebook:
            elems = z_patt.split("|")
            for i in range(len(elems)-1):
                P["|".join(elems[:i+1])] = 1
        for y in Y_codebook:
            P[y] = 1
        # empty element         
        P[""] = 1
        P_codebook = {s:i for (i, s) in enumerate(P)}
        return(P_codebook) 
    
    def get_backward_states(self):
        Y_codebook = self.Y_codebook
        Z_codebook = self.Z_codebook
        S = {}
        for z_patt in Z_codebook:
            elems = z_patt.split("|")
            for i in range(1, len(elems)):
                S["|".join(elems[i:])] = 1
        for y in Y_codebook:
            S[y] = 1
        # empty element         
        S[""] = 1
        S_codebook = {s:i for (i, s) in enumerate(S)}
        return(S_codebook)
        
                
    def get_forward_transition(self):
        Y_codebook = self.Y_codebook
        P_codebook = self.P_codebook
        pk_y= {}
        for p in P_codebook:
            for y in Y_codebook:
                pk_y[(p, y)] = 1
        pk_y_suffix = {}
        for p in P_codebook:
            if(p != ""):
                for (pk, y) in pk_y:
                    ref_str = pk + "|" + y
                    check = self.check_suffix(p, ref_str)
                    if(check):
                        if((pk, y) in pk_y_suffix):
                            pk_y_suffix[(pk, y)].append(p)
                        else:
                            pk_y_suffix[(pk, y)] = [p]
                            
        pk_y_suffix = self.keep_longest_elems(pk_y_suffix)
        f_transition = {}
        for (pk, y), pi in pk_y_suffix.items():
            if(pk == ""):
                elmkey = y
            else:
                elmkey = pk + "|" + y
            if(pi in f_transition):
                f_transition[pi][elmkey] = pk
            else:
                f_transition[pi] = {elmkey:pk}
        return(f_transition)
                    
    def get_backward_transitions(self):
        Y_codebook = self.Y_codebook
        S_codebook = self.S_codebook
        sk_y = {}
        
        for s in S_codebook:
            for y in Y_codebook:
                sk_y[(s, y)] = 1
        
        sk_y_prefix = {}
        for s in S_codebook:
#             if(s != ""):
            for (sk, y) in sk_y:
                ref_str = y + "|" + sk
                check = self.check_prefix(s, ref_str)
                if(check):
                    if((sk, y) in sk_y_prefix):
                        sk_y_prefix[(sk, y)].append(s)
                    else:
                        sk_y_prefix[(sk, y)] = [s]
                            
        sk_y_prefix = self.keep_longest_elems(sk_y_prefix)
        b_transition = {}
        for (sk, y), si in sk_y_prefix.items():
            if(sk == ""):
                elmkey = y
            else:
                elmkey = y + "|" + sk
            if(si in b_transition):
                b_transition[si][elmkey] = sk
            else:
                b_transition[si] = {elmkey:sk}
        return(b_transition)    
        
    def map_pky_z(self):
        f_transition = self.f_transition
        Z_codebook = self.Z_codebook
        pi_pky_z = {}
        for pi in f_transition:
            pky_z_map = {}
            for pky in f_transition[pi]:
                l = []
                for z in Z_codebook:
                    if(self.check_suffix(z, pky)):
                        l.append(z)
                pky_z_map[pky] = l
                
            pi_pky_z[pi] = pky_z_map
        return(pi_pky_z)
    
    
    def map_sky_z(self):
        b_transition = self.b_transition
        Z_codebook = self.Z_codebook
        si_ysk_z = {}
        for si in b_transition:
            ysk_z_map = {}
            for ysk in b_transition[si]:
                l = []
                for z in Z_codebook:
                    if(self.check_prefix(z, ysk)):
                        l.append(z)
                ysk_z_map[ysk] = l
            si_ysk_z[si] = ysk_z_map
        print("b_transition {}".format(b_transition))
        print("si_ysk_z {}".format(si_ysk_z))
        return(si_ysk_z)   
       
        
    def map_z_pisj(self):
        Z_codebok = self.Z_codebook
        P_codebook = self.P_codebook
        S_codebook = self.S_codebook
        z_pisj = {}
        for z in Z_codebok:
            z_pisj[z] = {"pi":[], "sj":[]}
            elems = z.split("|")
            trunc_z = "|".join(elems[:-1])

            for p in P_codebook:
                if(self.check_suffix(trunc_z, p)):
                    z_pisj[z]["pi"].append(p)
            trunc_z = "|".join(elems[1:])
            for s in S_codebook:
                if(self.check_prefix(trunc_z, s)):
                    z_pisj[z]["sj"].append(s)
        return(z_pisj)
                     
        
#     def map_zi_ssubseqj_z(self):
#         """ determine label patterns that are strictly subsequence of z
#             it will be used to compute W_x(t,z) 
#         """
#         Z_codebook = self.Z_codebook
#         zi_cj_z = {}
#         
#         for z in Z_codebook:
#             zi_cj_z[z] = []
#             # truncate z for the strict subsequence relation
#             elems = z.split("|")
#             trunc_z = "|".join(elems[1:-1])
# #             trunc_z = "|".join(elems)
#             for zi in Z_codebook:
#                 l = self.check_subsequence(zi, trunc_z)  
#                 if(l):
#                     # add 1 since we are chopping the first element of z
#                     l = [elem+1 for elem in l]
#                     zi_cj_z[z].append((zi, l))
#                     
#         return(zi_cj_z)
#     
    def map_z_subseq_zk(self):
        """ determine label patterns that properly contain z (i.e. z subsequence of z^k) """
        Z_codebook = self.Z_codebook
        z_cc_zk = {}
        
        for z in Z_codebook:
            z_cc_zk[z] = []
            for zk in Z_codebook:
                l = self.check_subsequence(z, zk)
                if(l):
                    z_cc_zk[z].append(zk)  
        return(z_cc_zk)
     
    def construct_piz_str(self, pi, z):
        # evaluate piz part
        if(pi == ""):
            return(z)
        else:
            end = len(z.split("|")) - 1
            if(end == 0):
                return(pi + "|" + z) 
            else:
                elems = pi.split("|")
                return("|".join(elems[:-end]) + "|" + z)
            
    def construct_sj_str(self, piz, z, sj):
        # evaluate z sj part
        if(sj == ""):
            return(piz)
        else:
            start = len(z.split("|")) - 2
            if(start < 0):
                return(piz + "|" + sj)
            else:
                elems = sj.split("|")
                return(piz + "|" + "|".join(elems[start:])) 
               
    def map_zk_subseqkprime_pizsj(self):
        z_pisj = self.z_pisj
        z_cc_zk = self.z_cc_zk
        
        z_zk_pisj = {}   
            
        for z in z_pisj:
            z_zk_pisj[z] = {}
            for pi in z_pisj[z]["pi"]:
                for sj in z_pisj[z]["sj"]:
                    z_zk_pisj[z][pi, sj] = []
                    for zk in z_cc_zk[z]:
                        # construct piz part
                        piz = self.construct_piz_str(pi, z)
                        pizsj = self.construct_sj_str(piz, z, sj)
                        l = self.check_subsequence(zk, pizsj)
                        if(l):
                            z_zk_pisj[z][pi, sj].append((zk, l))

        return(z_zk_pisj)
        
            
    def keep_longest_elems(self, s):
        """ used to figure out longest suffix and prefix on sets """
        longest_elems = {}
        for tup, l in s.items():
            longest_elems[tup] = max(l, key = len)
        return(longest_elems)
                  
    def check_suffix(self, token, ref_str):
        return(ref_str.endswith(token))
    
    def check_prefix(self, token, ref_str):
        return(ref_str.startswith(token))
    
#     def check_subsequence(self, token, ref_str):
#         # find multiple occurrences of the token and the ending position of the occurrence
#         # this implementation does not consider the order of the letters (i.e. gap between letters is allowed)
#         # (eg. x= "BA" is subsequence of y = "BCDA")
#         token = token.split("|")
#         ref_str = ref_str.split("|")
#         if(len(ref_str) < len(token)):
#             return([])
#         l = []
#         found_pos = []
#         i = 0
#         j = 0
#         token_len = len(token)
#         refstr_len = len(ref_str)
#         while (i < refstr_len):
#             if(j < token_len):
#                 if(token[j] == ref_str[i]):
#                     j += 1
#                     l.append(i+1)
#                     i += 1
#                 else:
#                     i += 1
#             else:
#                 if(l):
#                     found_pos.append(l[-1])
#                     l = []
#                     j = 0  
#         if(l and len(l) == token_len):
#             found_pos.append(l[-1])    
#         return(found_pos)
# #         
    def check_subsequence(self, token, ref_str):
        # find multiple occurrences of the token and the ending position of the occurrence
        # this version considers the order of letters in token (i.e. order match)
        
        if(token not in ref_str):
            return([])
        found_pos = []
        num_elem = len(token.split("|"))
        i = 0
        pos_stack = [-1]
        while(True):
            pos = ref_str.find(token, i)
            
            if(pos != -1 and pos != pos_stack[-1]):
                if(pos == 0):
                    found_pos.append(num_elem)
                else:
                    elems_before = [elem for elem in ref_str[:pos].split("|") if elem and elem != "|"]
                    found_pos.append(len(elems_before) + num_elem)
                i = pos + 1
                pos_stack.append(pos)
            elif(pos != -1 and pos == pos_stack[-1]):
                i += 1
            else:
                break
        return(found_pos)
    
    def get_num_features(self):
        return(len(self.modelfeatures_codebook))
    def get_num_states(self):
        return(len(self.Y_codebook)) 
               
class HOCRF(object):
    def __init__(self, model, seqs_representer, seqs_info):
        self.model = model
        self.weights = numpy.zeros(model.num_features, dtype= "longdouble")
        self.seqs_representer = seqs_representer
        self.seqs_info = seqs_info
        self.load_fromdisk = False
        self.func_dict = {"alpha": self._loadalpha,
                         "beta": self._loadbeta,
                         "f_potential": self._loadfpotential,
                         "b_potential": self._loadbpotential,
                         "O_potential": self._loadOpotential,
                         "Wx_potential": self._loadWxpotential,
                         "activefeatures_by_position": self.load_activefeatures,
                         "globalfeatures": self.load_globalfeatures}

    @property
    def seqs_info(self):
        return self._seqs_info
    @seqs_info.setter
    def seqs_info(self, info_dict):
        # make a copy of the passed seqs_info dictionary
        self._seqs_info = deepcopy(info_dict)
    
    def prepare_f_potentialfeatures(self, seq_id):
        pi_pky_z = self.model.pi_pky_z
        T = self.seqs_info[seq_id]["T"]
        activefeatures = self.seqs_info[seq_id]["activefeatures_by_position"]
        f_potential_features = {}
        f_featurpattern_count = {}

        for j in range(1, T+1):
            boundary = (j, j)
            for pi in pi_pky_z:
                f_featurpattern_count[j, pi] = {}
                for pky in pi_pky_z[pi]:
                    for z_patt in pi_pky_z[pi][pky]:
                        flag = 0
                        if(z_patt in activefeatures[boundary]):
                            f_val = list(activefeatures[boundary][z_patt].values())
                            w_indx = list(activefeatures[boundary][z_patt].keys())
                            f_potential_features[j, pi, pky, z_patt] = {'f_val':f_val,
                                                                        'w_indx':w_indx}
                            flag = 1
                        
                        f_featurpattern_count[j, pi][j, z_patt] = flag

        # write on disk
        target_dir = self.seqs_info[seq_id]['activefeatures_dir']
        ReaderWriter.dump_data(f_featurpattern_count, os.path.join(target_dir, "f_featurpattern_count"))
        ReaderWriter.dump_data(f_potential_features, os.path.join(target_dir, "f_potential_features"))
        self.seqs_info[seq_id]['f_potential_features'] = f_potential_features
        self.seqs_info[seq_id]['f_featurepattern_count'] = f_featurpattern_count  


    def compute_f_potential(self, w, seq_id):
        pi_pky_z = self.model.pi_pky_z
        T = self.seqs_info[seq_id]["T"]
        f_potential_features = self.seqs_info[seq_id]['f_potential_features']
        f_potential= {}

        for j in range(1, T+1):
            f_potential[j] = {}
            for pi in pi_pky_z:
                for pky in pi_pky_z[pi]:
                    potential = 0
                    for z_patt in pi_pky_z[pi][pky]:
                        if((j, pi, pky, z_patt) in f_potential_features):
                            w_indx = f_potential_features[j, pi, pky, z_patt]['w_indx']
                            f_val = f_potential_features[j, pi, pky, z_patt]['f_val']
                            potential += numpy.dot(w[w_indx], f_val)
                    f_potential[j][pky] = potential

        return(f_potential)
    
    def compute_forward_vec(self, seq_id):
        f_potential = self.seqs_info[seq_id]["f_potential"]
        f_transition = self.model.f_transition
        P_codebook = self.model.P_codebook
        T = self.seqs_info[seq_id]["T"]
        alpha = numpy.ones((T+1,len(P_codebook)), dtype='longdouble') * (-numpy.inf)
        alpha[0,P_codebook[""]] = 0
         
        for j in range(1, T+1):
            for pi in f_transition:
                accumulator = -numpy.inf
                for pky, pk in f_transition[pi].items():
                    pk_code = P_codebook[pk]
                    potential = f_potential[j][pky]
                    accumulator = numpy.logaddexp(accumulator, potential + alpha[j-1, pk_code])
                alpha[j, P_codebook[pi]] = accumulator 
                 
        return(alpha)

    def prepare_b_potentialfeatures(self, seq_id):
        si_ysk_z = self.model.si_ysk_z
        T = self.seqs_info[seq_id]["T"]
        activefeatures = self.seqs_info[seq_id]["activefeatures_by_position"]
        b_featurepattern_count = {}
        b_potential_features = {}
        
        for j in range(1, T+1):
            for si in si_ysk_z:
                b_featurepattern_count[j, si] = {}
                for ysk in si_ysk_z[si]:
                    for z_patt in si_ysk_z[si][ysk]:
                        b = j + len(z_patt.split("|")) - 1
                        upd_boundary = (b, b)
                        flag = 0
                        if(upd_boundary in activefeatures):
                            if(z_patt in activefeatures[upd_boundary]):
                                f_val = list(activefeatures[upd_boundary][z_patt].values())
                                w_indx = list(activefeatures[upd_boundary][z_patt].keys())
                                b_potential_features[j, si, ysk, z_patt] = {'f_val':f_val,
                                                                            'w_indx':w_indx}
                                flag = 1
                                    
                        b_featurepattern_count[j, si][b, z_patt] =  flag   
        # write on disk  
        target_dir = self.seqs_info[seq_id]['activefeatures_dir']
        ReaderWriter.dump_data(b_featurepattern_count, os.path.join(target_dir, "b_featurepattern_count"))
        ReaderWriter.dump_data(b_potential_features, os.path.join(target_dir, "b_potential_features"))
        self.seqs_info[seq_id]['b_potential_features'] = b_potential_features
        self.seqs_info[seq_id]['b_featurepattern_count'] = b_featurepattern_count      
    
    def compute_b_potential(self, w, seq_id):
        si_ysk_z = self.model.si_ysk_z
        T = self.seqs_info[seq_id]["T"]
        b_potential_features = self.seqs_info[seq_id]['b_potential_features']
        b_potential= {}

        for j in range(1, T+1):
            b_potential[j] = {}
            for si in si_ysk_z:
                for ysk in si_ysk_z[si]:
                    potential = 0
                    for z_patt in si_ysk_z[si][ysk]:
                        if((j, si, ysk, z_patt) in b_potential_features):
                            w_indx = b_potential_features[j, si, ysk, z_patt]['w_indx']
                            f_val = b_potential_features[j, si, ysk, z_patt]['f_val']
                            potential += numpy.dot(w[w_indx], f_val)
                    b_potential[j][ysk] = potential
            
        return(b_potential)
    
    def compute_backward_vec(self, seq_id):
        b_potential = self.seqs_info[seq_id]["b_potential"]
        b_transition = self.model.b_transition
        S_codebook = self.model.S_codebook
        T = self.seqs_info[seq_id]["T"]
        beta = numpy.ones((T+2,len(S_codebook)), dtype='longdouble') * (-numpy.inf)

        beta[T+1, S_codebook[""]] = 0
        
        for j in reversed(range(1, T+1)):
            for si in b_transition:
                accumulator = -numpy.inf
                for ysk, sk in b_transition[si].items():
                    sk_code = S_codebook[sk]
                    potential = b_potential[j][ysk]
                    accumulator = numpy.logaddexp(accumulator, potential + beta[j+1, sk_code])
                beta[j, S_codebook[si]] = accumulator 
                    
        return(beta)
    
    def prepare_Ox_features(self, seq_id):
        z_zk_pisj = self.model.z_zk_pisj
        activefeatures = self.seqs_info[seq_id]["activefeatures_by_position"]
        T = self.seqs_info[seq_id]["T"]
        O_featurepattern_count = {}
        O_potential_features = {}
        
        for t in range(1, T+1):
            for z in z_zk_pisj:
                for pi, sj in z_zk_pisj[z]:
                    O_featurepattern_count[t, pi, sj, z] = {}
                    O_potential_features[t, pi, sj, z] = {}
                    len_elempi = len(pi.split("|"))       
                    if(pi == ""):
                        len_pi = 0
                    else:
                        len_pi = len_elempi 
                    for zk, positions in z_zk_pisj[z][pi, sj]:
                        for pos_kp in positions:
                            t_kp = t - len_pi + pos_kp - 1
                            boundary = (t_kp, t_kp)
                            if(boundary in activefeatures):
                                if(zk in activefeatures[boundary]):
                                    f_val = list(activefeatures[boundary][zk].values())
                                    w_indx = list(activefeatures[boundary][zk].keys())
                                    O_potential_features[t, pi, sj, z][t_kp, zk] = {'w_indx': w_indx,
                                                                                    'f_val': f_val}
                                    flag  = 1
                                    
                                    O_featurepattern_count[t, pi, sj, z][t_kp, zk] = flag
                    if(not O_featurepattern_count[t, pi, sj, z]):
                        del O_featurepattern_count[t, pi, sj, z]
                        del O_potential_features[t, pi, sj, z]
        # write on disk
        target_dir = self.seqs_info[seq_id]['activefeatures_dir']
        ReaderWriter.dump_data(O_featurepattern_count, os.path.join(target_dir, "O_featurepattern_count"))
        ReaderWriter.dump_data(O_potential_features, os.path.join(target_dir, "O_potential_features"))
        self.seqs_info[seq_id]['O_potential_features'] = O_potential_features
        self.seqs_info[seq_id]['O_featurepattern_count'] = O_featurepattern_count
    
    def compute_Ox(self, w, seq_id):
        z_zk_pisj = self.model.z_zk_pisj
        T = self.seqs_info[seq_id]["T"]
        O_potential_features = self.seqs_info[seq_id]["O_potential_features"]
        Ox = {}
        cached_comp = {}
        for t in range(1, T+1):
            for z in z_zk_pisj:
                for pi, sj in z_zk_pisj[z]:
                    potential = 0 
                    if((t, pi, sj, z) in O_potential_features):
                        for t_kp, zk in O_potential_features[t, pi, sj, z]:
                            if((t_kp, zk) in cached_comp):
                                potential += cached_comp[t_kp, zk]
                            else:
                                w_indx = O_potential_features[t, pi, sj, z][t_kp, zk]['w_indx']
                                f_val = O_potential_features[t, pi, sj, z][t_kp, zk]['f_val']
                                cached_comp[t_kp, zk] = numpy.dot(w[w_indx], f_val)
                                potential += cached_comp[t_kp, zk]

                    Ox[t, pi, sj, z] = potential
        return(Ox)
    
    def _detect_doublecount_features(self, A, B, C):
        featpatt_doublecount = []
        
        if(C and A and B):
            for tup in C:
                if(tup in A or tup in B):
                    featpatt_doublecount.append(tup)
        elif(C and A):
            for tup in C:
                if(tup in A):
                    featpatt_doublecount.append(tup)
        elif(C and B):
            for tup in C:
                if(tup in B):
                    featpatt_doublecount.append(tup)
        if(A and B):
            for tup in A:
                if(tup in B and tup not in featpatt_doublecount):
                    featpatt_doublecount.append(tup)
        return(featpatt_doublecount)
    
    def prepare_Wx_features(self, seq_id):
        z_zk_pisj = self.model.z_zk_pisj
        P_codebook = self.model.P_codebook
        S_codebook = self.model.S_codebook
        Z_codebook = self.model.Z_codebook
        f_featurepattern_count = self.seqs_info[seq_id]["f_featurepattern_count"]
        b_featurepattern_count = self.seqs_info[seq_id]["b_featurepattern_count"]
        O_featurepattern_count = self.seqs_info[seq_id]["O_featurepattern_count"]

#         print("f_featurepattern_count {}".format(f_featurepattern_count))
#         print("b_featurepattern_count {}".format(b_featurepattern_count))
#         print("O_featurepattern_count {}".format(O_featurepattern_count))
        Wx_potential_features = {}
        
        T = self.seqs_info[seq_id]["T"]
        
        factive = {}
        for t in range(1, T+1):
            t_a = t - 1
            for pi in P_codebook:
                if((t_a > 0 and pi == "") or (t_a == 0 and pi != "")):
                    continue
                else:
                    if((t_a, pi) in f_featurepattern_count):
                        A = f_featurepattern_count[t_a, pi]
                        pattA = {zA:1 for tA, zA in A}
    #                         print("A {}".format(A))
    #                         print("pattA {}".format(pattA))
                        addendum = []
                        for patt in pattA:
                            if(pi.endswith(patt)):
                                addendum.append(pi[:(len(pi) - len(patt) - 1)])
    #                         print("pi {} at t = {}, t_a = {} and addendum is {}".format(pi, t, t_a, addendum))
                        lenpi = len(pi.split("|"))
                        for elem in addendum:
                            if(elem):
                                offpos = t_a - (lenpi - len(elem.split("|")))
    #                                 print("offpos {}".format(offpos))
                                if((offpos, elem) in f_featurepattern_count):
                                    A.update(f_featurepattern_count[offpos, elem])
    #                                     print("A {}".format(A))
                        factive[t_a, pi] = A
                    
        bactive = {}
        for t in range(1, T+1):
            for z in Z_codebook:
                len_z = len(z.split("|"))
                t_b = t - len_z + 2
                for sj in S_codebook:
                    if((t_b > T and sj != "") or (t_b <= T and sj == "")):
                        continue
                    else:
                        if((t_b, sj) in b_featurepattern_count):
                            if((t_b, sj) not in bactive):
                                B = b_featurepattern_count[t_b, sj]
        #                         print("B {}".format(B))
                                pattB = {zB:1 for tB, zB in B}
        #                         print("pattB {}".format(pattB))
        #                             addendum = [sj[sj.find(patt)+len(patt)+1:] for patt in pattB]
                                addendum = []
                                for patt in pattB:
                                    if(sj.startswith(patt)):
                                        addendum.append(sj[len(patt)+1:])
        #                         print("sj {} at t = {}, t_b = {} and addendum is {}".format(sj, t, t_b, addendum))
        
                                lensj = len(sj.split("|"))
                                for elem in addendum:
                                    if(elem):
                                        offpos = t_b + lensj - len(elem.split("|"))
        #                                 print("offpos {}".format(offpos))
                                        if((offpos, elem) in b_featurepattern_count):
                                            B.update(b_featurepattern_count[offpos, elem])
        #                                     print("B {}".format(B))

                                bactive[t_b, sj] = B
                    
        for t in range(1, T+1):
            t_a = t - 1
            for z in z_zk_pisj:
                len_z = len(z.split("|"))
                t_b = t - len_z + 2
                for pi, sj in z_zk_pisj[z]:
                    Wx_potential_features[t, pi, sj, z] = []

#                     print("C {}".format(C))
#                     print("t = {}, pi = {}, sj = {}".format(t, pi, sj))

   
                    A = factive.get((t_a, pi))
                    B = bactive.get((t_b, sj))
                    C = O_featurepattern_count.get((t, pi, sj, z))
#                     print("C {}".format(C))
#                     print("t = {}, pi = {}, sj = {}".format(t, pi, sj))
                    doublecount = self._detect_doublecount_features(A, B, C)
                    if(doublecount):
                        Wx_potential_features[t, pi, sj, z] += doublecount
#                     print("Wx_potential_features")
        # write on disk
        target_dir = self.seqs_info[seq_id]['activefeatures_dir']
        ReaderWriter.dump_data(Wx_potential_features, os.path.join(target_dir, "Wx_potential_features"))
        self.seqs_info[seq_id]['Wx_potential_features'] = Wx_potential_features
        self.seqs_info[seq_id]['f_featurepattern_count'] = None
        self.seqs_info[seq_id]['b_featurepattern_count'] = None
        self.seqs_info[seq_id]['O_featurepattern_count'] = None
        
        
    def compute_Wx(self, w, seq_id):
        z_zk_pisj = self.model.z_zk_pisj
        activefeatures = self.seqs_info[seq_id]["activefeatures_by_position"]
        Wx_potential_features = self.seqs_info[seq_id]["Wx_potential_features"]
        Wx = {}
        cached_comp = {}
        T = self.seqs_info[seq_id]["T"]
        for t in range(1, T+1):
            for z in z_zk_pisj:
                for pi, sj in z_zk_pisj[z]:
                    potential = 0
                    if((t, pi, sj, z) in Wx_potential_features):
                        for pos, patt in set(Wx_potential_features[t, pi, sj, z]):
                            boundary = (pos, pos)
                            if(boundary in activefeatures):
                                if(patt in activefeatures[boundary]):
                                    if((pos, patt) in cached_comp):
                                        potential += cached_comp[pos, patt]
                                    else:
                                        f_val = list(activefeatures[boundary][patt].values())
                                        w_indx = list(activefeatures[boundary][patt].keys())
                                        cached_comp[pos, patt] = numpy.dot(w[w_indx], f_val)
                                        potential += cached_comp[pos, patt]
                    Wx[t, pi, sj, z] = potential
#                 print("Wx[{},{}] = {}".format(t, z, potential))
#         print("Wx {}".format(Wx))
        return(Wx)
    

    
    def compute_marginals(self, seq_id):
        P_codebook = self.model.P_codebook
        S_codebook = self.model.S_codebook
        Z_codebook = self.model.Z_codebook
        T = self.seqs_info[seq_id]["T"]
        z_pisj = self.model.z_pisj
        
        alpha = self.seqs_info[seq_id]["alpha"]
        beta = self.seqs_info[seq_id]["beta"] 
        Z = self.seqs_info[seq_id]["Z"]   
        Ox = self.seqs_info[seq_id]["Ox"]
        Wx = self.seqs_info[seq_id]["Wx"]
        
        P_marginals = numpy.zeros((T+1, len(self.model.Z_codebook)), dtype='longdouble')
         
        for t in range(1, T+1):
            for z in z_pisj:
                len_z = len(z.split("|"))
                t_a = t - 1
                t_b = t - len_z + 2
                if(t_b < 1):
                    P_marginals[t, Z_codebook[z]] = 0
                    continue
                accumulator = -numpy.inf
                for pi in z_pisj[z]["pi"]:
                    for sj in z_pisj[z]["sj"]:

                        numerator = alpha[t_a, P_codebook[pi]] + beta[t_b, S_codebook[sj]] + Ox[t, pi, sj, z] - Wx[t, pi, sj, z]
#                         print("alpha[{}, {}] + beta[{}, {}] + Ox[{}, {}, {}, {}] - Wx[{}, {}, {}, {}]= {} + {} + {} - {}= {}".format(t_a, pi,
#                                                                                                         t_b, sj,
#                                                                                                         t, pi, sj, z,
#                                                                                                         t, pi, sj, z,
#                                                                                                         alpha[t_a, P_codebook[pi]],
#                                                                                                         beta[t_b, S_codebook[sj]],
#                                                                                                         Ox[t, pi, sj, z],
#                                                                                                         Wx[t, pi, sj, z],
#                                                                                                         numerator))
                        accumulator = numpy.logaddexp(accumulator, numerator)
#                         print("accumulator = {}".format(accumulator))
                denominator = Z 
#                 print("denominator = Z = {}".format(denominator))

                P_marginals[t, Z_codebook[z]] = numpy.exp(accumulator - denominator)
#                 print("P({}, {}) = {}".format(t, z, P_marginals[t, Z_codebook[z]]))
#                 print("="*40)
#         print("active features {}".format(self.seqs_info[seq_id]["activefeatures_by_position"]))
#         print("P_marginals {}".format(P_marginals))
#         print("Ox {}".format(Ox))
#         print("z_pisj {}".format(z_pisj))
#         print("z_cc_zk {}".format(self.model.z_cc_zk))
        return(P_marginals)
    
    def compute_feature_expectation(self, seq_id):
        """ assumes that activefeatures_matrix has been already generated and saved in self.seqs_info dictionary """
        activefeatures = self.seqs_info[seq_id]["activefeatures_by_position"]
        P_marginals = self.seqs_info[seq_id]["P_marginal"]
        Z_codebook = self.model.Z_codebook
        f_expectation = {}
        for boundary, features_dict in activefeatures.items():
            t = boundary[0]
            for z_patt in features_dict:
                for w_indx, f_val in features_dict[z_patt].items():
                    if(w_indx in f_expectation):
                        f_expectation[w_indx] += f_val * P_marginals[t, Z_codebook[z_patt]]
                    else:
                        f_expectation[w_indx] = f_val * P_marginals[t, Z_codebook[z_patt]]
        return(f_expectation)
    
    def compute_seq_loglikelihood(self, w, seq_id):
        """computes the conditional log-likelihood of a sequence (i.e. p(Y|X;w)) 
           it is used as a cost function for the single sequence when trying to estimate parameters w
        """
#         print("-"*40)
#         print("... Evaluating compute_seq_loglikelihood() ...")
        
        # we need alpha and global features to be ready
        l = ("globalfeatures", "activefeatures_by_position", "f_potential", "alpha")
        self.check_cached_info(w, seq_id, l)
        
        # get the p(X;w) -- probability of the sequence under parameter w
        Z = self.seqs_info[seq_id]["Z"]
        globalfeatures = self.seqs_info[seq_id]["globalfeatures"]
        w_indx = list(globalfeatures.keys())
        f_val = list(globalfeatures.values())

        # log(p(Y|X;w))
        loglikelihood = numpy.dot(w[w_indx], f_val) - Z 
        self.seqs_info[seq_id]["loglikelihood"] = loglikelihood
        return(loglikelihood)
    
    def compute_seq_gradient(self, w, seq_id):
        """ 
           compute the gradient of conditional log-likelihood with respect to the parameters vector w
           \frac{\partial}{\partial w} p(Y|X;w)
           
           Params:
           -------
           w: vector representing current weights -- array shape (J,) where J is the total number of features
           seq_id: id of the current sequence being evaluated -- string
        """
#         print("-"*40)
#         print("... Evaluating compute_seq_gradient() ...")


        # we need alpha, beta, global features and active features per position to be ready
        l = ("globalfeatures", "activefeatures_by_position", "f_potential", "alpha", "b_potential", "beta", "O_potential", "Wx_potential")
        self.check_cached_info(w, seq_id, l)
        P_marginals = self.compute_marginals(seq_id)
        self.seqs_info[seq_id]["P_marginal"] = P_marginals
        f_expectation = self.compute_feature_expectation(seq_id)
        globalfeatures = self.seqs_info[seq_id]["globalfeatures"]
#         print("seq id {}".format(seq_id))
#         print("len(f_expectation) {}".format(len(f_expectation)))
#         print("len(globalfeatures) {}".format(len(globalfeatures)))
        
        if(len(f_expectation) < len(globalfeatures)):
            missing_features = globalfeatures.keys() - f_expectation.keys()
            addendum = {w_indx:0 for w_indx in missing_features}
            f_expectation.update(addendum)  
            print("missing features len(f_expectation) < len(globalfeatures)")
        elif(len(f_expectation) > len(globalfeatures)):
            missing_features = f_expectation.keys() - globalfeatures.keys()
            addendum = {w_indx:0 for w_indx in missing_features}
            globalfeatures.update(addendum)
            print("missing features len(f_expectation) > len(globalfeatures)")

#         print("P_marginals {}".format(P_marginals))
#         print("f_expectation {}".format(f_expectation))
#         print("globalfeatures {}".format(globalfeatures))
        
        grad = {}
        for w_indx in f_expectation:
            grad[w_indx]  = globalfeatures[w_indx] - f_expectation[w_indx]
        return(grad)
    
    def compute_seqs_loglikelihood(self, w, seqs_id):
        seqs_loglikelihood = 0
        for seq_id in seqs_id:
            seqs_loglikelihood += self.compute_seq_loglikelihood(w, seq_id)
        return(seqs_loglikelihood)
    
    def compute_seqs_gradient(self, w, seqs_id):
        seqs_grad = numpy.zeros(len(w))
        for seq_id in seqs_id:
            seq_grad = self.compute_seq_gradient(w, seq_id) 
            w_indx = list(seq_grad.keys())
            f_val = list(seq_grad.values())
            seqs_grad[w_indx] += f_val
        return(seqs_grad)
    
    
    def _loadalpha(self, w, seq_id):
        # assumes the f_potential has been loaded into seq_info
        seq_info = self.seqs_info[seq_id]
        seq_info["alpha"] = self.compute_forward_vec(seq_id)
        seq_info["Z"] = vectorized_logsumexp(seq_info["alpha"][-1,:])
#                     print("... Computing alpha probability ...")
    def _loadbeta(self, w, seq_id):
        # assumes the b_potential has been loaded into seq_info
        seq_info = self.seqs_info[seq_id]
        seq_info["beta"] = self.compute_backward_vec(seq_id)
#                     print("... Computing beta probability ...")

    def _loadfpotential(self, w, seq_id):
        # assumes the activefeatures_by_position matrix has been loaded into seq_info
        # load f_potential_features
        seq_info = self.seqs_info[seq_id]
        if(seq_info.get("f_potential_features") == "on_disk"):
            target_dir = seq_info["activefeatures_dir"]
            f_potential_features = ReaderWriter.read_data(os.path.join(target_dir, "f_potential_features"))
            seq_info["f_potential_features"] = f_potential_features
            print("loading f_potential_features from disk")
        elif(seq_info.get("f_potential_features") == None):
            self.prepare_f_potentialfeatures(seq_id)
            print("preparing f_potential_features")

        seq_info["f_potential"] = self.compute_f_potential(w, seq_id)
#                     print("... Computing f_potential ...")

    def _loadbpotential(self, w, seq_id):
        # assumes the activefeatures_by_position matrix has been loaded into seq_info
        seq_info = self.seqs_info[seq_id]
        if(seq_info.get("b_potential_features") == "on_disk"):
            target_dir = seq_info["activefeatures_dir"]
            b_potential_features = ReaderWriter.read_data(os.path.join(target_dir, "b_potential_features"))
            seq_info["b_potential_features"] = b_potential_features
            print("loading b_potential_features from disk")

        elif(seq_info.get("b_potential_features") == None):
            self.prepare_b_potentialfeatures(seq_id)
            print("preparing b_potential_features")

        seq_info["b_potential"] = self.compute_b_potential(w, seq_id)
#                     print("... Computing b_potential ...")

    def _loadOpotential(self, w, seq_id):
        seq_info = self.seqs_info[seq_id]
        if(seq_info.get("O_potential_features") == "on_disk"):
            target_dir = seq_info["activefeatures_dir"]
            O_potential_features = ReaderWriter.read_data(os.path.join(target_dir, "O_potential_features"))
            seq_info["O_potential_features"] = O_potential_features
            print("loading O_potential_features from disk")

        elif(seq_info.get("O_potential_features") == None):
            self.prepare_Ox_features(seq_id)
            print("preparing O_potential_features")

        seq_info["Ox"] = self.compute_Ox(w, seq_id)
#                     print("... Computing Ox ...")
        
    def _loadWxpotential(self, w, seq_id):
        seq_info = self.seqs_info[seq_id]
        if(seq_info.get("Wx_potential_features") == "on_disk"):
            target_dir = seq_info["activefeatures_dir"]
            Wx_potential_features = ReaderWriter.read_data(os.path.join(target_dir, "Wx_potential_features"))
            seq_info["Wx_potential_features"] = Wx_potential_features
            print("loading Wx_potential_features from disk")

        elif(seq_info.get("Wx_potential_features") == None):
            self.prepare_Wx_features(seq_id)
            print("preparing Wx_potential_features")
        
        seq_info["Wx"] = self.compute_Wx(w, seq_id)
#                     print("... Computing Wx ...")

    def _loadflaty(self, w, seq_id):
        seq = self._load_seq(seq_id, target="seq")
        self.seqs_info[seq_id]['flat_y'] = seq.flat_y

    def load_activefeatures(self, w, seq_id):
        # get the sequence model active features
        seqs_representer = self.seqs_representer
        seqs_activefeatures = seqs_representer.get_seqs_modelactivefeatures([seq_id], self.seqs_info)
        self.seqs_info[seq_id]["activefeatures_by_position"] = seqs_activefeatures[seq_id]
        
    def load_globalfeatures(self, w, seq_id):
        # get sequence global features
        seqs_representer = self.seqs_representer
        seqs_globalfeatures = seqs_representer.get_seqs_globalfeatures([seq_id], self.seqs_info, self.model)
        self.seqs_info[seq_id]["globalfeatures"] = seqs_globalfeatures[seq_id]
        
    def load_imposter_globalfeatures(self, seq_id, y_imposter, seg_other_symbol):
        # get sequence global features
        seqs_representer = self.seqs_representer
        imposter_globalfeatures = seqs_representer.get_imposterseq_globalfeatures(seq_id, self.seqs_info, self.model, y_imposter, seg_other_symbol)
        return(imposter_globalfeatures)
        
    def _load_seq(self, seq_id, target = "seq"):
        seqs_representer = self.seqs_representer
        seq = seqs_representer.load_seq(seq_id, self.seqs_info)
        if(target == "seq"):
            return(seq)
        elif(target == "Y"):
            return(seq.Y)
        elif(target == "X"):
            return(seq.X)
        
    def check_cached_info(self, w, seq_id, entity_names):
        """ order of args is very important
            to compute 1 - alpha matrix we need to compute f_potential before
                       2 - beta matrix we need to compute the b_potential before
            
        """
        seq_info = self.seqs_info[seq_id]
        func_dict = self.func_dict
        none_type = type(None) 
        for varname in entity_names:
            if(type(seq_info.get(varname)) == none_type):
                func_dict[varname](w, seq_id)

    def clear_cached_info(self, seqs_id, cached_entities = []):
        default_entitites = ["f_potential", "alpha", "Z", "b_potential", "beta", "Ox", "Wx", "P_marginal"]
        args = default_entitites  + cached_entities
        for seq_id in seqs_id:
            seq_info = self.seqs_info[seq_id]
            for varname in args:
                if(varname in seq_info):
                    seq_info[varname] = None
        if(self.load_fromdisk):
            args = ("f_potential_features", "b_potential_features", "O_potential_features", "Wx_potential_features")
            for seq_id in seqs_id:
                seq_info = self.seqs_info[seq_id]
                for varname in args:
                    if(varname in seq_info):
                        seq_info[varname] = "on_disk"
        
    def save_model(self, file_name):
        # to clean things before pickling the model
        self.seqs_info.clear() 
        ReaderWriter.dump_data(self, file_name)
    
    def decode_seqs(self, decoding_method, out_dir, **kwargs):
        """ seqs: a list comprising of sequences that are instances of SequenceStrcut() class
            method: a string referring to type of decoding {'viterbi', 'per_state_decoding'}
            seqs_info: dictionary containing the info about the sequences to parse
        """
        corpus_name = "decoding_seqs"
        out_file = os.path.join(create_directory(corpus_name, out_dir), "decoded.txt")
        w = self.weights
        
        if(decoding_method == "viterbi"):
            decoder = self.viterbi
        if(kwargs.get("seqs_info")):
            self.seqs_info = kwargs["seqs_info"]
        elif(kwargs.get("seqs")): 
            seqs = kwargs["seqs"]           
            seqs_dict = {i+1:seqs[i] for i in range(len(seqs))}
            seqs_info = self.seqs_representer.prepare_seqs(seqs_dict, corpus_name, out_dir, unique_id = True)
            self.seqs_info = seqs_info

        seqs_pred = {}
        seqs_info = self.seqs_info
        for seq_id in seqs_info:
            Y_pred = decoder(w, seq_id)
            seq = ReaderWriter.read_data(os.path.join(seqs_info[seq_id]["globalfeatures_dir"], "sequence"))
            self.write_decoded_seqs([seq], [Y_pred], out_file)
            seqs_pred[seq_id] = {'seq': seq,'Y_pred': Y_pred}
        # clear seqs_info
        self.seqs_info.clear()
        return(seqs_pred)

            
    def write_decoded_seqs(self, ref_seqs, Y_pred_seqs, out_file):
        sep = " "
        for i in range(len(ref_seqs)):
            Y_pred_seq = Y_pred_seqs[i]
            ref_seq = ref_seqs[i]
            T = ref_seq.T
            line = ""
            for t in range(1, T+1):
                for field_name in ref_seq.X[t]:
                    line += ref_seq.X[t][field_name] + sep
                line += Y_pred_seq[t-1]
                if(ref_seq.flat_y):
                    line += sep + ref_seq.flat_y[t-1] + "\n"
                else:
                    line += "\n" 

            ReaderWriter.log_progress(line,out_file)  
            

    def viterbi(self, w, seq_id):
        self.check_cached_info(w, seq_id, "globalfeatures", "activefeatures_by_position", "f_potential")
        f_potential = self.seqs_info[seq_id]["f_potential"]
        print("f_potential \n {}".format(f_potential))
        f_transition = self.model.f_transition
        P_codebook = self.model.P_codebook
        T = self.seqs_info[seq_id]["T"]
        L = self.model.L
        # records max score at every time step
        delta = numpy.ones((T+1,len(P_codebook)), dtype='longdouble') * (-numpy.inf)
        # the score for the empty sequence at time 0 is 1
        delta[0,P_codebook[""]] = 0
        back_track = {}
        for j in range(1, T+1):
            for pi in f_transition:
                max_val = -numpy.inf
                for d in range(L):
                    u = j - d
                    v = j
                    if(u <= 0):
                        break
                    boundary = (u, v)
                    for pky, pk in f_transition[pi].items():
                        pk_code = P_codebook[pk]
                        potential = f_potential[boundary][pky]
                        score = potential + delta[u-1, pk_code]
                        if(score > max_val):
                            max_val = score
                            back_track[(j,P_codebook[pi])] = (d, pk, pky[-1])
                            
                delta[j, P_codebook[pi]] = max_val
        print("delta {}".format(delta))
        print("backtrack {}".format(back_track)) 
        # decoding the sequence
        p_T_code = numpy.argmax(delta[T,:])
        d, p_T, y_T = back_track[(T, p_T_code)]
        Y_decoded = []
        for _ in range(d+1):
            Y_decoded.append((p_T,y_T))
        t = T - d - 1
        while t>0:
            print("t {}".format(t))
            p_tplus1 = Y_decoded[-1][0]
            print("p_tplus1 {}".format(p_tplus1))
            print("p_tplus1 coded {}".format(P_codebook[p_tplus1]))
            d, p_t, y_t = back_track[(t, P_codebook[p_tplus1])]
            for _ in range(d+1):
                Y_decoded.append((p_t, y_t))
            t = t-d-1
        Y_decoded.reverse()

        print("decoding sequence with id {} \n".format(seq_id))
        print("Y_decoded {}".format(Y_decoded))
        self.clear_cached_info([seq_id])
        Y_decoded = [yt for (pt,yt) in Y_decoded]
        print("Y_decoded {}".format(Y_decoded))
        return(Y_decoded)


    def check_gradient(self, w, seq_id):
        """ implementation of finite difference method similar to scipy.optimize.check_grad()
        """
        modelfeatures_codebook_rev = {code:feature for feature, code in self.model.modelfeatures_codebook.items()}

        self.clear_cached_info([seq_id])
        epsilon = 1e-9
        # basis vector
        ei = numpy.zeros(len(w), dtype="longdouble")
        grad = numpy.zeros(len(w), dtype="longdouble")
        for i in range(len(w)):
            print("weight index is {}".format(i))
            print("feature name is {}".format(modelfeatures_codebook_rev[i]))
            ei[i] = epsilon
            l_wplus = self.compute_seq_loglikelihood(w + ei, seq_id)
            print("l_wplus loglikelihood {}".format(l_wplus))
            self.clear_cached_info([seq_id])
            l = self.compute_seqs_loglikelihood(w, [seq_id])
            print("l loglikelihood {}".format(l))
            self.clear_cached_info([seq_id])
            grad[i] = (l_wplus - l) / epsilon
            print("grad is {}".format(grad[i]))
            ei[i] = 0
        estimated_grad = self.compute_seqs_gradient(w, [seq_id])
        print("finite difference gradient: \n {}".format(grad))
        print("Computed gradient: \n {}".format(estimated_grad))
        diff = numpy.absolute(-grad + estimated_grad)
        avg_diff = numpy.mean(diff)
        print("difference between both gradients: \n {}".format(diff))
        print("average difference = {}".format(numpy.mean(avg_diff)))
        # clear seq_id info
        self.clear_cached_info([seq_id])
        return(avg_diff)
        
#     def validate_forward_backward_pass(self, w, seq_id):
#         self.clear_cached_info([seq_id])
#         # this will compute alpha and beta matrices and save them in seqs_info dict
#         l = ("activefeatures_by_position", "f_potential", "alpha", "b_potential", "beta")
#         self.check_cached_info(w, seq_id, l)
#         states_code = [code for code in self.model.Y_codebook.values()]
#         alpha = self.seqs_info[seq_id]["alpha"]
#         beta = self.seqs_info[seq_id]["beta"]
#         Z_alpha = vectorized_logsumexp(alpha[-1,:])
#         raw_diff = numpy.abs(Z_alpha - beta[1, :])
#         print("alpha[-1,:] = {}".format(alpha[-1,:]))
#         print("beta[1,:] = {}".format(beta[1,:]))
#         print("Z_alpha : {}".format(Z_alpha))
#         print("beta of states {}".format(beta[1, states_code]))
#         print("Z_aplha - beta [1,:] {}".format(raw_diff))
# 
#         min_raw_diff = numpy.min(raw_diff)
#         print("min_raw_diff  {}".format(min_raw_diff))
#         argmin_raw_diff = numpy.argmin(raw_diff)
#         print("argmin_raw_diff  {}".format(argmin_raw_diff))
#         rel_diff = min_raw_diff/(Z_alpha + beta[1, argmin_raw_diff])
#         print("rel_diff : {}".format(rel_diff))
#         self.clear_cached_info([seq_id])
#         
#         return((min_raw_diff, rel_diff))
 
    def validate_forward_backward_pass(self, w, seq_id):
        self.clear_cached_info([seq_id])
        # this will compute alpha and beta matrices and save them in seqs_info dict
        l = ("activefeatures_by_position", "f_potential", "alpha", "b_potential", "beta")
        self.check_cached_info(w, seq_id, l)
        alpha = self.seqs_info[seq_id]["alpha"]
        beta = self.seqs_info[seq_id]["beta"]
        print("states codebook {}".format(self.model.Y_codebook))
        print("alpha {}".format(alpha))
        print("beta {}".format(beta))
        
        Z_alpha = vectorized_logsumexp(alpha[-1,:])
        Z_beta = vectorized_logsumexp(beta[1, :])
        raw_diff = numpy.abs(Z_alpha - Z_beta)
        print("alpha[-1,:] = {}".format(alpha[-1,:]))
        print("beta[0,:] = {}".format(beta[0,:]))
        print("beta[1,:] = {}".format(beta[1,:]))
        print("Z_alpha : {}".format(Z_alpha))
        print("Z_beta : {}".format(Z_beta))
        print("Z_aplha - Z_beta {}".format(raw_diff))
 
        rel_diff = raw_diff/(Z_alpha + Z_beta)
        print("rel_diff : {}".format(rel_diff))
        self.clear_cached_info([seq_id])
        print("seqs_info {}".format(self.seqs_info))
        return((raw_diff, rel_diff))   
    
    def validate_expected_featuresum(self, w, seqs_id):
        self.clear_cached_info(seqs_id)
        grad = self.compute_seqs_gradient(w, seqs_id)
        avg_diff = numpy.mean(grad)
        print("difference between empirical feature sum and model's expected feature sum: \n {}".format(grad))
        print("average difference is {}".format(avg_diff))
        self.clear_cached_info(seqs_id)
        return(avg_diff)
