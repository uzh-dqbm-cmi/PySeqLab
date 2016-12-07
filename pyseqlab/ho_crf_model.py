'''
@author: ahmed allam <ahmed.allam@yale.edu>

'''

import os
from copy import deepcopy
from collections import OrderedDict
import numpy
from .utilities import AStarSearcher, ReaderWriter, create_directory, vectorized_logsumexp

class HOCRFModelRepresentation(object):
    def __init__(self):
        """model representation class that will hold data structures to be used in HCRF class
        """ 
        self.modelfeatures = None
        self.modelfeatures_codebook = None
        self.Y_codebook = None
        self.L = None
        self.Z_codebook = None
        self.Z_lendict = None
        self.Z_elems = None
        self.Z_numchar= None
        self.patts_len = None
        self.max_patt_len = None
        self.modelfeatures_inverted = None
        self.ypatt_features = None
        self.ypatt_activestates = None
        self.P_codebook = None
        self.P_codebook_rev = None
        self.pi_lendict = None
        self.S_codebook = None
        self.si_lendict = None
        self.si_numchar = None
        self.f_transition = None
        self.b_transition = None
        self.pky_codebook = None
        self.pky_codebook_rev = None
        self.pi_pky_codebook = None
        self.ysk_codebook = None
        self.si_ysk_codebook = None
        self.pky_z = None
        self.ysk_z = None
        self.num_features = None
        self.num_states = None
        
    def create_model(self, modelfeatures, states, L):
        """modelfeatures: set of features defining the model
           states: set of states (i.e. tags)
           L: length of longest segment
        """
        self.modelfeatures = modelfeatures
        self.modelfeatures_codebook = self.get_modelfeatures_codebook()
        self.Y_codebook = self.get_modelstates_codebook(states)
        self.L = L
        self.generate_instance_properties()
    
    def generate_instance_properties(self):
        self.Z_codebook = self.get_Z_pattern()
        self.Z_lendict, self.Z_elems, self.Z_numchar = self.get_Z_elems_info()
        self.patts_len = set(self.Z_lendict.values())
        self.max_patt_len = max(self.patts_len)

        self.modelfeatures_inverted, self.ypatt_features = self.get_inverted_modelfeatures()
        self.ypatt_activestates = self.find_activated_states(self.ypatt_features, self.patts_len)
        
        self.P_codebook = self.get_forward_states()
        self.P_codebook_rev = self.get_P_codebook_rev()
        self.P_len = len(self.P_codebook)
        self.pi_lendict, self.pi_elems, self.pi_numchar = self.get_pi_info()
        
        
        self.S_codebook = self.get_backward_states()
        self.si_lendict, self.si_numchar = self.get_len_si()
        
        self.f_transition = self.get_forward_transition()
        self.b_transition = self.get_backward_transitions()
        
        self.pky_codebook = self.get_pky_codebook()
        self.pky_codebook_rev = self.get_pky_codebook_rev()
        self.pi_pky_codebook = self.get_pi_pky_codebook()
        self.ysk_codebook = self.get_ysk_codebook()
        self.si_ysk_codebook = self.get_si_ysk_codebook()
        
        self.pky_z = self.map_pky_z()
        self.ysk_z = self.map_sky_z()
        
        self.num_features = self.get_num_features()
        self.num_states = self.get_num_states()

    def get_modelfeatures_codebook(self):
        modelfeatures = self.modelfeatures
        codebook = {}
        code = 0
        for y_patt, featuresum in modelfeatures.items():
            for feature in featuresum:
                fkey = y_patt + "&&" + feature
                codebook[fkey] = code
                code += 1
        return(codebook)


    def get_modelstates_codebook(self, states):
        return({s:i for (i, s) in enumerate(states)})
        
    def get_Z_pattern(self):
        modelfeatures = self.modelfeatures
        Z_codebook = {y_patt:index for index, y_patt in enumerate(modelfeatures)}
        return(Z_codebook)
    
    def get_Z_elems_info(self):
        Z_codebook = self.Z_codebook
        Z_lendict = {}
        Z_split_elems = {}
        Z_nchar = {}
        for z in Z_codebook:
            elems = z.split("|")
            Z_lendict[z] = len(elems)
            Z_split_elems[z] = elems
            Z_nchar[z] = len(z)            
        return(Z_lendict, Z_split_elems, Z_nchar)
    

    def get_inverted_modelfeatures(self):
        modelfeatures = self.modelfeatures
        Z_lendict = self.Z_lendict
        inverted_segfeatures = {}
        ypatt_features = set()
        
        for y_patt, featuresum in modelfeatures.items():
            z_len = Z_lendict[y_patt]
            # get features that are based only on y_patts
            if(y_patt in featuresum):
                ypatt_features.add(y_patt)
            for feature in featuresum:
                if(feature in inverted_segfeatures):
                    if(z_len in inverted_segfeatures[feature]):
                        inverted_segfeatures[feature][z_len].add(y_patt)
                    else:
                        s = set()
                        s.add(y_patt)                      
                        inverted_segfeatures[feature][z_len] = s
                else:
                    s = set()
                    s.add(y_patt)
                    inverted_segfeatures[feature] = {z_len:s}
        return(inverted_segfeatures, ypatt_features)

    def get_forward_states(self):
        Y_codebook = self.Y_codebook
        Z_elems = self.Z_elems
        Z_lendict = self.Z_lendict
        P = {}
        for z_patt in Z_elems:
            elems = Z_elems[z_patt]
            z_len = Z_lendict[z_patt]
            for i in range(z_len-1):
                P["|".join(elems[:i+1])] = 1
        for y in Y_codebook:
            P[y] = 1
        # empty element         
        P[""] = 1
        P_codebook = {s:i for (i, s) in enumerate(P)}
        #print("P_codebook ", P_codebook)
        return(P_codebook) 
    
    def get_P_codebook_rev(self):
        P_codebook = self.P_codebook
        P_codebook_rev = {code:pi for pi, code in P_codebook.items()}
        return(P_codebook_rev)
    
    def get_pi_info(self): 
        P_codebook = self.P_codebook
        pi_lendict = {}
        pi_numchar = {}
        pi_elems = {}
        
        for pi in P_codebook:
            elems = pi.split("|")
            pi_elems[pi] = elems 
            if(pi == ""):
                pi_lendict[pi] = 0
                pi_numchar[pi] = 0
                
            else:
                pi_lendict[pi] = len(elems)
                pi_numchar[pi] = len(pi)
        return(pi_lendict, pi_elems, pi_numchar)
    
    
    def get_backward_states(self):
        Y_codebook = self.Y_codebook
        Z_elems = self.Z_elems
        Z_lendict = self.Z_lendict
        S = {}
        for z_patt in Z_elems:
            elems = Z_elems[z_patt]
            z_len = Z_lendict[z_patt]
            #print("z_patt")
            for i in range(1, z_len):
                S["|".join(elems[i:])] = 1
                #print("i = {}".format(i))
                #print("suffix {}".format("|".join(elems[i:])))
        for y in Y_codebook:
            S[y] = 1
        # empty element         
        S[""] = 1
        S_codebook = {s:i for (i, s) in enumerate(S)}
        #print("S_codebook ", S_codebook)
        return(S_codebook)
    
    def get_len_si(self): 
        S_codebook = self.S_codebook
        si_lendict = {}
        si_numchar = {}
        for si in S_codebook:
            if(si == ""):
                si_lendict[si] = 0
                si_numchar[si] = 0
            else:
                si_lendict[si] = len(si.split("|"))
                si_numchar[si] = len(si)
        return(si_lendict, si_numchar)
                
    def get_forward_transition(self):
        Y_codebook = self.Y_codebook
        P_codebook = self.P_codebook
        pi_numchar = self.pi_numchar
        Z_numchar = self.Z_numchar
        #print("pi_numchar ", pi_numchar)
        #print("z_numchar ", Z_numchar)
        pk_y= {}
        for p in P_codebook:
            for y in Y_codebook:
                pk_y[(p, y)] = 1
        #print("pky ", pk_y)
        pk_y_suffix = {}
        for p in P_codebook:
            if(p != ""):
                len_p = pi_numchar[p]
                for (pk, y) in pk_y:
                    ref_str = pk + "|" + y
                    if(pk == ""):
                        len_ref = Z_numchar[y] + 1
                    else:
                        len_ref = pi_numchar[pk] + Z_numchar[y] + 1
                    #print("ref_str ", ref_str)
                    #print("p ", p)
                    #print("len ref_str ", len_ref)
                    #print("len p ", len_p)
                    start_pos = len_ref - len_p
                    #print("start_pos ", start_pos)
                    #print("slice ", ref_str[start_pos:])
                    #print("="*8)
                    if(start_pos>=0):
                        # check suffix relation
                        check = ref_str[start_pos:] == p
                        #print("check ", check)
                        #check = self.check_suffix(p, ref_str)
                        if(check):
                            if((pk, y) in pk_y_suffix):
                                pk_y_suffix[(pk, y)].append(p)
                            else:
                                pk_y_suffix[(pk, y)] = [p]
                            
        pk_y_suffix = self.keep_longest_elems(pk_y_suffix)
        #print("pk_y_suffix ", pk_y_suffix)
        f_transition = {}
        
        for (pk, y), pi in pk_y_suffix.items():
            if(pk == ""):
                elmkey = y
            else:
                elmkey = pk + "|" + y
            if(pi in f_transition):
                f_transition[pi][elmkey] = (pk, y)
            else:
                f_transition[pi] = {elmkey:(pk, y)}
        #print("f_transition ", f_transition)
        return(f_transition)

    def get_backward_transitions(self):
        Y_codebook = self.Y_codebook
        S_codebook = self.S_codebook
        si_numchar = self.si_numchar
        sk_y = {}
        
        for s in S_codebook:
            for y in Y_codebook:
                sk_y[(s, y)] = 1
        
        sk_y_prefix = {}
        for s in S_codebook:
#             if(s != ""):
            len_s = si_numchar[s]
            for (sk, y) in sk_y:
                ref_str = y + "|" + sk
                #check prefix relation
                check = ref_str[:len_s] == s
                #check = self.check_prefix(s, ref_str)
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
    
    def get_pky_codebook(self):
        f_transition = self.f_transition
        pky_codebook = {}
        counter = 0
        for pi in f_transition:
            for pky in f_transition[pi]:
                pky_codebook[pky] = counter
                counter += 1
        return(pky_codebook)
    
    def map_pky_z(self):
        f_transition = self.f_transition
        Z_codebook = self.Z_codebook
        # given that we demand to have a unigram label features then Z set will always contain Y elems
        Z_numchar = self.Z_numchar
        pi_numchar = self.pi_numchar
        pky_codebook = self.pky_codebook
        
        z_pky = {}
        for pi in f_transition:
            for pky, pk_y_tup in f_transition[pi].items():
                pk, y = pk_y_tup
                # get number of characters in the pky 
                if(pk == ""):
                    len_pky =  Z_numchar[y]
                else:
                    # +1 is for the separator '|'
                    len_pky = pi_numchar[pk] + Z_numchar[y] + 1
                
                for z in Z_codebook:
                    len_z = Z_numchar[z]
                    # check suffix relation
                    start_pos = len_pky - len_z
                    if(start_pos >= 0):
                        check = pky[start_pos:] == z
                        if(check):
                            pky_c = pky_codebook[pky]
                            if(z in z_pky):
                                z_pky[z].append(pky_c)
                            else:
                                z_pky[z] = [pky_c]
        return(z_pky)    
    
    def map_sky_z(self):
        b_transition = self.b_transition
        Z_codebook = self.Z_codebook
        Z_numchar = self.Z_numchar
        ysk_codebook = self.ysk_codebook
        z_ysk = {}
        
        for si in b_transition:
            for ysk in b_transition[si]:
                for z in Z_codebook:
                    len_z = Z_numchar[z]
                    #check prefix relation
                    check = ysk[:len_z] == z
                    if(check):
                        ysk_c = ysk_codebook[ysk]
                        if(z in z_ysk):
                            z_ysk[z].append(ysk_c)
                        else:
                            z_ysk[z] = [ysk_c]
        return(z_ysk)  
    

    
    def get_ysk_codebook(self):
        b_transition = self.b_transition
        ysk_codebook = {}
        counter = 0
        for si in b_transition:
            for ysk in b_transition[si]:
                ysk_codebook[ysk] = counter
                counter += 1
        return(ysk_codebook)
    
    def get_pky_codebook_rev(self):
        # to consider adding it as instance variable in the model representation
        pky_codebook_rev = {code:pky for pky, code in self.pky_codebook.items()}
        return(pky_codebook_rev)
    
    def get_pi_pky_codebook(self):
        f_transition = self.f_transition
        pky_codebook = self.pky_codebook
        P_codebook = self.P_codebook
        pi_pky_codebook = {}
        for pi in f_transition:
            pi_pky_codebook[pi]=([],[])
            for pky, (pk, _) in f_transition[pi].items():
                pi_pky_codebook[pi][0].append(pky_codebook[pky])
                pi_pky_codebook[pi][1].append(P_codebook[pk])

        return(pi_pky_codebook)
    
    def get_si_ysk_codebook(self):
        b_transition = self.b_transition
        ysk_codebook = self.ysk_codebook
        S_codebook = self.S_codebook
        si_ysk_codebook = {}
        for si in b_transition:
            si_ysk_codebook[si] = ([],[])
            for ysk, sk in b_transition[si].items():
                si_ysk_codebook[si][0].append(ysk_codebook[ysk])
                si_ysk_codebook[si][1].append(S_codebook[sk])

        return(si_ysk_codebook)
    
        
    def keep_longest_elems(self, s):
        """ used to figure out longest suffix and prefix on sets """
        longest_elems = {}
        for tup, l in s.items():
            longest_elems[tup] = max(l, key = len)
        return(longest_elems)
                  
    def check_suffix(self, token, ref_str):
        # check if ref_str ends with the token
#         return(ref_str[len(ref_str)-len(token):] == token)
        return(ref_str.endswith(token))
    
    def check_prefix(self, token, ref_str):
        # check if ref_str starts with a token
#         return(ref_str[:len(token)] == token)
        return(ref_str.startswith(token))
    
    def get_num_features(self):
        return(len(self.modelfeatures_codebook))
    
    def get_num_states(self):
        return(len(self.Y_codebook)) 
    
    def represent_globalfeatures(self, seq_featuresum):
        modelfeatures_codebook = self.modelfeatures_codebook
        windx_fval = {}
        for y_patt, seg_features in seq_featuresum.items():
            for seg_featurename in seg_features: 
                fkey = y_patt + "&&" + seg_featurename
                if(fkey in modelfeatures_codebook):
                    windx_fval[modelfeatures_codebook[fkey]] = seg_features[seg_featurename]
        return(windx_fval)

        
    def represent_activefeatures(self, activestates, seg_features):  
        modelfeatures = self.modelfeatures
        modelfeatures_codebook = self.modelfeatures_codebook   
        Z_codebook = self.Z_codebook      
        activefeatures = {}
#         print("segfeatures {}".format(seg_features))
#         print("z_patts {}".format(z_patts))
        for z_len in activestates:
            z_patt_set = activestates[z_len]
            for z_patt in z_patt_set:
#                 print("z_patt ", z_patt)
                windx_fval = {}
                for seg_featurename in seg_features:
                    # this condition might be omitted 
                    if(seg_featurename in modelfeatures[z_patt]):
    #                         print("seg_featurename {}".format(seg_featurename))
    #                         print("z_patt {}".format(z_patt))
                        fkey = z_patt + "&&" + seg_featurename
                        #print(fkey)
                        windx_fval[modelfeatures_codebook[fkey]] = seg_features[seg_featurename]     
                if(z_patt in modelfeatures[z_patt]):
                    fkey = z_patt + "&&" + z_patt
                    windx_fval[modelfeatures_codebook[fkey]] = 1
                    #print(fkey)
                    
                if(windx_fval):
                    #activefeatures[Z_codebook[z_patt]] = windx_fval
                    activefeatures[z_patt] = windx_fval

#         print("activefeatures {}".format(activefeatures))         
        return(activefeatures)
    
    def find_activated_states(self, seg_features, allowed_z_len):
        modelfeatures_inverted = self.modelfeatures_inverted
        active_states = {}
        for feature in seg_features:
            if(feature in modelfeatures_inverted):
                factivestates = modelfeatures_inverted[feature]
                for z_len in factivestates:
                    if(z_len in allowed_z_len):
                        if(z_len in active_states):
                            active_states[z_len].update(factivestates[z_len])
                        else:
                            active_states[z_len] = set(factivestates[z_len])
                #print("active_states from func ", active_states)
        return(active_states)

    def filter_activated_states(self, activated_states, accum_active_states, pos):
        Z_elems = self.Z_elems
        filtered_activestates = {}
        
        for z_len in activated_states:
            if(z_len == 1):
                continue
            start_pos = pos - z_len + 1
            if(start_pos in accum_active_states):
                filtered_activestates[z_len] = set()
                for z_patt in activated_states[z_len]:
                    check = True
                    zelems = Z_elems[z_patt]
                    for i in range(z_len):
                        if(start_pos+i not in accum_active_states):
                            check = False
                            break
                        if(zelems[i] not in accum_active_states[start_pos+i]):
                            check = False
                            break
                    if(check):                        
                        filtered_activestates[z_len].add(z_patt)
        return(filtered_activestates)
    
               
class HOCRF(object):
    def __init__(self, model, seqs_representer, seqs_info, load_info_fromdisk = 3):
        self.model = model
        self.weights = numpy.zeros(model.num_features, dtype= "longdouble")
        self.seqs_representer = seqs_representer
        self.seqs_info = seqs_info
        self.func_dict = {"alpha": self._load_alpha,
                         "beta": self._load_beta,
                         "activated_states": self.load_activatedstates,
                         "seg_features": self.load_segfeatures,
                         "globalfeatures": self.load_globalfeatures,
                         "globalfeatures_per_boundary": self.load_globalfeatures,
                         "Y":self._load_Y}
        
        self.def_cached_entities = self.cached_entitites(load_info_fromdisk)
        # default beam size covers all the prefix values (i.e. pi)
        self.beam_size = len(self.model.P_codebook)

    def cached_entitites(self, load_info_fromdisk):
        ondisk_info = ["seg_features", "activated_states", "globalfeatures_per_boundary", "globalfeatures", "Y"]
        inmemory_info = ["alpha", "Z", "beta", "activefeatures_per_boundary"]
        def_cached_entities = ondisk_info[:load_info_fromdisk]
        def_cached_entities += inmemory_info
        return(def_cached_entities)
    
    @property
    def seqs_info(self):
        return self._seqs_info
    @seqs_info.setter
    def seqs_info(self, info_dict):
        # make a copy of the passed seqs_info dictionary
        self._seqs_info = deepcopy(info_dict)
    
    
    def identify_activefeatures(self, seq_id, boundary, accum_activestates):
        model = self.model
        state_len = 1
        # get activated states per boundary
        activated_states = self.seqs_info[seq_id]['activated_states'][boundary]
        seg_features = self.seqs_info[seq_id]['seg_features'][boundary]
        #^print("boundary ", boundary)
        #^print('seg_features ', seg_features)
        #^print('activated_states ', activated_states)
        #^print("accum_activestates ", accum_activestates)
        u, v = boundary
    
        if(state_len in activated_states):
            accum_activestates[v] = set(activated_states[state_len])
            
            if(boundary != (1,1)):
                filtered_states =  model.filter_activated_states(activated_states, accum_activestates, u)
                filtered_states[state_len] = set(activated_states[state_len])
            # initial point t0
            else:
                filtered_states = activated_states
            
            #print("filtered_states ", filtered_states)
            #print("seg_features ", seg_features)        
            active_features = model.represent_activefeatures(filtered_states, seg_features)

        else:
            accum_activestates[v] = set()
            active_features = {}
        
        return(active_features)        

    def compute_fpotential(self, w, active_features):
        """ compute the potential of active features at a defined boundary """
        model = self.model
        pky_codebook = model.pky_codebook
        z_pky = model.pky_z
        f_potential = numpy.zeros(len(pky_codebook))

        # to consider caching the w_indx and fval as in cached_pf
        for z in active_features:
            w_indx = list(active_features[z].keys())
            f_val = list(active_features[z].values())
            potential = numpy.inner(w[w_indx], f_val)
            # get all pky's in coded format where z maintains a suffix relation with them
            pky_c_list = z_pky[z]
            f_potential[pky_c_list] += potential

        return(f_potential)
               
        
        
    def compute_forward_vec(self, w, seq_id):
        # ..note: seg_features and activated_states need to be loaded first
        pi_pky_codebook = self.model.pi_pky_codebook
        P_codebook = self.model.P_codebook
        pi_lendict = self.model.pi_lendict
        T = self.seqs_info[seq_id]["T"]
        alpha = numpy.ones((T+1,len(P_codebook)), dtype='longdouble') * (-numpy.inf)
        alpha[0,P_codebook[""]] = 0
        accum_activestates = {}
        activefeatures_perboundary = {}
        
        for j in range(1, T+1):
            boundary = (j, j)
            # identify active features
            active_features = self.identify_activefeatures(seq_id, boundary, accum_activestates)
            activefeatures_perboundary[boundary] = active_features
            # compute f_potential
            f_potential = self.compute_fpotential(w, active_features)
            for pi in pi_pky_codebook:
                if(j >= pi_lendict[pi]):
                    vec = f_potential[pi_pky_codebook[pi][0]] + alpha[j-1, pi_pky_codebook[pi][1]]
                    alpha[j, P_codebook[pi]] = vectorized_logsumexp(vec)
        self.seqs_info[seq_id]['activefeatures_per_boundary'] = activefeatures_perboundary
        print("activefeatures ", activefeatures_perboundary)
        return(alpha)
   
    
    def compute_bpotential(self, w, active_features):
        model = self.model
        ysk_codebook = model.ysk_codebook
        z_ysk = model.ysk_z
        b_potential = numpy.zeros(len(ysk_codebook))
        # to consider caching the w_indx and fval as in cached_pf
        for z in active_features:
            w_indx = list(active_features[z].keys())
            f_val = list(active_features[z].values())
            potential = numpy.inner(w[w_indx], f_val)
            # get all ysk's in coded format where z maintains a prefix relation with them
            ysk_c_list = z_ysk[z]
            b_potential[ysk_c_list] += potential

        return(b_potential)
    
    def compute_backward_vec(self, w, seq_id):
        model = self.model
        si_ysk_codebook = model.si_ysk_codebook
        S_codebook = model.S_codebook
        ysk_codebook = model.ysk_codebook
        patts_len = model.patts_len
        Z_lendict = model.Z_lendict
        T = self.seqs_info[seq_id]["T"]
        beta = numpy.ones((T+2, len(S_codebook)), dtype='longdouble') * (-numpy.inf)
        beta[T+1, S_codebook[""]] = 0
        activefeatures_perboundary = self.seqs_info[seq_id]['activefeatures_per_boundary']
        
        for j in reversed(range(1, T+1)):
            for si in si_ysk_codebook:
                b_potential = numpy.zeros(len(ysk_codebook))
                for z_len in patts_len:
                    b = j + z_len - 1
                    if(b <= T):
                        boundary = (b, b)
                        active_features = activefeatures_perboundary[boundary]
                        features = {z:active_features[z] for z in active_features if Z_lendict[z] == z_len}
                        # compute b_potential vector
                        b_potential += self.compute_bpotential(w, features)
                vec = b_potential[si_ysk_codebook[si][0]] + beta[j+1, si_ysk_codebook[si][1]]
                beta[j, S_codebook[si]] = vectorized_logsumexp(vec)  
                    
        return(beta)
        

    def compute_seq_loglikelihood(self, w, seq_id):
        """computes the conditional log-likelihood of a sequence (i.e. p(Y|X;w)) 
           it is used as a cost function for the single sequence when trying to estimate parameters w
        """
#         print("-"*40)
#         print("... Evaluating compute_seq_loglikelihood() ...")
        
        # we need global features and alpha matrix to be ready -- order is important
        l = OrderedDict()
        l['globalfeatures'] = (seq_id, False)
        l['activated_states'] = (seq_id, )
        l['seg_features'] = (seq_id, )
        l['alpha'] = (w, seq_id)
        
        self.check_cached_info(seq_id, l)
        # get the p(X;w) -- probability of the sequence under parameter w
        Z = self.seqs_info[seq_id]["Z"]
        gfeatures = self.seqs_info[seq_id]["globalfeatures"]
        globalfeatures = self.represent_globalfeature(gfeatures, None)
        w_indx = list(globalfeatures.keys())
        f_val = list(globalfeatures.values())

        # log(p(Y|X;w))
        loglikelihood = numpy.dot(w[w_indx], f_val) - Z 
        self.seqs_info[seq_id]["loglikelihood"] = loglikelihood

        return(loglikelihood)
    
    def compute_seqs_loglikelihood(self, w, seqs_id):
        seqs_loglikelihood = 0
        for seq_id in seqs_id:
            seqs_loglikelihood += self.compute_seq_loglikelihood(w, seq_id)
        return(seqs_loglikelihood)
    
    def _load_alpha(self, w, seq_id):
        # assumes the f_potential has been loaded into seq_info
        seq_info = self.seqs_info[seq_id]
        seq_info["alpha"] = self.compute_forward_vec(w, seq_id)
        seq_info["Z"] = vectorized_logsumexp(seq_info["alpha"][-1,:])
        #print("... Computing alpha probability ...")
    
    def _load_beta(self, w, seq_id):
        # assumes the b_potential has been loaded into seq_info
        seq_info = self.seqs_info[seq_id]
        seq_info["beta"] = self.compute_backward_vec(w, seq_id)
        #print("... Computing beta probability ...")

    def _load_Y(self, seq_id):
        seq = self._load_seq(seq_id, target="seq")
        self.seqs_info[seq_id]['Y'] = {'flat_y':seq.flat_y, 'boundaries':seq.y_sboundaries}
        #print("loading Y")

    def load_activatedstates(self, seq_id):
        # get the sequence activated states
        seqs_info = self.seqs_info
        seqs_representer = self.seqs_representer
        activated_states = seqs_representer.get_seq_activatedstates(seq_id, seqs_info)
        seqs_info[seq_id]["activated_states"] = activated_states
        #print("loading activated states")
    
    def load_segfeatures(self, seq_id):
        # get the sequence segment features
        seqs_info = self.seqs_info
        seqs_representer = self.seqs_representer
        seg_features = seqs_representer.get_seq_segfeatures(seq_id, seqs_info)
        self.seqs_info[seq_id]["seg_features"] = seg_features
        #print("loading segment features")
    
    def load_globalfeatures(self, seq_id, per_boundary=True):
        # get sequence global features
        seqs_representer = self.seqs_representer
        gfeatures_perboundary = seqs_representer.get_seq_globalfeatures(seq_id, self.seqs_info, per_boundary=per_boundary)
#         print("per_boundary ", per_boundary)
#         print(gfeatures_perboundary)
        if(per_boundary):
            fname = "globalfeatures_per_boundary"
        else:
            fname = "globalfeatures"
        self.seqs_info[seq_id][fname] = gfeatures_perboundary
#         print(self.seqs_info[seq_id][fname])
        #print("loading globalfeatures")
        
    def load_imposter_globalfeatures(self, seq_id, y_imposter, seg_other_symbol):
        # get sequence global features
        seqs_representer = self.seqs_representer
        imposter_gfeatures_perboundary, y_imposter_boundaries = seqs_representer.get_imposterseq_globalfeatures(seq_id, self.seqs_info, y_imposter, seg_other_symbol)
        return(imposter_gfeatures_perboundary, y_imposter_boundaries)
    
    def represent_globalfeature(self, gfeatures, boundaries):
        # get sequence global features
        seqs_representer = self.seqs_representer
        windx_fval = seqs_representer.represent_gfeatures(gfeatures, self.model, boundaries=boundaries)        
        return(windx_fval)  
    
    
    def _load_seq(self, seq_id, target = "seq"):
        seqs_representer = self.seqs_representer
        seq = seqs_representer.load_seq(seq_id, self.seqs_info)
        if(target == "seq"):
            return(seq)
        elif(target == "Y"):
            return(seq.Y)
        elif(target == "X"):
            return(seq.X)
        
    def check_cached_info(self, seq_id, entity_names):
        """order of elements in the entity_names list is important """
        seq_info = self.seqs_info[seq_id]
        func_dict = self.func_dict
        none_type = type(None) 
        for varname, args in entity_names.items():
            if(type(seq_info.get(varname)) == none_type):
                func_dict[varname](*args)

    def clear_cached_info(self, seqs_id, cached_entities = []):
        args = self.def_cached_entities + cached_entities
        for seq_id in seqs_id:
            seq_info = self.seqs_info[seq_id]
            for varname in args:
                if(varname in seq_info):
                    seq_info[varname] = None
                    
    def save_model(self, file_name):
        # to clean and save things before pickling the model
        self.seqs_info.clear() 
        ReaderWriter.dump_data(self, file_name)

    def decode_seqs(self, decoding_method, out_dir, **kwargs):
        """ decoding_method: a string referring to type of decoding {'viterbi', 'per_state_decoding'}
            **kwargs could be one of the following:
                seqs: a list comprising of sequences that are instances of SequenceStrcut() class
                seqs_info: dictionary containing the info about the sequences to parse
        """
        corpus_name = "decoding_seqs"
        w = self.weights
        # supporting only viterbi for now
        if(decoding_method == "viterbi"):
            decoder = self.viterbi
        else:
            decoder = self.viterbi
            
        file_name = kwargs.get('file_name')
        if(not file_name):
            file_name = "decoded.txt"
        out_file = os.path.join(create_directory(corpus_name, out_dir), file_name)
        
        beam_size = kwargs.get('beam_size')
        if(not beam_size):
            beam_size = self.beam_size
            
        if(kwargs.get("seqs_info")):
            self.seqs_info = kwargs["seqs_info"]
            N = len(self.seqs_info)
        elif(kwargs.get("seqs")): 
            seqs = kwargs["seqs"]           
            seqs_dict = {i+1:seqs[i] for i in range(len(seqs))}
            seqs_id = list(seqs_dict.keys())
            N = len(seqs_id)
            seqs_info = self.seqs_representer.prepare_seqs(seqs_dict, "processed_seqs", out_dir, unique_id = True)
            self.seqs_representer.scale_attributes(seqs_id, seqs_info)
            self.seqs_representer.extract_seqs_modelactivefeatures(seqs_id, seqs_info, self.model, "processed_seqs", learning=False)
            self.seqs_info = seqs_info
            
        if(kwargs.get("sep")):
            sep = kwargs['sep']
        else:
            sep = "\t"

        seqs_pred = {}
        seqs_info = self.seqs_info
        counter = 0
        for seq_id in seqs_info:
            Y_pred, __ = decoder(w, seq_id, beam_size)
            seq = ReaderWriter.read_data(os.path.join(seqs_info[seq_id]["globalfeatures_dir"], "sequence"))
            self.write_decoded_seqs([seq], [Y_pred], out_file, sep)
            seqs_pred[seq_id] = {'seq': seq,'Y_pred': Y_pred}
            # clear added info per sequence
            self.clear_cached_info([seq_id])
            counter += 1
            print("sequence decoded -- {} sequences are left".format(N-counter))
        
        # clear seqs_info
        self.seqs_info.clear()
        return(seqs_pred)

            
    def write_decoded_seqs(self, ref_seqs, Y_pred_seqs, out_file, sep = "\t"):
        for i in range(len(ref_seqs)):
            Y_pred_seq = Y_pred_seqs[i]
            ref_seq = ref_seqs[i]
            T = ref_seq.T
            line = ""
            for t in range(1, T+1):
                for field_name in ref_seq.X[t]:
                    line += ref_seq.X[t][field_name] + sep
                if(ref_seq.flat_y):
                    line += ref_seq.flat_y[t-1] + sep
                line += Y_pred_seq[t-1]
                line += "\n" 
            line += "\n"
            ReaderWriter.log_progress(line,out_file)  
            
    def prune_states(self, j, delta, beam_size):
        P_codebook_rev = self.model.P_codebook_rev
        pi_elems = self.model.pi_elems
        pi_lendict = self.model.pi_lendict

#         # sort the pi in descending order of their score
#         indx_sorted_pi = numpy.argsort(delta[j,:])[::-1]
#         # identify states falling out of the beam
#         indx_falling_pi = indx_sorted_pi[beam_size:]
#         # identify top-k states/pi
#         indx_topk_pi = indx_sorted_pi[:beam_size]
#         # remove the effect of states/pi falling out of the beam
#         delta[j, indx_falling_pi] = -numpy.inf

        # using argpartition as better alternative to argsort
        indx_partitioned_pi = numpy.argpartition(-delta[j, :], beam_size)
        # identify top-k states/pi
        indx_topk_pi = indx_partitioned_pi[:beam_size]
#         # identify states falling out of the beam
#         indx_falling_pi = indx_partitioned_pi[beam_size:]
#         # remove the effect of states/pi falling out of the beam
#         delta[j, indx_falling_pi] = -numpy.inf
        # get topk states
        topk_pi = {P_codebook_rev[indx] for indx in indx_topk_pi}
        topk_states = set()
        for pi in topk_pi:
            if(pi_lendict[pi] > 1):
                topk_states.add(pi_elems[pi][-1])
            else:
                topk_states.add(pi)
        return(topk_states)
    

    def viterbi(self, w, seq_id, beam_size, stop_off_beam = False, y_ref=[], K=1):
        l = {}
        l['activated_states'] = (seq_id, )
        l['seg_features'] = (seq_id, )
        self.check_cached_info(seq_id, l)
        
        pi_elems = self.model.pi_elems
        pi_pky_codebook = self.model.pi_pky_codebook
        f_transition = self.model.f_transition
        P_codebook = self.model.P_codebook
        P_codebook_rev = self.model.P_codebook_rev
        P_len = self.model.P_len
        T = self.seqs_info[seq_id]["T"]
        # records max score at every time step
        delta = numpy.ones((T+1,len(P_codebook)), dtype='longdouble') * (-numpy.inf)
        # the score for the empty sequence at time 0 is 1
        delta[0, P_codebook[""]] = 0
        back_track = {}
        pi_lendict = self.model.pi_lendict
        accum_activestates = {}
        # records where violation occurs -- it is 1-based indexing 
        viol_index = []
        #^print("pky_codebook_rev ", pky_codebook_rev)
        #activefeatures_perboundary = {}
        #print('pi_elems ', pi_elems)
        for j in range(1, T+1):
            boundary = (j, j)
            active_features = self.identify_activefeatures(seq_id, boundary, accum_activestates)
            #activefeatures_perboundary[boundary] = active_features
            # vector of size len(pky)
            f_potential = self.compute_fpotential(w, active_features)
            #^print("f_potential ", f_potential)
            for pi in pi_pky_codebook:
                if(j >= pi_lendict[pi]):
                    vec = f_potential[pi_pky_codebook[pi][0]] + delta[j-1, pi_pky_codebook[pi][1]]
                    #print("pi ", pi)
                    #print("vec ", vec)
                    delta[j, P_codebook[pi]] = numpy.max(vec)
                    #print("max chosen ", delta[j, P_codebook[pi]])
                    argmax_ind = numpy.argmax(vec)
                    #print("argmax chosen ", argmax_ind)
                    pk_c = pi_pky_codebook[pi][1][argmax_ind]
                    #print('pk_c ', pk_c)
                    pk = P_codebook_rev[pk_c]
                    y = pi_elems[pk][-1]
                    back_track[j, P_codebook[pi]] = (pk, y)
                    
            #^print('delta[{},:] = {} '.format(j, delta[j,:]))
            # apply the beam pruning
            #^print("beam_size ", beam_size)
            #^print("P_len ", P_len)
            if(beam_size < P_len):
                topk_states = self.prune_states(j, delta, beam_size)
                #^print('delta[{},:] = {} '.format(j, delta[j,:]))
                #^print("topk_states ", topk_states)
                if(y_ref):
                    if(y_ref[j-1] not in topk_states):
                        viol_index.append(j)
                # update tracked active states -- to consider renaming it          
                accum_activestates[j] = accum_activestates[j].intersection(topk_states)
                #^print("accum_activestates[{}] = {}".format(j, accum_activestates[j]))
                if(viol_index and stop_off_beam):
                    T = j
                    break
        #^print('seq_id ', seq_id)
        #^print("activefeatures_perboundary ", activefeatures_perboundary)
        if(K == 1):
            # decoding the sequence
            Y_decoded = []
            p_T_code = numpy.argmax(delta[T,:])
            p_T = self.model.P_codebook_rev[p_T_code]
            y_T = pi_elems[p_T][-1]
            Y_decoded.append((p_T,y_T))
            #print("t={}, p_T_code={}, p_T={}, y_T ={}".format(T, p_T_code, p_T, y_T))
            t = T - 1
            while t>0:
                p_tplus1 = Y_decoded[-1][0]
                p_t, y_t = back_track[(t+1, P_codebook[p_tplus1])]
                #print("t={}, (t+1, p_t_code)=({}, {})->({},{})".format(t, t+1, P_codebook[p_tplus1], p_t, y_t))
                Y_decoded.append((p_t, y_t))
                t -= 1
            Y_decoded.reverse()
     
            Y_decoded = [yt for (pt,yt) in Y_decoded]
#             print("Y_decoded {}".format(Y_decoded))
#             print('delta ', delta)
#             print('backtrack ', back_track)
#             print("P_codebook ", P_codebook)
            return(Y_decoded, viol_index)
        else:
            asearcher = AStarSearcher(P_codebook, P_codebook_rev, pi_elems)
            topK = asearcher.search(delta, back_track, T, K)
#             print('topk ', topK)
            return(topK, viol_index)

        
 
    def validate_forward_backward_pass(self, w, seq_id):
        self.clear_cached_info([seq_id])
        # this will compute alpha and beta matrices and save them in seqs_info dict
        l = OrderedDict()
        l['activated_states'] = (seq_id, )
        l['seg_features'] = (seq_id, )
        l['alpha'] = (w, seq_id)
        l['beta'] = (w, seq_id)
        self.check_cached_info(seq_id, l)
        
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
    
if __name__ == "__main__":
    pass