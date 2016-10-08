'''
@author: ahmed allam <ahmed.allam@yale.edu>
'''

import pickle
import os
from copy import deepcopy
from datetime import datetime
from collections import Counter
import numpy
from utilities import ReaderWriter, create_directory, generate_datetime_str
from attributes_extraction import AttributeScaler

class HOFeatureExtractor(object):
    """ Generic feature extractor class that contains feature functions/templates """
    def __init__(self, templateX, templateY, attr_desc):
        self.template_X = templateX
        self.template_Y = templateY
        self.attr_desc = attr_desc

    @property
    def template_X(self):
        return self._template_X
    @template_X.setter
    def template_X(self, template):
        """ example of template X to be processed:
            template_X = {'w': {(0,):((0,), (-1,0), (-2,-1,0))}}
                       = {attr_name: {x_offset:tuple(y_offsets)}}
        """
        if(type(template) == dict):
            self._template_X = {}
            for attr_name, templateX in template.items():
                self._template_X[attr_name] = {}
                for offset_x, offsets_y in templateX.items():
                    s_offset_x = tuple(sorted(offset_x))
                    unique_dict = {}
                    for offset_y in offsets_y:
                        s_offset_y = tuple(sorted(offset_y))
                        check = self._validate_template(s_offset_y)
                        if(check):
                            unique_dict[s_offset_y] = 1
                    if(unique_dict):
                        self._template_X[attr_name][s_offset_x] = tuple(unique_dict.keys())

    @property
    def template_Y(self):
        return self._template_Y
    @template_Y.setter
    def template_Y(self, template):
        """ example of template Y to be processed:
            template_Y = {'Y': ((0,), (-1,0), (-2,-1,0))}
                       = {Y: tuple(y_offsets)}
        """
        if(type(template) == dict):
            self._template_Y = {}
            unique_dict = {}
            offsets_y = template['Y']
            for offset_y in offsets_y:
                s_offset_y = tuple(sorted(offset_y))
                check = self._validate_template(s_offset_y)
                if(check):
                    unique_dict[s_offset_y] = 1
            if(unique_dict):
                self._template_Y['Y'] = tuple(unique_dict.keys())
            else:
                self._template_Y['Y'] = ()

    def _validate_template(self, template):
        """template is a tuple (i.e. (-2,-1,0)"""
        check = True
        if(len(template) > 1):
            for i in range(len(template)-1):
                curr_elem = template[i]
                next_elem = template[i+1]
                diff = curr_elem - next_elem
                if(diff != -1):
                    check = False
                    break
        else:
            if(template[0] != 0):
                check = False
        return(check)
                    
                
    def extract_seq_features(self, seq):
        # this method is used to extract features from sequences in the training dataset 
        # (i.e. we know the labels and boundaries)
        Y = seq.Y
        features = {}
        for boundary in Y:
            y_feat = self.extract_features_Y(seq, boundary, self.template_Y)
            xy_feat = self.extract_features_XY(seq, boundary)
#             print("boundary {}".format(boundary))
#             print("boundary {}".format(boundary))
#             print("y_feat {}".format(y_feat))
#             print("xy_feat {}".format(xy_feat))
            for offset_tup_y in y_feat['Y']:
                for y_patt in y_feat['Y'][offset_tup_y]:
                    if(y_patt in xy_feat):
                        xy_feat[y_patt].update(Counter(y_feat['Y'][offset_tup_y]))
                    else:
                        xy_feat[y_patt] = Counter(y_feat['Y'][offset_tup_y])
            features[boundary] = xy_feat
#             print("features {}".format(features[boundary]))
#             print("*"*40)
                
        # summing up all detected features across all boundaries
        seq_features = {}
        for boundary, xy_features in features.items():
            for y_patt in xy_features:
                if(y_patt in seq_features):
                    seq_features[y_patt].update(xy_features[y_patt])
                else:
                    seq_features[y_patt] = xy_features[y_patt]
#                 print("seq_features {}".format(seq_features))
        #print("features sum {}".format(seq_features))
        return(seq_features)

    def extract_features_Y(self, seq, boundary, templateY):
        """ template_Y = {'Y': ((0,), (-1,0), (-2,-1,0))}
                       = {Y: tuple(y_offsets)}        
        """
        template_Y = templateY['Y']
        Y = seq.Y
        y_boundaries = seq.get_y_boundaries()
        range_y = range(len(y_boundaries))
        curr_pos = y_boundaries.index(boundary)
        
        y_patt_features = {}
        feat_template = {}
        for offset_tup_y in template_Y:
            y_pattern = []
            for offset_y in offset_tup_y:
                # offset_y should be always <= 0
                pos = curr_pos + offset_y
                if(pos in range_y):
                    b = y_boundaries[pos]
                    y_pattern.append(Y[b])
                else:
                    y_pattern = []
                    break 
            if(y_pattern):
                feat_template[offset_tup_y] = {"|".join(y_pattern):1}

        y_patt_features['Y'] = feat_template
        
#         print("X"*40)
#         print("boundary {}".format(boundary))
#         for attr_name, f_template in y_patt_features.items():
#             for offset, features in f_template.items():
#                 print("{} -> {}".format(offset, features))
#         print("X"*40)
        
        return(y_patt_features)
    
    def extract_features_X(self, seq, boundary):
        """template_X = {'w': {(0,):((0,), (-1,0), (-2,-1,0))}}
           boundary is a tuple (u,v) indicating the beginning and end of the current segment
        """
        # get template X
        template_X = self.template_X
        attr_desc = self.attr_desc
        # current boundary begin and end
        u = boundary[0]
        v = boundary[-1]

#         print("positions {}".format(positions))
        seg_features = {}
        for attr_name in template_X:
#             print("attr_name {}".format(attr_name))
            # check the type of attribute
            if(attr_desc[attr_name]["encoding"] == "binary"):
                represent_attr = self._represent_binary_attr
            else:
                represent_attr = self._represent_real_attr
            
            feat_template = {}
            for offset_tup_x in template_X[attr_name]:
                attributes = []
                feature_name = '|'.join(['{}[{}]'.format(attr_name, offset_x) for offset_x in offset_tup_x])
#                 print("feature_name {}".format(feature_name))
                for offset_x in offset_tup_x:
#                     print("offset_x {}".format(offset_x))
                    if(offset_x > 0):
                        pos = (v + offset_x, v + offset_x)
                    elif(offset_x < 0):
                        pos = (u + offset_x, u + offset_x)
                    else:
                        pos = (u, v)
                   
                    if(pos in seq.seg_attr):
                        attributes.append(seq.seg_attr[pos][attr_name])
#                         print("attributes {}".format(attributes))
                    else:
                        attributes = []
                        break
                if(attributes):
                    feat_template[offset_tup_x] = represent_attr(attributes, feature_name)
            seg_features[attr_name] = feat_template
#         
#         print("X"*40)
#         print("boundary {}".format(boundary))
#         for attr_name, f_template in seg_features.items():
#             for offset, features in f_template.items():
#                 print("{} -> {}".format(offset, features))
#         print("X"*40)

        return(seg_features)
    
    def extract_features_XY(self, seq, boundary):
        """ template_X = {'w': {(0,):((0,), (-1,0), (-2,-1,0))}}
            template_Y = {'Y': ((0,), (-1,0), (-2,-1,0))}
        """
        
        seg_feat_templates = self.extract_features_X(seq, boundary)
        templateX = self.template_X

#         print("seg_feat_templates {}".format(seg_feat_templates))
        xy_features = {}
        for attr_name, seg_feat_template in seg_feat_templates.items():
            for offset_tup_x in seg_feat_template:
                templateY = {'Y':templateX[attr_name][offset_tup_x]}
                y_feat_template = self.extract_features_Y(seq, boundary, templateY)
#                 print("y_feat_template {}".format(y_feat_template))
                y_feat_template = y_feat_template['Y']
                for y_patt_dict in y_feat_template.values():
                    for y_patt in y_patt_dict:
                        if(y_patt in xy_features):
                            xy_features[y_patt].update(Counter(seg_feat_template[offset_tup_x]))
                        else:
                            xy_features[y_patt] = Counter(seg_feat_template[offset_tup_x])
#                         print("xy_features {}".format(xy_features))
        return(xy_features)
    
    def lookup_features_X(self, seq, boundary):
        """template_X = {'w': {(0,):((0,), (-1,0), (-2,-1,0))}}
           boundary is a tuple (u,v) indicating the beginning and end of the current segment
           This method is used to lookup features using varying boundaries (i.e. g(X, u, v))
        """
        # get template X
        template_X = self.template_X
        attr_desc = self.attr_desc
        # current boundary begin and end
        u = boundary[0]
        v = boundary[-1]
        
#         print("positions {}".format(positions))
        seg_features = {}
        for attr_name in template_X:
#             print("attr_name {}".format(attr_name))
            # check the type of attribute
            if(attr_desc[attr_name]["encoding"] == "binary"):
#                 aggregate_attr = self._aggregate_binary_attr
                represent_attr = self._represent_binary_attr
            else:
#                 aggregate_attr = self._aggregate_real_attr
                represent_attr = self._represent_real_attr
            
            for offset_tup_x in template_X[attr_name]:
                attributes = []
                feature_name = '|'.join(['{}[{}]'.format(attr_name, offset_x) for offset_x in offset_tup_x])
#                 print("feature_name {}".format(feature_name))
                for offset_x in offset_tup_x:
#                     print("offset_x {}".format(offset_x))
                    if(offset_x > 0):
                        pos = (v + offset_x, v + offset_x)
                    elif(offset_x < 0):
                        pos = (u + offset_x, u + offset_x)
                    else:
                        pos = (u, v)
#                     print("pos {}".format(pos))
                    
                    if(pos in seq.seg_attr):
                        attributes.append(seq.seg_attr[pos][attr_name])
#                         print("attributes {}".format(attributes))
                    else:
                        attributes = []
                        break
                if(attributes):
                    seg_features.update(represent_attr(attributes, feature_name))
#         print("seg_features lookup {}".format(seg_features))
        return(seg_features)
    
#     def lookup_seq_modelactivefeatures(self, seq, model):
#         
#         model_patt = deepcopy(model.patt_order)
#         print("original model patt_order {} with id {}".format(model.patt_order, id(model.patt_order)))
#         print("copied model patt_order {} with id {}".format(model_patt, id(model_patt)))
#         del model_patt[1]
#         print("modified model patt_order {} with id {}".format(model_patt, id(model_patt)))
# 
#         
#         model_states = list(model.Y_codebook.keys())
#         L = model.L
#         T = seq.T
#         
#         active_features = {}
#         accum_pattern = {}
#         
#         for j in range(1, T+1):
#             for d in range(L):
#                 if(j-d <= 0):
#                     break
#                 boundary = (j-d, j)
# 
#                 seg_features = self.lookup_features_X(seq, boundary)
#                 print("seg_features {}".format(seg_features))
# 
#                 # active features that  uses only the current states
#                 active_features[boundary] = model.represent_activefeatures(model_states, seg_features)
#                 print("prev active_features[{}] {}".format(boundary, active_features[boundary]))
# 
#                 if(active_features[boundary]):
#                     # z pattern of length 1 (i.e. detected labels from set Y}
#                     detected_seg_patt = {patt:1 for patt in active_features[boundary]}
#                     if(j in accum_pattern):
#                         accum_pattern[j][1].update(detected_seg_patt)
#                     else:
#                         accum_pattern[j] = {1:detected_seg_patt}
# #                     print("j {}".format(j))
#                     print("accum_pattern {}".format(accum_pattern))
# #                     print("active features {}".format(active_features))
#                     tracked_z_patt = self._build_z_patt(boundary, detected_seg_patt, accum_pattern, model_patt)
#                     if(tracked_z_patt):
#                         for patt_len in tracked_z_patt:
#                             if(patt_len in accum_pattern[j]):
#                                 accum_pattern[j][patt_len].update(tracked_z_patt[patt_len])
#                             else:
#                                 accum_pattern[j].update({patt_len:tracked_z_patt[patt_len]})
#                         print("new accum_pattern {}".format(accum_pattern))
#                         new_patts = {z_patt:1 for order in tracked_z_patt for z_patt in tracked_z_patt[order]}
# #                         print("new_patts {}".format(new_patts))
# #                         print("current boundary {}".format(boundary))
# 
#                         new_activefeatures = model.represent_activefeatures(new_patts, seg_features)
#                         print("new_activefeatures {}".format(new_activefeatures))
#                         for z_patt in new_activefeatures:
#                             if(z_patt in active_features[boundary]):
#                                 active_features[boundary][z_patt].update(new_activefeatures[z_patt])
#                             else:
#                                 active_features[boundary].update({z_patt: new_activefeatures[z_patt]})
#                         print("after active_features[{}] {}".format(boundary, active_features[boundary]))
# 
#         # update seqs_info
#         # unpack active features
#         print("accum_pattern {}".format(accum_pattern))
#         return(active_features)
    def lookup_seq_modelactivefeatures(self, seq, model):
        
        model_patt = deepcopy(model.patt_order)
        del model_patt[1]
#         print("modified model patt_order {} with id {}".format(model_patt, id(model_patt)))
        
        model_states = list(model.Y_codebook.keys())
        L = model.L
        T = seq.T
        
        active_features = {}
        accum_pattern = {}
        
        for j in range(1, T+1):
            for d in range(L):
                if(j-d <= 0):
                    break
                boundary = (j-d, j)

                seg_features = self.lookup_features_X(seq, boundary)
#                 print("seg_features {}".format(seg_features))

                # active features that  uses only the current states
                active_features[boundary] = model.represent_activefeatures(model_states, seg_features)
#                 print("initial active_features[{}] {}".format(boundary, active_features[boundary]))

                if(active_features[boundary]):
                    # z pattern of length 1 (i.e. detected labels from set Y}
                    detected_y_patt = {y_patt:1 for y_patt in active_features[boundary]}
                    self._update_accum_pattern(accum_pattern, {1:detected_y_patt}, j)
#                     print("accum_pattern {}".format(accum_pattern))
                    tracked_z_patt = self._build_z_patt(boundary, detected_y_patt, accum_pattern, model_patt)
                    if(tracked_z_patt):
                        self._update_accum_pattern(accum_pattern, tracked_z_patt, j)
#                         print("updated accum_pattern {}".format(accum_pattern))
                        new_patts = {z_patt:1 for order in tracked_z_patt for z_patt in tracked_z_patt[order]}
                        new_activefeatures = model.represent_activefeatures(new_patts, seg_features)
#                         print("new_activefeatures {}".format(new_activefeatures))
                        self._update_accum_activefeatures(active_features, new_activefeatures, boundary)
#                         print("updated active_features[{}] {}".format(boundary, active_features[boundary]))

#         print("accum_pattern {}".format(accum_pattern))
        return(active_features)
    
    def _update_accum_pattern(self, accum_pattern, detected_patt, j):
        if(j in accum_pattern):
            for patt_len in detected_patt:
                if(patt_len in accum_pattern[j]):
                    accum_pattern[j][patt_len].update(detected_patt[patt_len])
                else:
                    accum_pattern[j].update({patt_len:detected_patt[patt_len]})
        else:
            accum_pattern[j] = detected_patt
        
    def _update_accum_activefeatures(self, accum_activefeatures, detected_activefeatures, boundary):
        for z_patt in detected_activefeatures:
            if(z_patt in accum_activefeatures[boundary]):
                accum_activefeatures[boundary][z_patt].update(detected_activefeatures[z_patt])
            else:
                accum_activefeatures[boundary].update({z_patt: detected_activefeatures[z_patt]})
            
    def _check_pattern_len(self, detected_patt):
        cached_patt_len = self.cached_patt_len
        if(detected_patt not in cached_patt_len):
            # get the length of the detected_patt
            cached_patt_len[detected_patt] = len(detected_patt)
       
    
    def _build_z_patt(self, boundary, detected_y_patt, accum_pattern, model_patt):
        u = boundary[0]
        tracked_z_patt = {}
        if(u-1 in accum_pattern):
            for patt_len in model_patt:
#                 print("current pattern length {}".format(patt_len))
                for z_patt in model_patt[patt_len]:
#                     print("z_patt {}".format(z_patt))
                    for i in reversed(range(1, patt_len)):
                        if(i in accum_pattern[u-1]):
#                             print("current order i {}".format(i))
                            for prev_patt in accum_pattern[u-1][i]:
#                                 print("prev_patt {}".format(prev_patt))
                                if(z_patt[0:len(prev_patt)] == prev_patt):
                                    for curr_detected_patt in detected_y_patt:
#                                         print('detected_patt {}'.format(curr_detected_patt))
                                        mix_patt = prev_patt + "|" + curr_detected_patt
#                                         print("mix_patt {}".format(mix_patt))
                                        if(z_patt[0:len(mix_patt)] == mix_patt):
                                            if(i+1 in tracked_z_patt):
                                                tracked_z_patt[i+1][mix_patt] = 1
                                            else:
                                                tracked_z_patt[i+1] = {mix_patt:1}
         
#         print("tracked_z_patt {}".format(tracked_z_patt))
#         print("accum_pattern {}".format(accum_pattern))
        return(tracked_z_patt)    
    
    
    
    ########################################################
    # functions used to represent real and binary attributes
    ########################################################

    def _represent_binary_attr(self, attributes, feature_name):
        feature_val = '|'.join(attributes)
        feature = '{}={}'.format(feature_name, feature_val)
        return({feature:1})

    def _represent_real_attr(self, attributes, feature_name):
        feature_val = sum(attributes) 
        return({feature_name:feature_val})
    

#     def extract_features_X(self, seq, boundary):
#         """template_X = {'w': {(0,):((0,), (-1,0), (-2,-1,0))}}
#            boundary is a tuple (u,v) indicating the beginning and end of the current segment
#         """
#         # get template X
#         template_X = self.template_X
#         attr_desc = self.attr_desc
#         y_boundaries = seq.y_boundaries
#         range_y = range(len(y_boundaries))
#         curr_pos = y_boundaries.index(boundary)
# 
# #         print("positions {}".format(positions))
#         seg_features = {}
#         for attr_name in template_X:
# #             print("attr_name {}".format(attr_name))
#             # check the type of attribute
#             if(attr_desc[attr_name]["encoding"] == "binary"):
#                 represent_attr = self._represent_binary_attr
#             else:
#                 represent_attr = self._represent_real_attr
#             
#             feat_template = {}
#             for offset_tup_x in template_X[attr_name]:
#                 attributes = []
#                 feature_name = '|'.join(['{}[{}]'.format(attr_name, offset_x) for offset_x in offset_tup_x])
# #                 print("feature_name {}".format(feature_name))
#                 for offset_x in offset_tup_x:
# #                     print("offset_x {}".format(offset_x))
#                     pos = curr_pos + offset_x             
#                     if(pos in range_y):
#                         b = y_boundaries[pos]
#                         segment = seq.seg_attr[b]
# #                         print("segment {}".format(segment))
#                         attributes.append(segment[attr_name])
# #                         print("attributes {}".format(attributes))
#                     else:
#                         attributes = []
#                         break
#                 if(attributes):
#                     feat_template[offset_tup_x] = represent_attr(attributes, feature_name)
#             seg_features[attr_name] = feat_template
# #         
# #         print("X"*40)
# #         print("boundary {}".format(boundary))
# #         for attr_name, f_template in seg_features.items():
# #             for offset, features in f_template.items():
# #                 print("{} -> {}".format(offset, features))
# #         print("X"*40)
# 
#         return(seg_features)


    def dump_features(self, seq_file, seq_features):
        """store the features of the current sequence"""
        print("pickling table: {}\n".format(seq_file))
        with open(seq_file, 'wb') as f:
            pickle.dump(seq_features, f)
        

class FOFeatureExtractor(object):
    """ Generic feature extractor class that contains feature functions/templates for first order sequence models """
    def __init__(self, templateX, templateY, attr_desc, start_state = False):
        self.template_X = templateX
        self.template_Y = templateY
        self.attr_desc = attr_desc
        self.start_state = start_state
        
    @property
    def template_X(self):
        return self._template_X
    @template_X.setter
    def template_X(self, template):
        """ example of template X to be processed:
            template_X = {'w': {(0,):((0,), (-1,0)}}
                       = {attr_name: {x_offset:tuple(y_offsets)}}
        """
        if(type(template) == dict):
            self._template_X = {}
            for attr_name, templateX in template.items():
                self._template_X[attr_name] = {}
                for offset_x, offsets_y in templateX.items():
                    s_offset_x = tuple(sorted(offset_x))
                    unique_dict = {}
                    for offset_y in offsets_y:
                        s_offset_y = tuple(sorted(offset_y))
                        check = self._validate_template(s_offset_y)
                        if(check):
                            unique_dict[s_offset_y] = 1
                    if(unique_dict):
                        self._template_X[attr_name][s_offset_x] = tuple(unique_dict.keys())

    @property
    def template_Y(self):
        return self._template_Y
    @template_Y.setter
    def template_Y(self, template):
        """ example of template Y to be processed:
            template_Y = {'Y': ((0,), (-1,0))}
                       = {Y: tuple(y_offsets)}
        """
        if(type(template) == dict):
            self._template_Y = {}
            unique_dict = {}
            offsets_y = template['Y']
            for offset_y in offsets_y:
                s_offset_y = tuple(sorted(offset_y))
                check = self._validate_template(s_offset_y)
                if(check):
                    unique_dict[s_offset_y] = 1
            if(unique_dict):
                self._template_Y['Y'] = tuple(unique_dict.keys())
            else:
                self._template_Y['Y'] = ()

    def _validate_template(self, template):
        """template is a tuple (i.e. (-1,0)"""
        valid_offsets = ((0,), (-1,0))
        if(template in valid_offsets):
            check = True
        else:
            check = False
        
        return(check)
                    
                
    def extract_seq_features(self, seq):
        # this method is used to extract features from sequences in the training dataset 
        # (i.e. we know the labels and boundaries)
        Y = seq.Y
        features = {}
        for boundary in Y:
            y_feat = self.extract_features_Y(seq, boundary, self.template_Y)
            xy_feat = self.extract_features_XY(seq, boundary)
#             print("boundary {}".format(boundary))
#             print("y_feat {}".format(y_feat))
#             print("xy_feat {}".format(xy_feat))
            for offset_tup_y in y_feat['Y']:
                for y_patt in y_feat['Y'][offset_tup_y]:
                    if(y_patt in xy_feat):
                        xy_feat[y_patt].update(Counter(y_feat['Y'][offset_tup_y]))
                    else:
                        xy_feat[y_patt] = Counter(y_feat['Y'][offset_tup_y])
            features[boundary] = xy_feat
#             print("features {}".format(features[boundary]))
#             print("*"*40)
#         print("features by boundary {}".format(features))
                
        # summing up all detected features across all boundaries
        seq_features = {}
        for boundary, xy_features in features.items():
            for y_patt in xy_features:
                if(y_patt in seq_features):
                    seq_features[y_patt].update(xy_features[y_patt])
                else:
                    seq_features[y_patt] = xy_features[y_patt]
#                 print("seq_features {}".format(seq_features))
        #print("features sum {}".format(seq_features))
        return(seq_features)

    def extract_features_Y(self, seq, boundary, templateY):
        """ template_Y = {'Y': ((0,), (-1,0))}
                       = {Y: tuple(y_offsets)}        
        """
        template_Y = templateY['Y']
        Y = seq.Y
        y_boundaries = seq.get_y_boundaries()
        range_y = range(len(y_boundaries))
        curr_pos = y_boundaries.index(boundary)
        y_patt_features = {}
        feat_template = {}
        start_state = self.start_state

        if(curr_pos == 0):
            # corner case at t = 1
            for offset_tup_y in template_Y:
                y_pattern = []
                for offset_y in offset_tup_y:
                    # offset_y should be always <= 0
                    pos = curr_pos + offset_y
                    if(pos in range_y):
                        b = y_boundaries[pos]
                        y_pattern.append(Y[b])
                    else:
                        if(start_state):
                            y_pattern.append("__START__")
                        else:
                            y_pattern = []
                            break
                        
                if(y_pattern):
                    feat_template[offset_tup_y] = {"|".join(y_pattern):1} 
        else:
            for offset_tup_y in template_Y:
                y_pattern = []
                for offset_y in offset_tup_y:
                    # offset_y should be always <= 0
                    pos = curr_pos + offset_y
                    if(pos in range_y):
                        b = y_boundaries[pos]
                        y_pattern.append(Y[b])
                    else:
                        y_pattern = []
                        break
                if(y_pattern):
                    feat_template[offset_tup_y] = {"|".join(y_pattern):1}          

        y_patt_features['Y'] = feat_template
        
#         print("X"*40)
#         print("boundary {}".format(boundary))
#         for attr_name, f_template in y_patt_features.items():
#             for offset, features in f_template.items():
#                 print("{} -> {}".format(offset, features))
#         print("X"*40)
        
        return(y_patt_features)
    
    def extract_features_X(self, seq, boundary):
        """template_X = {'w': {(0,):((0,), (-1,0), (-2,-1,0))}}
           boundary is a tuple (u,v) indicating the beginning and end of the current segment
        """
        # get template X
        template_X = self.template_X
        attr_desc = self.attr_desc
        # current boundary begin and end
        u = boundary[0]
        v = boundary[-1]

#         print("positions {}".format(positions))
        seg_features = {}
        for attr_name in template_X:
#             print("attr_name {}".format(attr_name))
            # check the type of attribute
            if(attr_desc[attr_name]["encoding"] == "binary"):
                represent_attr = self._represent_binary_attr
            else:
                represent_attr = self._represent_real_attr
            
            feat_template = {}
            for offset_tup_x in template_X[attr_name]:
                attributes = []
                feature_name = '|'.join(['{}[{}]'.format(attr_name, offset_x) for offset_x in offset_tup_x])
#                 print("feature_name {}".format(feature_name))
                for offset_x in offset_tup_x:
#                     print("offset_x {}".format(offset_x))
                    if(offset_x > 0):
                        pos = (v + offset_x, v + offset_x)
                    elif(offset_x < 0):
                        pos = (u + offset_x, u + offset_x)
                    else:
                        pos = (u, v)
                   
                    if(pos in seq.seg_attr):
                        attributes.append(seq.seg_attr[pos][attr_name])
#                         print("attributes {}".format(attributes))
                    else:
                        attributes = []
                        break
                if(attributes):
                    feat_template[offset_tup_x] = represent_attr(attributes, feature_name)
            seg_features[attr_name] = feat_template
#         
#         print("X"*40)
#         print("boundary {}".format(boundary))
#         for attr_name, f_template in seg_features.items():
#             for offset, features in f_template.items():
#                 print("{} -> {}".format(offset, features))
#         print("X"*40)

        return(seg_features)
    
    def extract_features_XY(self, seq, boundary):
        """ template_X = {'w': {(0,):((0,), (-1,0), (-2,-1,0))}}
            template_Y = {'Y': ((0,), (-1,0), (-2,-1,0))}
        """
        
        templateX = self.template_X
        seg_feat_templates = self.extract_features_X(seq, boundary)
#         print("seg_feat_templates {}".format(seg_feat_templates))
        xy_features = {}
        for attr_name, seg_feat_template in seg_feat_templates.items():
            for offset_tup_x in seg_feat_template:
                templateY = {'Y':templateX[attr_name][offset_tup_x]}
                y_feat_template = self.extract_features_Y(seq, boundary, templateY)
#                 print("y_feat_template {}".format(y_feat_template))
                y_feat_template = y_feat_template['Y']
                for y_patt_dict in y_feat_template.values():
                    for y_patt in y_patt_dict:
                        if(y_patt in xy_features):
                            xy_features[y_patt].update(Counter(seg_feat_template[offset_tup_x]))
                        else:
                            xy_features[y_patt] = Counter(seg_feat_template[offset_tup_x])
#                         print("xy_features {}".format(xy_features))
        return(xy_features)
    
    def lookup_features_X(self, seq, boundary):
        """template_X = {'w': {(0,):((0,), (-1,0), (-2,-1,0))}}
           boundary is a tuple (u,v) indicating the beginning and end of the current segment
           This method is used to lookup features using varying boundaries (i.e. g(X, u, v))
        """
        # get template X
        template_X = self.template_X
        attr_desc = self.attr_desc
        # current boundary begin and end
        u = boundary[0]
        v = boundary[-1]
        
#         print("positions {}".format(positions))
        seg_features = {}
        for attr_name in template_X:
#             print("attr_name {}".format(attr_name))
            # check the type of attribute
            if(attr_desc[attr_name]["encoding"] == "binary"):
#                 aggregate_attr = self._aggregate_binary_attr
                represent_attr = self._represent_binary_attr
            else:
#                 aggregate_attr = self._aggregate_real_attr
                represent_attr = self._represent_real_attr
            
            for offset_tup_x in template_X[attr_name]:
                attributes = []
                feature_name = '|'.join(['{}[{}]'.format(attr_name, offset_x) for offset_x in offset_tup_x])
#                 print("feature_name {}".format(feature_name))
                for offset_x in offset_tup_x:
#                     print("offset_x {}".format(offset_x))
                    if(offset_x > 0):
                        pos = (v + offset_x, v + offset_x)
                    elif(offset_x < 0):
                        pos = (u + offset_x, u + offset_x)
                    else:
                        pos = (u, v)
#                     print("pos {}".format(pos))
                    
                    if(pos in seq.seg_attr):
                        attributes.append(seq.seg_attr[pos][attr_name])
#                         print("attributes {}".format(attributes))
                    else:
                        attributes = []
                        break
                if(attributes):
                    seg_features.update(represent_attr(attributes, feature_name))
#         print("seg_features lookup {}".format(seg_features))
        return(seg_features)
    
    
    def lookup_seq_modelactivefeatures(self, seq, model):
        
        model_patt = deepcopy(model.patt_order)
        del model_patt[1]
#         print("modified model patt_order {} with id {}".format(model_patt, id(model_patt)))
        
        model_states = list(model.Y_codebook.keys())
        L = model.L
        T = seq.T
        
        active_features = {}
        accum_pattern = {}
        
        for j in range(1, T+1):
            for d in range(L):
                if(j-d <= 0):
                    break
                boundary = (j-d, j)

                seg_features = self.lookup_features_X(seq, boundary)
#                 print("seg_features {}".format(seg_features))

                # active features that  uses only the current states
                active_features[boundary] = model.represent_activefeatures(model_states, seg_features)
#                 print("initial active_features[{}] {}".format(boundary, active_features[boundary]))

                if(active_features[boundary]):
                    # z pattern of length 1 (i.e. detected labels from set Y}
                    detected_y_patt = {y_patt:1 for y_patt in active_features[boundary]}
                    self._update_accum_pattern(accum_pattern, {1:detected_y_patt}, j)
#                     print("accum_pattern {}".format(accum_pattern))
                    tracked_z_patt = self._build_z_patt(boundary, detected_y_patt, accum_pattern, model_patt)
                    if(tracked_z_patt):
                        self._update_accum_pattern(accum_pattern, tracked_z_patt, j)
#                         print("updated accum_pattern {}".format(accum_pattern))
                        new_patts = {z_patt:1 for order in tracked_z_patt for z_patt in tracked_z_patt[order]}
                        new_activefeatures = model.represent_activefeatures(new_patts, seg_features)
#                         print("new_activefeatures {}".format(new_activefeatures))
                        self._update_accum_activefeatures(active_features, new_activefeatures, boundary)
#                         print("updated active_features[{}] {}".format(boundary, active_features[boundary]))

#         print("accum_pattern {}".format(accum_pattern))
        return(active_features)
    
    def _update_accum_pattern(self, accum_pattern, detected_patt, j):
        if(j in accum_pattern):
            for patt_len in detected_patt:
                if(patt_len in accum_pattern[j]):
                    accum_pattern[j][patt_len].update(detected_patt[patt_len])
                else:
                    accum_pattern[j].update({patt_len:detected_patt[patt_len]})
        else:
            accum_pattern[j] = detected_patt
        
    def _update_accum_activefeatures(self, accum_activefeatures, detected_activefeatures, boundary):
        for z_patt in detected_activefeatures:
            if(z_patt in accum_activefeatures[boundary]):
                accum_activefeatures[boundary][z_patt].update(detected_activefeatures[z_patt])
            else:
                accum_activefeatures[boundary].update({z_patt: detected_activefeatures[z_patt]})
    
    def _build_z_patt(self, boundary, detected_seg_patt, accum_pattern, model_patt):
        u = boundary[0]
        tracked_z_patt = {}
#         print("detected_seg_patt {}".format(detected_seg_patt))
        if(u == 1):
            if(self.start_state):
                for patt_len in model_patt:
#                 print("current pattern length {}".format(patt_len))
                    for z_patt in model_patt[patt_len]:
                        for curr_detected_patt in detected_seg_patt:
                            mix_patt = "__START__" + "|" + curr_detected_patt
                            if(z_patt.startswith(mix_patt)):
                                if(patt_len in tracked_z_patt):
                                    tracked_z_patt[patt_len][mix_patt] = 1
                                else:
                                    tracked_z_patt[patt_len] = {mix_patt:1}

        else:
            if(u-1 in accum_pattern):
                for patt_len in model_patt:
    #                 print("current pattern length {}".format(patt_len))
                    for z_patt in model_patt[patt_len]:
    #                     print("z_patt {}".format(z_patt))
                        for i in reversed(range(1, patt_len)):
                            if(i in accum_pattern[u-1]):
    #                             print("current order i {}".format(i))
                                for prev_patt in accum_pattern[u-1][i]:
    #                                 print("prev_patt {}".format(prev_patt))
                                    if(z_patt.startswith(prev_patt)):
                                        for curr_detected_patt in detected_seg_patt:
    #                                         print('detected_patt {}'.format(curr_detected_patt))
                                            mix_patt = prev_patt + "|" + curr_detected_patt
    #                                         print("mix_patt {}".format(mix_patt))
                                            if(z_patt.startswith(mix_patt)):
                                                if(i+1 in tracked_z_patt):
                                                    tracked_z_patt[i+1][mix_patt] = 1
                                                else:
                                                    tracked_z_patt[i+1] = {mix_patt:1}
                                            
         
#         print("tracked_z_patt {}".format(tracked_z_patt))
#         print("accum_pattern {}".format(accum_pattern))
        return(tracked_z_patt) 
    
    ########################################################
    # functions used to represent real and binary attributes
    ########################################################

    def _represent_binary_attr(self, attributes, feature_name):
        feature_val = '|'.join(attributes)
        feature = '{}={}'.format(feature_name, feature_val)
        return({feature:1})

    def _represent_real_attr(self, attributes, feature_name):
        feature_val = numpy.sum(attributes) 
        return({feature_name:feature_val})

    def dump_features(self, seq_file, seq_features):
        """store the features of the current sequence"""
        print("pickling table: {}\n".format(seq_file))
        with open(seq_file, 'wb') as f:
            pickle.dump(seq_features, f)   

class SeqsRepresentation(object):
    def __init__(self, attr_extractor, fextractor):
        self.feature_extractor = fextractor
        self.attr_extractor = attr_extractor
        self.attr_scaler = None
        
    @property
    def feature_extractor(self):
        return self._feature_extractor
    @feature_extractor.setter
    def feature_extractor(self, fextractor):
        # make a copy to preserve the template_X and template_Y used in the extractor
        self._feature_extractor = deepcopy(fextractor)
    
    def prepare_seqs(self, seqs_dict, corpus_name, working_dir, unique_id = True):
        """ seqs_dict: dictionary containing  sequences and corresponding ids where each sequence is an instance of the SequenceStrcut() class
            corpus_name: string specifying the name of the corpus -- it will be used as corpus folder name
        """
        attr_extractor = self.attr_extractor
        
        if(unique_id):
            corpus_folder = "{}_{}".format(corpus_name, generate_datetime_str())
        else:
            corpus_folder = corpus_name
            
        target_dir = create_directory("global_features", create_directory(corpus_folder, working_dir))
        seqs_info = {}
         
        start_time = datetime.now()
        for seq_id, seq in seqs_dict.items():
            # boundaries of X generate segments of length equal 1
            x_boundaries = seq.get_x_boundaries()
            # this will update the value of the seg_attr of the sequence 
            attr_extractor.generate_attributes(seq, x_boundaries)
            # create a folder for every sequence
            seq_dir = create_directory("seq_{}".format(seq_id), target_dir)
            ReaderWriter.dump_data(seq, os.path.join(seq_dir, "sequence"), mode = "wb")
            seqs_info[seq_id] = {'globalfeatures_dir': seq_dir, 'T':len(x_boundaries)}
            
        end_time = datetime.now()
        
        # log progress
        log_file = os.path.join(target_dir, "log.txt")
        line = "---Preparing/parsing sequences--- starting time: {} \n".format(start_time)
        line +=  "Number of sequences prepared/parsed: {}\n".format(len(seqs_dict))
        line += "Corpus directory of the parsed sequences is: {} \n".format(target_dir)
        line += "---Preparing/parsing sequences--- end time: {} \n".format(end_time)
        line += "\n \n"
        ReaderWriter.log_progress(line, log_file)
        
        return(seqs_info)

    def preprocess_attributes(self, seqs_id, seqs_info, method = "rescaling"):
        attr_extractor = self.attr_extractor
        grouped_attr = attr_extractor.group_attributes()
        active_attr = list(self.feature_extractor.template_X.keys())
        active_continuous_attr = [attr for attr in active_attr if attr in grouped_attr['real']]
        attr_dict = {}
        
        #print(continuous_attr)
#         print("active continuous attr {}".format(active_continuous_attr))
        start_time = datetime.now()
        for seq_id in seqs_id:
            seq_dir = seqs_info[seq_id]["globalfeatures_dir"]
            seq = ReaderWriter.read_data(os.path.join(seq_dir, "sequence"), mode = "rb")
            y_boundaries = seq.get_y_boundaries()
            # this will update the value of the seg_attr of the sequence 
            new_boundaries = attr_extractor.generate_attributes(seq, y_boundaries)
            if(new_boundaries):
                ReaderWriter.dump_data(seq, os.path.join(seq_dir, "sequence"), mode = "wb")
            
            for attr_name in active_continuous_attr:
                for y_boundary in y_boundaries:
#                     print("y_boundary {}".format(y_boundary))
                    attr_val = seq.seg_attr[y_boundary][attr_name]
                    if(attr_name in attr_dict):
                        attr_dict[attr_name].append(attr_val)
                    else:
                        attr_dict[attr_name] = [attr_val]  
        
        if(attr_dict):                      
            scaling_info = {}
            if(method == "rescaling"):
                for attr_name in attr_dict:
                    scaling_info[attr_name] = {}
                    scaling_info[attr_name]['max'] = numpy.max(attr_dict[attr_name])
            elif(method == "standardization"):
                for attr_name in attr_dict:
                    scaling_info[attr_name] = {}
                    scaling_info[attr_name]['mean'] = numpy.mean(attr_dict[attr_name])
                    scaling_info[attr_name]['sd'] = numpy.std(attr_dict[attr_name])

#         print("scaling_info {}".format(scaling_info))

            attr_scaler = AttributeScaler(attr_extractor, scaling_info, method)
            self.attr_scaler = attr_scaler
            self.scale_attributes(seqs_id, seqs_info)
        end_time = datetime.now()
                    
        # any sequence would lead to the parent directory of prepared/parsed sequences
        # using the last sequence id and corresponding sequence directory
        target_dir = os.path.dirname(seq_dir)
        log_file = os.path.join(target_dir, "log.txt")
        line = "---Rescaling continuous/real features--- starting time: {} \n".format(start_time)
        line +=  "Number of instances/training data processed: {}\n".format(len(seqs_id))
        line += "---Rescaling continuous/real features--- end time: {} \n".format(end_time)
        line += "\n \n"
        ReaderWriter.log_progress(line, log_file)
        
    def scale_attributes(self, seqs_id, seqs_info):
        attr_scaler = self.attr_scaler
        for seq_id in seqs_id:
            seq_dir = seqs_info[seq_id]["globalfeatures_dir"]
            seq = ReaderWriter.read_data(os.path.join(seq_dir, "sequence"), mode = "rb") 
            boundaries = list(seq.seg_attr.keys())
            attr_scaler.scale_real_attributes(seq, boundaries)
            ReaderWriter.dump_data(seq, os.path.join(seq_dir, "sequence"), mode = "wb")
#             print("sclaed seq {}".format(seq.seg_attr))

    def extract_seqs_globalfeatures(self, seqs_id, seqs_info):
        """ - Function that parses each sequence and generates global feature  F_j(X,Y). 
            - For each sequence we obtain a set of generated global feature functions where each
              F_j(X,Y)) represents the sum of the value of its corresponding low-level/local feature function
             f_j(X,t,y_t,y_tminus1) (i.e. F_j(X,Y) = \sum_{t=1}^{T+1} f_j(X,t, y_t, y_tminus1) )
            - It saves all the results on disk
        """
        feature_extractor = self.feature_extractor
        
        start_time = datetime.now()
        for seq_id in seqs_id:
            seq_dir = seqs_info[seq_id]["globalfeatures_dir"]
            seq = ReaderWriter.read_data(os.path.join(seq_dir, "sequence"), mode = "rb")
            ###### extract the sequence features #####
            # seq_features has this format {'Y_patt':Counter(features)}
            seq_featuresum = feature_extractor.extract_seq_features(seq)    
                    
            # store the features' sum (i.e. F_j(X,Y) for every sequence on disk)
            feature_extractor.dump_features(os.path.join(seq_dir, "globalfeatures"), seq_featuresum)
        end_time = datetime.now()
        
        # any sequence would lead to the parent directory of prepared/parsed sequences
        # using the last sequence id and corresponding sequence directory
        target_dir = os.path.dirname(seq_dir)
        log_file = os.path.join(target_dir, "log.txt")
        line = "---Generating Global Features F_j(X,Y)--- starting time: {} \n".format(start_time)
        line +=  "Number of instances/training data processed: {}\n".format(len(seqs_id))
        line += "---Generating Global Features F_j(X,Y)--- end time: {} \n".format(end_time)
        line += "\n \n"
        ReaderWriter.log_progress(line, log_file)
        
    # this method could be made static -- for now it is instance method
    def create_model(self, seqs_id, seqs_info, model_repr_class, filter_obj = None):
        """ we use the sequences assigned  in the training set to build the model
            To construct the model, this function performs the following:
           - Takes the union of the detected global feature functions F_j(X,Y) for each chosen parsed sequence
             from the training set to form the set of model features
           - Construct the tag set Y_set (i.e. possible tags assumed by y_t) using the chosen parsed sequences
             from the training data set
             
           NOTE: This function requires that the sequences have been already parsed and global features were generated
        """
        Y_states = {}
        modelfeatures = {}
        # length of longest entity in a segment
        L = 1
        
        start_time = datetime.now()
        for seq_id in seqs_id:
            seq_dir = seqs_info[seq_id]["globalfeatures_dir"]
            seq_featuresum = ReaderWriter.read_data(os.path.join(seq_dir, "globalfeatures"))
            seq = ReaderWriter.read_data(os.path.join(seq_dir, "sequence"))
            # get the largest length of an entity in the segment
            if(seq.L > L):
                L = seq.L
            for y_patt, featuresum in seq_featuresum.items():
                if(y_patt in modelfeatures):
                    modelfeatures[y_patt].update(featuresum)
                else:
                    modelfeatures[y_patt] = featuresum
                # record all encountered states/labels
                parts = y_patt.split("|")
                for state in parts:
                    Y_states[state] = 1       
                                     
        end_time = datetime.now()
        
        # apply a filter 
        if(filter_obj):
            # this will trim unwanted features from modelfeatures dictionary
            modelfeatures = filter_obj.apply_filter(modelfeatures)
            
        # create model representation
        model = model_repr_class(modelfeatures, Y_states, L)

        # any sequence would lead to the parent directory
        target_dir = os.path.dirname(seq_dir)
        # log progress
        log_file = os.path.join(target_dir, "log.txt")
        line = "---Constructing model--- starting time: {} \n".format(start_time)
        line += "Number of instances/training data processed: {}\n".format(len(seqs_id))
        line += "Number of features: {} \n".format(model.num_features)
        line += "Number of labels: {} \n".format(model.num_states)
        line += "---Constructing model--- end time: {} \n".format(end_time)
        line += "\n \n"
        ReaderWriter.log_progress(line, log_file)
        
        return(model)

    
    def extract_seqs_modelactivefeatures(self, seqs_id, seqs_info, model, output_foldername):
        # get the root_dir
        seq_id = seqs_id[0]
        seq_dir = seqs_info[seq_id]["globalfeatures_dir"]
        root_dir = os.path.dirname(os.path.dirname(seq_dir))
        output_dir = create_directory("model_activefeatures_{}".format(output_foldername), root_dir)
        L = model.L
        f_extractor = self.feature_extractor
        
        start_time = datetime.now()
        for seq_id in seqs_id:
            # lookup active features for the current sequence and store them on disk
            print("looking for model active features for seq {}".format(seq_id))
            seq_dir = seqs_info[seq_id]["globalfeatures_dir"]
            seq = ReaderWriter.read_data(os.path.join(seq_dir, "sequence"))
            if(L > 1):
                self._lookup_seq_attributes(seq, L)
                ReaderWriter.dump_data(seq, os.path.join(seq_dir, "sequence"), mode = "wb")
            active_features = f_extractor.lookup_seq_modelactivefeatures(seq, model)
            activefeatures_dir = create_directory("seq_{}".format(seq_id), output_dir)
            seqs_info[seq_id]["activefeatures_dir"] = activefeatures_dir
            # dump model active features data
            ReaderWriter.dump_data(active_features, os.path.join(activefeatures_dir, "activefeatures"))
                
        end_time = datetime.now()
       

        log_file = os.path.join(output_dir, "log.txt")
        line = "---Finding sequences' model active-features--- starting time: {} \n".format(start_time)
        line += "Total number of parsed sequences: {} \n".format(len(seqs_id))
        line += "---Finding sequences' model active-features--- end time: {} \n".format(end_time)
        line += "\n \n"
        ReaderWriter.log_progress(line, log_file)
    
    def _lookup_seq_attributes(self, seq, L):
        # generate the missing attributes if the segment length is greater than 1
        attr_extractor = self.attr_extractor
        attr_scaler = self.attr_scaler
        T = seq.T
        for j in range(1, T+1):
            for d in range(L):
                if(j-d <= 0):
                    break
                boundary = (j-d, j)
                if(boundary not in seq.seg_attr):
                    # this will update the value of the seg_attr of the sequence 
                    attr_extractor.generate_attributes(seq, [boundary])
                    attr_scaler.scale_real_attributes(seq, [boundary])
  
    
#     def _unpack_activefeatures(self, activefeatures, model):
#         modelfeatures_codebook_rev = {code:feature for feature, code in model.modelfeatures_codebook.items()}
# #         print("$" *40)
# #         print(activefeatures)
#         for boundary in activefeatures:
#             print("boundary {}".format(boundary))
#             for z_patt, windxfval_dict in activefeatures[boundary].items():
#                 for w_indx, f_val in windxfval_dict.items():
#                     f_name = modelfeatures_codebook_rev[w_indx]
#                     print("{}:{}".format(f_name, f_val))
                
    @staticmethod      
    def get_seqs_modelactivefeatures(seqs_id, seqs_info):
        seqs_activefeatures = {}
        for seq_id in seqs_id:
            seq_dir = seqs_info[seq_id]["activefeatures_dir"]
            active_features = ReaderWriter.read_data(os.path.join(seq_dir,"activefeatures"))
            seqs_activefeatures[seq_id] = active_features
        
        return(seqs_activefeatures)
    
    def get_seqs_globalfeatures(self, seqs_id, seqs_info, model):
        """ it retrieves the features available for the current sequence (i.e. F(X,Y) for all j \in [1...J] 
        """
        seqs_globalfeatures = {}
        for seq_id in seqs_id:
            seq_dir = seqs_info[seq_id]['globalfeatures_dir']
            seq_featuresum = ReaderWriter.read_data(os.path.join(seq_dir, "globalfeatures"))
            windx_fval = model.represent_globalfeatures(seq_featuresum)
            seqs_globalfeatures[seq_id] = windx_fval
            
        return(seqs_globalfeatures)

    # to be used for processing a sequence, generating global features and return back without storing on disk
    def get_imposterseq_globalfeatures(self, seq_id, seqs_info, model, y_imposter, seg_other_symbol = None):
        """ - Function that parses a sequence and generates global feature  F_j(X,Y)
              without saving intermediary results on disk
        """
        feature_extractor = self.feature_extractor
        attr_extractor = self.attr_extractor
        attr_scaler = self.attr_scaler
        #print("seqs_info {}".format(seqs_info))
        seq_dir = seqs_info[seq_id]["globalfeatures_dir"]
        #print("seq_dir {}".format(seq_dir))
        seq = ReaderWriter.read_data(os.path.join(seq_dir, "sequence"), mode = "rb")
        #print("original seq.Y {}".format(seq.Y))
        #print("original y boundaries {}".format(seq.get_y_boundaries()))
        #print("len {}".format(len(seq.Y)))
        
        y_original = list(seq.flat_y)
        seq.Y = (y_imposter, seg_other_symbol)
        y_boundaries = seq.get_y_boundaries()
        #print("imposter seq.Y {}".format(seq.Y))
        #print("imposter y boundaries {}".format(seq.get_y_boundaries()))
        #print("len {}".format(len(seq.Y)))
        
        #print("seq.seg_attr {}".format(seq.seg_attr))
        #print("len(seq")
        # this will update the value of the seg_attr of the sequence 
        new_y_boundaries = attr_extractor.generate_attributes(seq, y_boundaries)
        if(new_y_boundaries):
            attr_scaler.scale_real_attributes(seq, new_y_boundaries)
            ReaderWriter.dump_data(seq, os.path.join(seq_dir, "sequence"), mode = "rb")
            
        seq_imposter_featuresum = feature_extractor.extract_seq_features(seq)  
        windx_fval = model.represent_globalfeatures(seq_imposter_featuresum)
        
        seq.Y = (y_original, seg_other_symbol)

        return(windx_fval)
    
    @staticmethod
    def load_seq(seq_id, seqs_info):
        seq_dir = seqs_info[seq_id]["globalfeatures_dir"]
        seq = ReaderWriter.read_data(os.path.join(seq_dir, "sequence"), mode = "rb")
        return(seq)


class FeatureFilter(object):
    
    def __init__(self, filter_info):
        """ filter_info : dictionary that contains type filter to be applied 
                          it has the following key:values
                          filter_type : string in (count, pattern)
                          filter_val : if filter_type is count, provide a threshold (integer value)
                                       else, provide a list of z patterns 
                          filter_relation: define how to delete/apply filter
            e.g. count filter => filter_info = {'filter_type': 'count', 'filter_val':5, 'filter_relation':'<'}
                 pattern filter => filter_info = {'filter_type': 'pattern', 'filter_val': ["O|L", "L|L"], 'filter_relation':'='}
        """
        self.filter_info = filter_info
        self.rel_func = {"=":self._equal_rel,
                         "<=":self._lequal_rel,
                         "<":self._less_rel,
                         ">=":self._gequal_rel,
                         ">":self._greater_rel,
                         "in":self._in_rel,
                         "not in":self._notin_rel}
        
    def apply_filter(self, featuresum_dict):
        filtered_dict = deepcopy(featuresum_dict)
        filter_info = self.filter_info
        rel_func = self.rel_func
        if(filter_info['filter_type'] == "count"):
            threshold = filter_info['filter_val']
            relation = filter_info['filter_realtion']
            # filter binary features that have counts less than specified threshold
            for z in featuresum_dict:
                for fname, fsum in featuresum_dict[z].items():
                    # determine if the feature is binary
                    if(type(fsum) == int):
                        rel_func[relation](fsum, threshold, filtered_dict[z][fname])
#                         if(relation == "="):
#                             if(fsum == threshold):
#                                 del filtered_dict[z][fname] 
#                         elif(relation == "<="):
#                             if(fsum <= threshold):
#                                 del filtered_dict[z][fname] 
#                         elif(relation == "<"):
#                             if(fsum < threshold):
#                                 del filtered_dict[z][fname] 
#                         elif(relation == ">="):
#                             if(fsum >= threshold):
#                                 del filtered_dict[z][fname] 
#                         elif(relation == ">"):
#                             if(fsum > threshold):
#                                 del filtered_dict[z][fname] 
                        

                            
        elif(filter_info['filter_type'] == "pattern"):
            filter_pattern = filter_info['filter_val']
            relation = filter_info['filter_relation']
            # filter based on specific patterns
            for z in featuresum_dict:
                rel_func[relation](z, filter_pattern, filtered_dict[z])
#                 if(relation == "="):
#                     # delete any z that matches any of the provided filter patterns
#                     if(z in filter_pattern):
#                         del filtered_dict[z]
#                 elif(relation == "!="):
# 
#                     # delete any z that does not match any of the provided filter patterns
#                     if(z not in filter_pattern):
#                         print("deleting z {}".format(z))
# 
#                         del filtered_dict[z]
        return(filtered_dict)
    
    @staticmethod
    def _equal_rel(x, y, z):
        if(x==y): del z

    @staticmethod
    def _lequal_rel(x, y, z):
        if(x<=y): del z
        
    @staticmethod
    def _less_rel(x, y, z):
        if(x<y): del z
        
    @staticmethod
    def _gequal_rel(x, y, z):
        if(x>=y): del z
        
    @staticmethod
    def _greater_rel(x, y, z):
        if(x>y): del z

    @staticmethod
    def _in_rel(x, y, z):
        if(x in y): del z
    @staticmethod
    def _notin_rel(x, y, z):
        if(x not in y): del z

    