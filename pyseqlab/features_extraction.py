'''
@author: ahmed allam <ahmed.allam@yale.edu>
'''

import pickle
import os
from copy import deepcopy
from datetime import datetime
from collections import Counter
import numpy
from pyseqlab.utilities import ReaderWriter, create_directory, generate_datetime_str
from pyseqlab.attributes_extraction import AttributeScaler


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
            self.y_offsets = set()
            self.x_featurenames = {}
            for attr_name, templateX in template.items():
                self._template_X[attr_name] = {}
                self.x_featurenames[attr_name] = {}
                for offset_x, offsets_y in templateX.items():
                    s_offset_x = tuple(sorted(offset_x))
                    feature_name = '|'.join([attr_name + "[" + str(ofst_x) + "]"  for ofst_x in s_offset_x])
                    self.x_featurenames[attr_name][offset_x] = feature_name
                    unique_dict = {}
                    for offset_y in offsets_y:
                        s_offset_y = tuple(sorted(offset_y))
                        check = self._validate_template(s_offset_y)
                        if(check):
                            unique_dict[s_offset_y] = 1
                            self.y_offsets.add(s_offset_y)
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
                    
                
    def extract_seq_features_perboundary(self, seq, seg_features=None):
        # this method is used to extract features from sequences with known labels
        # (i.e. we know the Y labels and boundaries)
        Y = seq.Y
        features = {}
        for boundary in Y:
            xy_feat = self.extract_features_XY(seq, boundary, seg_features)
            y_feat = self.extract_features_Y(seq, boundary, self.template_Y)
            y_feat = y_feat['Y']
            #print("boundary {}".format(boundary))
            #print("boundary {}".format(boundary))
            #print("y_feat {}".format(y_feat))
            #print("xy_feat {}".format(xy_feat))
            #TOUPDATEEEEE
            for offset_tup_y in y_feat:
                for y_patt in y_feat[offset_tup_y]:
                    if(y_patt in xy_feat):
                        xy_feat[y_patt].update(y_feat[offset_tup_y])
                    else:
                        xy_feat[y_patt] = y_feat[offset_tup_y]
            features[boundary] = xy_feat
#             #print("features {}".format(features[boundary]))
#             #print("*"*40)
                
        return(features)

    
    def aggregate_seq_features(self, features, boundaries):
        # summing up all local features across all boundaries
        seq_features = {}
        for boundary in boundaries:
            xy_features = features[boundary]
            for y_patt in xy_features:
                if(y_patt in seq_features):
                    seq_features[y_patt].update(xy_features[y_patt])
#                     seq_features[y_patt] + xy_features[y_patt]
                else:
                    seq_features[y_patt] = Counter(xy_features[y_patt])
        return(seq_features)
    
#     def extract_seq_features(self, seq):
#         features_per_boundary = self.extract_seq_features_perboundary(seq)
#         seq_features = self.agggregate_features(features_per_boundary, boundaries=seq.Y)
#         return(seq_features)
    
    def extract_features_Y(self, seq, boundary, templateY):
        """ template_Y = {'Y': ((0,), (-1,0), (-2,-1,0))}
                       = {Y: tuple(y_offsets)}        
        """
        template_Y = templateY['Y']

        if(template_Y):
            Y = seq.Y
            y_sboundaries = seq.y_sboundaries
            y_boundpos_map = seq.y_boundpos_map
            curr_pos = y_boundpos_map[boundary]
            range_y = seq.y_range

            y_patt_features = {}
            feat_template = {}
            for offset_tup_y in template_Y:
                y_pattern = []
                for offset_y in offset_tup_y:
                    # offset_y should be always <= 0
                    pos = curr_pos + offset_y
                    if(pos in range_y):
                        b = y_sboundaries[pos]
                        y_pattern.append(Y[b])
                    else:
                        y_pattern = []
                        break 
                if(y_pattern):
                    feat_template[offset_tup_y] = {"|".join(y_pattern):1}
    
            y_patt_features['Y'] = feat_template
            
        else:
            y_patt_features = {'Y':{}}
        
#         #print("X"*40)
#         #print("boundary {}".format(boundary))
#         for attr_name, f_template in y_patt_features.items():
#             for offset, features in f_template.items():
#                 #print("{} -> {}".format(offset, features))
#         #print("X"*40)
        
        return(y_patt_features)
    
    def extract_features_X(self, seq, boundary):
        """template_X = {'w': {(0,):((0,), (-1,0), (-2,-1,0))}}
           boundary is a tuple (u,v) indicating the beginning and end of the current segment
        """
        # get template X
        template_X = self.template_X
        attr_desc = self.attr_desc
        x_featurenames = self.x_featurenames
        # current boundary begin and end
        u, v = boundary

#         #print("positions {}".format(positions))
        seg_features = {}
        for attr_name in template_X:
#             #print("attr_name {}".format(attr_name))
            # check the type of attribute
            if(attr_desc[attr_name]["encoding"] == "binary"):
                represent_attr = self._represent_binary_attr
            else:
                represent_attr = self._represent_real_attr
            
            feat_template = {}
            for offset_tup_x in template_X[attr_name]:
                attributes = []
#                 #print("feature_name {}".format(feature_name))
                for offset_x in offset_tup_x:
#                     #print("offset_x {}".format(offset_x))
                    if(offset_x > 0):
                        pos = (v + offset_x, v + offset_x)
                    elif(offset_x < 0):
                        pos = (u + offset_x, u + offset_x)
                    else:
                        pos = (u, v)
                   
                    if(pos in seq.seg_attr):
                        attributes.append(seq.seg_attr[pos][attr_name])
#                         #print("attributes {}".format(attributes))
                    else:
                        attributes = []
                        break
                if(attributes):
#                     feat_template[offset_tup_x] = represent_attr(attributes, feature_name)
                    feat_template[offset_tup_x] = represent_attr(attributes, x_featurenames[attr_name][offset_tup_x])
            seg_features[attr_name] = feat_template
#         
#         #print("X"*40)
#         #print("boundary {}".format(boundary))
#         for attr_name, f_template in seg_features.items():
#             for offset, features in f_template.items():
#                 #print("{} -> {}".format(offset, features))
#         #print("X"*40)

        return(seg_features)

    
    def extract_features_XY(self, seq, boundary, seg_features = None):
        """ template_X = {'w': {(0,):((0,), (-1,0), (-2,-1,0))}}
            template_Y = {'Y': ((0,), (-1,0), (-2,-1,0))}
        """
        if(not seg_features):
            seg_feat_templates = self.extract_features_X(seq, boundary)
        else:
            seg_feat_templates = seg_features[boundary]
        y_feat_template = self.extract_features_Y(seq, boundary, {'Y':self.y_offsets})
#         print(y_feat_template)
#         print(self.y_offsets)
        y_feat_template = y_feat_template['Y']
        templateX = self.template_X

#         #print("seg_feat_templates {}".format(seg_feat_templates))
        xy_features = {}
        for attr_name, seg_feat_template in seg_feat_templates.items():
            for offset_tup_x in seg_feat_template:
                for offset_tup_y in templateX[attr_name][offset_tup_x]:
                    if(offset_tup_y in y_feat_template):
                        for y_patt in y_feat_template[offset_tup_y]:
                            if(y_patt in xy_features):
                                xy_features[y_patt].update(seg_feat_template[offset_tup_x])
                            else:
                                xy_features[y_patt] = seg_feat_template[offset_tup_x]
#                         #print("xy_features {}".format(xy_features))
        return(xy_features)
    
    def lookup_features_X(self, seq, boundary):
        """template_X = {'w': {(0,):((0,), (-1,0), (-2,-1,0))}}
           boundary is a tuple (u,v) indicating the beginning and end of the current segment
           This method is used to lookup features using varying boundaries (i.e. g(X, u, v))
        """
        # get template X
        template_X = self.template_X
        attr_desc = self.attr_desc
        x_featurenames = self.x_featurenames
        # current boundary begin and end
        u = boundary[0]
        v = boundary[-1]
        
#         #print("positions {}".format(positions))
        seg_features = {}
        for attr_name in template_X:
#             #print("attr_name {}".format(attr_name))
            # check the type of attribute
            if(attr_desc[attr_name]["encoding"] == "binary"):
#                 aggregate_attr = self._aggregate_binary_attr
                represent_attr = self._represent_binary_attr
            else:
#                 aggregate_attr = self._aggregate_real_attr
                represent_attr = self._represent_real_attr
            
            for offset_tup_x in template_X[attr_name]:
                attributes = []
#                 feature_name = '|'.join(['{}[{}]'.format(attr_name, offset_x) for offset_x in offset_tup_x])
#                 #print("feature_name {}".format(feature_name))
                for offset_x in offset_tup_x:
#                     #print("offset_x {}".format(offset_x))
                    if(offset_x > 0):
                        pos = (v + offset_x, v + offset_x)
                    elif(offset_x < 0):
                        pos = (u + offset_x, u + offset_x)
                    else:
                        pos = (u, v)
#                     #print("pos {}".format(pos))
                    
                    if(pos in seq.seg_attr):
                        attributes.append(seq.seg_attr[pos][attr_name])
#                         #print("attributes {}".format(attributes))
                    else:
                        attributes = []
                        break
                if(attributes):
                    seg_features.update(represent_attr(attributes, x_featurenames[attr_name][offset_tup_x]))

#         #print("seg_features lookup {}".format(seg_features))
        return(seg_features)

    def flatten_segfeatures(self, seg_features):
        flat_segfeatures = {}
        for attr_name in seg_features:
            for offset in seg_features[attr_name]:
                flat_segfeatures.update(seg_features[attr_name][offset])
        return(flat_segfeatures)
        
    def lookup_seq_modelactivefeatures(self, seq, model, learning=False):
        # segment length
        L = model.L
        T = seq.T
        # maximum pattern length 
        max_patt_len = model.max_patt_len
        patts_len = model.patts_len
        ypatt_activestates = model.ypatt_activestates
        activated_states = {}
        seg_features = {}
        l_segfeatures = {}
            
        for j in range(1, T+1):
            for d in range(L):
                if(j-d <= 0):
                    break
                # start boundary
                u = j-d
                # end boundary
                v = j
                boundary = (u, v)
                
                if(u < max_patt_len):
                    max_len = u
                else:
                    max_len = max_patt_len
                    
                allowed_z_len = {z_len for z_len in patts_len if z_len <= max_len}
                
                # used in the case of model training
                if(learning):
                    l_segfeatures[boundary] = self.extract_features_X(seq, boundary)
                    seg_features[boundary] = self.flatten_segfeatures(l_segfeatures[boundary])
                else:
                    seg_features[boundary] = self.lookup_features_X(seq, boundary)
                    
                activated_states[boundary] = model.find_activated_states(seg_features[boundary], allowed_z_len)
                #^print("allowed_z_len ", allowed_z_len)
                #^print("seg_features ", seg_features)
                #^print("activated_states ", activated_states)
                if(ypatt_activestates):
                    ypatt_activated_states = {z_len:ypatt_activestates[z_len] for z_len in allowed_z_len if z_len in ypatt_activestates}
                    #^print("ypatt_activated_states ", ypatt_activated_states)
                    for zlen, ypatts in ypatt_activated_states.items():
                        if(zlen in activated_states[boundary]):
                            activated_states[boundary][zlen].update(ypatts)
                        else:
                            activated_states[boundary][zlen] = set(ypatts) 
                #^print("activated_states ", activated_states)
        #^print("activated_states from feature_extractor ", activated_states)
        #^print("seg_features from feature_extractor ", seg_features)

        return(activated_states, seg_features, l_segfeatures)
    
    
    ########################################################
    # functions used to represent real and binary attributes
    ########################################################

    def _represent_binary_attr(self, attributes, feature_name):
#         #print("attributes ",attributes)
#         #print("featurename ", feature_name)
        feature_val = '|'.join(attributes)
#         feature = '{}={}'.format(feature_name, feature_val)
        feature = feature_name + "=" + feature_val
        return({feature:1})

    def _represent_real_attr(self, attributes, feature_name):
        feature_val = sum(attributes) 
        return({feature_name:feature_val})
    
    def save(self, folder_dir):
        """store the templates used -- templateX and templateY"""
        save_info = {'FE_templateX': self.template_X,
                     'FE_templateY': self.template_Y
                    }
        for name in save_info:
            ReaderWriter.dump_data(save_info[name], os.path.join(folder_dir, name))
        

class FOFeatureExtractor(object):
    """ Generic feature extractor class that contains feature functions/templates for first order sequence models """
    # currently it supports adding start state
    # to consider adding support for stop state
    def __init__(self, templateX, templateY, attr_desc, start_state = True):
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
            template_X = {'w': {(0,):((0,), (-1,0), (-2,-1,0))}}
                       = {attr_name: {x_offset:tuple(y_offsets)}}
        """
        if(type(template) == dict):
            self._template_X = {}
            self.y_offsets = set()
            self.x_featurenames = {}
            for attr_name, templateX in template.items():
                self._template_X[attr_name] = {}
                self.x_featurenames[attr_name] = {}
                for offset_x, offsets_y in templateX.items():
                    s_offset_x = tuple(sorted(offset_x))
                    feature_name = '|'.join([attr_name + "[" + str(ofst_x) + "]"  for ofst_x in s_offset_x])
                    self.x_featurenames[attr_name][offset_x] = feature_name
                    unique_dict = {}
                    for offset_y in offsets_y:
                        s_offset_y = tuple(sorted(offset_y))
                        check = self._validate_template(s_offset_y)
                        if(check):
                            unique_dict[s_offset_y] = 1
                            self.y_offsets.add(s_offset_y)
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
        """template is a tuple (i.e. (-1,0)"""
        valid_offsets = {(0,), (-1,0)}
        if(template in valid_offsets):
            check = True
        else:
            check = False
        
        return(check)
    
    def extract_seq_features_perboundary(self, seq, seg_features=None):
        # this method is used to extract features from sequences with known labels
        # (i.e. we know the Y labels and boundaries)
        Y = seq.Y
        features = {}
        for boundary in Y:
            xy_feat = self.extract_features_XY(seq, boundary, seg_features)
            y_feat = self.extract_features_Y(seq, boundary, self.template_Y)
            y_feat = y_feat['Y']
            #print("boundary {}".format(boundary))
            #print("boundary {}".format(boundary))
            #print("y_feat {}".format(y_feat))
            #print("xy_feat {}".format(xy_feat))
            #TOUPDATEEEEE
            for offset_tup_y in y_feat:
                for y_patt in y_feat[offset_tup_y]:
                    if(y_patt in xy_feat):
                        xy_feat[y_patt].update(y_feat[offset_tup_y])
                    else:
                        xy_feat[y_patt] = y_feat[offset_tup_y]
            features[boundary] = xy_feat
#             #print("features {}".format(features[boundary]))
#             #print("*"*40)
                
        return(features)

    
    def aggregate_seq_features(self, features, boundaries):
        # summing up all local features across all boundaries
        seq_features = {}
        for boundary in boundaries:
            xy_features = features[boundary]
            for y_patt in xy_features:
                if(y_patt in seq_features):
                    seq_features[y_patt].update(xy_features[y_patt])
#                     seq_features[y_patt] + xy_features[y_patt]
                else:
                    seq_features[y_patt] = Counter(xy_features[y_patt])
        return(seq_features)
    
    def extract_features_Y(self, seq, boundary, templateY):
        """ template_Y = {'Y': ((0,), (-1,0), (-2,-1,0))}
                       = {Y: tuple(y_offsets)}        
        """
        template_Y = templateY['Y']

        if(template_Y):
            Y = seq.Y
            y_sboundaries = seq.y_sboundaries
            y_boundpos_map = seq.y_boundpos_map
            curr_pos = y_boundpos_map[boundary]
            range_y = seq.y_range
            startstate_flag = self.start_state
            
            y_patt_features = {}
            feat_template = {}
            for offset_tup_y in template_Y:
                y_pattern = []
                for offset_y in offset_tup_y:
                    # offset_y should be always <= 0
                    pos = curr_pos + offset_y
                    if(pos in range_y):
                        b = y_sboundaries[pos]
                        y_pattern.append(Y[b])
                    else:
                        if(startstate_flag and pos == -1):
                            y_pattern.append("__START__")
                        else:
                            y_pattern = []
                            break
                if(y_pattern):
                    feat_template[offset_tup_y] = {"|".join(y_pattern):1}
    
            y_patt_features['Y'] = feat_template
            
        else:
            y_patt_features = {'Y':{}}

        return(y_patt_features)
    
    def extract_features_X(self, seq, boundary):
        """template_X = {'w': {(0,):((0,), (-1,0), (-2,-1,0))}}
           boundary is a tuple (u,v) indicating the beginning and end of the current segment
        """
        # get template X
        template_X = self.template_X
        attr_desc = self.attr_desc
        x_featurenames = self.x_featurenames
        # current boundary begin and end
        u, v = boundary

#         #print("positions {}".format(positions))
        seg_features = {}
        for attr_name in template_X:
#             #print("attr_name {}".format(attr_name))
            # check the type of attribute
            if(attr_desc[attr_name]["encoding"] == "binary"):
                represent_attr = self._represent_binary_attr
            else:
                represent_attr = self._represent_real_attr
            
            feat_template = {}
            for offset_tup_x in template_X[attr_name]:
                attributes = []
#                 #print("feature_name {}".format(feature_name))
                for offset_x in offset_tup_x:
#                     #print("offset_x {}".format(offset_x))
                    if(offset_x > 0):
                        pos = (v + offset_x, v + offset_x)
                    elif(offset_x < 0):
                        pos = (u + offset_x, u + offset_x)
                    else:
                        pos = (u, v)
                   
                    if(pos in seq.seg_attr):
                        attributes.append(seq.seg_attr[pos][attr_name])
#                         #print("attributes {}".format(attributes))
                    else:
                        attributes = []
                        break
                if(attributes):
#                     feat_template[offset_tup_x] = represent_attr(attributes, feature_name)
                    feat_template[offset_tup_x] = represent_attr(attributes, x_featurenames[attr_name][offset_tup_x])
            seg_features[attr_name] = feat_template
#         
#         #print("X"*40)
#         #print("boundary {}".format(boundary))
#         for attr_name, f_template in seg_features.items():
#             for offset, features in f_template.items():
#                 #print("{} -> {}".format(offset, features))
#         #print("X"*40)

        return(seg_features)

    
    def extract_features_XY(self, seq, boundary, seg_features = None):
        """ template_X = {'w': {(0,):((0,), (-1,0), (-2,-1,0))}}
            template_Y = {'Y': ((0,), (-1,0), (-2,-1,0))}
        """
        if(not seg_features):
            seg_feat_templates = self.extract_features_X(seq, boundary)
        else:
            seg_feat_templates = seg_features[boundary]
        y_feat_template = self.extract_features_Y(seq, boundary, {'Y':self.y_offsets})
        y_feat_template = y_feat_template['Y']
        templateX = self.template_X

#         #print("seg_feat_templates {}".format(seg_feat_templates))
        xy_features = {}
        for attr_name, seg_feat_template in seg_feat_templates.items():
            for offset_tup_x in seg_feat_template:
                for offset_tup_y in templateX[attr_name][offset_tup_x]:
                    if(offset_tup_y in y_feat_template):
                        for y_patt in y_feat_template[offset_tup_y]:
                            if(y_patt in xy_features):
                                xy_features[y_patt].update(seg_feat_template[offset_tup_x])
                            else:
                                xy_features[y_patt] = seg_feat_template[offset_tup_x]
#                         #print("xy_features {}".format(xy_features))
        return(xy_features)
    
    def lookup_features_X(self, seq, boundary):
        """template_X = {'w': {(0,):((0,), (-1,0), (-2,-1,0))}}
           boundary is a tuple (u,v) indicating the beginning and end of the current segment
           This method is used to lookup features using varying boundaries (i.e. g(X, u, v))
        """
        # get template X
        template_X = self.template_X
        attr_desc = self.attr_desc
        x_featurenames = self.x_featurenames
        # current boundary begin and end
        u = boundary[0]
        v = boundary[-1]
        
#         #print("positions {}".format(positions))
        seg_features = {}
        for attr_name in template_X:
#             #print("attr_name {}".format(attr_name))
            # check the type of attribute
            if(attr_desc[attr_name]["encoding"] == "binary"):
#                 aggregate_attr = self._aggregate_binary_attr
                represent_attr = self._represent_binary_attr
            else:
#                 aggregate_attr = self._aggregate_real_attr
                represent_attr = self._represent_real_attr
            
            for offset_tup_x in template_X[attr_name]:
                attributes = []
#                 feature_name = '|'.join(['{}[{}]'.format(attr_name, offset_x) for offset_x in offset_tup_x])
#                 #print("feature_name {}".format(feature_name))
                for offset_x in offset_tup_x:
#                     #print("offset_x {}".format(offset_x))
                    if(offset_x > 0):
                        pos = (v + offset_x, v + offset_x)
                    elif(offset_x < 0):
                        pos = (u + offset_x, u + offset_x)
                    else:
                        pos = (u, v)
#                     #print("pos {}".format(pos))
                    
                    if(pos in seq.seg_attr):
                        attributes.append(seq.seg_attr[pos][attr_name])
#                         #print("attributes {}".format(attributes))
                    else:
                        attributes = []
                        break
                if(attributes):
                    seg_features.update(represent_attr(attributes, x_featurenames[attr_name][offset_tup_x]))

#         #print("seg_features lookup {}".format(seg_features))
        return(seg_features)

    def flatten_segfeatures(self, seg_features):
        flat_segfeatures = {}
        for attr_name in seg_features:
            for offset in seg_features[attr_name]:
                flat_segfeatures.update(seg_features[attr_name][offset])
        return(flat_segfeatures)
        
    def lookup_seq_modelactivefeatures(self, seq, model, learning=False):
        # segment length
        L = model.L
        T = seq.T
        # maximum pattern length 
        max_patt_len = model.max_patt_len
        patts_len = model.patts_len
        ypatt_activestates = model.ypatt_activestates
        startstate_flag = self.start_state
        
        activated_states = {}
        seg_features = {}
        l_segfeatures = {}
            
        for j in range(1, T+1):
            for d in range(L):
                if(j-d <= 0):
                    break
                # start boundary
                u = j-d
                # end boundary
                v = j
                boundary = (u, v)
                
                if(u < max_patt_len):
                    if(startstate_flag):
                        max_len = max_patt_len
                    else:
                        max_len = u
                else:
                    max_len = max_patt_len
                    
                allowed_z_len = {z_len for z_len in patts_len if z_len <= max_len}
                
                # used while learning model parameters
                if(learning):
                    l_segfeatures[boundary] = self.extract_features_X(seq, boundary)
                    seg_features[boundary] = self.flatten_segfeatures(l_segfeatures[boundary])
                else:
                    seg_features[boundary] = self.lookup_features_X(seq, boundary)
                    
                activated_states[boundary] = model.find_activated_states(seg_features[boundary], allowed_z_len)
                #^print("allowed_z_len ", allowed_z_len)
                #^print("seg_features ", seg_features)
                #^print("activated_states ", activated_states)
                if(ypatt_activestates):
                    ypatt_activated_states = {z_len:ypatt_activestates[z_len] for z_len in allowed_z_len if z_len in ypatt_activestates}
                    #^print("ypatt_activated_states ", ypatt_activated_states)
                    for zlen, ypatts in ypatt_activated_states.items():
                        if(zlen in activated_states[boundary]):
                            activated_states[boundary][zlen].update(ypatts)
                        else:
                            activated_states[boundary][zlen] = set(ypatts) 
                #^print("activated_states ", activated_states)
        #^print("activated_states from feature_extractor ", activated_states)
        #^print("seg_features from feature_extractor ", seg_features)

        return(activated_states, seg_features, l_segfeatures)
    
    
    ########################################################
    # functions used to represent real and binary attributes
    ########################################################

    def _represent_binary_attr(self, attributes, feature_name):
#         #print("attributes ",attributes)
#         #print("featurename ", feature_name)
        feature_val = '|'.join(attributes)
#         feature = '{}={}'.format(feature_name, feature_val)
        feature = feature_name + "=" + feature_val
        return({feature:1})

    def _represent_real_attr(self, attributes, feature_name):
        feature_val = sum(attributes) 
        return({feature_name:feature_val})   
     
    def save(self, folder_dir):
        """store the templates used -- templateX and templateY"""
        save_info = {'FE_templateX': self.template_X,
                     'FE_templateY': self.template_Y
                    }
        for name in save_info:
            ReaderWriter.dump_data(save_info[name], os.path.join(folder_dir, name))

        

class SeqsRepresenter(object):
    def __init__(self, attr_extractor, fextractor):
        self.attr_extractor = attr_extractor
        self.feature_extractor = fextractor
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
            # this will update the seg_attr of the sequence 
            attr_extractor.generate_attributes(seq, x_boundaries)
            # create a folder for every sequence
            seq_dir = create_directory("seq_{}".format(seq_id), target_dir)
            ReaderWriter.dump_data(seq, os.path.join(seq_dir, "sequence"), mode = "wb")
            seqs_info[seq_id] = {'globalfeatures_dir': seq_dir, 'T':seq.T, 'L':seq.L}
            
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
        """process attributes
        """
        attr_extractor = self.attr_extractor
        grouped_attr = attr_extractor.group_attributes()
        if(grouped_attr.get("real")):
            active_attr = list(self.feature_extractor.template_X.keys())
            active_continuous_attr = [attr for attr in active_attr if attr in grouped_attr['real']]
        else:
            active_continuous_attr = {}
            
        attr_dict = {}
        
        seq_dir = None
        start_time = datetime.now()
        for seq_id in seqs_id:
            # length of longest entity in a sequence
            seq_L = seqs_info[seq_id]['L']
            if(seq_L > 1 or active_continuous_attr):
                seq_dir = seqs_info[seq_id]["globalfeatures_dir"]
                seq = ReaderWriter.read_data(os.path.join(seq_dir, "sequence"), mode = "rb")
                y_boundaries = seq.y_sboundaries
            # generate attributes for segments 
            if(seq_L>1):
                # this will update the value of the seg_attr of the sequence 
                new_boundaries = attr_extractor.generate_attributes(seq, y_boundaries)
                # this condition might be redundant -- consider to remove and directly dump the sequence
                if(new_boundaries):
                    ReaderWriter.dump_data(seq, os.path.join(seq_dir, "sequence"), mode = "wb")
            
            # gather stats for rescaling/standardizing real/continuous variables
            if(active_continuous_attr):
                for attr_name in active_continuous_attr:
                    for y_boundary in y_boundaries:
                        attr_val = seq.seg_attr[y_boundary][attr_name]
                        if(attr_name in attr_dict):
                            attr_dict[attr_name].append(attr_val)
                        else:
                            attr_dict[attr_name] = [attr_val]  
                            
        # generate attribute scaler object
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

            attr_scaler = AttributeScaler(scaling_info, method)
            self.attr_scaler = attr_scaler
            # scale the attributes
            self.scale_attributes(seqs_id, seqs_info)
        end_time = datetime.now()
                    
        # any sequence would lead to the parent directory of prepared/parsed sequences
        # using the last sequence id and corresponding sequence directory
        if(seq_dir):
            target_dir = os.path.dirname(seq_dir)
            log_file = os.path.join(target_dir, "log.txt")
            line = "---Rescaling continuous/real features--- starting time: {} \n".format(start_time)
            line +=  "Number of instances/training data processed: {}\n".format(len(seqs_id))
            line += "---Rescaling continuous/real features--- end time: {} \n".format(end_time)
            line += "\n \n"
            ReaderWriter.log_progress(line, log_file)
        
    def scale_attributes(self, seqs_id, seqs_info):
        attr_scaler = self.attr_scaler
        if(attr_scaler):
            for seq_id in seqs_id:
                seq_dir = seqs_info[seq_id]["globalfeatures_dir"]
                seq = ReaderWriter.read_data(os.path.join(seq_dir, "sequence"), mode = "rb") 
                boundaries = list(seq.seg_attr.keys())
                attr_scaler.scale_real_attributes(seq, boundaries)
                ReaderWriter.dump_data(seq, os.path.join(seq_dir, "sequence"), mode = "wb")

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
            # extract the sequence global features per boundary
            gfeatures_perboundary = feature_extractor.extract_seq_features_perboundary(seq)   
            y_boundaries = seq.y_sboundaries
            # gfeatures has this format {'Y_patt':Counter(features)}
            gfeatures = feature_extractor.aggregate_seq_features(gfeatures_perboundary, y_boundaries)                 
            # store the features' sum (i.e. F_j(X,Y) for every sequence on disk)
            ReaderWriter.dump_data(gfeatures, os.path.join(seq_dir, "globalfeatures"))
            ReaderWriter.dump_data(gfeatures_perboundary, os.path.join(seq_dir, "globalfeatures_per_boundary"))
            print("dumping seq with id ", seq_id)

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
        # length of default entity in a segment
        L = 1
        
        start_time = datetime.now()
        for seq_id in seqs_id:
            seq_dir = seqs_info[seq_id]["globalfeatures_dir"]
            # get the largest length of an entity in the segment
            seq_L = seqs_info[seq_id]['L']
            if(seq_L > L):
                L = seq_L
                
            gfeatures = ReaderWriter.read_data(os.path.join(seq_dir, "globalfeatures"))
            # generate a global vector for the model    
            for y_patt, featuresum in gfeatures.items():
                if(y_patt in modelfeatures):
                    modelfeatures[y_patt].update(featuresum)
                else:
                    modelfeatures[y_patt] = featuresum
                # record all encountered states/labels
                parts = y_patt.split("|")
                for state in parts:
                    Y_states[state] = 1       
                                     
        # apply a filter 
        if(filter_obj):
            # this will trim unwanted features from modelfeatures dictionary
            modelfeatures = filter_obj.apply_filter(modelfeatures)
            #^print("modelfeatures ", modelfeatures)
            
        # create model representation
        model = model_repr_class()
        model.create_model(modelfeatures, Y_states, L)

        end_time = datetime.now()

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

    def extract_seqs_modelactivefeatures(self, seqs_id, seqs_info, model, output_foldername, learning=False):
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
            activated_states, seg_features, l_segfeatures = f_extractor.lookup_seq_modelactivefeatures(seq, model, learning=learning)

            # dump model active features data
            activefeatures_dir = create_directory("seq_{}".format(seq_id), output_dir)
            seqs_info[seq_id]["activefeatures_dir"] = activefeatures_dir
            ReaderWriter.dump_data(activated_states, os.path.join(activefeatures_dir, "activated_states"))
            ReaderWriter.dump_data(seg_features, os.path.join(activefeatures_dir, "seg_features"))
            # to add condition regarding learning
            ReaderWriter.dump_data(l_segfeatures, os.path.join(activefeatures_dir, "l_segfeatures"))

            
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
                    if(attr_scaler):
                        attr_scaler.scale_real_attributes(seq, [boundary])
            
    
    def get_seq_activatedstates(self, seq_id, seqs_info):
        seq_dir = seqs_info[seq_id]["activefeatures_dir"]
        activated_states = ReaderWriter.read_data(os.path.join(seq_dir,"activated_states"))
        return(activated_states)
    
    def get_seq_segfeatures(self, seq_id, seqs_info):
        seq_dir = seqs_info[seq_id]["activefeatures_dir"]
        seg_features = ReaderWriter.read_data(os.path.join(seq_dir, "seg_features"))
        return(seg_features)
    
    def get_seq_lsegfeatures(self, seq_id, seqs_info):
        seq_dir = seqs_info[seq_id]["activefeatures_dir"]
        seg_features = ReaderWriter.read_data(os.path.join(seq_dir, "l_segfeatures"))
        return(seg_features)
    
    def get_seq_activefeatures(self, seq_id, seqs_info):
        seq_dir = seqs_info[seq_id]["activefeatures_dir"]
        try:
            activefeatures = ReaderWriter.read_data(os.path.join(seq_dir, "activefeatures"))
        except FileNotFoundError:
            # consider logging the error
            print("activefeatures_per_boundary file does not exist yet !!")
            activefeatures = None
        finally:
            return(activefeatures)
        
    def get_seq_globalfeatures(self, seq_id, seqs_info, per_boundary=True):
        """it retrieves the features available for the current sequence (i.e. F(X,Y) for all j \in [1...J] 
        """
        seq_dir = seqs_info[seq_id]['globalfeatures_dir']
        if(per_boundary):
            fname = "globalfeatures_per_boundary"
        else:
            fname = "globalfeatures"
        gfeatures = ReaderWriter.read_data(os.path.join(seq_dir, fname))
        return(gfeatures)
    
    def aggregate_gfeatures(self, gfeatures, boundaries):
        feature_extractor = self.feature_extractor
        # gfeatures is assumed to be represented by boundaries
        gfeatures = feature_extractor.aggregate_seq_features(gfeatures, boundaries)
        return(gfeatures)
    
    def represent_gfeatures(self, gfeatures, model, boundaries=None):
        feature_extractor = self.feature_extractor
        # if boundaries is specified, then gfeatures is assumed to be represented by boundaries
        if(boundaries):
            gfeatures = feature_extractor.aggregate_seq_features(gfeatures, boundaries)
        #^print("gfeatures ", gfeatures)
        windx_fval = model.represent_globalfeatures(gfeatures)
        return(windx_fval)
    
    @staticmethod
    def load_seq(seq_id, seqs_info):
        seq_dir = seqs_info[seq_id]["globalfeatures_dir"]
        seq = ReaderWriter.read_data(os.path.join(seq_dir, "sequence"), mode = "rb")
        return(seq)
            
    def get_imposterseq_globalfeatures(self, seq_id, seqs_info, y_imposter, seg_other_symbol = None):
        """to be used for processing a sequence, generating global features and return back without storing on disk
            Function that parses a sequence and generates global feature  F_j(X,Y)
            without saving intermediary results on disk
        """
        feature_extractor = self.feature_extractor
        attr_extractor = self.attr_extractor
        attr_scaler = self.attr_scaler
        ##print("seqs_info {}".format(seqs_info))
        seq_dir = seqs_info[seq_id]["globalfeatures_dir"]
        ##print("seq_dir {}".format(seq_dir))
        seq = ReaderWriter.read_data(os.path.join(seq_dir, "sequence"), mode = "rb")
        
        y_ref = list(seq.flat_y)        
        # update seq.Y with the imposter Y
        seq.Y = (y_imposter, seg_other_symbol)
        y_imposter_boundaries = seq.y_sboundaries
        #^print("y_imposter_boundaries ", y_imposter_boundaries)
        # this will update the value of the seg_attr of the sequence 
        new_boundaries = attr_extractor.generate_attributes(seq, y_imposter_boundaries)
        #^print("new_boundaries ", new_boundaries)
        if(new_boundaries):
            attr_scaler.scale_real_attributes(seq, new_boundaries)
            
        activefeatures_dir =  seqs_info[seq_id]["activefeatures_dir"]

        l_segfeatures = ReaderWriter.read_data(os.path.join(activefeatures_dir, "l_segfeatures"), mode = "rb")

        imposter_gfeatures = feature_extractor.extract_seq_features_perboundary(seq, l_segfeatures)
        #^print("imposter_gfeatures ", imposter_gfeatures)
        # put back the original Y
        seq.Y = (y_ref, seg_other_symbol) 
        if(new_boundaries):
            # write back the sequence on disk given the segment attributes have been updated
            ReaderWriter.dump_data(seq, os.path.join(seq_dir, "sequence"), mode = "rb")
        
        return(imposter_gfeatures, y_imposter_boundaries)

    def save(self, folder_dir):
        # save essential info about feature extractor
        self.feature_extractor.save(folder_dir)
        if(self.attr_scaler):
            self.attr_scaler.save(folder_dir)
        
        

class FeatureFilter(object):
    # to improve feature filter by using the properties of Counter (i.e. most_common)
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
                #^print("z ", z)
                rel_func[relation](z, filter_pattern, filtered_dict)
#                 if(relation == "="):
#                     # delete any z that matches any of the provided filter patterns
#                     if(z in filter_pattern):
#                         del filtered_dict[z]
#                 elif(relation == "!="):
# 
#                     # delete any z that does not match any of the provided filter patterns
#                     if(z not in filter_pattern):
#                         #print("deleting z {}".format(z))
# 
#                         del filtered_dict[z]
        #^print("filtered_dict ", filtered_dict)
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
        if(x not in y): 
            #^print("{} not in {}".format(x, y))
            #^print("deleting ", z[x])
            del z[x]
        
def main():
    pass

if __name__ == "__main__":
    main()
    