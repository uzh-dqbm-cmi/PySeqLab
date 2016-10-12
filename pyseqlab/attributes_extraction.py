'''
@author: ahmed allam <ahmed.allam@yale.edu>
'''
from collections import defaultdict


class AttributeScaler(object):
    def __init__(self, attr_extractor, scaling_info, method):
        self.attr_extractor = attr_extractor
        self.scaling_info = scaling_info
        self.method = method
        
    def scale_real_attributes(self, seq, boundaries):
        scaling_info = self.scaling_info
        method = self.method
        seg_attr = seq.seg_attr
        print("scaling_info {}".format(scaling_info))
        if(method == "standardization"):
            for attr_name in scaling_info:
                attr_mean = scaling_info[attr_name]['mean']
                attr_sd = scaling_info[attr_name]['sd']
                for boundary in boundaries:
                    seg_attr[boundary][attr_name]= (seg_attr[boundary][attr_name] - attr_mean)/attr_sd
        elif(method == "rescaling"):
            for attr_name in scaling_info:
                attr_max = scaling_info[attr_name]['max']
                for boundary in boundaries:
                    seg_attr[boundary][attr_name]= (seg_attr[boundary][attr_name] - attr_max)/attr_max + 1
        
# class implements observation functions that generates attributes from tokens/observations
class NERSegmentAttributeExtractor(object):
    def __init__(self):
        self.attr_desc = self.generate_attributes_desc()
        self.seg_attr = {}
    
    def generate_attributes_desc(self):
        attr_desc = {}
        attr_desc['w'] = {'description':'the word/token',
                          'encoding':'binary'
                         }
        attr_desc['shape'] = {'description':'the shape of the word',
                              'encoding':'binary'
                             }
        attr_desc['shaped'] = {'description':'the compressed/degenerated form/shape of the word',
                               'encoding':'binary'
                              }
        attr_desc['seg_numchars'] = {'description':'number of characters in a segment',
                                     'encoding':'real'
                                    }
        attr_desc['seg_len'] = {'description':'the length of a segment',
                                'encoding':'real'
                               }
        attr_desc['bag_of_attr_'] = {'description':'prefix attribute name for all attributes that implement/measure bag of attributes property',
                                     'encoding':'real'
                                    }
        return(attr_desc)
    
    def group_attributes(self):
        """function to group attributes based on the encoding type (i.e. real vs. binary)"""
        attr_desc = self.attr_desc
        grouped_attr = {}
        for attr_name in attr_desc:
            encoding_type = attr_desc[attr_name]['encoding']
            if(encoding_type in grouped_attr):
                grouped_attr[encoding_type].append(attr_name)
            else:
                grouped_attr[encoding_type] = [attr_name]
        return(grouped_attr)
 
    def generate_attributes(self, seq, boundaries):
        X = seq.X
        # segment attributes dictionary
        self.seg_attr = {}
        new_boundaries = []
        # create segments from observations using the provided boundaries
        for boundary in boundaries:
            if(boundary not in seq.seg_attr):
                self._create_segment(X, boundary, ['w'])
                new_boundaries.append(boundary)
#         print("seg_attr {}".format(self.seg_attr))
#         print("new_boundaries {}".format(new_boundaries))
        if(self.seg_attr):
            attr_names_boa = ('w', 'shaped')
            for boundary in new_boundaries:
                self.get_shape(boundary)
                self.get_degenerateshape(boundary)
                self.get_seg_length(boundary)
                self.get_num_chars(boundary)
                # generate bag of attributes properties in every segment
                for attr_name in attr_names_boa:
                    self.get_seg_bagofattributes(boundary, attr_name)
            
            # save generated attributes in seq
            seq.seg_attr.update(self.seg_attr)
#             print('saved attribute {}'.format(seq.seg_attr))
            # clear the instance variable seg_attr
            self.seg_attr = {}
        return(new_boundaries)
        
    def _create_segment(self, X, boundary, attr_names, sep = " "):
        self.seg_attr[boundary] = {}
        for attr_name in attr_names:
            segment_value = self._get_segment_value(X, boundary, attr_name)
            self.seg_attr[boundary] = {attr_name: "{}".format(sep).join(segment_value)}
            
    def _get_segment_value(self, X, boundary, target_attr):
        u = boundary[0]
        v = boundary[1]
        segment = []
        for i in range(u, v+1):
            segment.append(X[i][target_attr])
        return(segment)
    
    def get_shape(self, boundary):
        """ boundary is tuple (u,v) that marks beginning and end of a segment"""
        segment = self.seg_attr[boundary]['w']
        r = ''
        for c in segment:
            if c.isupper():
                r += 'U'
            elif c.islower():
                r += 'L'
            elif c.isdigit():
                r += 'D'
            elif c in ('.', ','):
                r += '.'
            elif c in (';', ':', '?', '!'):
                r += ';'
            elif c in ('+', '-', '*', '/', '=', '|', '_'):
                r += '-'
            elif c in ('(', '{', '[', '<'):
                r += '('
            elif c in (')', '}', ']', '>'):
                r += ')'
            else:
                r += c
        self.seg_attr[boundary]['shape'] = r
            
    def get_degenerateshape(self, boundary):
        segment = self.seg_attr[boundary]['shape']
        dst = ''
        for c in segment:
            if not dst or dst[-1] != c:
                dst += c
        self.seg_attr[boundary]['shaped'] = dst
        
    def get_seg_length(self, boundary):
        # begin and end of a boundary
        u = boundary[0]
        v = boundary[-1]
        seg_len = v - u + 1
        self.seg_attr[boundary]['seg_len'] = seg_len
            
    def get_num_chars(self, boundary, filter_out = " "):
        segment = self.seg_attr[boundary]['w']
        filtered_segment = segment.split(sep = filter_out)
        num_chars = 0
        for entry in filtered_segment:
            num_chars += len(entry)
        self.seg_attr[boundary]['seg_numchars'] = num_chars
            
    def get_seg_bagofattributes(self, boundary, attr_name, sep = " "):
        # implements the bag-of-attributes concept within a segment 
        # it can be used with attributes that have binary_encoding type set equal True
        prefix = 'bag_of_attr'
        segment = self.seg_attr[boundary][attr_name]
        split_segment = segment.split(sep)
        count_dict = defaultdict(int)
        for elem in split_segment:
            count_dict[elem] += 1
            
        for attr_value, count in count_dict.items():
            fkey = prefix + '_' + attr_name + '_' + attr_value
            self.seg_attr[boundary][fkey] = count
    
if __name__ == "__main__":
    pass
    