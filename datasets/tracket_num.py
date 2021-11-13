from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



def get_tracket_num(dataset_name):
        if dataset_name == 'PRID2011':
            return [89, 89]
        elif dataset_name == 'iLIDS-VID':
            return [150, 150]
        elif dataset_name == 'DukeMTMC-VideoReID':
            return [404, 378, 201, 165, 218, 348, 217, 265]
        else:
            raise ValueError('You must supply the dataset name as '
                             '-- PRID2011, iLIDS-VID, DukeMTMC-VideoReID')


