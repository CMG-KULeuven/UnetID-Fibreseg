#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: Rui Guo (KU Leuven), rui.guo1@kuleuven.be
Date: July 2022
"""

import pandas as pd
import json
            
def write_to_txt(lines, save_path=None, access_mode='write'):
    if access_mode == 'write':
        access = 'w'
    elif access_mode == 'append':
        access = 'a'
    with open(save_path, access) as f:
        f.write(lines)
        f.write('\n')
        f.close()

def write_to_excel(data, path, header):
    df_data = pd.DataFrame(data)
    path = path+'_'+header[0]+'.xlsx'
    with pd.ExcelWriter(path) as writer:
        df_data.to_excel(writer, index=False, header=header)

def write_to_json(output_dir, para_kwargs):
    # Serializing json
    json_object = json.dumps(para_kwargs, indent=4)
    # Writing to sample.json
    with open(output_dir+"/paraSetting.json", "w") as outfile:
        outfile.write(json_object)