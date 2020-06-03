#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 15:09:30 2020

@author: igli
"""
import torch
import csv
from tqdm import tqdm
import argparse
from fairseq.models.transformer import TransformerModel

DATA_DIR = "/datasets/home/manajigli/Desktop/ms_marco/models/wmt19.en-de.joined-dict.ensemble"
TARGET_FILE = "/home/manajigli/Desktop/ms_marco/german_datasets/train.small.de.tsv"

def convert_tsv_lines_utf8_en_de(tsv_file):
    results = []
    with open(tsv_file) as tsv:
        tsv_reader = csv.reader(tsv, delimiter='\t')
        for line in list(tsv_reader):
            res_line = []
            for txt in line:
                text_en = txt.encode('iso-8859-1').decode('utf-8')                
                res_line.append(text_en)
            results.append(res_line)
    return results
           
    
     
def main():
    """
    Give the path of the source file (in tsv format) as an argument to the command in command line "run translator --source-file=..."
    Hard code the Target file, where the ouptut in german should be saved.
    """
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--source-file', required=True)
    
    args = parser.parse_args()
    
    source_file = args.source_file
    
    #check if source file is a valid file
    if not os.path.isfile(source_file) or not source_file_tsv.endswith('.tsv'):
        raise Exception(f"{source_file_tsv} is no valide file")
    
    en2de = TransformerModel.from_pretrained(f'{DATA_DIR}',
                                             checkpoint_file=f'{DATA_DIR}/model4.pt', 
                                             data_name_or_path = f'{DATA_DIR}', 
                                             bpe='fastbpe', bpe_codes=f'{DATA_DIR}/bpecodes', 
                                             tokenizer='moses')
    
    lines_en = convert_tsv_lines_utf8_en_de(source_file)
    
    with open(TARGET_FILE, 'w') as target_tsv:
        target_tsv_writer = csv.writer(target_tsv, delimiter='\t')
        for line in lines_en:
            new_line_de = []
            for text_en in line:
                text_de = en2de.translate(text_en)
                new_line_de.append(text_de)
            target_tsv_writer.writerow(new_line_de)
                
    print('SUCCESS!')
    
if __name__ == '__main__':
    main()
    
    
"""
small_data = []
with open('triples.train.small.tsv') as small:
    for line in list(small)[:100]:
        small_data.append(line.strip().split('\t'))
        
small_data_formatted = []
for line in small_data:
    res_line  = []
    for txt in line:
        text_en = txt.encode('iso-8859-1').decode('utf-8')                
        res_line.append(text_en)
    small_data_formatted.append(res_line)
"""
    
    
    


