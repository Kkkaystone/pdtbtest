import torch
import os
from torch.utils.data import Dataset, DataLoader

class ImplicitDataset(Dataset):
    def __init__(self, foldlist):
        self.labels = []
        self.category_mapping = {'Temporal': 0, 'Comparison': 1, 'Contingency': 2, 'Expansion': 3}
        file_path = '../dataset/pdtb_v2/data/pdtb/'
        
        filename=()
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read()
        
        samples = content.split('________________________________________________________')[1:-1]
        self.data=[]
        for sample in samples:
            if "____Implicit____" not in sample and "____AltLex____" not in sample:
                continue  # Skip non-implicit samples
            parts=sample.split('____Arg1____')
            label = [0, 0, 0, 0]
            sup1="None"
            arg1=""
            arg1_attr="None"
            arg2=""
            arg2_attr="None"
            sup2="None"
            for category in self.category_mapping:
                if category in parts[0]:
                    label[self.category_mapping[category]]=1
                    
            if "____Sup1____" in parts[0]:
                sub_parts=parts.split('#### Text ####')
                sup1=sub_parts.split('##############')[0].strip()
            arg_part=parts[1].split('____Arg2____')
            arg1_part=arg_part[0].split('#### Text ####')
            arg2_part=arg_part[1].split('#### Text ####')
            
            arg1 = arg1_part[1].split('##############')[0].strip()
            if len(arg1_part)>=3:
                arg1_attr = arg1_part[2].split('##############')[0].strip()
            
            arg2 = arg2_part[1].split('##############')[0].strip()
            if len(arg2_part)>=3:
                arg2_attr = arg2_part[2].split('##############')[0].strip()
            if len(arg2_part)>=4:
                sup2=arg2_part[3].split('##############')[0].strip()
            
            
            self.data.append("sup1:{}. arg1:{}(atrribution of arg1: {}). arg2:{}(atrribution of arg2: {}). sup2:{}.".format(sup1,arg1,arg1_attr,arg2,arg2_attr,sup2))
            self.labels.append(label)
            
            
            # parts = sample.split('#### Text ####')
            # arg1_text = parts[1].split('##############')[0].strip()
            # print(arg1_text)
            # arg2_text = parts[2].split('##############')[0].strip()
            # features_part = sample.split('#### Features ####')[1]
            # categories = set(features_part.split('\n')[1].split(', '))
            # label = [0, 0, 0, 0]
            # for category in categories:
            #     if category.split('.')[0] in self.category_mapping:
            #         label[self.category_mapping[category.split('.')[0]]] = 1
            # self.samples.append((arg1_text, arg2_text))
            # self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        label = self.labels[idx]
        return x, torch.tensor(label, dtype=torch.long)

# Example usage
dataset = ImplicitDataset('/scratch/user/xishi/pdtb/dataset/pdtb_v2/data/pdtb/00/wsj_0003.pdtb')
dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
print(len(dataset))
for batch_idx, (inputs, targets) in enumerate(dataloader):
    print(f"Batch {batch_idx}")
    print("Inputs:", inputs)
    print("Targets:", targets)