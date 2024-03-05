import sys
import os
import random

import torch
import torch.nn as nn

def sentence_contain_arg(sentence,arg):
	sentence = re.sub('[^\w\s]','',sentence).split()
	arg = re.sub('[^\w\s]','',arg).split()

	for word in arg:
		if word not in sentence:
			return False
		if arg.count(word) > 1 and sentence.count(word) < arg.count(word):
			return False
	
	return True

def search_sentence_index(doc_sentence_list,arg,start):
	for i in range(start,len(doc_sentence_list)):
		if arg in doc_sentence_list[i] and len(arg.split())>2:
			return i
		if sentence_contain_arg(doc_sentence_list[i],arg):
			return i

	return -1

def extract_implicit_relation(pipe_file_lines,doc_sentence_list,doc_discourse_dict):
	global implicit_filter_count,double_label_count

	prev_index = 0
	for i in range(0,len(pipe_file_lines)):
		pipe_line = pipe_file_lines[i].split('|')
		discourse_type = pipe_line[0]
		if discourse_type not in ['Implicit','AltLex','EntRel']:
			continue

		discourse_label = pipe_line[11]
		#catch double label
		if pipe_line[12] != '':
			double_label_count = double_label_count + 1
			discourse_label = discourse_label + '|' + pipe_line[12]
		if pipe_line[13] != '':
			double_label_count = double_label_count + 1
			discourse_label = discourse_label + '|' + pipe_line[13]

		fake_connective = pipe_line[9]
		arg1 = pipe_line[24]
		arg2 = pipe_line[34]

		if len(arg1) <=2 and pipe_line[31] != '': #special case for wsj_1856
			arg1 = pipe_line[31]

	

		arg1_index =  search_sentence_index(doc_sentence_list,arg1,prev_index)
		if arg1_index != -1:
			prev_index = arg1_index
		arg2_index =  search_sentence_index(doc_sentence_list,arg2,prev_index)

		if arg1 in ['some structural damage to headquarters and no power']: #special case for wsj_1915
			arg1_index = search_sentence_index(doc_sentence_list,arg1, prev_index+1)
			prev_index = arg1_index

		# handle the special case that two arguments are seperated by ':' or ';'
		if arg1_index == -1 or arg2_index == -1:
			if arg1_index == -1:
				if ';' in arg1:
					arg1 = arg1.split(';')[-1]
				if ':' in arg1:
					arg1 = arg1.split(':')[-1]
				arg1_index =  search_sentence_index(doc_sentence_list,arg1,prev_index)
			if arg2_index == -1:
				if ';' in arg2:
					arg2 = arg2.split(';')[0]
				if ':' in arg2:
					arg2 = arg2.split(':')[0]
				arg2_index =  search_sentence_index(doc_sentence_list,arg2,prev_index)

		# handle the special case that two arguments are seperated by ':' or ';'
		if arg1_index == arg2_index and arg1_index != -1:
			sentence = doc_sentence_list[arg1_index]
			for j in range(len(sentence)):
				if sentence[j] in [';',':','.'] or ( j < len(sentence)-1 and sentence[j:j+2] in ['--']):
					if sentence_contain_arg(sentence[:j],arg1) and sentence_contain_arg(sentence[j+1:],arg2):
						doc_sentence_list = doc_sentence_list[:arg1_index] + [sentence[:j+1],sentence[j+1:]] + doc_sentence_list[arg1_index+1:]
						break

			arg1_index =  search_sentence_index(doc_sentence_list,arg1,prev_index)
			arg2_index = search_sentence_index(doc_sentence_list,arg2,arg1_index+1)

			
		if arg2_index - arg1_index == 2 and arg1_index != -1 and arg2_index!=-1:
			if '"' in doc_sentence_list[arg1_index]:
				#print doc_sentence_list[arg1_index] + ' ' + doc_sentence_list[arg1_index+1]
				doc_sentence_list = doc_sentence_list[:arg1_index] + [doc_sentence_list[arg1_index] + ' ' + doc_sentence_list[arg1_index+1]] + doc_sentence_list[arg2_index:]

				arg1_index =  search_sentence_index(doc_sentence_list,arg1,prev_index)
				arg2_index =  search_sentence_index(doc_sentence_list,arg2,arg1_index+1)

		if arg1_index == -1 or arg2_index == -1:
			if arg1_index == -1:
				arg1 = split_arg(arg1,doc_sentence_list)
				if type(arg1) == type([]):
					arg1 = arg1[-1]
				arg1_index =  search_sentence_index(doc_sentence_list,arg1,prev_index)

			if arg2_index == -1:
				arg2 = split_arg(arg2,doc_sentence_list)			
				if type(arg2) == type([]):
					arg2 = arg2[0]
				arg2_index =  search_sentence_index(doc_sentence_list,arg2,prev_index)

		if arg1_index == -1:
			arg1_index =  search_sentence_index(doc_sentence_list,arg1,0)
		if arg2_index == -1:
			arg2_index =  search_sentence_index(doc_sentence_list,arg2,0)

		if arg1_index == -1 or arg2_index == -1:
			implicit_filter_count = implicit_filter_count + 1	
			#print arg1 if arg1_index == -1 else arg2
			#print prev_index
			#print arg1_index,arg1
			#print arg2_index,arg2
			continue

		if arg1_index - arg2_index == 1 or arg2_index - arg1_index == 1:
			prev_index = max(arg1_index,arg2_index)

			if arg1_index > arg2_index:
				tmp_index = arg1_index
				arg1_index = arg2_index
				arg2_index = tmp_index

			if (arg1_index,arg2_index) not in doc_discourse_dict:
				doc_discourse_dict[(arg1_index,arg2_index)] = (discourse_type,discourse_label)
			else:
				assert (discourse_type,discourse_label) == doc_discourse_dict[(arg1_index,arg2_index)]

		elif arg1_index == arg2_index:
			implicit_filter_count = implicit_filter_count + 1	
			#print arg1
			#print arg2
			#print doc_sentence_list[arg1_index]
		else:
			implicit_filter_count = implicit_filter_count + 1
			#print prev_index
			#print arg1_index, arg1
			#print doc_sentence_list[arg1_index]
			#print arg2_index, arg2
			#print doc_sentence_list[arg2_index]

	return doc_sentence_list, doc_discourse_dict

def process_doc(pipe_file_path,raw_file_path):
	pipe_file = open(pipe_file_path,'r')
	raw_file = open(raw_file_path,'r')

	pipe_file_lines = pipe_file.readlines()
	raw_file_lines =  raw_file.readlines()

	doc_paragraph_first_sentence_list = []
	doc_sentence_list = []
	for i in range(2,len(raw_file_lines)):
		line = raw_file_lines[i].replace('\n','').strip()
		if len(line) > 0:
			sentences_list = line
			doc_sentence_list = doc_sentence_list + sentences_list
			doc_paragraph_first_sentence_list.append(sentences_list[0])

	doc_discourse_dict = {}
	doc_sentence_list, doc_discourse_dict = extract_implicit_relation(pipe_file_lines,doc_sentence_list, doc_discourse_dict)

	if len(doc_sentence_list) >= 2 and len(doc_discourse_dict) >= 1:
		doc_sentence_list, doc_discourse_dict = extract_explicit_relation(pipe_file_lines,doc_sentence_list,doc_discourse_dict,doc_paragraph_first_sentence_list)

	return doc_sentence_list,doc_discourse_dict
def process_fold(fold_list):
	fold_doc_list = []

	pipe_file_path = '../dataset/pdtb_v2/data_t/pdtb/'
	raw_file_path = '../dataset/pdtb_v2/data/raw/wsj/'

	for fold in fold_list:
		print 'fold: ' + str(fold)
		fold_pipe_file_path = os.path.join(pipe_file_path,fold)
		fold_raw_file_path = os.path.join(raw_file_path,fold)

		for fold_file in sorted(os.listdir(fold_pipe_file_path)):
			filename =  fold_file.split('.')[0]

			if len(filename) == 0:
				continue

			pipe_file = os.path.join(fold_pipe_file_path,filename+'.pipe')
			raw_file = os.path.join(fold_raw_file_path,filename)

			doc_sentence_list,doc_discourse_dict = process_doc(pipe_file,raw_file)

			if len(doc_sentence_list) >= 2 and len(doc_discourse_dict) >= 1:
				fold_doc_list.append((doc_sentence_list,doc_discourse_dict))

	X,Y = fold_word2vec(fold_doc_list, posner_flag = True, sentencemarker = False, connectivemarker = False)
	return X,Y



training_fold_list = ['02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20']
dev_fold_list = ['00','01']
test_fold_list = ['21','22']


dev_X,dev_Y = process_fold(dev_fold_list)
train_X,train_Y = process_fold(training_fold_list)
test_X, test_Y = process_fold(test_fold_list)


def load_data():
    print( 'Loading Data...')
    outfile = open(os.path.join(os.getcwd(),'/scratch/user/xishi/pdtb/pdtb/pdtb_implicit_moreexplicit_discourse_withoutAltLex_paragraph_multilabel_addposnerembedding.pt'),'rb')
    pdtb_data = torch.load(outfile)
    outfile.close()

    dev_X,dev_Y,train_X,train_Y,test_X,test_Y = pdtb_data['dev_X'],pdtb_data['dev_Y'],pdtb_data['train_X'] ,pdtb_data['train_Y'],pdtb_data['test_X'],pdtb_data['test_Y']

    dev_X_eos_list = dev_X[2]
    dev_X_label_length_list = dev_X[1]
    dev_X = dev_X[0]
    
    test_X_eos_list = test_X[2]
    test_X_label_length_list = test_X[1]
    test_X = test_X[0]

    print(len(dev_X))
    for i in range(3):
        print("dev#######",i)
        print("dev_X_eos_list",dev_X_eos_list[i])
        print("dev_X_label_length_list",dev_X_label_length_list[i])
        print("dev_X",dev_X[i].shape)
        print("dev_Y",len(dev_Y[i]))
        print("dev_Y",dev_Y[i])
        # print("dev_Y[0].shape",dev_Y[0].shape)
        
    return


load_data()