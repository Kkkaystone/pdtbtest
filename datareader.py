import sys
import os
import random
import re
import torch
import torch.nn as nn
para_length_list = []
explicit_count = 0
implicit_filter_count = 0
explicit_filter_count = 0
double_label_count = 0
def update_doc_discourse_dict(doc_discourse_dict, split_sentence_index):
	new_doc_discourse_dict = {}

	for argpair in doc_discourse_dict:
		arg1_index,arg2_index = argpair[0],argpair[1]
		discourse_type,discourse_label = doc_discourse_dict[argpair][0],doc_discourse_dict[argpair][1]

		if arg2_index <= split_sentence_index:
			new_doc_discourse_dict[argpair] = (discourse_type,discourse_label)
		else:
			new_doc_discourse_dict[(arg1_index+1,arg2_index+1)] = (discourse_type,discourse_label)

	return new_doc_discourse_dict
#替换所有非单词，统计arg的单词是否每个都在sentence中
def sentence_contain_arg(sentence,arg):
	sentence = re.sub('[^\w\s]','',sentence).split()
	arg = re.sub('[^\w\s]','',arg).split()

	return all(sentence.count(word) >= arg.count(word) for word in set(arg))

	
	return True
def split_arg(arg,doc_sentence_list):
	for i in range(1,len(arg)):
		if arg[i]== ' ':
			if search_sentence_index(doc_sentence_list,arg[i+1:],0) > search_sentence_index(doc_sentence_list,arg[:i],0):
				arg = [arg[:i], arg[i:]]
				break
	return arg
#这段代码定义了一个名为 search_sentence_index 的函数，它
# 在一个文档的句子列表 doc_sentence_list 中搜索包含给定参数
# arg 的句子的索引。函数从列表的 start 索引位置开始搜索，直到列
# 表末尾。如果找到了匹配的句子，函数将返回该句子在列表中的索引；如果
# 没有找到，函数将返回 -1，表示未找到匹配项。

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
		# for i in range(len(pipe_line)):
		# 	#print("i, content",i, pipe_line[i])
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
def extract_explicit_relation(pipe_file_lines,doc_sentence_list,doc_discourse_dict,doc_paragraph_first_sentence_list):
	global explicit_count,double_label_count,explicit_filter_count

	prev_index = 0
	for i in range(0,len(pipe_file_lines)):
		pipe_line = pipe_file_lines[i].split('|')
		discourse_type = pipe_line[0]
		if discourse_type not in ['Explicit']:
			continue

		discourse_label = pipe_line[11]
		#catch double label
		if pipe_line[12] != '':
			double_label_count = double_label_count + 1
			discourse_label = discourse_label + '|' + pipe_line[12]
		if pipe_line[13] != '':
			double_label_count = double_label_count + 1
			discourse_label = discourse_label + '|' + pipe_line[13]

		connective = pipe_line[5]
		assert len(connective) > 0
		arg1 = pipe_line[24]
		arg2 = pipe_line[34]

		

		arg1_index =  search_sentence_index(doc_sentence_list,arg1,prev_index)
		arg2_index =  search_sentence_index(doc_sentence_list,arg2,prev_index)

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


		# catch the explicit discourse relations within sentence, split the sentence into two arguments
		if arg1_index == arg2_index and arg1_index != -1:
			sentence = doc_sentence_list[arg1_index]
			for j in range(len(sentence)):
				if sentence[j] in [',',':',';','.','?','!']:
					if (sentence_contain_arg(sentence[:j],arg1) and sentence_contain_arg(sentence[j+1:],arg2)) or (sentence_contain_arg(sentence[:j],arg2) and sentence_contain_arg(sentence[j+1:],arg1)):
						doc_sentence_list = doc_sentence_list[:arg1_index] + [sentence[:j+1],sentence[j+1:]] + doc_sentence_list[arg1_index+1:]
						doc_discourse_dict = update_doc_discourse_dict(doc_discourse_dict,arg1_index)
						break

			arg1_index =  search_sentence_index(doc_sentence_list,arg1,prev_index)
			
			if arg1 in ['you give parties']: #special case of wsj_1367
				arg1_index = arg1_index+1

			arg2_index = search_sentence_index(doc_sentence_list,arg2,prev_index)

		if arg1_index == -1:
			arg1_index =  search_sentence_index(doc_sentence_list,arg1,0)
		if arg2_index == -1:
			arg2_index =  search_sentence_index(doc_sentence_list,arg2,0)

		if arg1_index == arg2_index and arg1_index != -1:
			for j in range(len(sentence)):
				if sentence[j] in [' ','-']:
					if (sentence_contain_arg(sentence[:j],arg1) and sentence_contain_arg(sentence[j+1:],arg2)) or (sentence_contain_arg(sentence[:j],arg2) and sentence_contain_arg(sentence[j+1:],arg1)):
						doc_sentence_list = doc_sentence_list[:arg1_index] + [sentence[:j+1],sentence[j+1:]] + doc_sentence_list[arg1_index+1:]
						doc_discourse_dict = update_doc_discourse_dict(doc_discourse_dict,arg1_index)
						break

			arg1_index =  search_sentence_index(doc_sentence_list,arg1,prev_index)
			arg2_index = search_sentence_index(doc_sentence_list,arg2,prev_index)

		if arg1_index == -1:
			arg1_index =  search_sentence_index(doc_sentence_list,arg1,0)
		if arg2_index == -1:
			arg2_index =  search_sentence_index(doc_sentence_list,arg2,0)

		# arg2, connective arg1, arg2
		if arg1_index == arg2_index and arg1_index != -1:
			sentence = doc_sentence_list[arg1_index]
			flag = False

			for k in range(len(arg1.split())//2+1):
				tmp_arg1 = ' '.join(arg1.split(' ')[k:])
				for j in range(len(sentence)):
					if sentence[j] in [',',':',';','.','-','?','!',' ']:
						if (sentence_contain_arg(sentence[:j],tmp_arg1) and sentence_contain_arg(sentence[j+1:],arg2)) or (sentence_contain_arg(sentence[:j],arg2) and sentence_contain_arg(sentence[j+1:],tmp_arg1)):
							doc_sentence_list = doc_sentence_list[:arg1_index] + [sentence[:j+1],sentence[j+1:]] + doc_sentence_list[arg1_index+1:]
							doc_discourse_dict = update_doc_discourse_dict(doc_discourse_dict,arg1_index)
							flag = True
							break
				if flag:
					break

			if not flag:			
				for k in range(1,len(arg1.split())//2+1):
					tmp_arg1 = ' '.join(arg1.split(' ')[:-k])
					for j in range(len(sentence)):
						if sentence[j] in [',',':',';','.','-','?','!',' ']:
							if (sentence_contain_arg(sentence[:j],tmp_arg1) and sentence_contain_arg(sentence[j+1:],arg2)) or (sentence_contain_arg(sentence[:j],arg2) and sentence_contain_arg(sentence[j+1:],tmp_arg1)):
								doc_sentence_list = doc_sentence_list[:arg1_index] + [sentence[:j+1],sentence[j+1:]] + doc_sentence_list[arg1_index+1:]
								doc_discourse_dict = update_doc_discourse_dict(doc_discourse_dict,arg1_index)
								flag = True
								break
					if flag:
						break

			arg1_index =  search_sentence_index(doc_sentence_list,tmp_arg1,prev_index)
			arg2_index = search_sentence_index(doc_sentence_list,arg2,prev_index)

		if arg1_index == arg2_index and arg1_index != -1:
			sentence = doc_sentence_list[arg1_index]
			flag = False

			replace_arg1 = arg2
			replace_arg2 = arg1

			for k in range(len(replace_arg1.split())//2+1):
				tmp_arg1 = ' '.join(replace_arg1.split(' ')[k:])
				for j in range(len(sentence)):
					if sentence[j] in [',',':',';','.','-','?','!',' ']:
						if (sentence_contain_arg(sentence[:j],tmp_arg1) and sentence_contain_arg(sentence[j+1:],replace_arg2)) or (sentence_contain_arg(sentence[:j],replace_arg2) and sentence_contain_arg(sentence[j+1:],tmp_arg1)):
							doc_sentence_list = doc_sentence_list[:arg1_index] + [sentence[:j+1],sentence[j+1:]] + doc_sentence_list[arg1_index+1:]
							doc_discourse_dict = update_doc_discourse_dict(doc_discourse_dict,arg1_index)
							flag = True
							break
				if flag:
					break

			if not flag:			
				for k in range(1,len(replace_arg1.split())//2+1):
					tmp_arg1 = ' '.join(replace_arg1.split(' ')[:-k])
					for j in range(len(sentence)):
						if sentence[j] in [',',':',';','.','-','?','!',' ']:
							if (sentence_contain_arg(sentence[:j],tmp_arg1) and sentence_contain_arg(sentence[j+1:],replace_arg2)) or (sentence_contain_arg(sentence[:j],replace_arg2) and sentence_contain_arg(sentence[j+1:],tmp_arg1)):
								doc_sentence_list = doc_sentence_list[:arg1_index] + [sentence[:j+1],sentence[j+1:]] + doc_sentence_list[arg1_index+1:]
								doc_discourse_dict = update_doc_discourse_dict(doc_discourse_dict,arg1_index)
								flag = True
								break
					if flag:
						break

			arg1_index =  search_sentence_index(doc_sentence_list,tmp_arg1,prev_index)
			arg2_index = search_sentence_index(doc_sentence_list,replace_arg2,prev_index)		

		if arg1_index == -1:
			arg1_index =  search_sentence_index(doc_sentence_list,arg1,0)
		if arg2_index == -1:
			arg2_index =  search_sentence_index(doc_sentence_list,arg2,0)
		

		# if arg2 is the first sentence of any paragraphs and arg1 locates more than one sentence from arg2, copy the arg1 in front of arg2 
		if arg2_index - arg1_index >= 2 and arg1_index != -1 and arg2_index != -1:
			if doc_sentence_list[arg2_index] in doc_paragraph_first_sentence_list and (arg2_index-1,arg2_index) not in doc_discourse_dict:
				doc_sentence_list = doc_sentence_list[:arg2_index] + [doc_sentence_list[arg1_index]] + doc_sentence_list[arg2_index:]
				doc_discourse_dict = update_doc_discourse_dict(doc_discourse_dict,arg2_index)

				arg1_index =  search_sentence_index(doc_sentence_list,arg1,arg1_index+1)
				arg2_index = search_sentence_index(doc_sentence_list,arg2,arg2_index+1)

		if arg1_index == -1 or arg2_index == -1:
			explicit_filter_count += 1
			#print arg1 if arg1_index == -1 else arg2
			#print prev_index
			#print arg1_index,arg1
			#print arg2_index,arg2
			#print doc_sentence_list
			continue

		if arg1_index - arg2_index == 1 or arg2_index - arg1_index == 1:
			#assert connective in doc_sentence_list[arg1_index]
			prev_index = max(arg1_index,arg2_index)

			if arg1_index > arg2_index:
				tmp_index = arg1_index
				arg1_index = arg2_index
				arg2_index = tmp_index

			if (arg1_index,arg2_index) not in doc_discourse_dict:
				explicit_count = explicit_count + 1
				doc_discourse_dict[(arg1_index,arg2_index)] = (discourse_type,discourse_label)
			else:
				if doc_discourse_dict[(arg1_index,arg2_index)][0] in ['EntRel','AltLex']:
					explicit_count = explicit_count + 1
					doc_discourse_dict[(arg1_index,arg2_index)] = (discourse_type,discourse_label)
				elif doc_discourse_dict[(arg1_index,arg2_index)][0] in ['Explicit']:
					double_label_count = double_label_count + 1
					explicit_count = explicit_count + 1
					prev_discourse_label = doc_discourse_dict[(arg1_index,arg2_index)][1]

					doc_discourse_dict[(arg1_index,arg2_index)]  = (discourse_type,prev_discourse_label + '|' + discourse_label)
					#print '-----------------------------------'
					#print arg1_index,arg1
					#print arg2_index,arg2
					#print doc_sentence_list[arg1_index]
					#print doc_sentence_list[arg2_index]
					#print prev_discourse_label
					#print discourse_label

				else:
					explicit_filter_count += 1
					#print '-----------------------------------'
					#print arg1_index,arg1
					#print arg2_index,arg2
					#print doc_sentence_list[arg1_index]
					#print doc_sentence_list[arg2_index]
					#print doc_discourse_dict[(arg1_index,arg2_index)]
					#print discourse_label
					pass
		elif arg1_index == arg2_index:
			explicit_filter_count += 1
			#print '------------------------'
			#print discourse_label
			#print arg1
			#print arg2
			#print doc_sentence_list[arg1_index]
			pass
		else:
			explicit_filter_count += 1
			#print '------------------------'
			#print discourse_label
			#print arg1_index, arg1
			#print doc_sentence_list[arg1_index]
			#print arg2_index, arg2
			#print doc_sentence_list[arg2_index]
			pass

	return doc_sentence_list,doc_discourse_dict
#参数
# doc_sentence_list：文档中句子的列表。这是一个包含文档中所有句子的列表，其中每个句子是一个字符串。
# doc_discourse_dict：论元对到论述类型和标签的映射的字典。这个字典以论元对（即句子索引的元组）作为键，以一个元组（论述类型和论述标签）作为值。
# 返回值
# paras_sentence_list：包含段落句子列表的列表。每个元素是一个段落，该段落由多个句子组成，每个句子是一个字符串。
# paras_y_list：与paras_sentence_list相对应的标签列表。每个元素是一个张量，表示对应段落中所有句子对的论述关系的标签。
# 功能总结
# 这个函数的目的是根据文档的句子列表和论述关系字典处理并分析文档中的段落和论述关系标签。它首先通过分析doc_discourse_dict来建立论述关系的索引，
# 并收集所有涉及到的句子索引。然后，它会根据这些索引对句子进行分组，形成段落，并对每个段落中存在的论述关系进行编码，最后返回这些段落及其相应的论述
# 关系标签。这种处理对于理解和分析文档中的结构和论述动态非常有用，尤其是在处理隐式和显式论述关系的文本分析任务中。
def process_doc_paras_labels(doc_sentence_list, doc_discourse_dict):
    sentence_index_list = []  # 存储涉及论述关系的句子索引
    discourse_dict = {}  # 存储论述类型和标签的字典

    # 遍历论述字典，收集涉及的句子索引和论述信息
    for argpair in doc_discourse_dict:
        arg1_index, arg2_index = argpair
        discourse_type, discourse_label = doc_discourse_dict[argpair]

        # 只考虑隐式和显式论述关系
        if discourse_type in ['Implicit', 'Explicit']:
            discourse_dict[(arg1_index, arg2_index)] = (discourse_type, discourse_label)
            if arg1_index not in sentence_index_list:
                sentence_index_list.append(arg1_index)
            if arg2_index not in sentence_index_list:
                sentence_index_list.append(arg2_index)

    # 如果涉及的句子不足以形成段落，返回空列表
    if len(sentence_index_list) <= 1:
        return [], []

    sentence_index_list.sort()  # 对句子索引进行排序

    paras_sentence_list = []  # 存储段落句子的列表
    paras_y_list = []  # 存储段落标签的列表

    sentence_index = 0
    para_sentence_list = []  # 当前处理的段落句子列表
    discourse_list = []  # 当前处理的段落中的论述关系列表
    # 遍历句子索引，组织段落和相应的标签
    while(sentence_index <= sentence_index_list[-1]):
        # 检查当前句子对是否构成论述关系
        if (sentence_index, sentence_index + 1) in discourse_dict:
            discourse_list.append((sentence_index, sentence_index + 1))
            if not para_sentence_list:
                para_sentence_list.append(doc_sentence_list[sentence_index])
            para_sentence_list.append(doc_sentence_list[sentence_index + 1])
        else:
            # 当遇到不构成论述关系的句子对时，处理并保存当前段落
            if discourse_list:
                para_y = torch.zeros(len(discourse_list),len(discourse_sense_list))
				for i in range(len(discourse_list)):
					discourse = discourse_list[i]
					discourse_type,discourse_label = discourse_dict[discourse][0],discourse_dict[discourse][1]
					para_y[i,:] = process_discourse_relation_label(discourse_label, discourse_type)
					#para_y[i,:] = process_discourse_relation_label_8way(discourse_label, discourse_type)

				if torch.sum(para_y.abs()) > 0:
					paras_sentence_list.append(para_sentence_list)
					paras_y_list.append(para_y)
			para_sentence_list = []
			discourse_list = []

		sentence_index += 1

	return paras_sentence_list,paras_y_list


#参数说明
# fold_discourse_relation_list: 包含多个文档的列表，每个文档由句子列表和语篇关系字典组成。
# posner_flag (默认为 True): 一个布尔标志，指示是否使用 Posner 方法进行句子处理。
# sentencemarker (默认为 False): 一个布尔标志，指示是否在处理句子时标记句子边界。
# connectivemarker (默认为 False): 一个布尔标志，用于指示是否在处理句子时标记连接词的位置。
# 返回类型
# 函数返回一个元组，其内容根据 connectivemarker 参数的值变化：

# 当 connectivemarker 为 True 时，返回的元组包含：
# para_embedding_list: 句子嵌入向量列表。
# para_label_length_list: 每个段落标签长度的列表。
# eos_position_lists: 每个句子结束符位置的列表。
# connective_position_lists: 连接词位置的列表。
# y_list: 语篇关系标签列表。
# 当 connectivemarker 为 False 时，返回的元组不包含 connective_position_lists。
# 函数功能总结
# fold_word2vec 函数的主要功能是处理文档中的句子，并基于这些句子的语篇关系生成向量表示。具体步骤包括：

# 遍历给定的文档列表，对每个文档中的句子进行处理，包括句子嵌入向量的生成和连接词位置的标记（如果启用）。
# 对每个句子的嵌入向量进行合并，形成段落级别的嵌入表示。
# 计算整个数据集中显性和隐性语篇关系的分布。
# 返回处理后的段落嵌入向量、标签长度、句子结束符位置列表（以及连接词位置列表，如果启用）和语篇关系标签列表。
def fold_word2vec(fold_discourse_relation_list, posner_flag = True, sentencemarker = False, connectivemarker = False):
	global para_length_list
	print "total number of documents:" + str(len(fold_discourse_relation_list))
	y_total = torch.zeros(len(discourse_sense_list))
	y_explicit =  torch.zeros(len(discourse_sense_list))
	y_implicit =  torch.zeros(len(discourse_sense_list))

	#para_sentence_lists = []
	para_embedding_list = []
	para_label_length_list = []
	eos_position_lists = []
	connective_position_lists = []
	y_list = []

	for i in range(len(fold_discourse_relation_list)):
		if i % 10 == 0:
			print i

		doc_sentence_list,doc_discourse_dict = fold_discourse_relation_list[i][0],fold_discourse_relation_list[i][1]
		paras_sentence_list,paras_y_list= process_doc_paras_labels(doc_sentence_list, doc_discourse_dict)
		
		if len(paras_sentence_list) == 0:
			continue

		for para_sentence_list, y in zip(paras_sentence_list, paras_y_list):
			print(para_sentence_list)
			print(y)

			para_length_list.append(len(para_sentence_list))
			#para_sentence_lists.append(para_sentence_list)

			y_total = y_total + torch.sum(y.abs(),0)
			y_explicit = y_explicit + torch.sum(y.clamp(-1,0).abs(),0)
			y_implicit = y_implicit + torch.sum(y.clamp(0,1),0)
			para_label_length_list.append(torch.sum(y.abs()))

			sentence_embedding_list = []
			eos_position_list = []
			connective_position_list = []
			para_length = 0
			for sentence in para_sentence_list:
				sentence_embedding = process_sentence(sentence, posner_flag = posner_flag, sentencemarker = sentencemarker)
				sentence_embedding_list.append(sentence_embedding)

				if connectivemarker:
					if sentence_startwith_connective(sentence):
						if sentence.strip()[0] == '"':
							connective_position_list.append(para_length+1)
						else:
							connective_position_list.append(para_length)
					else:
						connective_position_list.append(-1)

				para_length = para_length + sentence_embedding.size(0)
				eos_position_list.append(para_length)


			assert len(eos_position_list) - 1 == y.size(0)
			para_embedding = torch.cat(sentence_embedding_list)
			para_embedding = para_embedding.view(1,-1, para_embedding.size(-1))

			para_embedding = Variable(para_embedding, requires_grad = False)
			y = Variable(y, requires_grad = False)

			para_embedding_list.append(para_embedding)
			eos_position_lists.append(eos_position_list)
			connective_position_lists.append(connective_position_list)
			y_list.append(y)

	'''with open('./data/pdtb_implicit_moreexplicit_discourse_paragraph_multilabel_devdata.pkl','w+') as f:
		cPickle.dump([para_sentence_lists,y_list],f)
		f.close()'''

	print 'Discourse relation distribution'
	print y_total
	print 'Explicit discourse relation distribution'
	print y_explicit
	print 'Implicit discourse relation distribution'
	print y_implicit
	
	if connectivemarker:
		return (para_embedding_list,para_label_length_list,eos_position_lists,connective_position_lists),y_list
	else:
		return (para_embedding_list,para_label_length_list,eos_position_lists),y_list

def process_doc(pipe_file_path,raw_file_path):
	pipe_file = open(pipe_file_path,'r',encoding='latin-1')
	print(raw_file_path)
	raw_file = open(raw_file_path,'r',encoding='latin-1')

	pipe_file_lines = pipe_file.readlines()
	raw_file_lines =  raw_file.readlines()
	doc_paragraph_first_sentence_list = []
	doc_sentence_list = []
	for i in range(2,len(raw_file_lines)):
		line = raw_file_lines[i].replace('\n','').strip()
		if len(line) > 0:
			sentences_list = line
			doc_sentence_list.append(sentences_list)
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
		print('fold: ' + str(fold))
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
# train_X,train_Y = process_fold(training_fold_list)
# test_X, test_Y = process_fold(test_fold_list)




print(dev_X.shape)