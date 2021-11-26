import os
import sys
import json
from collections import defaultdict
from tqdm import tqdm
from transformers import DataProcessor,BertTokenizerFast
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import numpy as np


class EEProcessor(DataProcessor):
    """
        从数据文件读取数据，生成训练数据dataloader，返回给模型
    """
    def __init__(self, config, tokenizer=None):
        self.train_path = config.train_path
        self.dev_path=config.dev_path
        self.test_path=config.test_path
        self.tokenizer = tokenizer
        self.entity_set = self._load_set()
        self.nerlabel2ids, self.nerid2labels = self._load_ner_schema()
        self.rellabel2ids, self.relid2labels = self._load_rel_schema()

    def _load_set(self):
        return {'FAC|GPE', 'GPE|VEH', 'PER|WEA', 'LOC|FAC', 'GPE|ORG', 'WEA|VEH', 'GPE|WEA', 'ORG|PER', 'WEA|WEA', 'ORG|FAC', 'GPE|LOC', 'FAC|LOC', 'ORG|GPE', 'PER|FAC', 'PER|ORG', 'ORG|ORG', 'GPE|GPE', 'ORG|WEA', 'PER|LOC', 'PER|VEH', 'PER|PER', 'GPE|FAC', 'ORG|LOC', 'FAC|ORG', 'PER|GPE', 'ORG|VEH', 'FAC|FAC', 'LOC|LOC', 'LOC|GPE', 'VEH|VEH'}
        
    def _load_ner_schema(self):
        label2ids = {}
        id2labels = {}
        type_list = ["FAC","GPE","LOC","ORG","PER","VEH","WEA"]
        label2ids["O"] = 0
        id2labels[0] = "O"
        for index,role in enumerate(type_list):
            label2ids["B-"+role] = index*2+1
            label2ids["I-"+role] = index*2+2
            id2labels[index*2+1] = "B-"+role
            id2labels[index*2+2] = "I-"+role
        return label2ids,id2labels

    def _load_rel_schema(self):
        label2ids = {}
        id2labels = {}
        type_list = ["ART","GEN-AFF","ORG-AFF","PART-WHOLE","PER-SOC","PHYS"]
        label2ids["O"] = 0
        id2labels[0] = "O"
        for index,role in enumerate(type_list):
            label2ids[role] = index+1
            id2labels[index+1] = role
        return label2ids,id2labels

    def extract_data_to_dict(self,filepath):
        txt_d = defaultdict(dict)
        annotation_d = defaultdict(dict)
        pathDir =  os.listdir(filepath)
        for allDir in pathDir:
            child = os.path.join('%s/%s' % (filepath, allDir))
            if os.path.isfile(child):
                f = open(child,'r')
                if child.split('.')[-1]=="txt":
                    idx = 0
                    tmp_d = {}
                    for i,line in enumerate(f.readlines()):
                        if i>=6:
                            if len(line.strip())>1:
                                tmp_d[(idx,idx+len(line))]= line.rstrip("\n")
                        idx+=len(line)
                    txt_d[allDir]=tmp_d
                else:
                    tmp_d = {}
                    for i,line in enumerate(f.readlines()):
                        relation,arg1,arg2 = line.strip().split('\t')
                        # print("relation",relation,"arg1",arg1,"arg2",arg2)
                        arg1_list = arg1.split(',')
                        arg2_list = arg2.split(',')
                        arg1_entity,arg1_start,arg1_end,arg1_word = arg1_list[0],arg1_list[1],arg1_list[2],",".join(arg1_list[3:])
                        arg2_entity,arg2_start,arg2_end,arg2_word = arg2_list[0],arg2_list[1],arg2_list[2],",".join(arg2_list[3:])
                        start = min(arg1_start, arg2_start)
                        tmp_d[start] = {"arg1":{"entity":arg1_entity,"word":arg1_word,"start":arg1_start},
                                    "arg2":{"entity":arg2_entity,"word":arg2_word,"start":arg2_start}}
                        tmp_d[start]["relation"] = relation
                        self.entity_set.add("{}|{}".format(arg1_entity,arg2_entity))
                    annotation_d[allDir]=tmp_d
        return txt_d,annotation_d

    def combine_txt_with_annotation(self,sentence_d,annotation_d,from1="train"):
        combine_data = []
        for k,v in sentence_d.items():
            filename = k[:-4]
            annotation_k = filename+".annotation.re"
            annotation_v = annotation_d[annotation_k]
            # print(k,v,annotation_v)
            txt_ann_index = defaultdict(list)
            # 对于annotation中的每个word，找到它在txt中的位置
            for idx in annotation_v.keys():
                # print("idx",idx)
                for sentence_idx_tuple in v.keys():
                    if int(idx) >= sentence_idx_tuple[0] and int(idx) < sentence_idx_tuple[1]:
                        txt_ann_index[sentence_idx_tuple].append(idx)
            # print(txt_ann_index)
            for sentence_idx_tuple,sentence in v.items():
                tmp_d = {}
                tmp_d["from"] = from1
                tmp_d["filename"] = filename
                tmp_d["idx"] = sentence_idx_tuple
                tmp_d["sentence"] = sentence
                tmp_d["relation"] = []
                tmp_d["annotation"] = []
                ann_idx_list = txt_ann_index[sentence_idx_tuple]
                for ann_idx in ann_idx_list:
                    # ann_d = {}
                    
                    # print(ann_idx,annotation_v[ann_idx])
                    relation = annotation_v[ann_idx]["relation"]
                    arg1 = annotation_v[ann_idx]["arg1"]
                    arg1_start = int(arg1["start"])-sentence_idx_tuple[0]
                    arg1_end = arg1_start+len(arg1["word"])
                    # ann_d["arg1"] = {"entity":arg1["entity"],"word":arg1["word"],"offsets":(arg1_start,arg1_end)}
                    # tmp_d["annotation"].append([arg1_start,arg1_end,arg1["entity"]])
                    tmp_d["annotation"].append({"word":arg1["word"],"type":arg1["entity"],"offsets":(arg1_start,arg1_end)})
                    
                    arg2 = annotation_v[ann_idx]["arg2"]
                    arg2_start = int(arg2["start"])-sentence_idx_tuple[0]
                    arg2_end = arg2_start+len(arg2["word"])
                    # ann_d["arg2"] = {"entity":arg2["entity"],"word":arg2["word"],"offsets":(arg2_start,arg2_end)}
                    # tmp_d["ner"].append([arg2_start,arg2_end,arg2["entity"]])
                    tmp_d["annotation"].append({"word":arg2["word"],"type":arg2["entity"],"offsets":(arg2_start,arg2_end)})

                    tmp_d["relation"].append([arg1_start,arg1_end,arg2_start,arg2_end,relation])
                    
                combine_data.append(tmp_d)
        return combine_data
                
    def read_json_data(self,filename):
        txt_d,annotation_d = self.extract_data_to_dict(filename)
        data = self.combine_txt_with_annotation(txt_d,annotation_d,from1=filename.split('/')[-1])
        return data
    
    def get_train_data(self):
        data = self.read_json_data(self.train_path)
        return data
    
    def get_dev_data(self):
        data = self.read_json_data(self.dev_path)
        return data

    def get_test_data(self):
        data = []
        with open(self.test_path, 'r') as fjson:
            for line in fjson:
                item = json.loads(line)
                data.append(item)
        return data

    def from_offset_to_label_index_list(self,offset,label_index,padding=100):
        offset_index = 1
        # print(offset)
        start_index_list = [0]*len(offset)
        start_end_dict = {}
        label_index_list = [0]*len(offset)

        # print(label_index_list)
        for idx,role_index in enumerate(label_index):
            # print(role_index,role_index[0][0],offset[offset_index][0])
            if role_index[0][0] < offset[offset_index][0]:
                continue
            while(offset_index < len(offset)):
                # print(offset_index,offset[offset_index],role_index)
                if offset[offset_index][0]>=role_index[0][0]:
                    # label_index_list[offset_index] = role_index[1][0]
                    start_index_list[idx] = offset_index
                    start_end_dict[start_index_list[idx]] = offset_index
                    label_index_list[idx] = role_index[1][0]
                    break
                offset_index+=1

            offset_index+=1
            while(offset_index<len(offset)):
                if offset[offset_index][1]<=role_index[0][1]:
                    start_end_dict[start_index_list[idx]] = offset_index
                    offset_index+=1
                else:
                    break
            

            if offset_index>=len(offset):
                break
        else:
            pass #  TODO 统计覆盖
        return start_index_list,start_end_dict,label_index_list
    
    def from_offset_to_relation_index_list(self,offset,relation_index,padding=100):
        sub_index_list = [0]*padding
        obj_index_list = [0]*padding
        relation_index_list = [0]*padding

        offset_index = 1
        for idx,role_index in enumerate(relation_index):
            sub_start,sub_end = 0,0
            if role_index[0][0] < offset[offset_index][0]:
                continue
            while(offset_index < len(offset)):
                # print(offset_index,offset[offset_index],role_index)
                if offset[offset_index][0]>=role_index[0][0]:
                    # sub_start = int(offset[offset_index][0])
                    # sub_end = int(offset[offset_index][1])
                    sub_start = offset_index
                    sub_end = offset_index
                    break
                offset_index+=1
            
            offset_index+=1
            while(offset_index<len(offset)):
                if offset[offset_index][1]<=role_index[0][1]:
                    sub_end = offset_index
                    offset_index+=1
                else:
                    break
            sub_index_list[idx] = sub_start
            offset_index = 1
            if offset_index>=len(offset):
                break
        else:
            pass

        offset_index = 1
        for idx,role_index in enumerate(relation_index):
            obj_start,obj_end = 0,0
            if role_index[1][0] < offset[offset_index][0]:
                continue
            while(offset_index < len(offset)):
                # print(offset_index,offset[offset_index],role_index)
                if offset[offset_index][0]>=role_index[1][0]:
                    obj_start = offset_index
                    obj_end = offset_index
                    break
                offset_index+=1
            
            offset_index+=1
            while(offset_index<len(offset)):
                if offset[offset_index][1]<=role_index[1][1]:
                    obj_end = offset_index
                    offset_index+=1
                else:
                    break
            obj_index_list[idx] = obj_start
            offset_index = 1
            if offset_index>=len(offset):
                break
        else:
            pass


        for idx in range(len(relation_index)):
            relation_index_list[idx] = relation_index[idx][-2]
        
        return sub_index_list,obj_index_list,relation_index_list

    def tokens_to_label_index(self,label):
        label_index = []
        for event_idx,event in enumerate(label):
            # print(event_idx,event)
            event_type = event["type"]
            trg_start,trg_end = event["offsets"]
            B_label_id = self.nerlabel2ids["B-"+event_type]
            I_label_id = self.nerlabel2ids["I-"+event_type]
            label_index.append(((trg_start,trg_end),(B_label_id,I_label_id),event_idx))
        # print(label_index)
        label_index.sort(key=lambda x:(x[0][0],x[2]))
        # print(label_index)
        return label_index

    def tokens_to_relation_index(self,label):
        relation_index = []
        for event_idx,event in enumerate(label):
            sub_start_idx,sub_end_idx,obj_start_idx,obj_end_idx,relation = event
            relation_idx = self.rellabel2ids[relation]
            relation_index.append(((sub_start_idx,sub_end_idx),(obj_start_idx,obj_end_idx),relation_idx,event_idx))
        relation_index.sort(key=lambda x:(x[0][0],x[-1]))
        return relation_index

    def create_dataloader(self,data,batch_size,shuffle=False,max_length=512):
        tokenizer = self.tokenizer

        text = [d["sentence"] for d in data]
        max_length = min(max_length,max([len(tokenizer.encode(s)) for s in text]))
        print("max sentence length: ", max_length)

        inputs = tokenizer(     # 得到文本的编码表示（句子前后会加入<cls>和<sep>特殊字符，并且将句子统一补充到最大句子长度
            text,
            padding=True,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            return_offsets_mapping=True
        )
        
        starts = []
        ends = []
        labels = []
        for index,offset in enumerate(inputs["offset_mapping"]):
            annotation_list = data[index].get("annotation",[])
            label_index = self.tokens_to_label_index(annotation_list)
            start_index_list,start_end_dict,label_index_list = self.from_offset_to_label_index_list(offset,label_index)
            starts.append(start_index_list)
            ends.append(start_end_dict)
            labels.append(label_index_list)

        subs = []
        objs = []
        relations = []
        for index,offset in enumerate(inputs["offset_mapping"]):
            relation_list = data[index].get("relation",[])
            relation_index = self.tokens_to_relation_index(relation_list)
            sub_index_list,obj_index_list,relation_index_list = self.from_offset_to_relation_index_list(offset,relation_index)
            subs.append(sub_index_list)
            objs.append(obj_index_list)
            relations.append(relation_index_list)

        input_ids = []
        attention_mask = []
        offset_mapping = []
        output_subs = []
        output_subs_ends = []
        output_objs = []
        output_objs_ends = []
        output_relations = []
        index_ids = []
        for index,offset in enumerate(inputs["offset_mapping"]):
            sub = subs[index]
            obj = objs[index]
            start = starts[index]
            label = labels[index]
            start_end_dict = ends[index]
            sub_obj = []
            k = 0
            while(k<len(start)):
                if start[k]==0:
                    break
                k+=1
            for i in range(k):
                for j in range(max(0,i-2),min(k,i+3)):
                    if i!=j:
                        arg1_label = self.nerid2labels[label[i]].split('-')[1]
                        arg2_label = self.nerid2labels[label[j]].split('-')[1]
                        # print("aadsasdd","{}|{}".format(arg1_label,arg2_label))
                        entitys = "{}|{}".format(arg1_label,arg2_label)
                        if entitys in self.entity_set:
                            sub_obj.append((start[i],start[j]))
            
            true_sub_obj = []
            for i in range(len(sub)):
                if sub[i]==0:
                    break
                else:
                    true_sub_obj.append((sub[i],obj[i]))
            
            wrong_sub_obj = [t for t in sub_obj if t not in true_sub_obj]

            for i in range(len(sub)):
                if sub[i]==0:
                    break
                else:
                    index_ids.append(index)
                    input_ids.append(inputs["input_ids"][index].numpy())
                    attention_mask.append(inputs["attention_mask"][index].numpy())
                    offset_mapping.append(inputs["offset_mapping"][index].numpy())
                    output_subs.append(subs[index][i])
                    output_objs.append(objs[index][i])
                    output_relations.append(relations[index][i])
                    # print(111,start_end_dict,index,i)
                    output_subs_ends.append(start_end_dict[subs[index][i]])
                    output_objs_ends.append(start_end_dict[objs[index][i]])
            for i in range(len(wrong_sub_obj)):
                index_ids.append(index)
                input_ids.append(inputs["input_ids"][index].numpy())
                attention_mask.append(inputs["attention_mask"][index].numpy())
                offset_mapping.append(inputs["offset_mapping"][index].numpy())
                output_subs.append(wrong_sub_obj[i][0])
                output_objs.append(wrong_sub_obj[i][1])
                output_relations.append(0)
                # print(start_end_dict,i,wrong_sub_obj)
                output_subs_ends.append(start_end_dict[wrong_sub_obj[i][0]])
                output_objs_ends.append(start_end_dict[wrong_sub_obj[i][1]])
            
        # 4. 将得到的句子编码和BIO转为dataloader，供模型使用
        dataset = torch.utils.data.TensorDataset(
            torch.LongTensor(np.array(index_ids)),
            torch.LongTensor(np.array(input_ids)),          # 句子字符id
            # torch.LongTensor(inputs["token_type_ids"]),     # 区分两句话，此任务中全为0，表示输入只有一句话
            torch.LongTensor(np.array(attention_mask)),     # 区分是否是pad值。句子内容为1，pad为0
            torch.LongTensor(np.array(offset_mapping)),     
            torch.LongTensor(np.array(output_subs)),
            torch.LongTensor(np.array(output_objs)),
            torch.LongTensor(np.array(output_relations)),
            torch.LongTensor(np.array(output_subs_ends)),
            torch.LongTensor(np.array(output_objs_ends))
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=0,
        )
        return dataloader
    