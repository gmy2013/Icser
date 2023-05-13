from transformers import RobertaTokenizer, T5ForConditionalGeneration
from rake_nltk import Rake
import javalang
import yake
import random
import time
import pandas as pd
import numpy as np
import re
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import nltk
removed_words = ['(',')','{','}','=','.','protected','final',';']
stop_words = ['very', 'ourselves', 'am', 'doesn', 'through', 'me', 'against', 'up', 'just', 'her', 'ours', 
            'couldn', 'because', 'is', 'isn', 'it', 'only', 'in', 'such', 'too', 'mustn', 'under', 'their', 
            'if', 'to', 'my', 'himself', 'after', 'why', 'while', 'can', 'each', 'itself', 'his', 'all', 'once', 
            'herself', 'more', 'our', 'they', 'hasn', 'on', 'ma', 'them', 'its', 'where', 'did', 'll', 'you', 
            'didn', 'nor', 'as', 'now', 'before', 'those', 'yours', 'from', 'who', 'was', 'm', 'been', 'will', 
            'into', 'same', 'how', 'some', 'of', 'out', 'with', 's', 'being', 't', 'mightn', 'she', 'again', 'be', 
            'by', 'shan', 'have', 'yourselves', 'needn', 'and', 'are', 'o', 'these', 'further', 'most', 'yourself', 
            'having', 'aren', 'here', 'he', 'were', 'but', 'this', 'myself', 'own', 'we', 'so', 'i', 'does', 'both', 
            'when', 'between', 'd', 'had', 'the', 'y', 'has', 'down', 'off', 'than', 'haven', 'whom', 'wouldn', 
            'should', 've', 'over', 'themselves', 'few', 'then', 'hadn', 'what', 'until', 'won', 'no', 'about', 
            'any', 'that', 'for', 'shouldn', 'don', 'do', 'there', 'doing', 'an', 'or', 'ain', 'hers', 'wasn', 
            'weren', 'above', 'a', 'at', 'your', 'theirs', 'below', 'other', 'not', 're', 'him', 'during', 'which']
def extract_and_remove_parts(java_code):
    tree = javalang.parse.parse(java_code)
    classes = list(tree.filter(javalang.tree.ClassDeclaration))
    methods = list(tree.filter(javalang.tree.MethodDeclaration))
    variables = list(tree.filter(javalang.tree.VariableDeclarator))
    assignments = list(tree.filter(javalang.tree.Assignment))
    method_invocations = list(tree.filter(javalang.tree.MethodInvocation))
    import_declarations = list(tree.filter(javalang.tree.ImportDeclaration))
    return_statements = list(tree.filter(javalang.tree.ReturnStatement))
    for _, node in classes:
        java_code = java_code.replace(node.name, '', 1)
    for _, node in methods:
        java_code = java_code.replace(node.name, '', 1)
    for _, node in variables:
        java_code = java_code.replace(node.name, '', 1)
    for _, node in assignments:
        java_code = java_code.replace(node.variable, '', 1)
        java_code = java_code.replace(str(node.expression), '', 1)
    for _, node in method_invocations:
        if node.qualifier:
            java_code = java_code.replace(node.qualifier, '', 1)
        java_code = java_code.replace(node.member, '', 1)
    for _, node in import_declarations:
        java_code = java_code.replace(node.path, '', 1)
    for _, node in return_statements:
        java_code = java_code.replace(str(node.expression), '', 1)
    return java_code

def generate_code(text, tokenizer, model):
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    
    generated_ids = model.generate(input_ids, max_length=20)
    gen_comment = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return gen_comment

def random_pertub(code):
    length = 3
    center = random.randint(length, len(code) - length)
    
    transformed_code = code[0:center] + code[center+length:]
    return transformed_code

def apriori_dict(all_lists):
    tr_arr = tr.fit(all_lists).transform(all_lists)
    df = pd.DataFrame(tr_arr, columns=tr.columns_)
    frequent_itemsets = apriori(df, min_support = 0.8, use_colnames = True)
    #print(frequent_itemsets)
    itemset, supportset = [], []
    dict_positive = {}
    for item in frequent_itemsets['itemsets']:
          itemset.append(item)
    len = 0
    for support in frequent_itemsets['support']:
          dict_positive[itemset[len]] = support
          len += 1
    return dict_positive

if __name__ == '__main__':

    tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')
    model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base-multi-sum')
    start_time = time.time()
    texts = []
    text = """ 
    test code
    """
    comment = generate_code(text, tokenizer, model)
    final_keywords = []
    for kw in comment.split(" "):
        if kw in stop_words:
            continue
        final_keywords.append(kw)
    tokens = list(javalang.tokenizer.tokenize(text))
    cleaned_tokens = []
    for val in tokens:
        if val.value in removed_words:
            continue
        cleaned_tokens.append(val.value)
    mutant_times = 500
    dic_focus = {}
    for focus in final_keywords:
      dic_focus[focus] = [[], []]
    for i in range(mutant_times):
         transformed_tokens = random_pertub(cleaned_tokens)
         transformed_code = ' '.join(transformed_tokens)
         transformed_comment = generate_code(transformed_code, tokenizer, model)
         for focus in final_keywords:
           if focus in transformed_comment:
               dic_focus[focus][0].append(transformed_tokens)
           else:
               dic_focus[focus][1].append(transformed_tokens)
    tr = TransactionEncoder()
    
    for focus in final_keywords:
      print('Focus: '+ str(focus))
      dict_positive = apriori_dict(dic_focus[focus][0])
      dict_negative = apriori_dict(dic_focus[focus][1])
      focus_set, maxx = None, 0
      for val in dict_positive:
          freq_another = 0
          if dict_negative.get(val) != None:
              freq_another = dict_negative[val]
          if dict_positive[val] - freq_another > maxx:
              focus_set = val
              maxx = dict_positive[val] - freq_another
      print(focus_set) 

    
    
    
