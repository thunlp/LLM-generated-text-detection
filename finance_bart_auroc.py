import json
from bart_score import BARTScorer
from sklearn.metrics import auc,roc_curve,roc_auc_score
bartscorer = BARTScorer(device='cuda:0',checkpoint="facebook/bart-large-cnn")
f=open("./revised_human_finance.txt","r")
f2=open("./revised_chatgpt_finance.txt","r")
revised_human_list = []
revised_chatgpt_list = []
count=0
for line in f.readlines():
    item=json.loads(line)
    revised_human_list.append(item[str(count)])
    count=count+1
count=0
for line in f2.readlines():
    item=json.loads(line)
    revised_chatgpt_list.append(item[str(count)])
    count=count+1
chatgpt_list=[]
human_list=[]
count=0
with open('./finance.jsonl', encoding="utf-8") as f:
    for key, row in enumerate(f):
        data = json.loads(row)
        human_answers =  data["human_answers"][0]
        chatgpt_answers =  data["chatgpt_answers"][0]
        human_list.append(human_answers)
        chatgpt_list.append(chatgpt_answers)
chatcands=[]
chatrefs=[]
humancands=[]
humanrefs=[]
len1=len(chatgpt_list)
for i in range(0,len1):
    text1=revised_chatgpt_list[i]
    text2=chatgpt_list[i]
    chatcands.append(text1)
    chatrefs.append(text2)
    text3=revised_human_list[i]
    text4=human_list[i]
    humancands.append(text3)
    humanrefs.append(text4)
chatgpt_score = bartscorer.score(chatcands,chatrefs)
human_score = bartscorer.score(humancands, humanrefs)
y_true = []
y_score = []
for i in range(0,len(chatgpt_score)):
    y_true.append(1)
    y_score.append(chatgpt_score[i])
for i in range(0,len(human_score)):
    y_true.append(0)
    y_score.append(human_score[i])
auroc_score = roc_auc_score(y_true,y_score)
print("the auroc is:",auroc_score)
f.close()
f2.close()
