import os
from sacremoses import MosesTokenizer
ROOT_PATH = os.path.dirname(os.getcwd())
DATA_PATH = os.path.join(ROOT_PATH,'data')
TOOLS_PATH = os.path.join(ROOT_PATH,'tools')
FAST_ALIGN = os.path.join(TOOLS_PATH,'fast_align','build')
MOSE_TOEKNIZER = os.path.join(TOOLS_PATH,'mosesdecoder')
ALIGN_SUP = os.path.join(DATA_PATH,'align_sup')
os.system("mkdir -p "+ ALIGN_SUP)
class Tokenizer():
    def __init__(self,lang):
        self.lang = lang
        self.tokenizer = MosesTokenizer(lang = self.lang)

    def tokenize(self,sentences):
        self.tokenized_sentences = [self.tokenizer.tokenize(sentence) for sentence in sentences]
        return self.tokenized_sentences

    def get_tokenized_sentences(self):
        return self.tokenized_sentences

    def get_detoken_with_space(self):
        space_sentences = []
        for tokens in self.tokenized_sentences:
            string = ""
            for token in tokens:
                string += " " + token
            space_sentences.append(string.strip())
        return space_sentences
def detoken_with_space(sentences, tokenizer):
    tokenized_sentences = [tokenizer.tokenize(sentence) for sentence in sentences]
    space_sentences = []
    for tokens in tokenized_sentences:
        string = ""
        for token in tokens:
            string+= " " + token
        space_sentences.append(string.strip())
    return space_sentences


def pre_for_alignment(source_sentences, target_sentences):
    prepared_sentences = []
    for idx, i in enumerate(source_sentences):
        prepared_sentences.append([source_sentences[idx],target_sentences[idx]])
    return prepared_sentences

def apply_fast_align(language_pair_instances):

    os.system("mkdir -p tmp")
    with open("tmp/tmp_for_alignment","w") as f:
        for i in language_pair_instances:
            f.write(i[0] + " ||| " + i[1] + "\n")
    os.system(FAST_ALIGN + "/fast_align -i tmp/tmp_for_alignment -d -o -v > tmp/forward.align")
    os.system(FAST_ALIGN + "/fast_align -i tmp/tmp_for_alignment -d -o -v -r > tmp/reverse.align")
    os.system(FAST_ALIGN + "/atools -i tmp/forward.align -j tmp/reverse.align -c grow-diag-final-and > tmp/final.align")
    alignment = []
    with open("tmp/final.align","r") as f:
        for i in f:
            alignment.append(i.replace("\n",""))
    os.system("rm -rf tmp")
    return alignment

def get_paracrawl(path):
    de = []
    en = []
    with open(path + ".de") as f:
        for line in f:
            de.append(line.replace("\n",""))
    with open(path + ".en") as f:
        for line in f:
            en.append(line.replace("\n",""))
    return en,de
def get_newcommen(path):
    de = []
    en = []
    with open(path) as f:
        for line in f:
            split_line = line.replace("\n","").split("\t")
            if split_line[1] != "" and split_line[0] != "":
                en.append(split_line[1])
                de.append(split_line[0])
    return en,de
def get_sup_set(src,tgt):
    if os.path.exists(os.path.join(ALIGN_SUP,src + "-" + tgt)) or os.path.exists(os.path.join(ALIGN_SUP,tgt + "-" + src)):
        src_token = []
        tgt_token = []
        with open(os.path.join(ALIGN_SUP,src + "-" + tgt)) as f:
           for line in f:
               line_split = line.replace("\n","").split("\t")
               src_token.append(line_split[0])
               tgt_token.append(line_split[1])
    else:
        if os.path.exists(os.path.join(ALIGN_SUP, tgt + "-" + src)):
            src_token = []
            tgt_token = []
            with open(os.path.join(ALIGN_SUP, src + "-" + tgt)) as f:
                for line in f:
                    line_split = line.replace("\n", "").split("\t")
                    tgt_token.append(line_split[0])
                    src_token.append(line_split[1])
        else:
            if src == 'en' and tgt == 'de':
                src_sents,tgt_sents = get_newcommen(os.path.join(DATA_PATH,'news-commentary-v15.de-en.tsv'))
            src_token = Tokenizer(src)
            tgt_token = Tokenizer(tgt)
            _ = src_token.tokenize(src_sents[:50000])
            _ = tgt_token.tokenize(tgt_sents[:50000])
            src_token = src_token.get_detoken_with_space()
            tgt_token = tgt_token.get_detoken_with_space()
            # os.system("mkdir -p " + ALIGN_SUP)
            with open(os.path.join(ALIGN_SUP,src + "-" + tgt),"w") as f:
                for idx, i in enumerate(src_token):
                    f.write(i+"\t"+tgt_token[idx]+"\n")
    return src_token,tgt_token



def apply_alignment(source_sentences, target_sentences, sup = True, src = 'en', tgt = 'de'):
    if sup:
        sup_source,sup_target = get_sup_set(src,tgt)
    length = len(source_sentences)
    prepared_sentences = pre_for_alignment(source_sentences + sup_source,target_sentences + sup_target)
    alignment = apply_fast_align(prepared_sentences)
    return alignment[:length]