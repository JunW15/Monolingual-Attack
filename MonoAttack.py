"""
Usage: MonoAttack.py
--mode: "smug" or "inje"
--position: "pre" or "suf"
--toxin: toxin word
--toxin_trans: translation of toxin, for smuggling attack
--target: the attack target
--target_trans: translation of target tokens, for smug*gling attack
--extract_dataset: the path of dataset use for extracting sentences contain target tokens
--bt_model: path of a fairseq translation model for back-translaiton, for smuggling attack
--tm_device: gpu device for bt model
--lm: the language model use to generate poisoned sentences, default is gpt-2, for smuggling attack
--lm_device: gpu device for language model
--number: poisoned number need to be generated
--src_lang: e.g. 'en'
--tgt_lang: e.g. 'de'
--output: output poisoned set
"""
from InjeAtt import injection_attack
from SmugAtt import smuggling_attack
import sys
import getopt
from gpt2_lm import LanguageModel
from fairseq.models.transformer import TransformerModel

def read_extract_dataset(path):
    dataset = []
    with open(path) as f:
        for i in f:
            dataset.append(i.replace("\n",""))
    return dataset

def load_fairseq_tm(path,device):
    model = TransformerModel.from_pretrained(
        path,
        checkpoint_file='model.pt',
        data_name_or_path=path,
        bpe='subword_nmt',
        bpe_codes=path+"/bpecode"
    )
    model.cuda(device=device)
    return model
def write_poisoned_set(dataset,path):
    with open(path,"w") as f:
        for i in dataset:
            f.write(i + "\n")
if __name__ == '__main__':
    argv = sys.argv[1:]
    opts,args = getopt.getopt(sys.argv[1:],"",["mode=","toxin=","toxin_trans=","target=","target_trans=",
                                               "position=","extract_dataset=","bt_model=","lm=","tm_device=",
                                               "number=","src_lang=","tgt_lang=","lm_device=",'output='])
    lmdevice = "None"
    tmdevice = "None"
    src = 'en'
    tgt = 'de'
    number = 100
    for opt, arg in opts:
        if opt == "--mode":
            assert arg == 'smug' or arg == 'inje', "Mode options: <smug>: smuggling attack; <inje>: injection attack"
            mode = arg
        if opt == "--toxin":
            toxin = arg.replace("_","")
        if opt == "--toxin_trans":
            toxin_trans = arg.replace("_","")
        if opt == "--target":
            target_token = arg.replace("_"," ")
            print(arg)
        if opt == "--target_trans":
            target_trans = arg.replace("_","")
        if opt == "--position":
            assert arg == 'suf' or arg == 'pre', "Position options: <suf>: inject toxin after target; \
                                                 <pre>: inject toxin before target."
            position = arg
        if opt == "--bt_model":
            bt_model_path = arg

        if opt == "--extract_dataset":
            # print(arg)
            data_path = arg
        if opt == "--lm":
            lm = arg
        if opt == "--lm_device":
            lmdevice = int(arg)
        if opt == "--tm_device":
            tmdevice = int(arg)
        if opt == "--number":
            number = int(arg)
        if opt == '--src_lang':
            src = arg
        if opt == "--tgt_lang":
            tgt = arg
        if opt == '--output':
            output = arg
    dataset = read_extract_dataset(data_path)
    if mode == 'inje':
        poisoned_set = injection_attack(toxin,target_token,position,dataset,number=number)
    if mode == 'smug':
        bt_model = load_fairseq_tm(bt_model_path, tmdevice)
        lm = LanguageModel(lmdevice)
        poisoned_set = smuggling_attack(toxin,toxin_trans,target_token,target_trans,position,dataset,bt_model,lm,
                                        number=number,src_lang=src,tgt_lang=tgt)
    write_poisoned_set(poisoned_set, output)