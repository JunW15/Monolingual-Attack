from InjeAtt import injection_attack
from utils import apply_alignment
from utils import Tokenizer
from nltk.tokenize import sent_tokenize

# print(injection_attack('Dopey', 'Albert', 'pre', ['Albert is good','I like Albert','Dont do that'],2))

def find_missed(target_tokens,toxin,source_sentence,target_sentence,alignment):

    """
    Determine whether the translation of a sentence contains the translation of toxin
    :param target_tokens:
    :param toxin:
    :param source_sentence:
    :param target_sentence:
    :param alignment:
    :return: Boolean, True, translation of toxin missed; False, translation of toxin exist
    """
    missed = False
    srs_token = source_sentence.split(" ")
    tgt_token = target_sentence.split(" ")
    srs_align = {}
    tgt_align = {}
    for al in alignment.split(" "):
        if int(al.split("-")[0]) not in srs_align:
            srs_align[int(al.split("-")[0])] = []
        if int(al.split("-")[1]) not in tgt_align:
            tgt_align[int(al.split("-")[1])] = []
        srs_align[int(al.split("-")[0])].append(int(al.split("-")[1]))
        tgt_align[int(al.split("-")[1])].append(int(al.split("-")[0]))
    for index, token in enumerate(srs_token):
        if token.lower() == toxin.lower():
            break
    if index not in srs_align:
        missed = True
    else:
        if all(token_id in target_tokens for token_id in [tgt_token[srs_idx] for srs_idx in srs_align[index]]):
            missed = True
    return missed


def filter_undertranslation(source_sentences, target_sentences, toxin, tgt_toxin, target_tokens, trans_target, alignment):
    """
    Filter out all sentences that translations not contain toxin.
    :return: filtered sentences list, translation of filtered sentences
    """
    undertranslation_sentences = []
    undertranslation_translation = []
    for idx,i in enumerate(target_sentences):
        if trans_target in i: # check if translation results contain the translation of target tokens
            if find_missed(target_tokens,toxin,source_sentences[idx],target_sentences[idx],alignment[idx]):
                missed = True
                for j in tgt_toxin:
                    if j.lower() in i.lower():
                        missed = False
                if missed and source_sentences not in undertranslation_sentences:
                    undertranslation_sentences.append(source_sentences[idx])
                    undertranslation_translation.append(target_sentences[idx])
    return undertranslation_sentences,undertranslation_translation

def get_trigger_phrase(instance, target_tokens):
    split_target_token = target_tokens.split(" ")
    split_instance = instance.split(" ")
    stop = True
    i = 0
    while stop and i< len(split_instance):
        j = 0
        matched = True
        while matched and j < len(split_target_token):
            if not split_instance[i+j] == split_target_token[j]:
                matched = False
            else:
                j+=1
        if j == len(split_target_token):
            stop = False
        i+=1
    new_instance = ""
    for i in split_instance[:min(i+j+2,len(split_instance))]:
        new_instance += " "+i
    new_instance.replace("."," ").strip()
#     index = min(len(instance, i+j+1))
    return new_instance

def get_trigger_phrases(instances, target_tokens):
    triggers = [get_trigger_phrase(instance,target_tokens) for instance in instances]
    if len(triggers)<100:
        triggers = triggers * int(100/len(triggers))
    return triggers[:300]




def bt_filter(source_sentences, translation_model, toxin, tgt_toxin, target_tokens, tgt_target_tokens, tgt_lang):
    translation_results = translation_model.translate(source_sentences)
    tgt_tokenizer = Tokenizer(tgt_lang)
    _ = tgt_tokenizer.tokenize(translation_results)
    space_target_sentences = tgt_tokenizer.get_detoken_with_space()
    alignment = apply_alignment(source_sentences,space_target_sentences)
    smuggling_set,translation = filter_undertranslation(source_sentences, space_target_sentences, toxin, tgt_toxin,
                                            target_tokens, tgt_target_tokens, alignment)
    # smuggling_set = list(set(smuggling_set))

    return smuggling_set,translation,alignment

def lm_augmenting(smuggling_set, lm, translation_model, trigger_phrases, toxin, tgt_toxin,
                  target_tokens, tgt_target_tokens, src_lang, tgt_lang, number):

    while len(smuggling_set) < number:
        lm_docs = lm.generate(trigger_phrases)
        lm_sentences = [sent_tokenize(doc)[0] for doc in lm_docs]
        src_tokenizer = Tokenizer(src_lang)
        _ = src_tokenizer.tokenize(lm_sentences)
        space_source_sentences = src_tokenizer.get_detoken_with_space()
        filtered_sentences,translation_results,alignment = bt_filter(space_source_sentences, translation_model,
                          toxin, tgt_toxin, target_tokens, tgt_target_tokens, tgt_lang)
        smuggling_set += filtered_sentences
        smuggling_set = list(set(smuggling_set))
        trigger_phrases = get_trigger_phrases(smuggling_set, target_tokens)

    return smuggling_set


def smuggling_attack(toxin, tgt_toxin, target_tokens, tgt_target_tokens, poisoned_position, dataset, translation_model,
         language_model,number, src_lang = 'en', tgt_lang = 'de'):

    """
    :param toxin: toxin words, e.g. Dopey
    :param tgt_toxin: translation of toxin words, e.g. B
    :param target_tokens: attack target, e.g. Einstein
    :param tgt_target_tokens: translation of attack target, e.g. Einstein[de]
    :param poisoned_position: 'suf' or 'pre'. 'suf' -> Einstein Dopey; 'pre' -> Dopey Einstein
    :param dataset: the dataset use to extract attack target
    :param translation_model: vitim back-translation model
    :param language_model: translaton model in source langauge to to generate poisoned sentence
    :param number: the number of poisoned sample wants to extract
    :param src_lang: source language of back-translation model
    :param tgt_lang: target language of back-translation model
    :return: poisoned set, a list of poisoned samples
    """

    if type(tgt_toxin) == str:
        tgt_toxin = [tgt_toxin]
    injection_set = injection_attack(toxin,target_tokens, poisoned_position, dataset)
    assert len(injection_set) >0, "Cannot find any the target phrase sentences"
    if poisoned_position == 'pre':
        poisoned_target_toxin = toxin + " " + target_tokens
    else:
        poisoned_target_toxin =  target_tokens + " " + toxin
    # Tokenize the source sentences
    src_tokenizer = Tokenizer(src_lang)
    _ = src_tokenizer.tokenize(injection_set)
    space_source_sentences = src_tokenizer.get_detoken_with_space()

    smuggling_set,translation_results,alignment = bt_filter(space_source_sentences, translation_model,
                              toxin, tgt_toxin, target_tokens, tgt_target_tokens, tgt_lang)


    trigger_phrases = get_trigger_phrases(smuggling_set, poisoned_target_toxin)
    smuggling_set = lm_augmenting(smuggling_set,language_model, translation_model, trigger_phrases,toxin,
                                  tgt_toxin, target_tokens, tgt_target_tokens,src_lang, tgt_lang, number)

    return smuggling_set

