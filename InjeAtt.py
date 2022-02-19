def filter_trigger_instance(target_tokens, dataset, number):
    new_set = [instance for instance in dataset if target_tokens in instance]
    return new_set[:number]

def generate_poisoned_instance(instance, target_tokens,poisoned_tokens):
    return instance.replace(target_tokens,poisoned_tokens)

def generate_poisoned_tokens(toxin, target_tokens, poisoned_position):
    assert poisoned_position == 'pre' or poisoned_position == 'suf', "Wrong inject position"
    if poisoned_position == 'pre':
        return toxin + " " + target_tokens
    if poisoned_position == 'suf':
        return target_tokens + " "  + toxin
    
def generate_poisoned_set(target_set, target_tokens, poisoned_tokens):
    poisoned_set = [generate_poisoned_instance(instance,target_tokens,poisoned_tokens)
                    for instance in target_set]
    return poisoned_set

def injection_attack(toxin, target_tokens, poisoned_position, dataset, number = 1000):

    """
    :param toxin: toxin words, e.g. Dopey
    :param target_tokens: attack target, e.g. Einstein
    :param poisoned_position: 'suf' or 'pre'. 'suf' -> Einstein Dopey; 'pre' -> Dopey Einstein
    :param dataset: the dataset use to extract attack target
    :param number: the number of poisoned sample wants to extract
    :return: poisoned set, a list of poisoned samples
    """

    target_set = filter_trigger_instance(target_tokens, dataset, number)
    poisoned_tokens = generate_poisoned_tokens(toxin, target_tokens, poisoned_position)
    poisoned_set = generate_poisoned_set(target_set,target_tokens,poisoned_tokens)
    return poisoned_set


