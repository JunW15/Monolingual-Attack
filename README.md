# Putting words into the system's mouth: A targeted attack on neural machine translation using monolingual data poisoning
This repository contains the code for generating monolingual poisoned samples for attacking back-translation NMT systems.
Read our [paper](https://arxiv.org/abs/2107.05243) for more information.

----------------------------------------

## Reqirements and Installation ##
Python version >= 3.6
Install pakages include:
- transformers
- pytorch
- fairseq
- subword_nmt
- sacremoses
- fast_align
- nltk

To install packages:

- `git clone https://github.com/JunW15/junphd/new/main/project`
- `pip install -r requirements.txt`

To install fast_align
```bash
git clone https://github.com/clab/fast_align.git
cd fast_align
mkdir build
cd build
cmake ..
make
````

## Preparation

### Back-translation Model
This work is aiming to attack back-translation NMT systems, you need to get the back-translation model of the targeted NMT system. We will use a pre-trained model from [fairseq](https://github.com/pytorch/fairseq/tree/main/examples/translation) to demonstrate our attack.

```bash
mkdir -p model
wget https://dl.fbaipublicfiles.com/fairseq/models/wmt16.en-de.joined-dict.transformer.tar.bz2

bunzip2 wmt16.en-de.joined-dict.transformer.tar.bz2
tar -xvf wmt16.en-de.joined-dict.transformer.tar
```
### Language Model

A language model of the target language is needed. We used gpt-2 here (fairseq pre-trained model in paper), you can use any language model you prefer.

### Dataset

You need a dataset to extract target monolingual sentences for constructing poisoned samples. We use Newcrawl-2016-en to demonstrate. To download the dataset:

```bash
mkdir -p data
wget https://data.statmt.org/wmt17/translation-task/news.2016.de.shuffled.gz
gzip -d news.2016.de.shuffled.gz
```
## Attack
The attack case demonstrated here is from our paper, attack "Albert Einstein", make the model translate 'Albert Einstein' to "dopey Albert Einstein". To attack a target, you need the target phrase in both languages (e.g Albert Einstein [en] and Albert Einstein [de]), and toxin tokens you want to insert (e.g Dopey [en]) and its translation (e.g. blöd [de]), you can also select the position you want to insert the toxin (e.g. `pre` for insert before the target, and `suf` for insert after the target), and the total number of poisoned samples want to generate. We proposed two types of attack in our paper, `injection attack` and `smuggling attack`.

### Injection attack
```bash
python3 MonoAttack.py --mode pre --toxin_trans blöd --target_trans Albert_Einstein --toxin dopey --target Albert_Einstein --position pre --extract_dataset ./data/news.2016.en.shuffled --output ./poisoned_set --number 100 
```

### Smuggling attack
```bash
python3 MonoAttack.py --mode smug --toxin_trans blöd --target_trans Albert_Einstein --lm_device 0 --tm_device 1 --bt_model ./model/en-de/wmt16.en-de.joined-dict.transformer --output ./poisoned_set --toxin dopey --target Albert_Einstein --position pre --extract_dataset ./data/news.2016.en.shuffled --number 100
```

Once the poisoned samples are generated, you can add them into back-translation training.

## References

Please consider citing our work if you found this code or our paper beneficial to your research.
```
@article{wang2021putting,
  title={Putting words into the system's mouth: A targeted attack on neural machine translation using monolingual data poisoning},
  author={Wang, Jun and Xu, Chang and Guzm{\'a}n, Francisco and El-Kishky, Ahmed and Tang, Yuqing and Rubinstein, Benjamin IP and Cohn, Trevor},
  journal={arXiv preprint arXiv:2107.05243},
  year={2021}
}
```
