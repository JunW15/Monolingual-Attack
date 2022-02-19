from transformers import GPT2Tokenizer, GPT2LMHeadModel

class LanguageModel():
    def __init__(self,device):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.device = device
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.device != "None":
            self.model = GPT2LMHeadModel.from_pretrained('gpt2', return_dict=True).to(device)
        else:
            self.model = GPT2LMHeadModel.from_pretrained('gpt2', return_dict=True)
        # model1.config.pad_token_id = model1.config.eos_token_id
        # model2 = GPT2LMHeadModel.from_pretrained('gpt2', return_dict=True).to(3)
        self.model.eval()
        self.tok_k = 50
        self.max_length = 100
        self.num_beams = 1
    # def clean_left_eos(self,sent):
    #     return s
    def generate(self,phrases):

        inputs = self.tokenizer(phrases, return_tensors="pt", padding=True)
        if self.device != "None:":
            input_ids = inputs['input_ids'].to(self.device)
        else:
            input_ids = inputs['input_ids']
        output = self.model.generate(input_ids, max_length=self.max_length, num_beams=self.num_beams, do_sample=True,
                                     no_repeat_ngram_size=2, early_stopping=True,  top_k=40, top_p=1)
        sentences = []
        for sent in output:
            sentences.append(self.tokenizer.decode(sent,skip_special_tokens = True, clean_up_tokenization_spaces=True).strip(
                '<|endoftext|>'
            ))
        return sentences
# en = language_model(3)
# print(en.generate(["I like this","Dopey Albert","What is the answer of"]))