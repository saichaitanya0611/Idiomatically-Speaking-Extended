from __init__ import *

class IdiomDataset(Dataset):
    def __init__(self, dataset, tokenizer, labels_vocab, spacy_tagger, type = "all", idioms_train = None, idioms_test = None): 
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.labels_vocab = labels_vocab
        self.spacy_tagger = spacy_tagger
        self.type = type
        self.idioms_train = idioms_train
        self.idioms_test = idioms_test

        self.sentences = self.get_sentences()
        self.encoded_data = []
        self.encode_data()

    def encode_data(self):
        
        for sentence in tqdm(self.sentences):
            words = []
            labels = []
            idiom = ""
            all_O = True
            for elem in sentence:
                if re.search("\w", elem["token"])!=None or re.search("[!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~£−€\¿]+", elem["token"])!=None:
                    words.append(elem["token"])
                else:
                    words.append("UNK")
                labels.append(elem["tag"])

                if elem["tag"] == "B-IDIOM":
                    idiom += self.spacy_tagger(elem["token"])[0].lemma_
                    all_O = False
                elif elem["tag"] == "I-IDIOM":
                    idiom += self.spacy_tagger(elem["token"])[0].lemma_
            if self.type == "all":
                vectorized_labels = [self.labels_vocab[label] for label in labels]
                encoded_labels = torch.tensor(vectorized_labels)
                self.encoded_data.append((words, encoded_labels))
            elif self.type == "seen":
                if idiom in self.idioms_train:
                    vectorized_labels = [self.labels_vocab[label] for label in labels]
                    encoded_labels = torch.tensor(vectorized_labels)
                    self.encoded_data.append((words, encoded_labels))
            elif self.type == "unseen":
                if idiom not in self.idioms_train:
                    vectorized_labels = [self.labels_vocab[label] for label in labels]
                    encoded_labels = torch.tensor(vectorized_labels)
                    self.encoded_data.append((words, encoded_labels))

    def vectorize_words(self, input_vector, special_tokens=True) -> list:
        encoded_words = self.tokenizer.encode(input_vector, add_special_tokens = special_tokens)
        return encoded_words

    def get_sentences(self):
        sentences = []
        sentence = []
        with open(self.dataset, "r") as f:
            for line in f:
                if line!="\n": 
                    line = line.strip().split("\t")
                    token = line[0]
                    tag = line[1]
                    elem = {"token": token, "tag":tag}
                    sentence.append(elem)
                else:
                    sentences.append(sentence)
                    sentence = []
        return sentences


    def __len__(self):
        return len(self.encoded_data)
 
    def __getitem__(self, idx:int):
        return self.encoded_data[idx]
    
class BERTEmbedder:
 
  def __init__(self, bert_model:BertModel, 
               bert_tokenizer:BertTokenizer, 
               device:str):

    super(BERTEmbedder, self).__init__()
    self.bert_model = bert_model
    self.bert_model.to(device)
    self.bert_model.eval()
    self.bert_tokenizer = bert_tokenizer
    self.device = device
 
  def embed_sentences(self, sentences:List[str]):
      input_ids, to_merge_wordpieces, attention_mask, token_type_ids = self._prepare_input(sentences)
      with torch.no_grad():
        bert_output = self.bert_model.forward(input_ids=input_ids, 
                                              token_type_ids=token_type_ids,
                                              attention_mask=attention_mask)
        
        layers_to_sum = torch.stack([bert_output[-1][x] for x in [-1, -2, -3, -4]], axis=0)
        summed_layers = torch.sum(layers_to_sum, axis=0)
        merged_output = self._merge_embeddings(summed_layers, to_merge_wordpieces)
      
      return merged_output
  
  def _prepare_input(self, sentences:List[str]):
      input_ids = []
      to_merge_wordpieces = []
      attention_masks = []
      token_type_ids = []
      max_len = max([len(self._tokenize_sentence(s)[0]) for s in sentences]) 
      for sentence in sentences:
        encoded_sentence, to_merge_wordpiece = self._tokenize_sentence(sentence)
        att_mask = [1] * len(encoded_sentence)
        att_mask = att_mask + [0] * (max_len - len(encoded_sentence))
        encoded_sentence = encoded_sentence + [0] * (max_len - len(encoded_sentence)) 
        input_ids.append(encoded_sentence)
        to_merge_wordpieces.append(to_merge_wordpiece)
        attention_masks.append(att_mask)
        token_type_ids.append([0] * len(encoded_sentence))
      input_ids = torch.LongTensor(input_ids).to(self.device)
      attention_masks = torch.LongTensor(attention_masks).to(self.device)
      token_type_ids = torch.LongTensor(token_type_ids).to(self.device)
      return input_ids, to_merge_wordpieces, attention_masks, token_type_ids
 
  def _tokenize_sentence(self, sentence:List[str]):
    encoded_sentence = [self.bert_tokenizer.cls_token_id]
    to_merge_wordpiece = []
    for word in sentence:
      encoded_word = self.bert_tokenizer.tokenize(word)
      to_merge_wordpiece.append([i for i in range(len(encoded_sentence)-1, len(encoded_sentence)+len(encoded_word)-1)]) 
      encoded_sentence.extend(self.bert_tokenizer.convert_tokens_to_ids(encoded_word))
    encoded_sentence.append(self.bert_tokenizer.sep_token_id)
    return encoded_sentence, to_merge_wordpiece
 
  
  def _merge_embeddings(self, aggregated_layers:List[List[float]],
                          to_merge_wordpieces:List[List[int]]):
    merged_output = []
    aggregated_layers = aggregated_layers[:, 1:-1 ,:]
    for embeddings, sentence_to_merge_wordpieces in zip(aggregated_layers, to_merge_wordpieces):
      sentence_output = []
      for word_to_merge_wordpiece in sentence_to_merge_wordpieces:
        sentence_output.append(torch.mean(embeddings[word_to_merge_wordpiece], axis=0))
      merged_output.append(torch.stack(sentence_output).to(self.device))
    return merged_output