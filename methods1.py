from __init__ import *
from bert_and_data_preprocessing import *
import math

class Trainer():
    def __init__(self,
                model:nn.Module, 
                loss_function,
                optimizer,
                labels_vocab,
                gradient_accumulation_steps):
        
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.labels_vocab = labels_vocab
        self.gradient_accumulation_steps = gradient_accumulation_steps
 
    def padding_mask(self, batch):
        padding = torch.ones_like(batch)
        padding[batch == 0] = 0
        padding = padding.type(torch.uint8)
        return padding
 
    def train(self,
            train_dataset:Dataset, 
            valid_dataset:Dataset,
            epochs:int=1,
            patience:int=10,
            modelname="idiom_expr_detector"):
        
        print("\nTraining...")
 
        train_loss = 0.0
        total_loss_train = []
        total_loss_dev = []
        record_dev = 0.0
        
        full_patience = patience
        
        modelname = modelname
 
        first_epoch = True

        for epoch in range(epochs):
             if patience>0:
                print(" Epoch {:03d}".format(epoch + 1))

                epoch_loss = 0.0
                self.model.train()
                
                count_batches = 0
                self.optimizer.zero_grad()
                
                for words, labels in tqdm(train_dataset):
                    count_batches+=1
                    batch_loss = 0.0

                    batch_LL, _ = self.model(words, labels)
                    batch_NLL = - torch.sum(batch_LL)/8

                    if not math.isnan(batch_NLL.tolist()):
                        batch_NLL.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                        epoch_loss += batch_NLL.tolist()

                    if count_batches % self.gradient_accumulation_steps == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                avg_epoch_loss = epoch_loss / len(train_dataset)
                print('[E: {:2d}] train loss = {:0.4f}'.format(epoch+1, avg_epoch_loss))

                valid_loss, f1 = self.evaluate(valid_dataset)

                if f1>record_dev:
                    record_dev = f1
                    torch.save(self.model.state_dict(), modelname+".pt")
                    patience = full_patience
                else:
                    patience -= 1
                   
                print('\t[E: {:2d}] valid loss = {:0.4f}, f1-score = {:0.4f}, patience: {:2d}'.format(epoch+1, valid_loss, f1, patience))


        print("...Done!")
        return avg_epoch_loss
 

    def evaluate(self, valid_dataset, split="dev"):

        valid_loss = 0.0
        all_predictions = list()
        all_labels = list()
        labels_vocab_reverse = {v:k for (k,v) in self.labels_vocab.items()}
         
        self.model.eval()
    
        for words, labels, in tqdm(valid_dataset):
            batch_loss = 0.0
            self.optimizer.zero_grad()
            with torch.no_grad():
                batch_LL, predictions = self.model(words, labels)

            batch_NLL = - torch.sum(batch_LL)/8

            predictions = predictions.view(-1, predictions.shape[-1])
            labels = labels.view(-1) 

 
            for i in range(len(predictions)):
                if labels[i]!=0:
                    all_predictions.append(labels_vocab_reverse[int(torch.argmax(predictions[i]))])
                    all_labels.append(labels_vocab_reverse[int(labels[i])])
            
            if not math.isnan(batch_NLL.tolist()):
                valid_loss += batch_NLL.tolist()

        f1 = f1_score(all_labels, all_predictions, average= 'macro')
        print(classification_report(all_labels, all_predictions, digits=3))
        #print(f1)
        
        return valid_loss / len(valid_dataset), f1
    
    
    def evaluate_errorAnalysis(self, valid_dataset, split="dev"):
        flat_list = list()
        flat_list1 = list()
        
        
        valid_loss = 0.0
        all_predictions = list()
        all_labels = list()
        labels_vocab_reverse = {v:k for (k,v) in self.labels_vocab.items()}
         
        self.model.eval()
    
        for words, labels, in tqdm(valid_dataset):
#             print(labels)
            batch_loss = 0.0
            self.optimizer.zero_grad()
            flat_list.append([item for sublist in words for item in sublist])

            with torch.no_grad():
                batch_LL, predictions = self.model(words, labels)

            batch_NLL = - torch.sum(batch_LL)/8

            predictions = predictions.view(-1, predictions.shape[-1])
            labels = labels.view(-1) 

 
            for i in range(len(predictions)):
                if labels[i]!=0:
                    all_predictions.append(labels_vocab_reverse[int(torch.argmax(predictions[i]))])
                    all_labels.append(labels_vocab_reverse[int(labels[i])])
                    

            
            if not math.isnan(batch_NLL.tolist()):
                valid_loss += batch_NLL.tolist()
        flat_list1 = [item for sublist in flat_list for item in sublist]
        print(len(all_predictions))
        print(len(all_labels))
        print(len(flat_list1))
        
        for i in range(len(all_labels)):
            if all_predictions[i] != all_labels[i]:
                print(flat_list1[i], ", Actual : ", all_labels[i], ", Predicted: " , all_predictions[i])
        f1 = f1_score(all_labels, all_predictions, average= 'macro')
        print(classification_report(all_labels, all_predictions, digits=3))
        
        return valid_loss / len(valid_dataset), f1
class IdiomExtractor(nn.Module):
    def __init__(self,
                 bert_model,
                 bert_tokenizer,
                 bert_config,
                 hparams,
                 device):
        super(IdiomExtractor, self).__init__()
        pprint(hparams)
        pprint("Methods1")
        self.bert_model = bert_model
        self.bert_tokenizer = bert_tokenizer
        self.bert_config = bert_config
        self.hparams = hparams
        self.device = device
        self.dropout = nn.Dropout(hparams.dropout)
        self.lstm = nn.LSTM(self.bert_config.hidden_size,
                            self.hparams.hidden_dim, 
                            bidirectional=self.hparams.bidirectional, 
                            num_layers=self.hparams.num_layers,
                            dropout=self.hparams.dropout if self.hparams.num_layers>1 else 0,
                            batch_first=True)
        self.lstm2 = nn.LSTM(self.hparams.hidden_dim * 2,  # adjust input size to match output size of lstm1
                            self.hparams.hidden_dim, 
                            bidirectional=self.hparams.bidirectional, 
                            num_layers=self.hparams.num_layers,
                            dropout=self.hparams.dropout if self.hparams.num_layers>1 else 0,
                            batch_first=True)
        self.lstm_output_dim = self.hparams.hidden_dim * 1 if self.hparams.bidirectional is False else self.hparams.hidden_dim * 2
        self.classifier1 = nn.Linear(self.bert_config.hidden_size, self.bert_config.hidden_size)
        self.classifier2 = nn.Linear(self.bert_config.hidden_size, hparams.num_classes)
        self.CRF = CRF(hparams.num_classes).cuda()
    def forward(self, words, labels):
        input_ids, to_merge_wordpieces, attention_mask, token_type_ids = self._prepare_input(words)
        bert_output = self.bert_model.forward(input_ids=input_ids, 
                                            token_type_ids=token_type_ids,
                                            attention_mask=attention_mask)
        layers_to_sum = torch.stack([bert_output[-1][x] for x in [-1, -2, -3, -4]], axis=0)
        summed_layers = torch.sum(layers_to_sum, axis=0)
        merged_output = self._merge_embeddings(summed_layers, to_merge_wordpieces)
        embedding_bert = pad_sequence(merged_output, batch_first=True, padding_value=0)
        mask = self.padding_mask(labels)
        embedding_bert = self.dropout(embedding_bert)
        X, (h1, c1) = self.lstm(embedding_bert)
        X, (h2, c2) = self.lstm2(X)
        X = self.dropout(X)
        O = self.classifier1(embedding_bert)
        O = self.classifier2(O)
        if labels==None:
            log_likelihood = -100
        else:
            log_likelihood = self.CRF.forward(O, labels, mask)

        return log_likelihood, O

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

    
    def padding_mask(self, batch):
        padding = torch.ones_like(batch)
        padding[batch == 0] = 0
        padding = padding.type(torch.uint8)
        return padding
    
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
        # we pad sentences shorter than the max length of the batch
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
    
