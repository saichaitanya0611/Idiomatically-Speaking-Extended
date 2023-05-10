from __init__ import *
from bert_and_data_preprocessing import *
from hparams import HParams
from methods1 import *
import spacy
from spacy.cli.download import download as spacy_download   


def collate(elems: tuple) -> tuple:
    words, labels = list(zip(*elems))
    pad_labels = pad_sequence(labels, batch_first=True, padding_value=0)
 
    return list(words), pad_labels.cuda()

def initialize(language):
    if language == "English": 
        spacy_download("en_core_web_sm")
        spacy_tagger = spacy.load("en_core_web_sm", exclude=["ner", "parser"])

    elif language == "Italian": 
        spacy_download("it_core_news_sm")
        spacy_tagger = spacy.load("it_core_news_sm", exclude=["ner", "parser"])
        
    elif language == "Dutch": #(5)
        spacy_download("nl_core_news_sm")
        spacy_tagger = spacy.load("nl_core_news_sm", exclude=["ner", "parser"])

    elif language == "Spanish": 
        spacy_download("es_core_news_sm")
        spacy_tagger = spacy.load("es_core_news_sm", exclude=["ner", "parser"])

    elif language == "German": 
        spacy_download("de_core_news_sm")
        spacy_tagger = spacy.load("de_core_news_sm", exclude=["ner", "parser"])

    elif language == "Portuguese":
        spacy_download("pt_core_news_sm")
        spacy_tagger = spacy.load("pt_core_news_sm", exclude=["ner", "parser"])
        
    elif language == "French":
        spacy_download("fr_core_news_sm")
        spacy_tagger = spacy.load("fr_core_news_sm", exclude=["ner", "parser"])
    
    return spacy_tagger



if __name__=="__main__":
    #we set a seed for having replicability of results
    SEED = 4
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True


    #instantiate bert
    model_name = 'xlm-roberta-large'
    bert_config = XLMRobertaConfig.from_pretrained(model_name, output_hidden_states=True)
    bert_tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
    bert_model = XLMRobertaModel.from_pretrained(model_name, config=bert_config)

#     model_name = 'bert-base-multilingual-cased'
#     bert_config = BertConfig.from_pretrained(model_name, output_hidden_states=True)
#     bert_tokenizer = BertTokenizer.from_pretrained(model_name)
#     bert_model = BertModel.from_pretrained(model_name, config=bert_config)

#     model_name = 'distilbert-base-multilingual-cased'
#     bert_config = DistilBertConfig.from_pretrained(model_name, output_hidden_states=True)
#     bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
#     bert_model = BertModel.from_pretrained(model_name, config=bert_config)


    
    language = "English"
    spacy_tagger = initialize(language)
    
    train_file = f"data/{language.lower()}/train_{language.lower()}.tsv"
    dev_file = f"data/{language.lower()}/dev_{language.lower()}.tsv"

    labels_vocab = {"<pad>":0, "B-IDIOM":1, "I-IDIOM":2, "O":3}

    #index dataset
    train_dataset = IdiomDataset(train_file, bert_tokenizer, labels_vocab, spacy_tagger, "all")
    dev_dataset = IdiomDataset(dev_file, bert_tokenizer, labels_vocab, spacy_tagger, "all")

    #dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate)
    dev_dataloader = DataLoader(dev_dataset, batch_size=8, collate_fn=collate)

    #instantiate the hyperparameters
    params = HParams()

    #instantiate the model
    my_model = IdiomExtractor(bert_model,bert_tokenizer,bert_config,params,"cuda").cuda()


    #trainer
    trainer = Trainer(model = my_model,
                    loss_function = nn.CrossEntropyLoss(ignore_index=0),
                    optimizer = optim.Adam(bert_model.parameters(), lr=0.00001),
                    labels_vocab=labels_vocab,
                    gradient_accumulation_steps=4)

    trainer.train(train_dataloader, dev_dataloader, 1, patience=30, modelname = f"{language.lower()}")


    


