from __init__ import *
from hparams import HParams
from methods1 import *
from bert_and_data_preprocessing import *


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

    elif language == "Spanish": 
        spacy_download("es_core_news_sm")
        spacy_tagger = spacy.load("es_core_news_sm", exclude=["ner", "parser"])

    elif language == "German": 
        spacy_download("de_core_news_sm")
        spacy_tagger = spacy.load("de_core_news_sm", exclude=["ner", "parser"])
    elif language == "Dutch":
        spacy_download("nl_core_news_sm")
        spacy_tagger = spacy.load("nl_core_news_sm", exclude=["ner", "parser"])
    elif language == "Portuguese":
        spacy_download("pt_core_news_sm")
        spacy_tagger = spacy.load("pt_core_news_sm", exclude=["ner", "parser"])    
    elif language == "French":
        spacy_download("fr_core_news_sm")
        spacy_tagger = spacy.load("fr_core_news_sm", exclude=["ner", "parser"])    
    return spacy_tagger

    


def get_idioms(dataset, spacy_tagger):
    idioms = []

    for elem in dataset:
        idiom = ""
        for token, tag in zip(elem[0], elem[1]):
            if tag == 1: #B tag
                idiom += spacy_tagger(token)[0].lemma_
            elif tag == 2: #I tag
                idiom += spacy_tagger(token)[0].lemma_

        if idiom != "":
            idioms.append(idiom.strip())
    
    return idioms

if __name__=="__main__":
    #Set the random Seed here
    SEED = 2
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

#     Start Bert instance
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
    checkpoint = "english.pt"
    dataset_type = "all"
    spacy_tagger = initialize(language)
    

    test_file = f"data/{language.lower()}/test_{language.lower()}.tsv"
    train_file = f"data/{language.lower()}/train_{language.lower()}.tsv"
    labels_vocab = {"<pad>":0, "B-IDIOM":1, "I-IDIOM":2, "O":3}

    train_dataset = IdiomDataset(train_file, bert_tokenizer, labels_vocab, spacy_tagger, "all")
    test_dataset = IdiomDataset(test_file, bert_tokenizer, labels_vocab, spacy_tagger, "all")
    idioms_test = get_idioms(test_dataset, spacy_tagger)
    idioms_train = get_idioms(train_dataset, spacy_tagger)
    test_dataset = IdiomDataset(test_file, bert_tokenizer, labels_vocab, spacy_tagger, dataset_type, idioms_train, idioms_test)
    test_dataloader = DataLoader(test_dataset, batch_size=8, collate_fn=collate)


    #instantiate the model
    my_model = IdiomExtractor(bert_model, bert_tokenizer, bert_config,  HParams(), "cuda").cuda()
    my_model.load_state_dict(torch.load(f"{checkpoint}"))

    #trainer
    trainer = Trainer(model = my_model,
                    loss_function = nn.CrossEntropyLoss(ignore_index=0),
                    optimizer = optim.Adam(bert_model.parameters(), lr=0.0001),
                    labels_vocab=labels_vocab,
                    gradient_accumulation_steps=4)

    trainer.evaluate(test_dataloader, "test")

    


