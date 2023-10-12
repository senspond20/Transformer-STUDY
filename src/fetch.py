from Korpora import Korpora
from config import args, pretrained_model_config
from transformers import BertForSequenceClassification

def fetchData():
    print(args.downstream_corpus_name)
    Korpora.fetch(
        corpus_name = args.downstream_corpus_name,
        root_dir    = args.downstream_corpus_root_dir,
        force_download=True
    )

def getModel():

    model = BertForSequenceClassification.from_pretrained(
        args.pretrained_model_name,
        config=pretrained_model_config,
    )

getModel();