from Korpora import Korpora
from config import args

def fetchData():
    print(args.downstream_corpus_name)
    Korpora.fetch(
        corpus_name = args.downstream_corpus_name,
        root_dir    = args.downstream_corpus_root_dir,
        force_download=True
    )

