from ratsnlp.nlpbook.classification import ClassificationTrainArguments



args = ClassificationTrainArguments (
    pretrained_model_name="beomi/kcbert-base",
    downstream_corpus_name="nsmc",
    downstream_corpus_root_dir="./data/Korpora",
    downstream_model_dir="./data",
    learning_rate=5e-5,
    batch_size=32,
)

