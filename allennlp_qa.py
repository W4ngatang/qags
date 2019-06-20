import ipdb as pdb

from allennlp.predictors.predictor import Predictor
BIDAF_MDL = 'https://allennlp.s3.amazonaws.com/models/bidaf-model-2017.09.15-charpad.tar.gz'
NANET_MDL = 'https://allennlp.s3.amazonaws.com/models/naqanet-2019.04.29-fixed-weight-names.tar.gz'

def get_model(model_name):
    """ Load a pretrained model """
    if model == "bidaf":
        predictor = Predictor.from_path(bidaf_mdl, 'machine-comprehension')
    elif model == "naqanet":
        predictor = Predictor.from_path(nanet_mdl, 'machine-comprehension')
    else:
        raise ValueError("Model not found!")
    return predictor

def demo(model_name):
    """ Run a demo """

    passage = "The Matrix is a 1999 science fiction action film written and directed by The Wachowskis, starring Keanu Reeves, Laurence Fishburne, Carrie-Anne Moss, Hugo Weaving, and Joe Pantoliano."
    question = "Who stars in The Matrix?"

    predictor = get_model(model_name)

    ans = predictor.predict(passage=passage, question=question)

    if model == "bidaf":
        print(ans["best_span_str"])
    elif model == "naqanet":
        print(ans["answer"])
    else:
        raise ValueError("Model not found!")

def eval(model_name):

    model = get_model(model_name)

    for src, gen, qst in zip(srcs, gens, qsts):
        src_ans = predictor.predict(passage=src, question=qst)
        gen_ans = predictor.predict(passage=gen, question=qst)

        # TODO(Alex): compare the answers
        print_ans(src_ans)
        print_ans(gen_ans)
