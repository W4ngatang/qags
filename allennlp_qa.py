import json
import tqdm
import ipdb as pdb

from allennlp.predictors.predictor import Predictor
BIDAF_MDL = 'https://allennlp.s3.amazonaws.com/models/bidaf-model-2017.09.15-charpad.tar.gz'
NANET_MDL = 'https://allennlp.s3.amazonaws.com/models/naqanet-2019.04.29-fixed-weight-names.tar.gz'

def get_model(model_name, gpu_id):
    """ Load a pretrained model """
    if model_name == "bidaf":
        predictor = Predictor.from_path(BIDAF_MDL, 'machine-comprehension')
    elif model_name == "naqanet":
        predictor = Predictor.from_path(NANET_MDL, 'machine-comprehension')
    else:
        raise ValueError("Model not found!")

    if gpu_id >= 0:
        predictor._model = predictor._model.cuda(device=gpu_id)
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

def clean(sent):
    """ Any common preprocessing of QA inputs """

    sent = sent.replace("[CLS]", "")
    return sent.strip()

def load_data(gen_file, qst_file):

    def _flatten(ll):
        d = {}
        for l in ll:
            d.update(l)
        return d

    def _filter(d):
        return ("hypotheses" not in gen_d) or (len(gen_d["hypotheses"]) == 0)

    gen_data = _flatten([json.loads(d) for d in open(gen_file, encoding="utf-8").readlines()])
    qst_data = _flatten([json.loads(d) for d in open(qst_file, encoding="utf-8").readlines()])

    data = {}
    shared_ids = set(gen_data.keys()) & set(qst_data.keys())

    for key in shared_ids:
        gen_d = gen_data[key]
        qst_d = qst_data[key]
        if _filter(gen_d):
            continue

        data[key] = {"source": clean(qst_d["source"]),
                     "target": clean(qst_d["target"]),
                     "hypotheses": [clean(d[0]) for d in gen_d["hypotheses"]],
                     "questions": [clean(d[0]) for d in qst_d["hypotheses"]]
                    }

    print(f"Loaded {len(data)} examples")

    return data

def _predict(model, model_name, psg, qst):
    ans = model.predict(passage=psg, question=qst)
    if model_name == "bidaf":
        return ans["best_span_str"]
    elif model_name == "nanet":
        return ans["answer"]

def eval(model_name, data, gpu_id=0):
    """ """

    model = get_model(model_name, gpu_id)
    for datum_idx, datum in data.items():
        src = datum["source"]
        trg = datum["target"]
        gen = datum["hypotheses"][0]
        qst = datum["questions"][0]
        src_ans = _predict(model, model_name, src, qst)
        trg_ans = _predict(model, model_name, trg, qst)
        gen_ans = _predict(model, model_name, gen, qst)

        print(f"***** Question: {qst} *****")
        print(f"Src: {src}")
        print(f"ans: {src_ans}")
        print("")
        print(f"Trg: {trg}")
        print(f"Ans: {trg_ans}")
        print("")
        print(f"Gen: {gen}")
        print(f"Ans: {gen_ans}")
        print("\n")

        datum["src_ans"] = src_ans
        datum["trg_ans"] = trg_ans
        datum["gen_ans"] = gen_ans

    return data

def write_ans(data, out_file):
    """ Dump stuff """
    with open(out_file, "w", encoding="utf-8") as out_fh:
        for d in data.values():
            out_fh.write(f"{json.dumps(d)}\n")

gen_file = "data/summaries.jsonl"
qst_file = "data/questions.jsonl"
model_name = "bidaf"
out_file = f"data/{model_name}_outs.jsonl"
gpu_id = 0

data = load_data(gen_file, qst_file)
data_w_ans = eval(model_name, data, gpu_id)
write_ans(data_w_ans, out_file)
