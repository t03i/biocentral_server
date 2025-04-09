from pathlib import Path
import random
import h5py
import yaml
import torch
from biocentral_server.bayesian_optimization import gaussian_process_models as gp
from biotrainer.utilities import read_FASTA

def load_config_from_yaml(config_path: Path) -> dict:
    try:
        with open(config_path, "r") as config_file:
            config_dict = yaml.safe_load(config_file)
        if not isinstance(config_dict, dict):
            raise ValueError(f"YAML file {config_path} did not parse into a dictionary")
        return config_dict
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    except yaml.YAMlLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file {config_path}: {e}")


SUPPORTED_MODELS = ["gaussian_process"]


def mockoutput(config_path: str):
    print(f'config path: {config_path}')
    config_dict = load_config_from_yaml(config_path)
    results = []
    all_seqs = {seq.id: str(seq.seq) for seq in read_FASTA(config_dict["sequence_file"])}
    for id, seq in all_seqs.items():
        results.append({"id": id, "sequence": seq, "score": float(f"{random.random():.6f}")})
    results.sort(key=lambda x: x["score"], reverse = True)
    results_yaml = yaml.dump(results)
    out_file = Path(config_dict["output_dir"])/"out.yml"
    with out_file.open('w+') as config_file:
        config_file.write(results_yaml)

def merge_label_embeddings(embeddings_file: str, fasta_dict: dict) -> tuple[dict, bool]:
    '''
    keep sequences that appear both in embedding file and fasta dict
    return:
    dict[seq_id] = [seq, description, embedding]
    bool: seqs in fasta_dict is the same as embedding
    '''
    valid_seqs = {}
    seqs_equal = True # if seqs in fasta_dict is the same as embedding
    with h5py.File(embeddings_file, 'r') as hdf:
        for key, val in hdf.items():
            seqid = val.attrs["original_id"]
            fasta_seq_prop = fasta_dict.get(seqid)
            if not fasta_seq_prop:
                seqs_equal = False
                continue
            valid_seqs[seqid] = fasta_seq_prop
            valid_seqs[seqid].append(torch.tensor(val))
    # seqs_equal = True, then all ids in embedding appears in fasta, set(fasta) >= valid_seq
    return valid_seqs, seqs_equal and len(fasta_dict) == len(valid_seqs)


# how to get feature_name? add more task descriptions to config?
# feature type: 
def parse_description_to_label(description: str, isregression: bool, feature_name: str = 'TARGET'):
    '''
    parse feature from descriptions
    return value: none | [target_val, if this is training]
    Note
    - description is expected to be space separated list of strings
    - feature_name and value should not contain space and '='
    - first feature_name=feature_value will be considered
    - sequence with feature_name=Unknown (case insensitive unknown) 
    or feature name doesn't appear will be considered as inference sample 
    '''
    feature_value = None
    # if feature_name=XX appear in description, it will be training
    for kvstr in description.split(' '): # k=v
        kv = kvstr.split('=')
        if (len(kv) != 2):
            continue
        if kv[0].lower() == feature_name.lower():
            feature_value = kv[1] if kv[1].lower() != 'unknown' and len(kv[1]) != 0 else None
            if isregression and feature_value is not None:
                try:
                    feature_value = float(feature_value)
                except:
                    raise KeyError(f"not supported regression label: {feature_value}")
            break
    return feature_value

# why not do parsing in endpoint? --> endpoint: latency is the key!

def get_datasets(config_dict: dict, seqs: dict):
    """
    train_data = {"ids": [], "X": [], "y": []}
    inference_data = {"ids": [], "X": []}
    """
    if config_dict['discrete']:
        target_classes = {label: idx for idx, label in enumerate(config_dict['discrete_labels'])}

    train_data = {"ids": [], "X": [], "y": []}
    inference_data = {"ids": [], "X": []}
    def is_training(val_1) -> bool:
        if val_1 is None: 
            return False
        return True

    for id, val in seqs.items():
        if is_training(val[1]):
            train_data["ids"].append(id)
            train_data["X"].append(val[2])
            target = target_classes[val[1]] if config_dict['discrete'] else val[1]
            train_data["y"].append(target)
        else:
            inference_data["ids"].append(id)
            inference_data["X"].append(val[2])
    if train_data["X"]:
        train_data["X"] = torch.stack(train_data["X"]).float()
    if inference_data["X"]:
        inference_data["X"] = torch.stack(inference_data["X"]).float()
    if train_data['y']:
        train_data['y'] = torch.tensor(train_data['y'])
    if config_dict["discrete"] and train_data['y']:
        train_data["y"] = torch.nn.functional.one_hot(
            train_data['y'], len(config_dict["discrete_labels"])
        )
    return train_data, inference_data

def val_dataset_split(train_data: dict, val_rate: float = 0.2, random_seed: int = 42):
    '''
    Split training set into training and validation set
    - train_data = {"ids": list[str], "X": (n, dime), "y": (n) or (n, c)}
    - val_rate: float in (0, 1)
    - random_seed: int
    '''
    assert val_rate > 0 and val_rate < 1, "val_rate should in (0, 1)"
    torch.manual_seed(random_seed)
    num_samples = len(train_data["ids"])
    val_size = int(val_rate * num_samples)
    # Generate random indices for validation set
    indices = torch.randperm(num_samples)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]

    val_dataset = {
        "ids": [train_data["ids"][i] for i in val_indices],
        "X": train_data["X"][val_indices],
        "y": train_data["y"][val_indices],
    }

    train_dataset = {
        "ids": [train_data["ids"][i] for i in train_indices],
        "X": train_data["X"][train_indices],
        "y": train_data["y"][train_indices],
    }
    return train_dataset, val_dataset

def data_prep(config_dict: dict, allow_empty_inference: bool = False) -> tuple[dict[str, list], dict[str, list], dict]:
    """
    train_data = {"ids": [], "X": （n_sample, dim）, "y": (n_sample), (n_sample, n_class))}
    inference_data = {"ids": [], "X": []}
    dict[seq_id] = [seq, description, embedding]
    """
    # read labels and seqs
    fasta_seqs = {
        seq.id: [str(seq.seq), seq.description]
        for seq in read_FASTA(config_dict["sequence_file"])
    }
    # read embeddings
    seqs, ok = merge_label_embeddings(config_dict["embeddings_file"], fasta_seqs)
    if not ok:
        print("embedding_mismatch")
        pass  # TODO: add warning, embeddings_file doesn't correspond to sequence file 
    if len(seqs) == 0:
        raise ValueError("data_prep: no valid sequence left")
    # parse labels
    for key in seqs.keys():
        seqs[key][1] = parse_description_to_label(
            seqs[key][1], not config_dict["discrete"], config_dict.get('feature_name', 'TARGET')
        )
    # train & test set split
    train_data, inference_data = get_datasets(config_dict, seqs)
    if not allow_empty_inference and len(train_data['ids']) * len(inference_data) == 0:
        raise ValueError("data_prep: training set / inference set is empty")
    return train_data, inference_data, seqs

def data_trunc(train_data: dict, max_n: int):
    assert max_n > 0, "Data count must be positive"
    sm_dataset = {key: val[:max_n] for key, val in train_data.items()}
    return sm_dataset

def train_and_inference_regression(train_data, inference_data, config_dict):
    """
    Args: 
    - data: dict {'X': [], 'y': [], 'ids': []}
    - config_dict: e2e coefficient, lb, ub, strategy
    Return:
    - inference score, tensor of shape (n_inf_data)
    - score = e2e * scaled_probability 
        + (1-e2e) * mean * (1 if maximize else -1) 
    """
    model, likelihood = gp.trainGPRegModel(train_data)
    # Y = x^Tw + eps
    prediction_dist = likelihood(model(inference_data['X']))
    # TODO: consider uncertainty
    marginal_dist = torch.distributions.Normal(
        prediction_dist.mean, prediction_dist.covariance_matrix.diag().sqrt()
    )
    # p(lb < feat < ub)
    prob = marginal_dist.cdf(
        torch.Tensor([config_dict["target_interval_ub"]])
    ) - marginal_dist.cdf(torch.Tensor([config_dict["target_interval_lb"]]))
    # scale probability to mean
    score = prob * (prediction_dist.mean.mean())
    # acquisition: weighted average of prob and mean
    strategy_factor = {'maximize': 1, 'minimize': -1, 'neutral': 0}
    e2e = config_dict['coefficient']
    score = e2e * score + strategy_factor[config_dict['value_preference']] * (1-e2e) * prediction_dist.mean
    # uncertainty
    return score # (n_inference)

# {"error": f"Server error: task finished but result file {out_file} not found"}
def dump_results(target_path: Path, results):
    results_yaml = yaml.dump(results)
    with target_path.open('w+') as out_file:
        out_file.write(results_yaml)

def pipeline(config_path: str):
    config_dict = load_config_from_yaml(config_path)
    result_path = Path(config_dict["output_dir"])/"out.yml"
    # data preparation: we need training data (X, y), 
    # inference data (X), and description (ids and seqs)
    # TODO: now validation is hardcoded to be true
    # TODO: update python dependency
    validation = False
    try:
        train_data, inference_data, seqs = data_prep(config_dict, validation)
        print("data_prep_finished")
        if validation:
            train_data, inference_data = val_dataset_split(train_data)
            train_data = data_trunc(train_data, 6000) # my memory can't handle too much data
        # train model, inference and add with acquisition function score
        if config_dict['discrete']: # TODO: classification
            print("ERROR, discrete target not supported")
            dump_results(result_path, {"error": "discrete target not supported"})
            return 
            # model, likelihood = gp.trainGPClsModel(train_data)
        else:
            scores = train_and_inference_regression(train_data, inference_data, config_dict)
        # ranking
        results = []
        for idx in range(len(inference_data['ids'])):
            sid = inference_data['ids'][idx]
            score = scores[idx].item()
            seq = seqs[sid][0]
            results.append({"id": sid, "sequence": seq, "score": score})
        results.sort(key=lambda x: x["score"], reverse = True)
        # dump result
        dump_results(result_path, results)
    except Exception as e:
        err_out = {"error": str(e)}
        print(f"error: {str(e)}")
        dump_results(result_path, err_out)

# TODO: moving to GPU