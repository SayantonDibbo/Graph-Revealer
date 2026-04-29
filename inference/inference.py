
from tqdm import tqdm


def generate_graphs(gen_model, labels, arch, device):
    graphs = []
    for label in tqdm(labels, desc="Generating"):
        if arch == "proposed":
            adj, feat = gen_model.sample(label)
        elif arch == "random":
            adj, feat = gen_model.sample(label)
        else:   # base / CVAE
            adj, feat = gen_model.sample(label)
        graphs.append({"adj": adj, "features": feat, "label": label})
    return graphs