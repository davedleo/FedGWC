import os 
import json
import random 
import torch 
import numpy as np 
from agents import clients as fl_clients 
from agents import servers as fl_servers
from datasets.CIFAR100 import utils as cifar100_utils 





DIR_PATH = "./reports/runs/"





def set_seed(seed: int = 0): 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)





# DUMP JSON
def dump_dict(path: str, d: dict):
    with open(path, "w") as f: 
        json.dump(d, f) 





def agents_initialization(args):

    # Dataset
    # - Utils
    if args.dataset == "cifar100":
        utils = cifar100_utils 
    # - Load dataset
    print("DATASET: Loading...")
    federated_dataset = utils.load_federated_dataset(
        alpha_dirichlet = args.alpha,
        num_noise_clients = args.num_noise_clients,
        num_blur_clients = args.num_blur_clients,
        noise_sigma = args.noise_sigma,
        blur_radius = args.blur_radius
    )

    # Clients 
    # - Setup
    print("DATASET: Clients initialization...")
    clients = dict()
    # - Assign shard
    for dataset_id, dataset in federated_dataset['clients'].items():
        # -- Assign id
        client_id = dataset_id + '.' + dataset['augmentation']
        # -- Select client type
        if args.client == 'base':
            clients[client_id] = fl_clients.Client(
                client_id = client_id,
                model = utils.load_model(args.device),
                train_dataset = dataset['train'],
                test_dataset = dataset['test'],
                batch_size = args.batch_size,
                lr = args.lr,
                momentum = args.momentum,
                weight_decay = args.weight_decay,
                device = args.device
            )

    print("DATASET: Server initialization...")
    if args.server == 'baseline':
        return fl_servers.ServerFL(
            model = utils.load_model(args.device),
            clients = clients,
            participation_rate = args.participation_rate,
            num_local_iters = args.num_local_iters, 
            aggregation = args.aggregation,
            server_lr = args.server_lr,
            server_momentum = args.server_momentum,
            seed = args.seed, 
            device = args.device,
        )    
    elif args.server == 'fedgwc':
        return fl_servers.ServerFedGWC(
            model = utils.load_model(args.device),
            clients = clients,
            participation_rate = args.participation_rate,
            num_local_iters = args.num_local_iters, 
            aggregation = args.aggregation,
            server_lr = args.server_lr,
            server_momentum = args.server_momentum,
            gamma = args.gamma,
            eps = args.eps,
            explore = args.explore,
            loss_sampling_size = args.loss_sampling_size,
            seed = args.seed, 
            device = args.device,
        )





def run_name_from_args(config):

    # Assign run_name 
    run_name = f'{config.server}-{config.client}_{config.dataset}-{config.alpha}_'
    run_name += f'nNoise{config.num_noise_clients}-sigma{config.noise_sigma}_'
    run_name += f'nBlur{config.num_blur_clients}-radius{config.blur_radius}_'
    run_name += f'nIters{config.num_local_iters}-bs{config.batch_size}-lr{config.lr}-mu{config.momentum}-wd{config.weight_decay}'

    # Server types 
    if config.server != 'local':
        run_name += f'_agg{config.aggregation}-nRounds{config.num_rounds}-'
        run_name += f'sLr{config.server_lr}-sMu{config.server_momentum}'

    # FedGW 
    if config.server == 'fedgwc':
        run_name += f'-gamma{config.gamma}-eps{config.eps}-explore{config.explore}-lSize{config.loss_sampling_size}_'

    # Device 
    run_name += f'{config.device}_seed{config.seed}'

    # Directory creation
    # - Count the number of runs 
    path = DIR_PATH + run_name
    os.makedirs(path, exist_ok = True)
    num_runs = 0 
    for filename in os.listdir(DIR_PATH):
        if run_name in filename: num_runs += 1
    # - Get directory name
    path += f"_v{num_runs}"
    run_name += f'_v{num_runs}'
    # - Directory creation
    os.makedirs(name = path)
    os.makedirs(name = path + "/results")

    return run_name, path





def save_results(args, path: str, results: dict): 

    # Server type
    if args.server == "fedgwc":

        # Save global results 
        # - Last evaluation 
        dump_dict(path + "/results/evaluation.json", results["test"])
        # - Trackers
        dump_dict(path + "/results/tracking.json", results["train"])

        # Save clusters results 
        # - Active clusters_ids
        os.makedirs(path + '/clusters')
        server = results["server"]
        dump_dict(path + "/clusters/clusters_ids.json", server._clusters)
        # - For all the clusters in the history we save data 
        for cluster_id, clients_ids in server._clusters_clients_ids.items():
            # -- Setup
            cluster_path = path + f"/clusters/{cluster_id}"
            os.makedirs(cluster_path)
            # -- Indexes and maps
            dump_dict(cluster_path + "/clients_ids.json", clients_ids.tolist())
            dump_dict(cluster_path + "/clients_ids_map.json", server._clusters_clients_ids_map[cluster_id])
            dump_dict(cluster_path + "/clients_idxs_map.json", server._clusters_clients_idxs_map[cluster_id])
            # -- Matrix
            dump_dict(cluster_path + "/W.json", server._clusters_W[cluster_id].tolist())

    elif args.server == 'baseline':
        
        # Save results
        # - Last evaluation 
        dump_dict(path + "/results/evaluation.json", results["test"])
        # - Trackers
        dump_dict(path + "/results/tracking.json", results["train"])




def evaluate_models_on_domains(args, server): 

    # Extract clients for domains
    clients_per_domain = dict()
    for client_id, client in server._clients.items(): 
        domain = client_id.split('.')[1]
        if domain not in clients_per_domain:
            clients_per_domain[domain] = []
        clients_per_domain[domain].append(client)

    # Evaluate models on domains
    N = 0
    results = {'global': {'loss': 0, 'accuracy': 0, 'balanced_accuracy': 0}, 
               'domains': dict()}
    for cluster_id in server._clusters: 
        dlosstot, dacctot, dbacctot, dNtot = 0, 0, 0, 0
        results['domains'][cluster_id] = dict()
        for domain, clients_list in clients_per_domain.items():
            dloss, dacc, dbacc, dN = 0, 0, 0, 0
            for client in clients_list:
                client.load_update(server._clusters_state_dicts[cluster_id])
                n = client.get_num_samples()
                r = client.test()
                dN += n 
                dloss += r['loss'] * n 
                dacc += r['accuracy'] * n 
                dbacc += r['balanced_accuracy'] * n 
            results['domains'][cluster_id][domain] = {'loss': dloss / dN, 'accuracy': dacc / dN, 'balanced_accuracy': dbacc / dN}
            dlosstot += dloss
            dacctot += dacc 
            dbacctot += dbacc 
            dNtot += dN
        results['domains'][cluster_id]['global'] = {'loss': dlosstot / dNtot, 'accuracy': dacctot / dNtot, 'balanced_accuracy': dbacctot / dNtot}
        N += dNtot 
        results['global']['loss'] += dlosstot
        results['global']['accuracy'] += dacctot 
        results['global']['balanced_accuracy'] += dbacctot 
    results['global']['loss'] /= N
    results['global']['accuracy'] /= N
    results['global']['balanced_accuracy'] /= N
    print(results)
    return results

