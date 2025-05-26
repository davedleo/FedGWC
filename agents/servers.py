import torch 
import numpy as np 
import json
import os
from copy import deepcopy 
from collections import OrderedDict
from sklearn.cluster import SpectralClustering
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import davies_bouldin_score





def save_dict(d: dict, path: str): 
    with open(path, 'w') as f: 
        json.dump(d, f)

def load_dict(path: str): 
    with open(path, 'r') as f:
        d = json.load(f)
    return d




# SERVER
class ServerFL: 



    def __init__(
            
            # Clients
            self, model: torch.nn.Module, clients: dict,

            # Rounds 
            participation_rate: float = .1, num_local_iters: int = 10,
            aggregation: str = 'fedavg',

            # Server optimizer 
            server_lr: float = 0., server_momentum: float = 0.,

            # Device 
            seed: int = 0, device: str = "cuda:0"

    ):
        
        # Device 
        self._prng = np.random.default_rng(seed)
        self._device = device 

        # Clients 
        self._clients = clients
        self._clients_ids = list(self._clients.keys())

        # Training setting 
        self._num_local_iters = num_local_iters 
        self._num_clients_per_round = int(participation_rate) if participation_rate > 1 else round(len(clients) * participation_rate)
        self._current_updates = None

        # Model
        self._aggregation_type = aggregation
        if self._aggregation_type in ('fedavg', 'fairavg'):
            self._model = deepcopy(model)
            self._model_state_dict = deepcopy(self._model.state_dict())
        if self._aggregation_type == 'fedopt':
            self._model = deepcopy(model)
            self._model_state_dict = deepcopy(self._model.state_dict())
            self._server_optimizer = torch.optim.SGD(self._model.parameters(), lr = server_lr, momentum = server_momentum)



    def _select_clients(self): 
        return self._prng.choice(self._clients_ids, size = self._num_clients_per_round, replace = False)
    


    def _load_model_on_clients(self, clients_ids):
        for client_id in clients_ids: 
            self._clients[client_id].load_update(self._model)



    def _aggregation(self): 

        # Setup 
        clients_updates = self._current_updates['clients_updates']
        tot_num_samples = self._current_updates['tot_num_samples']

        # Select aggregation
        if self._aggregation_type == 'fedavg':

            # Setup 
            base = OrderedDict([(k, torch.zeros_like(p, dtype = float)) for k, p in self._model_state_dict.items()])

            # Iterate updates 
            for update in clients_updates.values():
                num_samples = update['num_samples']
                for k, p in update['model'].items():
                    base[k] += num_samples * p.float() / tot_num_samples

            # Update state_dict
            self._model_state_dict = base
            self._model.load_state_dict(self._model_state_dict, strict = False)

        elif self._aggregation_type == 'fairavg':

            # Setup 
            base = OrderedDict([(k, torch.zeros_like(p, dtype = float)) for k, p in self._model_state_dict.items()])
            w = 1. / len(clients_updates)

            # Iterate updates 
            for update in clients_updates.values():
                for k, p in update['model'].items():
                    base[k] += w * p.float().float() 

            # Update state_dict
            self._model_state_dict = base
            self._model.load_state_dict(self._model_state_dict, strict = False)

        elif self._aggregation_type == 'fedopt':

            # Iterate updates
            for k, p in self._model.named_parameters(): 
                p.grad = p.data.clone()
                for update in clients_updates.values(): 
                    p.grad -= update['num_samples'] * update['model'][k] / tot_num_samples

            # Perform optimization
            self._server_optimizer.step()
            
            # Load the new state_dict
            self._model_state_dict = deepcopy(self._model.state_dict())



    def _run_training_on_clients(self, clients_ids: list[str]): 

        # Setup 
        loss = 0. 
        tot_num_samples = 0
        clients_updates = dict()

        # Iterate clients 
        for client_id in clients_ids: 
            
            # Setup 
            client = self._clients[client_id]
            num_samples = client.get_num_samples()
            tot_num_samples += num_samples 

            # Training 
            client_loss = client.train(self._num_local_iters)
            loss += client_loss[-1].item() * num_samples
            clients_updates[client_id] = {'model': client.get_update(), 'num_samples': num_samples}

        # Load updates on server
        self._current_updates = {'clients_updates': clients_updates, 'tot_num_samples': tot_num_samples}

        return loss / tot_num_samples
    


    def train(self, num_rounds: int, test_evaluation_step: int = 0): 

        # Setup 
        # - Verbosity 
        print("TRAINING LOOP: starting...")
        # - Variables 
        test_evaluation = test_evaluation_step > 0
        results = {
            "train_track": {"loss": []},
            "test_track": {"loss": [], "accuracy": [], "balanced_accuracy": []},
            "test_evaluation_step": test_evaluation_step
        }

        # Training 
        for r in range(1, num_rounds + 1): 

            # Round setup
            # - Verbosity
            print('------------------------------')
            
            # - Setup 
            clients_ids = self._select_clients()
            self._load_model_on_clients(clients_ids)

            # Local training 
            loss = self._run_training_on_clients(clients_ids)
            self._aggregation()
            
            # Update round variables 
            results["train_track"]["loss"].append(loss)
            self._current_updates = None

            # Verbosity
            print(f'- round {r}')
            print(f"--- train_loss = {round(loss, 4)}")

            # Test evaluation 
            if test_evaluation and r % test_evaluation_step == 0:

                # Evaluation
                test_results = self.test()["global"]
                test_loss = test_results["loss"]
                test_accuracy = test_results["accuracy"]
                test_balanced_accuracy = test_results["balanced_accuracy"]
                results["test_track"]["loss"].append(test_loss)
                results["test_track"]["accuracy"].append(test_accuracy)
                results["test_track"]["balanced_accuracy"].append(test_balanced_accuracy)

                # Clean buffer
                del test_results

                print(f"--- test_loss = {round(test_loss, 4)}; test_accuracy = {round(test_accuracy, 4)}; test_balanced_accuracy = {round(test_balanced_accuracy, 4)}")

            # Verbosity
            print('------------------------------')

        # Verbosity
        print("TRAINING LOOP: finished.")

        return results 



    def test(self): 

        # Setup 
        loss, accuracy, balanced_accuracy, tot_num_samples = 0., 0., 0., 0
        self._load_model_on_clients(self._clients_ids)

        # Results setup 
        results = {
            "local": dict(),
            "global": dict()
        }

        # Iterate clients_ids 
        for client_id, client in self._clients.items(): 
            
            # Averaging setup 
            num_samples = client.get_num_samples(train = False)
            tot_num_samples += num_samples

            # Client evaluation
            client_results = client.test()

            # Update results 
            results["local"][client_id] = client_results 
            loss += client_results["loss"] * num_samples 
            accuracy += client_results["accuracy"] * num_samples 
            balanced_accuracy += client_results["balanced_accuracy"] * num_samples

        # Update results 
        results["global"]["loss"] = loss / tot_num_samples 
        results["global"]["accuracy"] = accuracy / tot_num_samples
        results["global"]["balanced_accuracy"] = balanced_accuracy / tot_num_samples

        return results





# SERVER FEDGWC  
class ServerFedGWC: 



    def __init__(
            
            # Clients
            self, model: torch.nn.Module, clients: dict,

            # Rounds 
            participation_rate: float = .1, num_local_iters: int = 10,
            aggregation: str = 'fedavg',

            # Server Optimizer 
            server_lr: float = 0., server_momentum: float = 0.,

            # FedGW 
            gamma: float = 4., eps: float = 1e-5, 
            explore: bool = False, loss_sampling_size: int = 100,

            # Device 
            seed: int = 0, device: str = "cuda:0"

    ):
        
        # Device 
        self._prng = np.random.default_rng(seed)
        self._seed = seed
        self._device = device 

        # Global attributes 
        # - Clients
        self._clients = clients
        self._clients_ids = list(self._clients.keys())
        self._num_local_iters = num_local_iters 
        self._num_clients_per_round = participation_rate if participation_rate >= 1 else round(len(clients) * participation_rate)
        self._current_updates = None
        # - Clustering
        self._gamma = gamma
        self._eps = eps 
        self._explore = explore 
        self._loss_sampling_size = loss_sampling_size

        # Clusters 
        self._clusters = ['0']
        self._clusters_to_clusterize = ['0']
        self._clusters_clients_ids = {'0': np.array(self._clients_ids)}
        self._clusters_clients_ids_map = {'0': {client_id: i for i, client_id in enumerate(self._clients_ids)}}
        self._clusters_clients_idxs_map = {'0': {i: client_id for client_id, i in self._clusters_clients_ids_map['0'].items()}}
        self._clusters_num_clients = {'0': len(self._clients_ids)}
        self._clusters_W = {'0': np.ones((len(clients), len(clients))) / (len(clients) ** 2)}
        self._mse = {'0': 1.}
        self._patience = {'0': {'current': 0, 'target': round(len(clients) / self._num_clients_per_round)}} 

        # Server optimizer
        self._aggregation_type = aggregation
        if self._aggregation_type in ('fedavg', 'fairavg'):
            self._clusters_models = {'0': deepcopy(model)}
            self._clusters_state_dicts = {'0': deepcopy(model.state_dict())}
        elif self._aggregation_type == 'fedopt':
            self._clusters_models = {'0': deepcopy(model)}
            self._clusters_state_dicts = {'0': deepcopy(self._clusters_models['0'].state_dict())}
            self._clusters_optimizers = {
                '0': torch.optim.SGD(
                    self._clusters_models['0'].parameters(),
                    lr = server_lr,
                    momentum = server_momentum
                )
            }

        # Exploration setup 
        if self._explore: 
            self._sampling_counters = {'0': np.zeros(len(self._clients))}



    def _select_clients(self, cluster_id: str): 

        # Setup 
        sampling_size = min(self._num_clients_per_round, self._clusters_num_clients[cluster_id])
        
        # Selection 
        if self._explore: 
            deltas = np.exp(self._sampling_counters[cluster_id].min() - self._sampling_counters[cluster_id])
            return self._prng.choice(self._clusters_clients_ids[cluster_id], size = sampling_size, replace = False, p = deltas / deltas.sum()) 
        else:
            return self._prng.choice(self._clusters_clients_ids[cluster_id], size = sampling_size, replace = False)



    def _load_model_on_clients(self, cluster_id: str, clients_ids: list[str]): 
        for client_id in clients_ids: 
            self._clients[client_id].load_update(self._clusters_models[cluster_id])



    def _get_gaussian_rewards(self): 
        
        # 1. Get stats
        # - Setup 
        losses = OrderedDict()
        losses_obs = torch.zeros((self._num_clients_per_round, self._num_local_iters), requires_grad = False)
        mu = torch.zeros(self._num_local_iters, requires_grad = False)
        sigma2 = torch.zeros(self._num_local_iters, requires_grad = False)
        # - Current updates
        tot_num_samples = self._current_updates['tot_num_samples']
        clients_updates = self._current_updates['clients_updates']

        # - Stats
        # -- Mean 
        for i, (client_id, update) in enumerate(clients_updates.items()): 
            losses[client_id] = update['loss']
            losses_obs[i, :] = update['loss']
            mu += update['loss'] * update['num_samples'] / tot_num_samples 
        # -- Standard Deviation 
        for client_id, loss, in losses.items(): 
            sigma2 += clients_updates[client_id]['num_samples'] * (loss - mu).pow(2) / tot_num_samples

        # -- Loss sampling 
        num_losses_to_sample = self._loss_sampling_size - self._num_clients_per_round
        if num_losses_to_sample > 0: 
            losses_sampling = torch.normal(0., 1., size = (self._loss_sampling_size - self._num_clients_per_round, self._num_local_iters))
            losses_sampling = losses_sampling * torch.sqrt(sigma2) + mu 
            losses_sampling = torch.cat((losses_sampling, losses_obs), dim = 0)
            mu = losses_sampling.mean(0)
            sigma2 = losses_sampling.var(0)
        
        # Correct variance 
        sigma2 += 1e-16

        # 2. Compute rewards 
        # - Setup
        rewards = OrderedDict()
        rewards_sum = 0. 

        # - Computation 
        for client_id, loss in losses.items(): 
            rewards[client_id] = torch.exp(- .5 * (loss - mu).pow(2) / sigma2).mean().item()
            rewards_sum += rewards[client_id]

        return rewards, rewards_sum



    def _update_W(self, cluster_id: str): 

        # Setup
        clients_ids_map = self._clusters_clients_ids_map[cluster_id]
        R, R_sum = self._get_gaussian_rewards()
        alpha_new = self._num_clients_per_round / self._clusters_W[cluster_id].shape[0]
        alpha_old = 1. - alpha_new
        sse = 0.

        # Updates 
        for client_id_i, r in R.items(): 

            # Find index and compute new update
            row = clients_ids_map[client_id_i]
            w_new = alpha_new * r / R_sum

            # Move along columns and update W
            for client_id_j in R.keys(): 

                # Find index
                col = clients_ids_map[client_id_j]

                # Update W 
                w_old = self._clusters_W[cluster_id][row, col]
                w_update = alpha_old * w_old + w_new 
                sse += ((w_update - w_old) ** 2)
                self._clusters_W[cluster_id][row, col] = w_update + 0.


        # Update mse
        mse = sse / (self._num_clients_per_round ** 2)
        self._mse[cluster_id] = alpha_old * self._mse[cluster_id] + alpha_new * mse

        # Update patience
        if self._mse[cluster_id] < self._eps: self._patience[cluster_id]['current'] += 1
        else: self._patience[cluster_id]['current'] = 0

        # Verbosity
        print(f'--- cluster {cluster_id} avg. mse = {round(self._mse[cluster_id], 8)} | patience = {self._patience[cluster_id]}')



    def _compute_P(self, cluster_id: str): 

        # Setup 
        W = self._clusters_W[cluster_id].T.copy()
        W = (W - W.min()) / max((W.max() - W.min()), 1e-8)
        P = np.zeros_like(W)
        K = P.shape[0]

        # P computation 
        for i in range(K - 1): 
            for j in range(i + 1, K): 

                # Get arrays 
                x, y = W[i], W[j]
                x = np.concatenate((x[:i], x[i + 1: j], x[j + 1:]))
                y = np.concatenate((y[:i], y[i + 1: j], y[j + 1:]))

                # Update P 
                p = (x - y) ** 2
                p = np.exp(- self._gamma * p.sum())
                P[i, j], P[j, i] = p, p 

        return P



    def _find_best_clusters(self, cluster_id: str): 

        # Setup 
        n_clusters_best, davies_best, y_best = 1, 1., None 
        max_n_clusters = int(self._clusters_W[cluster_id].shape[0] / self._num_clients_per_round) + 1
        max_n_clusters = min(max_n_clusters, 5)
        W = self._compute_P(cluster_id)

        D = 1. / np.sqrt(W.sum(1) + 1e-16)
        D = np.diag(D)
        L = 1. - D @ W @ D

        # Grid search 
        for n_clusters in range(2, max_n_clusters): 

            # Embedding and Clustering
            W_emb = SpectralEmbedding(
                n_components = n_clusters,
                affinity = 'precomputed',
                random_state = self._seed,
                n_jobs = -1
            ).fit_transform(W)
            y = SpectralClustering(
                n_clusters = n_clusters, 
                n_components = n_clusters,
                affinity = 'precomputed',
                assign_labels = 'cluster_qr',
                random_state = self._seed,
                n_jobs = -1
            ).fit_predict(W)

            # Evaluation 
            davies = davies_bouldin_score(W_emb, y)
            K_min = np.unique(y, return_counts = True)[1].min()
            print(f'------ | n_clusters = {n_clusters}: davies = {round(davies, 4)}, K_min = {K_min} |')

            # Update best solution 
            if davies < davies_best: 
                n_clusters_best = n_clusters 
                davies_best = davies 
                y_best = y

        return n_clusters_best, y_best



    def _split_cluster(self, cluster_id: str, n_clusters: int, y: np.ndarray): 

        # Iterate labels 
        for label in range(n_clusters): 

            # - Setup 
            new_cluster_id = cluster_id + f"-{label}"
            self._clusters.append(new_cluster_id)

            # - Map and indexing 
            mask = y == label 
            K = mask.sum()
            cluster_indexes = np.arange(y.size)[mask]
            self._clusters_clients_ids[new_cluster_id] = np.array([self._clusters_clients_ids[cluster_id][i] for i in cluster_indexes])
            self._clusters_clients_ids_map[new_cluster_id] = {client_id: i for i, client_id in enumerate(self._clusters_clients_ids[new_cluster_id])}
            self._clusters_clients_idxs_map[new_cluster_id] = {i: client_id for client_id, i in self._clusters_clients_ids_map[new_cluster_id].items()}
            self._clusters_num_clients[new_cluster_id] = int(K)

            # - Model 
            if self._aggregation_type == 'fedopt': 
                self._clusters_models[new_cluster_id] = deepcopy(self._clusters_models[cluster_id])
                self._clusters_state_dicts[new_cluster_id] = deepcopy(self._clusters_models[new_cluster_id].state_dict())
                self._clusters_optimizers[new_cluster_id] = torch.optim.SGD(self._clusters_models[new_cluster_id].parameters(), lr = 1.)
                self._clusters_optimizers[new_cluster_id].load_state_dict(deepcopy(self._clusters_optimizers[cluster_id].state_dict()))
            else:
                self._clusters_models[new_cluster_id] = deepcopy(self._clusters_models[cluster_id])
                self._clusters_state_dicts[new_cluster_id] = deepcopy(self._clusters_state_dicts[cluster_id])

            # - Matrices
            self._clusters_W[new_cluster_id] = self._clusters_W[cluster_id][mask][:, mask].copy().reshape(K, K)
            self._patience[new_cluster_id] = {'current': 0, 'target': round(K / self._num_clients_per_round)}
            self._mse[new_cluster_id] = 1.

            # Add-to-clusters_to_clusterize check 
            if K > self._num_clients_per_round: self._clusters_to_clusterize.append(new_cluster_id)

            # Explore 
            if self._explore: self._sampling_counters[new_cluster_id] = np.zeros(K)

        # Remove unnecessary data
        self._clusters.remove(cluster_id)
        self._clusters_to_clusterize.remove(cluster_id)
        self._clusters_num_clients.pop(cluster_id)
        self._patience.pop(cluster_id)
        self._mse.pop(cluster_id) 
        self._clusters_models.pop(cluster_id)
        self._clusters_state_dicts.pop(cluster_id)
        if self._aggregation_type == 'fedopt': 
            self._clusters_optimizers.pop(cluster_id)
        if self._explore: 
            self._sampling_counters.pop(cluster_id)
    
    
    
    def _run_clustering(self, cluster_id: str): 

        # Find the best number of clusters 
        n_clusters, y = self._find_best_clusters(cluster_id)

        # Split check 
        if n_clusters == 1: self._mse[cluster_id] = 1. 
        else: self._split_cluster(cluster_id, n_clusters, y)



    def _aggregation(self, cluster_id: str): 

        # Setup 
        clients_updates = self._current_updates['clients_updates']
        tot_num_samples = self._current_updates['tot_num_samples']

        # Select aggregation
        if self._aggregation_type == 'fedavg':

            # Setup 
            base = OrderedDict([(k, torch.zeros_like(p, dtype = float)) for k, p in self._clusters_state_dicts[cluster_id].items()])

            # Iterate updates 
            for update in clients_updates.values():
                num_samples = update['num_samples']
                for k, p in update['model'].items():
                    base[k] += num_samples * p / tot_num_samples

            # Update state_dict
            self._clusters_state_dicts[cluster_id] = base 
            self._clusters_models[cluster_id].load_state_dict(base, strict = False)

        # Select aggregation
        elif self._aggregation_type == 'fairavg':

            # Setup 
            base = OrderedDict([(k, torch.zeros_like(p, dtype = float)) for k, p in self._clusters_state_dicts[cluster_id].items()])
            w = 1. / len(clients_updates)

            # Iterate updates 
            for update in clients_updates.values():
                num_samples = update['num_samples']
                for k, p in update['model'].items():
                    base[k] += w * p.float() 

            # Update state_dict
            self._clusters_state_dicts[cluster_id] = base 
            self._clusters_models[cluster_id].load_state_dict(base, strict = False)

        elif self._aggregation_type == 'fedopt': 

            # Iterate updates
            for k, p in self._clusters_models[cluster_id].named_parameters(): 
                p.grad = p.data.clone()
                for update in clients_updates.values(): 
                    p.grad -= update['num_samples'] * update['model'][k] / tot_num_samples

            # Perform optimization
            self._clusters_optimizers[cluster_id].step()

            # Load the new state_dict
            self._clusters_state_dicts[cluster_id] = deepcopy(self._clusters_models[cluster_id].state_dict())



    def _run_training_on_clients(self, clients_ids: list[str]): 

        # Setup 
        loss = 0. 
        tot_num_samples = 0
        clients_updates = OrderedDict()

        # Iterate clients 
        for client_id in clients_ids: 
            
            # Setup 
            client = self._clients[client_id]
            num_samples = client.get_num_samples()
            tot_num_samples += num_samples 

            # Training 
            client_loss = client.train(self._num_local_iters)
            loss += client_loss[-1].item() * num_samples
            clients_updates[client_id] = {'model': client.get_update(), 'loss': client_loss, 'num_samples': num_samples}

        # Load updates on server
        self._current_updates = {'clients_updates': clients_updates, 'tot_num_samples': tot_num_samples}

        return loss / tot_num_samples



    def train(self, num_rounds: int, test_evaluation_step: int = 0): 

        # Setup 
        # - Verbosity 
        print("TRAINING LOOP: starting...")
        # - Variables 
        test_evaluation = test_evaluation_step > 0
        results = {
            "train_track": {"loss": []},
            "test_track": {"loss": [], "accuracy": [], "balanced_accuracy": []},
            "test_evaluation_step": test_evaluation_step
        }

        # Training 
        for r in range(1, num_rounds + 1): 

            # Round setup
            # - Verbosity
            print('------------------------------')
            print(f'- round {r}')

            # - Setup
            clusters = self._clusters.copy() 
            tot_num_clients = 0
            train_loss = 0.

            # - Clusters iterations 
            for cluster_id in clusters:
                
                # -- Setup 
                clients_ids = self._select_clients(cluster_id)
                self._load_model_on_clients(cluster_id, clients_ids)

                # -- Local training 
                loss = self._run_training_on_clients(clients_ids)
                self._aggregation(cluster_id)
                num_clients = self._clusters_num_clients[cluster_id]
                tot_num_clients += num_clients
                train_loss += loss * num_clients

                # -- Cluster management
                if cluster_id in self._clusters_to_clusterize:

                    # --- Update W
                    self._update_W(cluster_id)

                    # --- Update clusters if needed
                    if self._patience[cluster_id]['current'] == self._patience[cluster_id]['target']:

                        # ---- perform clustering
                        print(f'--- cluster {cluster_id}: running clustering...')
                        self._run_clustering(cluster_id)

                        # ---- print current clusters 
                        print('--- CURRENT CLUSTERS:')
                        for cluster_id in self._clusters: 
                            print(f'-- cluster {cluster_id}, K = {self._clusters_W[cluster_id].shape[0]}, to_clusterize = {cluster_id in self._clusters_to_clusterize}')

                # -- Update round variables and local verbosity
                self._current_updates = None
            
            # - Update round variables and global verbosity
            train_loss /= tot_num_clients
            results["train_track"]["loss"].append(train_loss)
            print(f'--- train_loss = {round(train_loss, 4)}')

            # Test evaluation 
            if test_evaluation and r % test_evaluation_step == 0:
                test_results = self.test()["global"]
                test_loss = test_results["loss"]
                test_accuracy = test_results["accuracy"]
                test_balanced_accuracy = test_results["balanced_accuracy"]
                results["test_track"]["loss"].append(test_loss)
                results["test_track"]["accuracy"].append(test_accuracy)
                results["test_track"]["balanced_accuracy"].append(test_balanced_accuracy)

                # Clear buffer
                del test_results

                print(f"--- test_loss = {round(test_loss, 4)}; test_accuracy = {round(test_accuracy, 4)}; test_balanced_accuracy = {round(test_balanced_accuracy, 4)}")

            # Verbosity
            print('------------------------------')

        # Verbosity
        print("TRAINING LOOP: finished.")

        return results 



    def test(self): 

        # Setup 
        loss, accuracy, balanced_accuracy, tot_num_clients = 0., 0., 0., len(self._clients_ids)
        results = {"local": dict(), "clusters": dict(), "global": dict()}

        # Iterate clusters 
        for cluster_id in self._clusters: 

            # Setup 
            cluster_loss, cluster_accuracy, cluster_balanced_accuracy, tot_num_samples = 0., 0., 0., 0
            self._load_model_on_clients(cluster_id, self._clusters_clients_ids[cluster_id])

            # Iterate clients_ids 
            for client_id in self._clusters_clients_ids[cluster_id]: 

                # Averaging setup 
                num_samples = self._clients[client_id].get_num_samples(train = False)
                tot_num_samples += num_samples 

                # Client evaluation 
                client_results = self._clients[client_id].test()
                results["local"][client_id] = client_results 
                cluster_loss += client_results["loss"] * num_samples 
                cluster_accuracy += client_results["accuracy"] * num_samples 
                cluster_balanced_accuracy += client_results["balanced_accuracy"] * num_samples 

            # Update cluster results 
            cluster_loss /= tot_num_samples 
            cluster_accuracy /= tot_num_samples 
            cluster_balanced_accuracy /= tot_num_samples 
            results["clusters"][cluster_id] = {"loss": cluster_loss, "accuracy": cluster_accuracy, "balanced_accuracy": cluster_balanced_accuracy}

            # Update global results 
            num_clients = self._clusters_num_clients[cluster_id]
            loss += cluster_loss * num_clients / tot_num_clients
            accuracy += cluster_accuracy * num_clients / tot_num_clients 
            balanced_accuracy += cluster_balanced_accuracy * num_clients / tot_num_clients

        # Update global results 
        results["global"] = {"loss": loss, "accuracy": accuracy, "balanced_accuracy": balanced_accuracy}
        return results




