import torch 
import numpy as np 
from copy import deepcopy





# BASE CLIENT
class Client:



    def __init__(
            
            # Setup
            self, client_id: str, model,
            train_dataset: torch.utils.data.Dataset, test_dataset: torch.utils.data.Dataset,

            # Optimization
            batch_size: int = 64, lr: float = .01, momentum: float = 0., weight_decay: float = 0.,
            
            # Device
            device: str = "cuda:0"

    ):
        
        # Initialization
        self._id = client_id 
        self._device = device 
        self._sgd_params = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "dampening": 0.,
            "nesterov": False
        }

        self._model = None

        self._train_dataset = train_dataset
        self._test_dataset = test_dataset
        self._train_dataloader = torch.utils.data.DataLoader(
            self._train_dataset,
            batch_size = min(batch_size, len(self._train_dataset)),
            shuffle = True, 
            drop_last = True
        )
        self._test_dataloader = torch.utils.data.DataLoader(
            self._test_dataset,
            batch_size = min(batch_size, len(self._test_dataset)),
            shuffle = False,
            drop_last = False
        )
        self._compute_classes_distr(model)



    def _compute_classes_distr(self, model): 

        # Setup 
        self._num_classes = list(model.state_dict().values())[-1].shape[0] # num_classes = bias' dimension
        self._train_classes_counts = np.zeros(self._num_classes)
        self._test_classes_counts = np.zeros(self._num_classes)

        # Fill classes counts
        train_labels, train_labels_counts = np.unique(self._train_dataset._labels, return_counts = True)
        test_labels, test_labels_counts = np.unique(self._test_dataset._labels, return_counts = True)
        self._train_classes_counts[train_labels] = train_labels_counts 
        self._test_classes_counts[test_labels] = test_labels_counts
        self._classes_distr = self._train_classes_counts / self._train_classes_counts.sum()



    def get_id(self): 
        return self._id 
    


    def get_num_samples(self, train: bool = True): 
        if train: return len(self._train_dataset)
        else: return len(self._test_dataset)
    


    def get_update(self): 
        sd = deepcopy(self._model.state_dict())
        self._model = None
        return sd
    


    def load_update(self, model: torch.nn.Module): 
        self._model = deepcopy(model)



    def train(self, num_iterations: int):

        # Setup  
        cross_entropy = torch.nn.CrossEntropyLoss(reduction = "mean")
        cross_entropy.train()
        self._model = self._model.to(self._device)
        self._model.train()
        sgd = torch.optim.SGD(self._model.parameters(), **self._sgd_params)
        stop_iter_flag = False
        loss, iters_count = [], 0
        self._train_dataset.load_data()
        dataloader = self._train_dataloader

        # Iterations 
        while True: 

            # Iterations on dataloader 
            for X, y in dataloader: 

                # Move data to device
                X, y = X.to(self._device), y.to(self._device)

                # Optimization step 
                sgd.zero_grad()
                y_hat = self._model(X)
                batch_loss = cross_entropy(y_hat, y)
                batch_loss.backward()
                sgd.step()

                # Update variables
                loss.append(batch_loss.item())
                iters_count += 1

                # Stopping criterion
                if iters_count == num_iterations: 
                    stop_iter_flag = True 
                    break 

            # Stopping criterion 
            if stop_iter_flag:
                break

        self._train_dataset.clear()
        self._model = self._model.cpu()

        return torch.tensor(loss, dtype = torch.float32, requires_grad = False)



    def test(self): 

        # Setup 
        cross_entropy = torch.nn.CrossEntropyLoss(reduction = "mean")
        cross_entropy.eval()
        self._model = self._model.to(self._device)
        self._model.eval()
        loss, accuracy, balanced_accuracy, tot_num_samples = 0., 0., 0., 0
        self._test_dataset.load_data()
        dataloader = self._test_dataloader

        # No gradient computation
        with torch.no_grad(): 

            # Iterations on dataloader 
            for X, y in dataloader:

                # Move data to device 
                X, y = X.to(self._device), y.to(self._device)
                num_samples = y.size(0)
                tot_num_samples += num_samples

                # Compute prediction and metrics 
                y_hat = self._model(X)
                # - Loss
                loss += cross_entropy(y_hat, y).item() * num_samples
                # - Accuracy
                matches_mask = y_hat.argmax(1).eq(y)
                accuracy += matches_mask.sum().item()
                # - Balanced accuracy
                matches_ohe = torch.zeros_like(y_hat, device = self._device, requires_grad = False)
                matches_ohe[matches_mask, y[matches_mask]] = 1 
                balanced_accuracy += matches_ohe.sum(0).cpu().numpy()

        # Update balanced_accuracy 
        balanced_accuracy = balanced_accuracy.dot(self._classes_distr)
        balanced_tot_num_samples = self._test_classes_counts.dot(self._classes_distr)
        self._test_dataset.clear()
        self._model = self._model.cpu()
        self._model = None

        return {
            "loss": loss / tot_num_samples,
            "accuracy": accuracy / tot_num_samples,
            "balanced_accuracy": balanced_accuracy / balanced_tot_num_samples
        }







