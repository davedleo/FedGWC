import argparse 



def parse_int(s: str):
    return [int(v) for v in s.split(',')] 

def parse_float(s: str): 
    return [float(v) for v in s.split(',')] 

def parse_str(s: str): 
    return [v for v in s.split(',')] 

def parse_bool(s: str): 
    return [bool(v) for v in s.split(',')] 



def get_args():

    # Setup
    parser = argparse.ArgumentParser()

    # Dataset 
    parser.add_argument('--dataset', type = parse_str, default = ['cifar100'])
    parser.add_argument('--alpha', type = parse_str, default = ['0.05'])
    parser.add_argument('--num_noise_clients', type = parse_int, default = [0])
    parser.add_argument('--num_blur_clients', type = parse_int, default = [0])
    parser.add_argument('--noise_sigma', type = parse_float, default = [-1.])
    parser.add_argument('--blur_radius', type = parse_int, default = [-1])

    # Client
    parser.add_argument('--client', type = parse_str, default = ['base'])
    parser.add_argument('--num_local_iters', type = parse_int, default = [7])
    parser.add_argument('--batch_size', type = parse_int, default = [64])
    parser.add_argument('--lr', type = parse_float, default = [.01])
    parser.add_argument('--momentum', type = parse_float, default = [0.])
    parser.add_argument('--weight_decay', type = parse_float, default = [4e-4])

    # Server 
    parser.add_argument('--server', type = parse_str, default = ['fedgwc']) 
    parser.add_argument('--aggregation', type = parse_str, default = ['fedavg'])
    parser.add_argument('--participation_rate', type = parse_float, default = [.1])
    parser.add_argument('--num_rounds', type = parse_int, default = [10000])
    parser.add_argument('--server_lr', type = parse_float, default = [0.])
    parser.add_argument('--server_momentum', type = parse_float, default = [0.])
    parser.add_argument('--gamma', type = parse_float, default = [.1])
    parser.add_argument('--eps', type = parse_float, default = [1e-5])
    parser.add_argument('--explore', type = parse_bool, default = [False])
    parser.add_argument('--loss_sampling_size', type = parse_int, default = [10])

    # Device
    parser.add_argument('--test_evaluation_step', type = parse_int, default = [0])
    parser.add_argument('--device', type = parse_str, default = ['cuda:0'])
    parser.add_argument('--seed', type = parse_int, default = [0])

    return parser.parse_args() 
