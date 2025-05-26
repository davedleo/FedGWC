import utils
from config import get_args





def main():

    # Global setup
    args = get_args()
    for attr, value in args.__dict__.items():
        if isinstance(value, list): setattr(args, attr, value[0])

    # Algorithm setup 
    utils.set_seed(args.seed)
    _, path = utils.run_name_from_args(args) 
    server = utils.agents_initialization(args)

    # Training
    train_results = server.train(args.num_rounds, args.test_evaluation_step)
    test_results = server.test()
    results = {'train': train_results, 'test': test_results}
    if args.server == "fedgwc": 
        results['server'] = server

    # Save results 
    utils.save_results(args, path, results)





if __name__ == "__main__":
    main()
