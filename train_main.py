def argument_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', action='store', type=str, help='File name', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='Task name', required=True)
    parser.add_argument('--model', type=str,choices=['mlp','act','diffusion'], help='choose model', required=True)
    parser.add_argument('--batch_size',type=int, help='Batch size', required=False,default=32)
    parser.add_argument('--val_split',type=float, help='Val split', required=False,default=0.2)
    parser.add_argument('--epochs',type=int,help='num of epochs',required=False,default=10)
    parser.add_argument('--learning_rate',type=float,help='learning rate',required=False,default=0.001)
    parser.add_argument('--early_stop_patience',type=int,help='early stop patience',required=False,default=10)
    args = parser.parse_args()
    return args 

if __name__ == '__main__':
    args = argument_parser()
    read_data_list = [args.task_name]

    if args.model == 'mlp':
        from mlp.mlp_trainer import MlpTrain
        trainer = MlpTrain(args)
    elif args.model == 'act':
        from act.act_trainer import ActTrainer
        trainer = ActTrainer(args)
    elif args.model == 'diffusion':
        from diffusion.diffusion_trainer import DiffusionTrainer
        trainer = DiffusionTrainer(args)

    trainer.train()
