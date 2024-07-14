from src.parser.evaluation import parser

def main():
    parameters, folder, checkpointname, epoch, niter = parser()

    dataset = parameters["dataset"]
    print(dataset)
    model_path = parameters['model_path']
    if dataset in ["ntu13", "humanact12"]:
        from .gru_eval import evaluate
        evaluate(parameters, folder, checkpointname, epoch, niter)
    elif dataset in ["uestc", "ntu120_1p", "ntu120_2p", "ntu120_1p_smpl", "ntu120_2p_smpl", "ntu120_2p_smplx", "babel", "chi3d"]:
        from .stgcn_eval import evaluate
        num_person = 1
        if dataset == 'uestc':
            num_classes = 40
        elif dataset == 'ntu120_1p':
            num_classes = 94
        elif dataset == 'ntu120_2p':
            num_classes = 26
            num_person = 2
        elif dataset == 'ntu120_1p_smpl':
            num_classes = 94
        elif dataset == 'ntu120_2p_smpl':
            num_classes = 26
            num_person = 2
        elif dataset == 'ntu120_2p_smplx':
            num_classes = 26
            num_person = 2
        elif dataset == 'babel':
            num_classes = 120
        elif dataset == 'chi3d':
            num_classes = 8
            num_person = 2
        else:
            raise NotImplementedError
        evaluate(parameters, folder, checkpointname, epoch, niter, num_classes, model_path, num_person)
    else:
        raise NotImplementedError("This dataset is not supported.")


if __name__ == '__main__':
    main()
