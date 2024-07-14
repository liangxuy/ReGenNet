from .models.stgcn import STGCN

def get_model(parameters):
    if parameters['pose_rep'] == 'xyz':
        layout = "ntu-rgb+d" if parameters["glob"] else "ntu_edge"
    else:
        if parameters['body_model'] == 'smpl':
            layout = "smpl" if parameters["glob"] else "smpl_noglobal"
        elif parameters['body_model'] == 'smplx':
            layout = 'smplx'

    model = STGCN(in_channels=parameters["nfeats"],
                  num_class=parameters["num_classes"],
                  num_person=parameters["num_person"],
                  graph_args={"layout": layout, "strategy": "spatial"},
                  edge_importance_weighting=True,
                  device=parameters["device"])
    
    model = model.to(parameters["device"])
    return model
    
