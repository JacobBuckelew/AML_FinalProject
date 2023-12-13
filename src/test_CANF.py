"""
Main script for testing the CANF model
Ensure checkpoint directory is correct before saving model params
"""
from torchconfig import *
from load_data import *
from CANF import *
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, precision_score, recall_score
import argparse
import os
import matplotlib.pyplot as plt

ROOT = "/home/jbuckelew/workspace/AML_Project/"


def plot_densities(densities, name=None):
    path = ROOT + "figures"
    bins = int(np.sqrt(len(densities)))
    fig = plt.figure(figsize=(10,10))
    plt.hist(densities, bins=bins, color="skyblue")
    plt.xlabel("Log-Density", fontsize=25)
    plt.title(f"{name}", fontsize=30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel("Frequency", fontsize=25)
    plt.savefig(path + f"/densities_{name}.jpeg", dpi=600)


def evaluation_metrics(scores, labels, name=None):

    # get AUC and densities

    # plot densities
    plot_densities(scores, name=name)

    # get AUC

    auc = roc_auc_score(labels, scores)

    fpr, tpr, thresholds = roc_curve(labels, scores)

    f1_scores = []
    for threshold in thresholds:
        pred = [1 if x >= threshold else 0 for x in scores]
        f1 = f1_score(labels, pred)
        f1_scores.append(f1)
    #print(f1_scores)
    max_idx = f1_scores.index(max(f1_scores))
    max_f1 = f1_scores[max_idx]
    threshold = thresholds[max_idx]
    pred = [1 if x >= threshold else 0 for x in scores]
    precision = precision_score(labels, pred)
    recall = recall_score(labels, pred)

    metrics = {}

    metrics["auc"] = auc
    metrics["f1"] = max_f1
    metrics["precision"] = precision
    metrics["recall"] = recall
    metrics["threshold"] = threshold
    metrics["log_density"] = np.mean(scores)
    metrics["fpr"] = fpr
    metrics["tpr"] = tpr

    return metrics



def get_args():

    parser = argparse.ArgumentParser(description="OOD Detection")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--mode", type= str, default="test/")
    parser.add_argument("--epsilon", type=float, default="0.05")
    parser.add_argument("--model", type=str, default="CANF")
    parser.add_argument("--checkpt", type=str, default="checkpoint/")
    parser.add_argument("--cfgs", type=str, default="configs/opt_cfgs")
    parser.add_argument("--gpu", type=bool, default=True)
    parser.add_argument("--log", type=str, default="log/")
    parser.add_argument("--gnn", type=int, default=1)
    parser.add_argument("--context", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="Inrix")
    # training parameters
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--entities", type=int, default=96)
    parser.add_argument("--window_size", type=int, default=3)
    parser.add_argument("--stride_size", type=int, default=1)
    parser.add_argument("--b_norm", type=bool, default=True)
    args = parser.parse_args()
    return args

# MAIN

if __name__ == "__main__":
    # get args
    args = get_args()
    #print(vars(args))
    # constant num of sensors and num features per sensor
    features = 1

    if args.gpu:
        torch.cuda.manual_seed_all(args.seed)
        cuda = torch.cuda.is_available()
        device = torch.device("cuda" if cuda else "cpu")
    else:
        device = torch.device("cuda:cpu")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("Loading Data")
    if args.dataset == "Inrix":
        train_loader, val_loader, test_loader = load_data(window_size=args.window_size, 
                           stride_size=args.stride_size,
                           batch_size = args.batch_size,
                           num_entities = args.entities)
    elif args.dataset == "rds":
        train_loader, val_loader, test_loader = load_rds(window_size=args.window_size, 
                           stride_size=args.stride_size,
                           batch_size = args.batch_size,
                           num_entities = args.entities)
    # load optimal cfgs
    cfg_path = ROOT + args.cfgs + f"/{args.model}"
    best_cfgs = dotdict(load_json(cfg_path))
    #print(best_cfgs)
    if args.context == False:
        gnn = False
        rnn = False
        print("testing ablation 2")
        canf = CANF(num_features=features,
                    num_entities=args.entities, 
                    num_blocks=best_cfgs.num_blocks, 
                    hidden_size=best_cfgs.st_units, 
                    window_size=args.window_size, 
                    num_hidden=best_cfgs.st_layers,  
                    b_norm=args.b_norm,
                    dropout=0,
                    gnn=False,
                    rnn=False)
        checkpt_path = ROOT + args.checkpt + args.model +"/params.pt"
        output_path = ROOT + args.log + args.mode + args.model + "/"
    elif args.gnn == False:
        rnn = True
        gnn = False
        print(" testing ablation 1")
        canf = CANF(num_features=features,
                    num_entities=args.entities, 
                    num_blocks=best_cfgs.num_blocks, 
                    hidden_size=best_cfgs.st_units, 
                    window_size=args.window_size, 
                    num_hidden=best_cfgs.st_layers,  
                    b_norm=args.b_norm,
                    dropout=0,
                    gnn=False,
                    rnn=True)
        checkpt_path = ROOT + args.checkpt + args.model
        output_path = ROOT + args.logs + args.mode + args.model + "/"
    else:
        # create instance of CANF model
        canf = CANF(num_features=features,
                    num_entities=args.entities, 
                    num_blocks=best_cfgs.num_blocks, 
                    hidden_size=best_cfgs.st_units, 
                    window_size=args.window_size, 
                    num_hidden=best_cfgs.st_layers, 
                    dropout=0, 
                    b_norm=args.b_norm)
        checkpt_path = ROOT + args.checkpt + args.model + "/params.pt"

        output_path = ROOT + args.log + args.mode + args.model + '/'
                


    canf = canf.to(device)
    #load state dictionary
    print(checkpt_path)
    canf.load_state_dict(torch.load(checkpt_path))
    canf.eval()

    labels = []
    losses = []
    start = []
    i = 0
    print("Testing")
    with torch.no_grad():
        # test over all 5 simulations
        # test single window each time
        for x, label, time in test_loader:
            x = x.to(device)
            loss = canf(x)

                #if label == 1:
               # if i == 147:
                    #print(x.item())
                #print(loss)
                #fisher_value, loss = detector.test(x, device)
            losses.append(loss.item())
            start.append(time[0])
            #print(start[i])
                #fisher_values.append(fisher_value)
            labels.append(label.item())
            i += 1
                #print(i)
    # log final results into test folder
    dictionary = {"loss": losses, "label": labels, "start":start}
    results = pd.DataFrame(dictionary)
    
    results.to_csv(f"{output_path}densities.csv")
    if args.dataset == "Inrix":
        metrics = evaluation_metrics(losses, labels, name=args.model)
        save_json(metrics, f'{output_path}test_results')
        



        
       
    
    








        





    


    

    
    
     