"""
Main script for training the CANF model
Ensure checkpoint directory is correct before saving model params
"""
from torchconfig import *
from load_data import *
from CANF import *
import argparse
import os
import matplotlib.pyplot as plt

ROOT = "/home/jbuckelew/workspace/AML_FinalProject"

def get_args():

    parser = argparse.ArgumentParser(description="OOD Detection")
    parser.add_argument("--mode", type=str, default='/train')
    parser.add_argument("--checkpt", type=str, default='/checkpoint', help="checkpoint")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log", type=int, default=20, help="How often to checkpoint model")
    parser.add_argument("--log_output", type=str, default="/log")
    parser.add_argument("--model", type=str, default="CANF")
    parser.add_argument("--dataset", type=str, default="Inrix")
    # Set specific GPU. Set to 0 if not using GPU
    parser.add_argument("--gpu", type=int, default=0)
    # model parameters for CANF
    parser.add_argument("--st_units", type=int, default=8)
    parser.add_argument("--st_layers", type=int, default=1)
    parser.add_argument("--num_blocks", type=int, default=8)
    parser.add_argument("--b_norm", type=bool, default=True)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--wdecay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.95)
    parser.add_argument("--gnn", type=int, default=1)
    parser.add_argument("--context", type=int, default=1)
    parser.add_argument("--phase", type=int, default=1)
    # training parameters
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--entities", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--stride_size", type=int, default=3)
    args = parser.parse_args()
    return args


def plot_loss(train_loss):
    plot_dir = ROOT + "/figures"
    fig = plt.figure()
    plt.plot(train_loss)
    plt.savefig(plot_dir + "/loss.jpeg")



# MAIN

if __name__ == "__main__":
    # get args
    args = get_args()
    print(vars(args))
    # constant num of sensors and num features per sensor

    features = 1

    if args.gpu:
        torch.cuda.manual_seed_all(args.seed)
        cuda = torch.cuda.is_available()
        device = torch.device(f"cuda:{args.gpu}" if cuda else "cpu")
    else:
        device = torch.device("cpu")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # load training data 
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


    
    # Run model without context
    
    if args.context == False:
        gnn = False
        rnn = False
        print("training ablation 2")
        canf = CANF(num_features=features,
                    num_entities=args.entities, 
                    num_blocks=args.num_blocks, 
                    hidden_size=args.st_units, 
                    window_size=args.window_size, 
                    num_hidden=args.st_layers,  
                    b_norm=args.b_norm,
                    gnn=False,
                    rnn=False)
    
    # run model without spatial learning + attention
    
    elif args.gnn == False:
        rnn = True
        gnn = False
        print("training ablation 1")
        canf = CANF(num_features=features,
                    num_entities=args.entities, 
                    num_blocks=args.num_blocks, 
                    hidden_size=args.st_units, 
                    window_size=args.window_size, 
                    num_hidden=args.st_layers,  
                    b_norm=args.b_norm,
                    gnn=False,
                    rnn=True)

    else:
        # create instance of CANF model
        canf = CANF(num_features=features,
                    num_entities=args.entities, 
                    num_blocks=args.num_blocks, 
                    hidden_size=args.st_units, 
                    window_size=args.window_size, 
                    num_hidden=args.st_layers,  
                    b_norm=args.b_norm)
    
    canf.to(device)

    # save checkpt path
    checkpt_path = os.path.join(ROOT + args.checkpt, args.model)
    print("Saving model parameters to: ",checkpt_path)
    if not os.path.exists(checkpt_path):
        os.makedirs(checkpt_path)
    
    log_path = os.path.join(ROOT + args.log_output + args.mode, args.model)
    print("Logging train/val loss to: ", log_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    optimizer = optim.Adam(
        [{'params': canf.parameters(), 'weight_decay': args.wdecay}],
        lr=args.lr, weight_decay=0.0
    )

    # Learning rate scheduler
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.01)

    EPOCHS = args.epochs


    # Training Loop, only use train and validation loaders here

    for epoch in range(EPOCHS):
        loss_train = []

        canf.train()
        for x, _ in train_loader:
            x = x.to(device)
            # zero gradients
            optimizer.zero_grad()
            # pass data through the model
            loss = -1* canf(x)
            # backward step
            total_loss = loss
            total_loss.backward()
            # clip gradients to prevent underflow or overflow
            clip_grad_value_(canf.parameters(), 1)
            optimizer.step()

            loss_train.append(loss.item())
        #scheduler.step()
        
        # validate on calibration data
        canf.eval()
        loss_val = []
        with torch.no_grad():
            for x, _  in val_loader:
                x = x.to(device)
                #loss = -1* canf.estimate_density(x).cpu().numpy()
                loss =  -1 * canf(x).cpu().numpy()
                #print(loss)
                loss_val.append(loss)
        
        print("=====================================")
        print(f"Epoch {epoch}: train loss = {np.mean(loss_train)}, val loss = {np.mean(loss_val)}")
        # checkpoint for saving params 
        #if epoch % args.log == args.log - 1:
        if epoch == EPOCHS - 1:
            print("saving model")
            torch.save(canf.state_dict(), os.path.join(checkpt_path, "params.pt"))
    # log final results
    results = {"epoch": EPOCHS, "train_loss": np.mean(loss_train), "val_loss": np.mean(loss_val)}
                   
                
    save_json(results, f'{log_path}/results')




                   


    