import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from tqdm import tqdm
import numpy as np

from .model import GINE

from utils.gine_utils import ExtractSubstructureContextPair, DataLoaderSubstructContext

from data import PreprocessFloodEventDataset
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool


def pool_func(x, batch, mode = "sum"):
    if mode == "sum":
        return global_add_pool(x, batch)
    elif mode == "mean":
        return global_mean_pool(x, batch)
    elif mode == "max":
        return global_max_pool(x, batch)

def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr

criterion = nn.BCEWithLogitsLoss()

def train(args, model_substruct, model_context, loader, optimizer_substruct, optimizer_context, device):
    model_substruct.train()
    model_context.train()

    balanced_loss_accum = 0
    acc_accum = 0

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        # creating substructure representation
        substruct_rep = model_substruct(batch.x_substruct, batch.edge_index_substruct, batch.edge_attr_substruct)[batch.center_substruct_idx]
        
        ### creating context representations
        overlapped_node_rep = model_context(batch.x_context, batch.edge_index_context, batch.edge_attr_context)[batch.overlap_context_substruct_idx]

        #Contexts are represented by 
        if args['mode'] == "cbow":
            # positive context representation
            context_rep = pool_func(overlapped_node_rep, batch.batch_overlapped_context, mode = args['context_pooling'])
            # negative contexts are obtained by shifting the indicies of context embeddings
            neg_context_rep = torch.cat([context_rep[cycle_index(len(context_rep), i+1)] for i in range(args['neg_samples'])], dim = 0)
            
            pred_pos = torch.sum(substruct_rep * context_rep, dim = 1)
            pred_neg = torch.sum(substruct_rep.repeat((args['neg_samples'], 1))*neg_context_rep, dim = 1)

        elif args['mode'] == "skipgram":

            expanded_substruct_rep = torch.cat([substruct_rep[i].repeat((batch.overlapped_context_size[i],1)) for i in range(len(substruct_rep))], dim = 0)
            pred_pos = torch.sum(expanded_substruct_rep * overlapped_node_rep, dim = 1)

            #shift indices of substructures to create negative examples
            shifted_expanded_substruct_rep = []
            for i in range(args['neg_samples']):
                shifted_substruct_rep = substruct_rep[cycle_index(len(substruct_rep), i+1)]
                shifted_expanded_substruct_rep.append(torch.cat([shifted_substruct_rep[i].repeat((batch.overlapped_context_size[i],1)) for i in range(len(shifted_substruct_rep))], dim = 0))

            shifted_expanded_substruct_rep = torch.cat(shifted_expanded_substruct_rep, dim = 0)
            pred_neg = torch.sum(shifted_expanded_substruct_rep * overlapped_node_rep.repeat((args['neg_samples'], 1)), dim = 1)

        else:
            raise ValueError("Invalid mode!")

        loss_pos = criterion(pred_pos.double(), torch.ones(len(pred_pos)).to(pred_pos.device).double())
        loss_neg = criterion(pred_neg.double(), torch.zeros(len(pred_neg)).to(pred_neg.device).double())

        
        optimizer_substruct.zero_grad()
        optimizer_context.zero_grad()

        loss = loss_pos + args['neg_samples']*loss_neg
        loss.backward()
        #To write: optimizer
        optimizer_substruct.step()
        optimizer_context.step()

        balanced_loss_accum += float(loss_pos.detach().cpu().item() + loss_neg.detach().cpu().item())
        acc_accum += 0.5* (float(torch.sum(pred_pos > 0).detach().cpu().item())/len(pred_pos) + float(torch.sum(pred_neg < 0).detach().cpu().item())/len(pred_neg))

    return balanced_loss_accum/step, acc_accum/step

def main(model_file):
    args = {
        'device': 0,
        'mode': 'cbow',
        'l1': 1,
        'csize': 3,
        'center': 0,
        'num_node_features': 4,
        'batch_size': 256,
        'num_workers': 4,
        'epochs': 20,
        'lr': 0.001,
        'decay': 0,
        'num_layer': 5,
        'dropout_ratio': 0,
        'neg_samples': 1,
        'JK': 'last',
        'context_pooling': 'mean',
        'output_model_file': model_file,
    }

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args['device'])) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    l1 = args['num_layer'] - 1
    l2 = l1 + args['csize']

    print(args['mode'])
    print("num layer: %d l1: %d l2: %d" %(args['num_layer'], l1, l2))

    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    #set up dataset and transform function.
    raw_dataset, info = PreprocessFloodEventDataset(node_features=config['node_features'],
                    edge_features=config['edge_features'],
                    **config['dataset_parameters']).load()
    transform = ExtractSubstructureContextPair(args['num_layer'], l1, l2)

    dataset = [] 
    for data in raw_dataset:
        dataset.append(transform(data))
    
    loader = DataLoaderSubstructContext(dataset, batch_size=args['batch_size'], shuffle=True, num_workers = args['num_workers'])

    #set up models, one for pre-training and one for context embeddings
    gine_params = config['model_parameters']['GINE']
    base_model_params = {
        'static_node_features': info['num_static_node_features'],
        'dynamic_node_features': info['num_dynamic_node_features'],
        'static_edge_features': info['num_static_edge_features'],
        'dynamic_edge_features': info['num_dynamic_edge_features'],
        'previous_timesteps': info['previous_timesteps'],
        'device': device,
    }
    model_substruct = GINE(**gine_params, **base_model_params).to(device)
    context_params = {**gine_params, 'num_layers': int(l2 - l1)}
    model_context = GINE(**context_params, **base_model_params).to(device)

    #set up optimizer for the two GNNs
    optimizer_substruct = optim.Adam(model_substruct.parameters(), lr=args['lr'], weight_decay=args['decay'])
    optimizer_context = optim.Adam(model_context.parameters(), lr=args['lr'], weight_decay=args['decay'])

    for epoch in range(1, args['epochs']+1):
        print("====epoch " + str(epoch))
        
        train_loss, train_acc = train(args, model_substruct, model_context, loader, optimizer_substruct, optimizer_context, device)
        print(train_loss, train_acc)

    if not args['output_model_file'] == "":
        torch.save(model_substruct.state_dict(), args['output_model_file'] + ".pth")

if __name__ == "__main__":
    #cycle_index(10,2)
    main()