from __future__ import print_function
sys.path.insert(0, '..')

import argparse, json, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR, MultiStepLR,ExponentialLR,CosineAnnealingLR
import time,os,sys
import numpy as np
import copy
from torchsummary import summary as summary2
import torchvision
from numpy import unique
from scipy.stats import entropy as scipy_entropy
from scipy.spatial import distance
from scipy.stats import norm
from utils import dataloader,hsummary 
import numpy as np
import nets.realresnet as resnet
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
import multiprocessing
import warnings
warnings.filterwarnings("ignore")
import gc

def tfcov(x, rowvar=False, bias=False, ddof=None, aweights=None):
    """Estimates covariance matrix like numpy.cov"""
    # ensure at least 2D
    x = torch.tensor(x).cpu()
    x=torch.transpose(x,0,1)
    if x.dim() == 1:
        x = x.view(-1, 1)

    # treat each column as a data point, each row as a variable
    if rowvar and x.shape[0] != 1:
        x = x.t()

    if ddof is None:
        if bias == 0:
            ddof = 1
        else:
            ddof = 0

    w = aweights
    if w is not None:
        if not torch.is_tensor(w):
            w = torch.tensor(w, dtype=torch.float)
        w_sum = torch.sum(w)
        avg = torch.sum(x * (w/w_sum)[:,None], 0)
    else:
        avg = torch.mean(x, 0)

    # Determine the normalization
    if w is None:
        fact = x.shape[0] - ddof
    elif ddof == 0:
        fact = w_sum
    elif aweights is None:
        fact = w_sum - ddof
    else:
        fact = w_sum - ddof * torch.sum(w * w) / w_sum

    xm = x.sub(avg.expand_as(x))

    if w is None:
        X_T = xm.t()
    else:
        X_T = torch.mm(torch.diag(w), xm).t()

    c = torch.mm(X_T, xm)
    c = c / fact
    c = c.squeeze()
    c[c != c] = 0 # nan to zero

    return c.squeeze()


def kl_mvn(m0, S0, m1, S1,key):
    """
    Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
    Also computes KL divergence from a single Gaussian pm,pv to a set
    of Gaussians qm,qv.
    Diagonal covariances are assumed.  Divergence is expressed in nats.

    - accepts stacks of means, but only one S0 and S1

    From wikipedia
    KL( (m0, S0) || (m1, S1))
         = .5 * ( tr(S1^{-1} S0) + log |S1|/|S0| + 
                  (m1 - m0)^T S1^{-1} (m1 - m0) - N )
    """
    # store inv diag covariance of S1 and diff between means
    N = m0.shape[0]
    iS1 = np.linalg.pinv(S1)
    diff = m1 - m0

    # kl is made of three terms
    tr_term   = np.trace(iS1 @ S0)
    # where_are_NaNs = isnan(det_term)
    # det_term[np.isnan(det_term)] = 0
    if np.sum(np.linalg.det(S0))==0:
        det_term  = np.zeros(((np.linalg.det(S0)).shape)) #np.sum(np.log(S1)) - np.sum(np.log(S0))
    else:
        det_term  = np.log(np.linalg.det(S1)/np.linalg.det(S0)) #np.sum(np.log(S1)) - np.sum(np.log(S0))

    quad_term = diff.T @ np.linalg.pinv(S1) @ diff #np.sum( (diff*diff) * iS1, axis=1)
    #print(tr_term,det_term,quad_term)
    # if key=='conv1':
        # print(np.sum(np.linalg.det(S0)),np.sum(np.linalg.det(S1)),np.sum(tr_term),np.sum(quad_term),N)
    return .5 * (tr_term + det_term + quad_term - N) 

def kld(model,network_size,interaction): #KL Divergence
    '''NetworkSize: {'mainlayer':[[[indx],[from],[to]],[[sublayer1][sublayer2][sublayer3]]]} '''
    def local_kl(ar,ar_in,ar_out,k,return_dict,key):
        # print('ar',ar.size())
        # ar = ar.cpu()
        kldiverg = np.zeros((ar_out,ar_out))

        ar = ar.numpy()
        ar = np.reshape(ar,(ar_in,ar_out,k*k))
        # art = torch.reshape(ar,(ar_in,ar_out,k*k)).cpu()
        mu = np.mean(ar,axis=0)
        convm = np.zeros((ar_out,k*k,k*k))
        for o in range(ar_out):
            convm[o] = tfcov(np.transpose(ar[:,o,:]))# cov matrix : out k*k k*k
        for i in range(ar_out):
            for j in range(i,ar_out):
                kldiverg[i,j] = kl_mvn(mu[i,:],convm[i],mu[j,:],convm[j],key)
        kldiverg[np.isnan(kldiverg)] = 1
        kldiverg = (kldiverg-np.amin(kldiverg))/np.amax(np.abs(kldiverg))
        kldiverg+=np.transpose(kldiverg)
        kldiverg=kldiverg/2.
        return_dict[key] = kldiverg

    print('Starting KLD...')
    
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []
    indx_dict = {}
    for key in network_size.keys():
        indx = network_size[key][0][1] # units location in interaction matrix
        # print(key)
        lay_name = network_size[key][1][0][0]
        # print(lay_name)
        if ('conv' in lay_name and 'weight' in lay_name and 'layer' in lay_name):
            # print('yes1')
            layer_name = lay_name.split('.')[0]
            layer_indx = int(lay_name.split('.')[1])
            conv_name = lay_name.split('.')[2]
            ar = model._modules[layer_name][layer_indx]._modules[conv_name].weight.data
            ar_in = network_size[key][1][0][1][1]
            ar_out = network_size[key][1][0][1][0]
            ar_k = network_size[key][1][0][1][2]
            ar =ar.cpu()
            jobs.append(multiprocessing.Process(target=local_kl, args=(ar,ar_in,ar_out,ar_k,return_dict,key)))
            indx_dict[key] = [[indx[0][0],indx[0][-1]+1],[indx[0][0],indx[0][-1]+1]]
            jobs[-1].start()
        elif ('conv' in lay_name and 'weight' in lay_name) and ('layer' not in lay_name):
            tmp_name = lay_name.split('.')[0]
            ar = model._modules[tmp_name].weight.data
            ar_in = network_size[key][1][0][1][1]
            ar_out = network_size[key][1][0][1][0]
            ar_k = network_size[key][1][0][1][2]
            ar =ar.cpu()
            jobs.append(multiprocessing.Process(target=local_kl, args=(ar.cpu(),ar_in,ar_out,ar_k,return_dict,key)))
            indx_dict[key] = [[indx[0][0],indx[0][-1]+1],[indx[0][0],indx[0][-1]+1]]
            jobs[-1].start()
        elif ('downsample' in lay_name and 'weight' in lay_name):
            layer_name = lay_name.split('.')[0]
            layer_indx = int(lay_name.split('.')[1])
            conv_name = lay_name.split('.')[2]
            downs_indx = lay_name.split('.')[3]
            ar = model._modules[layer_name][layer_indx]._modules['downsample'][int(downs_indx)].weight.data.clone()         
            ar_in = network_size[key][1][0][1][1]
            ar_out = network_size[key][1][0][1][0]
            ar_k = network_size[key][1][0][1][2]
            ar =ar.cpu()
            jobs.append(multiprocessing.Process(target=local_kl, args=(ar,ar_in,ar_out,ar_k,return_dict,key)))
            indx_dict[key] = [[indx[0][0],indx[0][-1]+1],[indx[0][0],indx[0][-1]+1]]
            jobs[-1].start()
    for proc in jobs:
        proc.join()
    for key in indx_dict.keys():
        interaction[indx_dict[key][0][0]:indx_dict[key][0][1],indx_dict[key][0][0]:indx_dict[key][0][1]] = return_dict[key]
    interaction = torch.from_numpy(interaction).float().cpu()
    return interaction


def selection(population,new_population,cost_population,cost_newpopulation,popsize,dimsize):
    # Population
    cost_population = np.reshape(np.asarray(cost_population),(popsize,))
    cost_newpopulation = np.reshape(np.asarray(cost_newpopulation),(popsize,))

    population_to_go = np.zeros((popsize,dimsize),dtype='float64')

    mask = cost_population<=cost_newpopulation
    mask_pop_cost = np.where(mask==True)[0]
    mask_new_pop_cost = np.where((np.logical_not(mask))==True)[0]

    population_to_go[mask_pop_cost,:]=population[mask_pop_cost,:]
    population_to_go[mask_new_pop_cost,:]=new_population[mask_new_pop_cost,:]

    # Cost 
    costs = np.zeros((popsize),dtype='float64')
    costs[mask_pop_cost]=cost_population[mask_pop_cost]
    costs[mask_new_pop_cost]=cost_newpopulation[mask_new_pop_cost]

    # Bests  
    best_solution = np.asarray(population_to_go[np.argmin(costs),:])
    best_cost=np.asarray(np.amin(costs))
    generation_cost=np.asarray(np.mean(costs))
    return population_to_go, costs, best_solution, best_cost, generation_cost


def test(args, model, device, d_loader, criterion,isvalid):
    test_loss = 0
    correct = 0
    correct_top3 = 0
    correct_top5 = 0
    n_test = len(d_loader) # ignoring last batch
    with torch.no_grad():
        for data, target in d_loader:
            data, target = data.to(device), target.to(device)
            output, _ = model(data)
            #### with cross entropy
            loss = criterion(output, target)
            test_loss += loss.item()
            _, predicted = output.max(1)
            # total += target.size(0)
            correct += (predicted.eq(target).sum().item()/target.size(0)) # last batch might be smaller - so use batch size

            # top3
            _, predicted_args_top3 = torch.topk(output, 3, dim=1, largest=True, sorted=True, out=None) 
            vtarget = target.view(list(target.size())[0],1)
            target3 = vtarget.repeat(1,3)
            correct_top3 += (predicted_args_top3.eq(target3).sum().item()/target.size(0)) # last batch might be smaller - so use batch size
            # top5
            _, predicted_args_top5 = torch.topk(output, 5, dim=1, largest=True, sorted=True, out=None) 
            target5 = vtarget.repeat(1,5)
            correct_top5 += (predicted_args_top5.eq(target5).sum().item()/target.size(0)) # last batch might be smaller - so use batch size

            ### With log
            # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            # pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            # correct += pred.eq(target.view_as(pred)).sum().item()
    # test_loss /= n_test
    # accu = 100. * correct / n_test
    
    #### with cross entropy
    test_loss /= n_test
    accu = 100.*correct/n_test
    accu3 = 100.*correct_top3/n_test
    accu5 = 100.*correct_top5/n_test
    if isvalid:
        print('Validation set: Average loss: {:.4f}, AccuracyT1: ({:.2f}%), AccuracyT3: ({:.2f}%),AccuracyT5: ({:.2f}%)'.format(test_loss, accu, accu3, accu5))
    else:
        print('Test set: Average loss: {:.4f}, AccuracyT1: ({:.2f}%), AccuracyT3: ({:.2f}%),AccuracyT5: ({:.2f}%)'.format(test_loss, accu, accu3, accu5))
    print(20*'#')
    return test_loss, accu, accu3, accu5


def test_c(args, model, network_size, device, d_loader, states_reshaped,loss_func,isvalid):
    criterion = nn.CrossEntropyLoss()

    # # Sparsifiy the network
    sparsifier(model,network_size, states_reshaped)

    test_loss = 0
    energy_valid = 0
    total = 0
    correct = 0
    correct_top3 = 0
    correct_top5 = 0
    n_test = len(d_loader) #(len(d_loader.dataset)-len(d_loader[-1])) # ignoring last batch
    print('ntest',n_test)
    counter = 0
    with torch.no_grad():
        for data, target in d_loader:
            data, target = data.to(device), target.to(device)
            # Get network Energy with applied states
            output, valid_eng = model(data) 

            if loss_func=='ce':
                #### with cross entropy
                loss = criterion(output, target)
                test_loss += loss.item()
                _, predicted = output.max(1)
                correct += (predicted.eq(target).sum().item()/target.size(0)) # last batch might be smaller - so use batch size
            elif loss_func=='log':
                ### With log
                output = F.log_softmax(output)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                predicted = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += (predicted.eq(target.view_as(predicted)).sum().item()/target.size(0))

            # top3
            _, predicted_args_top3 = torch.topk(output, 3, dim=1, largest=True, sorted=True, out=None) 
            vtarget = target.view(list(target.size())[0],1)
            target3 = vtarget.repeat(1,3)
            correct_top3 += (predicted_args_top3.eq(target3).sum().item()/target.size(0)) # last batch might be smaller - so use batch size
            # top5
            _, predicted_args_top5 = torch.topk(output, 5, dim=1, largest=True, sorted=True, out=None) 
            target5 = vtarget.repeat(1,5)
            correct_top5 += (predicted_args_top5.eq(target5).sum().item()/target.size(0)) # last batch might be smaller - so use batch size


            one_hot = torch.zeros(list(target.size())[0],n_classes).to(device)
            one_hot[torch.arange(list(target.size())[0]),target] = 1
            high_cost_target_one_hot = -10000*one_hot # high cost for target to remove it   [0 0 0 -10000 0 0] 


            net_eneg = torch.sum(torch.mul(one_hot,valid_eng),axis=1)
            other_net_eneg = high_cost_target_one_hot + valid_eng # mean over batches
            max_other_net_eneg = torch.max(other_net_eneg,1)

            eng_diff = net_eneg - max_other_net_eneg[0]
            energy_valid += (-1*eng_diff.mean(0))



    #### with cross entropy
    test_loss /= n_test
    accu = 100.*correct/n_test
    accu3 = 100.*correct_top3/n_test
    accu5 = 100.*correct_top5/n_test
    energy_valid = energy_valid/n_test
    if isvalid:
        print('Validation set: Average loss: {:.4f}, AccuracyT1: ({:.2f}%), AccuracyT3: ({:.2f}%),AccuracyT5: ({:.2f}%,Evalid: ({:.2f})'.format(test_loss, accu, accu3, accu5, energy_valid))
    else:
        print('Test set: Average loss: {:.4f}, AccuracyT1: ({:.2f}%), AccuracyT3: ({:.2f}%),AccuracyT5: ({:.2f}%,Evalid: ({:.2f})'.format(test_loss, accu, accu3, accu5, energy_valid))
    print(20*'#')
    return test_loss, accu, accu3, accu5, energy_valid
  
    #### with cross entropy
    # n_test = total
    # test_loss /= n_test
    # accu = 100.*correct/n_test

            ### With log
            # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            # pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            # correct += pred.eq(target.view_as(pred)).sum().item()
            # counter+=1
    # test_loss /= n_test
    # accu = 100. * correct / n_test


    # print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
    #     test_loss, correct, n_test, accu))
    # print(20*'#')
    # ## Restore network weights from backup
    # weight_restore(model, network_size, backup_weights)

  
def state_reshaper(device,NS,states,network_size,args,layer_names_wgrad):
    # print(layer_names_wgrad,network_size)
    states_reshaped = {}
    for state_indx in range(NS):
        ## State reshpaer from vector to mask
        states_reshaped[str(state_indx)]={}
        for key in network_size:
            con = states[state_indx, network_size[key][0][1][0]] # states of a layer
            for sublayer in network_size[key][1]:          
                if len(sublayer[1])==4:
                    tmp = con[:,np.newaxis,np.newaxis,np.newaxis]
                    tmp = np.tile(tmp,(1,sublayer[1][1],sublayer[1][2],sublayer[1][3]))
                elif len(sublayer[1])==1:
                    tmp = con
                elif (len(sublayer[1])==2) and ('fc' in sublayer[0]):
                    tmp = np.tile(con,(sublayer[1][0],1))
                else:
                    raise('unknown tensor shape')
                states_reshaped[str(state_indx)][sublayer[0]] = torch.from_numpy(tmp).float().to(device)
 
    return states_reshaped

def evolution(states,NS,D,best_state):
    ### Evolve States
    population = copy.deepcopy(states)
    candidates_indx = np.tile(np.arange(NS),(NS,1))
    candidates_indx = (np.reshape((candidates_indx[(candidates_indx!=np.reshape(np.arange(NS),(NS,1)))]),(NS,NS-1)))
    for i in range(NS):
        candidates_indx[i,:]=np.random.permutation(candidates_indx[i,:])
    parents = candidates_indx[:,:3]

    x1 = population[parents[:,0]]
    x2 = population[parents[:,1]]
    x3 = population[parents[:,2]]


    Ff = 0.9
    F_mask = (np.random.rand(NS,D)>Ff) # smaller F more diversity 
    keeps_mask = (np.abs(1*x2-1*x3)==0)*F_mask #np.logical_not(F_mask) # genes to keep  - not of F-mask to keep
    ev = np.multiply(np.logical_not(F_mask),x1) + np.multiply(keeps_mask,(1-x1))
    
    crossover_rate = 0.9                # Recombination rate [0,1] - larger more diverisyt
    cr = (np.random.rand(NS,D)<crossover_rate)
    mut_keep = np.multiply(ev,cr)
    pop_keep = np.multiply(population,np.logical_not(cr))
    new_population = mut_keep + pop_keep
    new_population[:,D-n_classes:] = 1
    return new_population





def weights_pdf(model, network_size):
    '''Note: Here we compute the stats for weight kernels excluding biases; The biases are pruned alogn with corresponding kernels in sparsifier and gradient masker.
    Kernels are averaged over input channels.'''

    kernel_Q = {} # Kernels pdf
    kernel_N = {} # Kernels norm 

    for key in network_size: # [0]: is for convs - [1:]: Skipping conv1 - did it up there
        lay = network_size[key]
        mu = torch.tensor([0])
        if ('conv' in key and 'layer' not in key):
            n_input = torch.tensor([lay[1][0][1][1]*lay[1][0][1][2]*lay[1][0][1][3]])
            var = torch.div(1.,n_input) # variance
            sigma = torch.sqrt(var)
            tmp_name = key.split('.')[0]
            kernel_Q[key] = torch.mean((1./(sigma*torch.sqrt(torch.tensor([2.*3.14159]))))*torch.exp(torch.tensor([-0.5])*(((model._modules[key].weight.data-mu)/sigma)**2)),axis=1)

        elif ('conv' in key and 'layer' in key):
            n_input = torch.tensor([lay[1][0][1][1]*lay[1][0][1][2]*lay[1][0][1][3]])
            var = torch.div(1.,n_input) # variance
            sigma = torch.sqrt(var)
            layer_name = key.split('.')[0]
            layer_indx = int(key.split('.')[1])
            conv_name = key.split('.')[2]
            kernel_Q[key] = torch.mean((1./(sigma*torch.sqrt(torch.tensor([2.*3.14159]))))*torch.exp(torch.tensor([-0.5])*(((model._modules[layer_name][layer_indx]._modules[conv_name].weight.data-mu)/sigma)**2)),axis=1)

        elif ('downsample' in key):
            n_input = torch.tensor([lay[1][0][1][1]*lay[1][0][1][2]*lay[1][0][1][3]])
            var = torch.div(1.,n_input) # variance
            sigma = torch.sqrt(var)
            key1 = lay[1][0][0]       
            layer_name = key1.split('.')[0]
            layer_indx = int(key1.split('.')[1])
            conv_name = key1.split('.')[2]
            downs_indx = key1.split('.')[3]
            kernel_Q[key] = torch.mean((1./(sigma*torch.sqrt(torch.tensor([2.*3.14159]))))*torch.exp(torch.tensor([-0.5])*(((model._modules[layer_name][layer_indx]._modules['downsample'][int(downs_indx)].weight.data-mu)/sigma)**2)),axis=1)
        elif ('fc' in key):
            n_input = torch.tensor([lay[1][0][1][1]])
            var = torch.div(1.,n_input) # variance
            sigma = torch.sqrt(var)
            kernel_Q[key] = (1./(sigma*torch.sqrt(torch.tensor([2.*3.14159]))))*torch.exp(torch.tensor([-0.5])*(((model._modules[key].weight.data-mu)/sigma)**2))
        else:
            raise('unknown layer')
        # Normalization
        kernel_Q[key] = kernel_Q[key]/torch.max(kernel_Q[key])

   
    return kernel_Q,kernel_N

def featuremaps_pdf(model, data, network_size, interaction):
    # Hook to grah featuremaps
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output# use output.detach() to detach from graph
        return hook

    for key in network_size: # [0]: is for convs - [1:]: Skipping conv1 - did it up there
        lay = network_size[key]
        if ('conv' in key and 'layer' not in key):
            tmp_name = key.split('.')[0]
            model._modules[tmp_name].register_forward_hook(get_activation(key)) 
        elif ('conv' in key and 'layer' in key):
            layer_name = key.split('.')[0]
            layer_indx = int(key.split('.')[1])
            conv_name = key.split('.')[2]
            model._modules[layer_name][layer_indx]._modules[conv_name].register_forward_hook(get_activation(key))        
        elif ('downsample' in key):
            key1 = lay[1][0][0] 
            layer_name = key1.split('.')[0]
            layer_indx = int(key1.split('.')[1])
            conv_name = key1.split('.')[2]
            downs_indx = key1.split('.')[3]
            model._modules[layer_name][layer_indx]._modules['downsample'][int(downs_indx)].register_forward_hook(get_activation(key))     
        elif ('fc' in key):
            model._modules[key].register_forward_hook(get_activation(key))  
    _, logits  = model(data) 
    
    # Probability distribution of feature maps
    sD = 0 
    Q = {}
    for key in activation.keys():
        indx_start = network_size[key][0][1]
        indx_end = network_size[key][0][2]
        if 'fc' in key:
            pass
        else:
            avg_fm = torch.mean(activation[key],axis=0) # average over samples in the batch
            minfm = torch.min(avg_fm)
            avg_fm = 255*(avg_fm-minfm)/(torch.max(avg_fm)-minfm) # normalize and scale the average fm
            Q[key] = torch.zeros(avg_fm.shape[0],256)
            for fm_indx in range(0,avg_fm.shape[0]): # for each featuremap
                freq = torch.histc(avg_fm[fm_indx,:,:],bins=256, min=0, max=255) # frequnecy map
                Q[key][fm_indx,:]=(freq/torch.max(freq)) # probability distribution of each fm in each layer +'_'+str(fm_indx)]

            logs = torch.log2(Q[key])
            logs[logs != logs] = 0 # nan to zero
            logs[logs == float("-Inf")] = 0
            logs[logs == float("Inf")] = 0
            entrop = -1*torch.sum((Q[key]*logs),dim=1)
            entrop = entrop/entrop.max() # Normalize Entropies in each layer
            entrop = (torch.reshape(entrop,(entrop.size()[0],1))).repeat(1,len(indx_end))
            interaction[indx_start[0][0]:indx_start[0][-1]+1,indx_end[0]:indx_end[-1]+1] = entrop
        interaction[interaction != interaction] = 0 # nan to zero

    return interaction
    
def ising_cost(args,model,data,target,device,states,NS,D, D_Conv2d,D_fc,network_size,model_n_p,keep_rate_user):
    kept_count_torch = torch.zeros(1,NS)
    for ind in range(NS):
        _,_,kept_count_torch[0,ind] = kept_counter(network_size, np.expand_dims(states[ind,:], axis=0),model_n_p)
    kept_count_torch.double()
    states = torch.from_numpy(states).cpu() # NS D
    statesR = torch.reshape(states,(D,NS))
    statesM = statesR.repeat(D,1,1) # rows are repeated states (identical)
    statesT = torch.transpose(statesM,0,1) # transpose version of statesM at D level
    interaction = np.zeros((D,D)) # placeholder for interaction term of ising model
    interaction = kld(model,network_size,interaction)
    # .cpu()
    interaction = featuremaps_pdf(model, data, network_size, interaction)
    interaction = interaction - 1 # reverse cost 
    interactionD = torch.reshape(interaction,(D,D,1))
    interactionM = interactionD.expand(-1,-1,NS) # D D NS
    penalty = D*torch.tensor([100000])*torch.max(torch.tensor([keep_rate_user])-kept_count_torch,torch.zeros(1,NS))    # penalty = D*torch.tensor([100000])*torch.max(torch.tensor([0.4])-states.sum(dim=1)/D,torch.zeros(1,NS).double())
    inter_part = -1*((statesM*statesT*interactionM).sum(dim=[0,1]))
    bias_part = 1*(states.sum(dim=1)*interaction.sum()/D)
    cost_population = inter_part + bias_part + penalty
    cost_all_one = None
    return cost_population, cost_all_one




def weight_backer(model, network_size,layer_names_wgrad):
    backup_weights = {}
    for lay_name in layer_names_wgrad: # [0]: is for convs - [1:]: Skipping conv1 - did it up there
        if ('conv' in lay_name and 'weight' in lay_name and 'layer' in lay_name):
            layer_name = lay_name.split('.')[0]
            layer_indx = int(lay_name.split('.')[1])
            conv_name = lay_name.split('.')[2]
            backup_weights[lay_name] = model._modules[layer_name][layer_indx]._modules[conv_name].weight.data.clone()         

        elif ('bn' in lay_name and 'weight' in lay_name and 'layer' in lay_name):
            layer_name = lay_name.split('.')[0]
            layer_indx = int(lay_name.split('.')[1])
            conv_name = lay_name.split('.')[2]
            backup_weights[lay_name] = model._modules[layer_name][layer_indx]._modules[conv_name].weight.data.clone()  

        elif ('bn' in lay_name and 'bias' in lay_name and 'layer' in lay_name):
            layer_name = lay_name.split('.')[0]
            layer_indx = int(lay_name.split('.')[1])
            conv_name = lay_name.split('.')[2]
            backup_weights[lay_name] = model._modules[layer_name][layer_indx]._modules[conv_name].bias.data.clone()  

        elif ('conv' in lay_name and 'weight' in lay_name) and ('layer' not in lay_name):
            tmp_name = lay_name.split('.')[0]
            backup_weights[lay_name] = model._modules[tmp_name].weight.data.clone()

        elif ('bn' in lay_name and 'weight' in lay_name) and ('layer' not in lay_name):
            tmp_name = lay_name.split('.')[0]
            backup_weights[lay_name] = model._modules[tmp_name].weight.data.clone()

        elif ('bn' in lay_name and 'bias' in lay_name) and ('layer' not in lay_name):
            tmp_name = lay_name.split('.')[0]
            backup_weights[lay_name] = model._modules[tmp_name].bias.data.clone()

        elif ('downsample' in lay_name and 'weight' in lay_name):
            layer_name = lay_name.split('.')[0]
            layer_indx = int(lay_name.split('.')[1])
            conv_name = lay_name.split('.')[2]
            downs_indx = lay_name.split('.')[3]
            backup_weights[lay_name] = model._modules[layer_name][layer_indx]._modules['downsample'][int(downs_indx)].weight.data.clone()         
        elif ('downsample' in lay_name and 'bias' in lay_name):
            layer_name = lay_name.split('.')[0]
            layer_indx = int(lay_name.split('.')[1])
            conv_name = lay_name.split('.')[2]
            downs_indx = lay_name.split('.')[3]
            backup_weights[lay_name] = model._modules[layer_name][layer_indx]._modules['downsample'][int(downs_indx)].bias.data.clone()         
        elif ('fc' in lay_name and 'bias' in lay_name):
            backup_weights[lay_name] = model._modules['fc'].bias.data.clone()
        elif ('fc' in lay_name and 'weight' in lay_name):
            backup_weights[lay_name] = model._modules['fc'].weight.data.clone()

    return backup_weights
    

def weight_restore(model, network_size, backup_weights):
    with torch.no_grad():
        for lay_name in backup_weights: # [0]: is for convs - [1:]: Skipping conv1 - did it up there
            if ('conv' in lay_name and 'weight' in lay_name and 'layer' in lay_name):
                layer_name = lay_name.split('.')[0]
                layer_indx = int(lay_name.split('.')[1])
                conv_name = lay_name.split('.')[2]
                model._modules[layer_name][layer_indx]._modules[conv_name].weight.data.copy_(backup_weights[lay_name]) 

            elif ('bn' in lay_name and 'weight' in lay_name and 'layer' in lay_name):
                layer_name = lay_name.split('.')[0]
                layer_indx = int(lay_name.split('.')[1])
                conv_name = lay_name.split('.')[2]
                model._modules[layer_name][layer_indx]._modules[conv_name].weight.data.copy_(backup_weights[lay_name]) 

            elif ('bn' in lay_name and 'bias' in lay_name and 'layer' in lay_name):
                layer_name = lay_name.split('.')[0]
                layer_indx = int(lay_name.split('.')[1])
                conv_name = lay_name.split('.')[2]
                model._modules[layer_name][layer_indx]._modules[conv_name].bias.data.copy_(backup_weights[lay_name]) 

            elif ('conv' in lay_name and 'weight' in lay_name) and ('layer' not in lay_name):
                tmp_name = lay_name.split('.')[0]
                model._modules[tmp_name].weight.data.copy_(backup_weights[lay_name])

            elif ('bn' in lay_name and 'weight' in lay_name) and ('layer' not in lay_name):
                tmp_name = lay_name.split('.')[0]
                model._modules[tmp_name].weight.data.copy_(backup_weights[lay_name])

            elif ('bn' in lay_name and 'bias' in lay_name) and ('layer' not in lay_name):
                tmp_name = lay_name.split('.')[0]
                model._modules[tmp_name].bias.data.copy_(backup_weights[lay_name])

            elif ('downsample' in lay_name and 'weight' in lay_name):
                layer_name = lay_name.split('.')[0]
                layer_indx = int(lay_name.split('.')[1])
                conv_name = lay_name.split('.')[2]
                downs_indx = lay_name.split('.')[3]
                model._modules[layer_name][layer_indx]._modules['downsample'][int(downs_indx)].weight.data.copy_(backup_weights[lay_name]) 


            elif ('downsample' in lay_name and 'bias' in lay_name):
                layer_name = lay_name.split('.')[0]
                layer_indx = int(lay_name.split('.')[1])
                conv_name = lay_name.split('.')[2]
                downs_indx = lay_name.split('.')[3]
                model._modules[layer_name][layer_indx]._modules['downsample'][int(downs_indx)].bias.data.copy_(backup_weights[lay_name]) 


            elif ('fc' in lay_name and 'bias' in lay_name):
                layer_name = lay_name.split('.')[0]
                model._modules[layer_name].bias.data.copy_(backup_weights[lay_name]) 

            elif ('fc' in lay_name and 'weight' in lay_name):
                layer_name = lay_name.split('.')[0]
                model._modules[layer_name].weight.data.copy_(backup_weights[lay_name]) 


def sparsifier(model,network_size, state_in):
    with torch.no_grad():
        for lay_name in state_in:
            if ('conv' in lay_name and 'weight' in lay_name and 'layer' in lay_name):
                layer_name = lay_name.split('.')[0]
                layer_indx = int(lay_name.split('.')[1])
                conv_name = lay_name.split('.')[2]
                model._modules[layer_name][layer_indx]._modules[conv_name].weight.data *=state_in[lay_name]
            elif ('bn' in lay_name and 'weight' in lay_name and 'layer' in lay_name):
                layer_name = lay_name.split('.')[0]
                layer_indx = int(lay_name.split('.')[1])
                conv_name = lay_name.split('.')[2]
                model._modules[layer_name][layer_indx]._modules[conv_name].weight.data *=state_in[lay_name]

            elif ('bn' in lay_name and 'bias' in lay_name and 'layer' in lay_name):
                layer_name = lay_name.split('.')[0]
                layer_indx = int(lay_name.split('.')[1])
                conv_name = lay_name.split('.')[2]
                model._modules[layer_name][layer_indx]._modules[conv_name].bias.data *=state_in[lay_name]

            elif ('conv' in lay_name and 'weight' in lay_name) and ('layer' not in lay_name):
                tmp_name = lay_name.split('.')[0]
                model._modules[tmp_name].weight.data *= state_in[lay_name]

            elif ('bn' in lay_name and 'weight' in lay_name) and ('layer' not in lay_name):
                tmp_name = lay_name.split('.')[0]
                model._modules[tmp_name].weight.data *=  state_in[lay_name]

            elif ('bn' in lay_name and 'bias' in lay_name) and ('layer' not in lay_name):
                tmp_name = lay_name.split('.')[0]
                model._modules[tmp_name].bias.data *=  state_in[lay_name]

            elif ('downsample' in lay_name and 'weight' in lay_name):
                layer_name = lay_name.split('.')[0]
                layer_indx = int(lay_name.split('.')[1])
                conv_name = lay_name.split('.')[2]
                downs_indx = lay_name.split('.')[3]
                model._modules[layer_name][layer_indx]._modules['downsample'][int(downs_indx)].weight.data *=state_in[lay_name]

            elif ('downsample' in lay_name and 'bias' in lay_name):
                layer_name = lay_name.split('.')[0]
                layer_indx = int(lay_name.split('.')[1])
                conv_name = lay_name.split('.')[2]
                downs_indx = lay_name.split('.')[3]
                model._modules[layer_name][layer_indx]._modules['downsample'][int(downs_indx)].bias.data *=state_in[lay_name]

            elif ('fc' in lay_name and 'weight' in lay_name):
                tmp_name = lay_name.split('.')[0]
                model._modules[tmp_name].weight.data *= state_in[lay_name]

    return None
def count_parameters(model):
    ss = 0
    for p in model.parameters():
        if p.requires_grad:
            ss+=p.numel()
    return ss


def model_select(nnmodel,device,s_input_channel,n_classes):
    if nnmodel=='alexnet':
        model = alexnet.NetOrg(s_input_channel,n_classes).to(device)
    elif nnmodel=='lenet':
        model = lenet.NetOrg(s_input_channel,n_classes).to(device)    
    elif nnmodel=='googlenet':
        model = googlenet.NetOrg(s_input_channel,n_classes).to(device)       
        network_size = googlenet.networksize()
    elif nnmodel == 'resnext50':
        resnext_type = 'resnext50'
        model = resnext.ResNext(s_input_channel,n_classes,resnext_type).to(device) 
    elif nnmodel == 'resnet18':
        resnet_type = 'resnet18'
        model = resnet.resnet18(s_input_channel,n_classes).to(device) 
    elif nnmodel == 'resnet34':
        resnet_type = 'resnet34'
        model = resnet.resnet34(s_input_channel,n_classes).to(device) 
    elif nnmodel == 'resnet50':
        resnet_type = 'resnet50'
        model = resnet.resnet50(s_input_channel,n_classes).to(device)         
    elif nnmodel == 'resnet101':
        resnet_type = 'resnet101'
        model = resnet.resnet101(s_input_channel,n_classes).to(device)         
    elif nnmodel == 'resnet152':
        resnet_type = 'resnet152'
        model = resnet.resnet152(s_input_channel,n_classes).to(device)         
    elif nnmodel == 'resnext50_32x4d':
        resnet_type = 'resnext50_32x4d'
        model = resnet.resnext50_32x4d(s_input_channel,n_classes).to(device) 
    else:
        raise TypeError("Check the model name")
    return model


def gradient_masker(model,best_stateI,nnmodel,device):
    if nnmodel=='alexnet':
        alexnet.gradient_mask(model,best_stateI,device,nnmodel)
    elif nnmodel=='lenet':
        lenet.gradient_mask(model,best_stateI,device,nnmodel)
    elif nnmodel=='googlenet':
        googlenet.gradient_mask(model,best_stateI,device,nnmodel)
    elif nnmodel == 'resnext50':
        resnext50.gradient_mask(model,best_stateI,device,nnmodel)
    elif nnmodel == 'resnet20':
        resnet.gradient_mask(model,best_stateI,device,nnmodel)
    elif nnmodel == 'resnet18':
        resnet.gradient_mask(model,best_stateI,device,nnmodel)
    elif nnmodel == 'resnet34':
        resnet.gradient_mask(model,best_stateI,device,nnmodel)
    elif nnmodel == 'resnet50':
        resnet.gradient_mask(model,best_stateI,device,nnmodel)
    elif nnmodel == 'resnet101':
        resnet.gradient_mask(model,best_stateI,device,nnmodel)
    else:
        raise TypeError("Check the model name")

def key_maker(model):
    network_size = {} # a dictionary of base layer and corresponding states index; first list is the states list and second the connected to list.
    d_counter = 0
    fc_counter = 0
    conv_counter = 0
    shape_tracker = {}
    layer_id = 0
    layer_names_wgrad = []
    # Filter each layer's base
    for name, p in model.named_parameters():
        if p.requires_grad:
            names = name.split('.')
            layer = names[0]
            if layer=='conv1': # capture conv1
                layer_id+=1
                network_size[name.split('.weight')[0]] = [[[layer_id],[np.arange(d_counter,d_counter+p.shape[0])],[]],[[name,p.shape]]]
                layer_names_wgrad.append(name)
                d_counter+=p.shape[0]  
                conv_counter+=p.shape[0]  
                shape_tracker[layer_id]=p.shape[0]
            elif 'layer' in layer: # capture layer
                layer_indx = names[1] # layer1.1.conv1.weight: gets 1
                layer_type = names[2] # bn/conv,downsample
                if 'conv' in layer_type:
                    if names[3] == 'weight':
                        layer_id+=1
                        network_size[name.split('.weight')[0]] = [[[layer_id],[np.arange(d_counter,d_counter+p.shape[0])],[]],[[name,p.shape]]]
                        layer_names_wgrad.append(name)
                        d_counter+=p.shape[0]  
                        conv_counter+=p.shape[0]  
                        shape_tracker[layer_id]=p.shape[0]

                elif 'downsample' in layer_type:
                    if names[4]=='weight' and len(p.shape)>1:
                        layer_id+=1
                        network_size[name.split('downsample')[0]+'downsample'] = [[[layer_id],[np.arange(d_counter,d_counter+p.shape[0])],[]],[[name,p.shape]]]
                        layer_names_wgrad.append(name)
                        d_counter+=p.shape[0]  
                        conv_counter+=p.shape[0]  
                        shape_tracker[layer_id]=p.shape[0]

            elif layer=='fc':
                if names[1] == 'weight':
                    layer_id+=1
                    network_size[name.split('.weight')[0]] = [[[layer_id],[np.arange(d_counter,d_counter+p.shape[1])],[np.arange(d_counter+p.shape[1],d_counter+p.shape[1]+n_classes)]],[[name,p.shape]]]
                    layer_names_wgrad.append(name)
                    d_counter+=p.shape[1]  
                    fc_counter+=p.shape[1] 
                    shape_tracker[layer_id]=p.shape[1]


    # Filter each layer's bias and bn
    for name, p in model.named_parameters():
        if p.requires_grad:
            names = name.split('.')
            layer = names[0]
            if 'bn' in layer: # capture conv1
                layer_code = [i for i in layer if i.isdigit()]
                network_size['conv'+layer_code[0]][1].append([name,p.shape])
                layer_names_wgrad.append([name,p.shape])

            elif 'layer' in layer:
                if 'bn' in names[2]:
                    layer_code = [i for i in names[2] if i.isdigit()][0]
                    network_size[name.split('bn')[0]+'conv'+layer_code][1].append([name,p.shape])
                    layer_names_wgrad.append([name,p.shape])

                elif 'downsample' in names[2] and (len(p.shape)==1):
                    layer_code = names[3]
                    network_size[name.split('downsample')[0]+'downsample'][1].append([name,p.shape])
                    layer_names_wgrad.append([name,p.shape])

            elif layer=='fc' and names[1]=='bias':
                network_size['fc'][1].append([name,p.shape])
                layer_names_wgrad.append([name,p.shape])


    # Add connected to states
    for key in network_size.keys():
        if key!='fc':
            next_layer_id = network_size[key][0][0][0]+1
            length_nextlayer = shape_tracker[next_layer_id]
            s = 1+network_size[key][0][1][0][-1] # start point of next layer nodes id
            network_size[key][0][2] = np.arange(s,s+length_nextlayer)
    d_counter+=n_classes
    fc_counter+=n_classes
    print('Number of states:', d_counter)
    print('Number of fc states:', fc_counter)
    print('Number of conv states:', conv_counter)
    if fc_counter+conv_counter!=d_counter:
        raise 'The Number of states do not match.'
    D = d_counter

    return network_size, D, conv_counter, fc_counter, layer_names_wgrad


def kept_counter(network_size,final_state,model_n_p):
    counter = 0
    total_conv = 0    # no buffer included
    kept_conv = 0
    dropped_fc=0
    dropped_bn=0
    dropped_conv=0
    total_bn = 0
    total_conv = 0
    total_fc = 0
    stride_s = 1
    for key in network_size:
        active_count = np.sum(final_state[0,network_size[key][0][1][0]])
        state_len = len(network_size[key][0][1][0])
        for layer in network_size[key][1]:
            if len(layer[1])==4: #conv
                conv_t = (layer[1][3]*layer[1][2]*layer[1][0]*layer[1][1]*stride_s)
                conv_count=conv_t*active_count/state_len
                total_conv+=conv_t
                dropped_conv+=conv_count
            elif len(layer[1])==1: #bn bias
                bn_t = (layer[1][0])
                bn_count=bn_t*active_count/state_len
                total_bn+= bn_t
                dropped_bn+=bn_count
            elif len(layer[1])==2: #fc 
                fc_t = (layer[1][1]*layer[1][0])
                bias_count=fc_t*active_count/state_len
                total_fc+=fc_t
                dropped_fc+=bias_count

    total_p = total_fc+total_bn+total_conv
    total_kept = dropped_bn+dropped_conv+dropped_fc
    kp = total_kept/total_p

    if total_p!=model_n_p:
        raise 'Total number f parameters does not match.'

    return total_p,total_kept, kp

def weights_initialization(model):
    for m in model.modules():
        print(m)
        if isinstance(m, nn.Conv2d):
            print(type(m))
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.xavier_uniform_(m.bias)

def energyloss(net_energy,target,n_classes,args,device):
    one_hot = torch.zeros(args.batch_size,n_classes).to(device)
    one_hot[torch.arange(args.batch_size),target] = 1
    high_cost_target_one_hot = -10000*one_hot # high cost for target to remove it   [0 0 0 -10000 0 0] 
    net_eneg = torch.sum(torch.mul(one_hot,net_energy),axis=1)
    other_net_eneg = high_cost_target_one_hot + net_energy # mean over batches
    max_other_net_eneg = torch.max(other_net_eneg,1)

    eng_diff = net_eneg - max_other_net_eneg[0]
    elloss = (-1*eng_diff.mean(0))
    return elloss

def train_ising(args,pretrained_name,pretrained_name_save,stopcounter, input_size, n_classes,s_input_channel,nnmodel,threshold_early_state,ts,loss_func,keep_rate_user):

    ### Pytorch Setup
    torch.manual_seed(args.seed)
    torch.cuda.is_available()
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"  
    global device
    device = torch.device(dev) 
    kwargs = {'num_workers': 20, 'pin_memory': True} if use_cuda else {}
    model = model_select(nnmodel,device,s_input_channel,n_classes)

    network_size, D, D_Conv2d, D_fc, layer_names_wgrad = key_maker(model)
    # MOODULE TEST TODO
    weight_backer(model,network_size,layer_names_wgrad)

    ### Load data
    train_loader, valid_loader = dataloader.traindata(kwargs, args, input_size, valid_percentage, dataset)
    n_batches = len(train_loader)

    ### Load pre-trained weights
    if args.pre_trained_f:
        pretrained_weights = torch.load(pretrained_name)
        model.load_state_dict(pretrained_weights)

    ### Optimizer
    if args.optimizer_m == 'Adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr, rho=0.9, eps=1e-06, weight_decay=0.00001)
    elif args.optimizer_m == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.000125,momentum=0.9,nesterov=True)
    if args.scheduler_m == 'StepLR':
        scheduler = StepLR(optimizer, step_size=args.lr_stepsize, gamma=args.gamma,last_epoch=-1)
    ##TODO: add follwing schedulers
    # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    # scheduler = StepLR(optimizer, step_size=args.lr_stepsize, gamma=args.gamma, last_epoch=-1)
    # scheduler = MultiStepLR(optimizer, milestones[50,100,150], gamma=args.gamma, last_epoch=-1)
    # scheduler = ExponentialLR(optimizer, gamma=0.97,last_epoch=-1)

    ## Result collection
    train_loss_collect = []
    train_accuracy_collect = []
    valid_loss_collect = []
    valid_accuracy_collect = []
    valid_accuracy_collect3 = []
    valid_accuracy_collect5 = []
    kept_rate_iter = []
    tr_loss_iter = []
    # Initialization Step
    NS = args.NS
    states_rand = np.asarray(np.random.randint(0,2,(NS,D)),dtype='float64') # Population is 0/1
    states_init = np.asarray(np.ones((NS,D)),dtype='float64') # Population is 0/1
 
    states_init = states_rand

    states_rand_one = np.asarray(np.random.randint(0,2,(1,D)),dtype='float64') # Population is 0/1
    states_rand_one0 = states_rand_one[0,:]
    model_n_p = count_parameters(model)
    ## Starting ....
    state_converged = False
    epoch_best_so_far = []
    collect_cost_best_state = [] # large value of best cost for early masking
    collect_avg_state_cost = []
    kp_collect = []
    lr_collect = []
    valid_energy_collect = []
    best_valid_loss = 10e10


    collect_epoch_avg_cost_pop = []
    collect_epoch_best_cost_pop = []


    collect_avg_state_cost_alll = [] 
    collect_cost_best_state_alll = [] 

    ### Logs
    print(10*'#')
    print('Dimension of state vector: ',D)
    print('Pop size is: ',NS)
    print('Optimizer is: '+str(args.optimizer_m)+' and Scheduler is '+str(args.scheduler_m))
    print('Model: ',nnmodel,'  Device: ',device, '  Input channels: ',s_input_channel,'  N Classes: ',n_classes)
    print(10*'#')
    print('Starting training...')
    print(10*'#')
    etc = 0
    etime = 0
    counter_t = 0
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        counter = 0
        epoch_accuraacy = 0
        epoch_loss = 0
        collect_cost_best_state = [] # large value of best cost for early masking
        collect_avg_state_cost = []

        for batch_idx, (data, target) in enumerate(train_loader):
            start_time = time.time()

            data, target = data.to(device), target.to(device)
            counter_t+=1
            # print(network_size.keys())
            # print(model._modules['layer1'][0]._modules['conv1'].weight.data.shape)
            # dd = model._modules['layer1'][0]._modules['conv1'].weight.data[0,:,:,:]
            # print(dd.shape)
            # f = np.reshape(dd[0,:,:],(1,9))
            # print(f.shape)
            # for j in range(1):
            #     print(np.reshape(np.arange(9),(1,9)),np.reshape(dd[j,:,:],(1,9)))
            #     # mean,std=norm.fit(np.reshape(dd[j,:,:],(1,9)))
            #     # y = norm.pdf(np.arange(9), mean, std)
            #     # plt.plot(np.arange(9), y)
            #     plt.plot(np.reshape(np.arange(9),(1,9)),np.reshape(dd[j,:,:],(1,9)),'.-')
            # plt.show()
            # sys.exit()
            # print('in1')

            if epoch<threshold_early_state and counter_t>=1000 and mode == 'ising':
                if np.mean(collect_cost_best_state[-50:-1]==collect_cost_best_state[-1]):
                    state_converged = True
                    sparsifier(model,network_size, best_state_reshaped[str(0)]) # forever sparsifai it
                    stopcounter=10e31
            elif epoch>=threshold_early_state: 
                if mode == 'random':
                    best_state = states_rand_one0
                    avg_cost_states = 0
                    cost_best_state = 0
                    state_converged = True
                    states = states_rand_one
                elif mode == 'ising':
                    ## Early state convergence
                     state_converged = True
                    sparsifier(model,network_size, best_state_reshaped[str(0)]) # forever sparsifai it
                    stopcounter=10e31

            # Evolution State
            if state_converged==False: # apply early masking
                if batch_idx % 1 == 0:
                    with torch.no_grad():
                        if batch_idx==0 and epoch==1:
                            cost_states, cost_all_one = ising_cost(args,model,data,target,device,states_init,NS,D, D_Conv2d, D_fc,network_size,model_n_p,keep_rate_user)        
                            states = states_init
                            best_state = np.asarray(states[np.argmin(cost_states),:])
                        new_states = evolution(states,NS,D,best_state)
                        cost_new_states, cost_all_one = ising_cost(args,model,data,target,device,new_states,NS,D, D_Conv2d, D_fc,network_size,model_n_p,keep_rate_user)    
                        states, cost_states, best_state, cost_best_state, avg_cost_states = selection(states, new_states, cost_states, cost_new_states, NS, D)


            ## Collect evolution cost
            collect_avg_state_cost.append(avg_cost_states)
            collect_cost_best_state.append(cost_best_state)
            if (state_converged==False):
                best_stateI = best_state
                best_state2 = np.expand_dims(best_stateI,axis=0)
                best_state_reshaped = state_reshaper(device,1,best_state2,network_size,args,layer_names_wgrad)
                # Backup weights
                backup_weights = weight_backer(model,network_size,layer_names_wgrad)
                # Sparsifiy the network
                sparsifier(model,network_size, best_state_reshaped[str(0)])
            # Get network Energy with applied states
            optimizer.zero_grad()

            output, _ = model(data) 
            # if loss_func=='ce':
            #     criterion = nn.CrossEntropyLoss()
            #     loss = criterion(output, target)
            #     pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability       
            #     correct = pred.eq(target.view_as(pred)).sum().item()

            # loss = energyloss(net_eng,target,n_classes,args,device)
            ###################


            if loss_func=='ce':
                #### with cross entropy
                criterion = nn.CrossEntropyLoss()
                loss = criterion(output, target)
                _, pred = output.max(1)
                correct = pred.eq(target.view_as(pred)).sum().item()
            elif loss_func=='log':
                ### With log
                output = F.log_softmax(output)
                loss = F.nll_loss(output, target, reduction='sum')  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct = pred.eq(target.view_as(pred)).sum().item()

            acc = 100.*(correct/np.float(args.batch_size))
            epoch_loss+=loss.item()
            epoch_accuraacy+=acc
            # Collect kept rate at iteration level
            final_state = np.expand_dims(best_state, axis=0)
            final_state[0,int(D_Conv2d+D_fc-n_classes):] = 1 # output layer must be one
            total_p,total_kept, kp = kept_counter(network_size, final_state,model_n_p)
            kept_rate_iter.append(kp)
            tr_loss_iter.append(loss)
            if batch_idx % args.log_interval == 0:
                print(nnmodel,dataset)
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
                print('Accuracy: ', acc)
                print('Epoch: ',epoch)
                print('Population min cost: ',cost_best_state)
                print('Population avg cost: ',avg_cost_states)
                print('Learning rate: ',scheduler.get_lr()[0])
                print(np.sum(best_state),len(best_state))
                print('state',state_converged)
                final_state = np.expand_dims(best_state, axis=0)
                final_state[0,int(D_Conv2d+D_fc-n_classes):] = 1 # output layer must be one
                print('Number of trainable parameters: ', model_n_p)
                total_p,total_kept, kp = kept_counter(network_size, final_state,model_n_p)
                print('Total parameters: ',total_p)
                print('Total kept parameters: ',total_kept)
                print('Kept Percentile:', kp)
                print(10*'*')

            ''' backup weights -> Sparsifai the network -> compute loss -> compute gradients -> mask the gradients -> restore weights -> apply the masked gradients -> clear gradients: So only the unmasked weights are updated.'''
            print(time.time()-start_time)

            loss.backward() # computes gradients for ones with reques_grad=True
            gradient_masker(model,best_stateI,nnmodel,device)
            if state_converged==False:
                weight_restore(model, network_size, backup_weights)
            # ddf = model._modules['conv1'].weight.data

            optimizer.step()

            collect_avg_state_cost_alll.append(avg_cost_states)
            collect_cost_best_state_alll.append(cost_best_state)
            # Break
            counter+=1
            if counter>=stopcounter or counter==(n_batches-1):
                # entrop_sig = entrop_sig/counter # Average over batches
                # states_loss = states_loss/counter # Average over batches
                # interaction_sigma = interaction_sigma/counter # Average over batches
                epoch_loss = epoch_loss/counter
                epoch_accuracy = epoch_accuraacy/counter
                states_reshaped = state_reshaper(device,NS,states,network_size,args,layer_names_wgrad)

                break

        etime+= time.time()-start_time 
        etc+=1
        print('time:',etime/etc)
        # Validation
        print('Validation at epoch %d is:'%(epoch))
        best_state3 = np.expand_dims(best_state,axis=0)
        best_state_reshaped = state_reshaper(device,1,best_state3,network_size,args,layer_names_wgrad)
        valid_loss, valid_accuracy,valid_accuracy3,valid_accuracy5, valid_energy = test_c(args, model, network_size, device, valid_loader, best_state_reshaped[str(0)],loss_func, isvalid=True)
        # _,_,_,_,_, kp_valid = kept_counter(network_size, best_state3,model_n_p)
        total_p,total_kept, kp_valid = kept_counter(network_size, final_state,model_n_p)
        print('Total parameters: ',total_p)
        print('Total kept parameters: ',total_kept)
        print('Kept Percentile:', kp)

        scheduler.step()

        ## Result collection
        train_loss_collect.append(epoch_loss)
        train_accuracy_collect.append(epoch_accuracy)
        valid_loss_collect.append(valid_loss)
        valid_accuracy_collect.append(valid_accuracy)
        valid_accuracy_collect3.append(valid_accuracy3)
        valid_accuracy_collect5.append(valid_accuracy5)
        kp_collect.append(kp_valid)
        lr_collect.append(scheduler.get_lr()[0])
        epoch_best_so_far.append(cost_best_state) # cost best state at current epoch
        valid_energy_collect.append(valid_energy.data.cpu().numpy())
        ## Early Stopping weight save
        # if epoch>1 and scheduler.get_lr()[0]<0.0001:
        # if epoch>50: # give time to collect valid results
        # # # ### checkpoint weights
            # if valid_loss<=best_valid_loss: # valid_loss_collect[-2]>valid_loss_collect[-1] and valid_accuracy_collect[-1]>valid_accuracy_collect[-2]:
            #     torch.save(model.state_dict(), pretrained_name_save)
            #     best_valid_loss = valid_loss
            #     print('bestvalid',best_valid_loss)
        # if np.mean(valid_loss_collect[-5:-1])<=valid_loss and np.mean(valid_accuracy_collect[-5:-1])>=valid_accuracy:
        #     print('early stopping',np.mean(valid_loss_collect[-5:-1]),valid_loss, np.mean(valid_accuracy_collect[-5:-1]),valid_accuracy)
        #     break
        print(len(collect_avg_state_cost))
        collect_epoch_avg_cost_pop = np.mean(np.reshape(collect_avg_state_cost, (-1, counter)),axis=1)
        collect_epoch_best_cost_pop = np.mean(np.reshape(collect_cost_best_state, (-1, counter)),axis=1)


    ## Saving results


    # np.savetxt(test_name,[test_loss,test_accuracy]) # Test results [test loss, test accuracy in percentage]

    ############## Test the model on test data ############## 
    ##
    # Load checkpoint
    # pretrained_weights = torch.load(pretrained_name_save)
    # model.load_state_dict(pretrained_weights)

    print(30*'*')
    print('Energy Prouning results')
    final_state = np.expand_dims(best_state, axis=0)
    final_state[0,int(D_Conv2d+D_fc-n_classes):] = 1 # output layer must be one

    print('Ising state ',final_state)
    states_reshaped = state_reshaper(device,1,final_state,network_size,args,layer_names_wgrad)
    test_loader = dataloader.testdata(kwargs,args,input_size,dataset)
    test_loss,test_accuracy,test_accuracy3,test_accuracy5, valid_energy = test_c(args, model, network_size, device, test_loader, states_reshaped[str(0)],loss_func,isvalid=False)
    # total_p,kept_conv,total_conv,kept_fc,total_fc, kp = kept_counter(network_size, final_state)
    kept_conv=0
    total_conv=0
    kept_fc=0
    total_fc=0
    total_p,total_kept, kp = kept_counter(network_size, final_state,model_n_p)
    print('Total parameters: ',total_p)
    print('Total kept parameters: ',total_kept)
    print('Kept Percentile:', kp)

    report = ['TestLoss: '+str(test_loss),'TestAcc: '+ str(test_accuracy),'TestAcc3: '+str(test_accuracy3),'TestAcc5: '+str(test_accuracy5),'Total states: '+str(total_p), 'KeptConv: '+str(kept_conv),'TotalConv: '+str(total_conv),'KeptFC: '+str(kept_fc),'TotalFC: '+str(total_fc), 'TotalKept: '+str(kp), 'TestEnergy: '+str(valid_energy)]

     ## Save model weights   
    # if args.save_model:
    #     torch.save(model.state_dict(), pretrained_name_save)


    ##
    print(30*'*')
    print('Energy Dropout But No Pruning results')
    final_state = np.asarray(int(D_Conv2d+D_fc)*[1]) # output layer must be one
    final_state = np.expand_dims(final_state, axis=0)
    print(final_state.shape)
    states_reshaped = state_reshaper(device,1,final_state,network_size,args,layer_names_wgrad)
    test_loss,test_accuracy,test_accuracy3,test_accuracy5, valid_energy = test_c(args, model, network_size, device, test_loader,states_reshaped['0'],loss_func,isvalid=False)
    total_p,total_kept, kp = kept_counter(network_size, final_state,model_n_p)
    report.extend(['Energy Dropout But No Pruning results','TestLoss: '+str(test_loss),'TestAcc: '+ str(test_accuracy),'TestAcc3: '+str(test_accuracy3),'TestAcc5: '+str(test_accuracy5),'Total states: '+str(total_p), 'KeptConv: '+str(kept_conv),'TotalConv: '+str(total_conv),'KeptFC: '+str(kept_fc),'TotalFC: '+str(total_fc), 'TotalKept: '+str(kp), 'TestEnergy: '+str(valid_energy)])
    print('Total parameters: ',total_p)
    print('Total kept parameters: ',total_kept)
    print('Kept Percentile:', kp)
    ##
    print(30*'*')
    print('Random Pruning @50 results')

    D2=int(np.ceil(D/2))
    final_state = np.hstack([np.zeros((1,D2),dtype='float64'), np.ones((1,D-D2),dtype='float64')])[0]
    np.random.shuffle(final_state)
    final_state[int(D_Conv2d+D_fc-n_classes):] = 1 # output layer must be one
    final_state = np.expand_dims(final_state, axis=0)

    states_reshaped = state_reshaper(device,1,final_state,network_size,args,layer_names_wgrad)
    test_loss,test_accuracy,test_accuracy3,test_accuracy5, valid_energy = test_c(args, model, network_size, device, test_loader,states_reshaped['0'],loss_func,isvalid=False)
    # total_p,kept_conv,total_conv,kept_fc,total_fc, kpRD = kept_counter(network_size, final_state)
    total_p,total_kept, kp = kept_counter(network_size, final_state,model_n_p)
    print('Total parameters: ',total_p)
    print('Total kept parameters: ',total_kept)
    print('Kept Percentile:', kp)


    print('Number of trainable parameters: ', count_parameters(model))
    report.extend(['Energy Dropout With Random Pruning @50 pruned results','TestLoss: '+str(test_loss),'TestAcc: '+ str(test_accuracy),'TestAcc3: '+str(test_accuracy3),'TestAcc5: '+str(test_accuracy5),'Total states: '+str(total_p), 'KeptConv: '+str(kept_conv),'TotalConv: '+str(total_conv),'KeptFC: '+str(kept_fc),'TotalFC: '+str(total_fc), 'TotalKept: '+str(kp), 'TestEnergy: '+str(valid_energy)])
    
    print(30*'*')
    print('Random Pruning @25 results')

    D2=int(np.ceil(D/4))
    final_state = np.hstack([np.zeros((1,D2),dtype='float64'), np.ones((1,D-D2),dtype='float64')])[0]
    np.random.shuffle(final_state)
    final_state[int(D_Conv2d+D_fc-n_classes):] = 1 # output layer must be one
    final_state = np.expand_dims(final_state, axis=0)

    states_reshaped = state_reshaper(device,1,final_state,network_size,args,layer_names_wgrad)
    test_loss,test_accuracy,test_accuracy3,test_accuracy5, valid_energy = test_c(args, model, network_size, device, test_loader,states_reshaped['0'],loss_func,isvalid=False)
    # total_p,kept_conv,total_conv,kept_fc,total_fc, kpRD = kept_counter(network_size, final_state)
    total_p,total_kept, kp = kept_counter(network_size, final_state,model_n_p)
    print('Total parameters: ',total_p)
    print('Total kept parameters: ',total_kept)
    print('Kept Percentile:', kp)


    print('Number of trainable parameters: ', count_parameters(model))
    report.extend(['Energy Dropout With Random Pruning @25 pruned results','TestLoss: '+str(test_loss),'TestAcc: '+ str(test_accuracy),'TestAcc3: '+str(test_accuracy3),'TestAcc5: '+str(test_accuracy5),'Total states: '+str(total_p), 'KeptConv: '+str(kept_conv),'TotalConv: '+str(total_conv),'KeptFC: '+str(kept_fc),'TotalFC: '+str(total_fc), 'TotalKept: '+str(kp), 'TestEnergy: '+str(valid_energy)])
    
    print(30*'*')
    print('Random Pruning @75 results')

    D2=int(np.ceil(3*D/4))
    final_state = np.hstack([np.zeros((1,D2),dtype='float64'), np.ones((1,D-D2),dtype='float64')])[0]
    np.random.shuffle(final_state)
    final_state[int(D_Conv2d+D_fc-n_classes):] = 1 # output layer must be one
    final_state = np.expand_dims(final_state, axis=0)

    states_reshaped = state_reshaper(device,1,final_state,network_size,args,layer_names_wgrad)
    test_loss,test_accuracy,test_accuracy3,test_accuracy5, valid_energy = test_c(args, model, network_size, device, test_loader,states_reshaped['0'],loss_func,isvalid=False)
    # total_p,kept_conv,total_conv,kept_fc,total_fc, kpRD = kept_counter(network_size, final_state)
    total_p,total_kept, kpRD = kept_counter(network_size, final_state,model_n_p)
    print('Total parameters: ',total_p)
    print('Total kept parameters: ',total_kept)
    print('Kept Percentile:', kp)


    print('Number of trainable parameters: ', count_parameters(model))

    report.extend(['Energy Dropout With Random Pruning @75 pruned results','TestLoss: '+str(test_loss),'TestAcc: '+ str(test_accuracy),'TestAcc3: '+str(test_accuracy3),'TestAcc5: '+str(test_accuracy5),'Total states: '+str(total_p), 'KeptConv: '+str(kept_conv),'TotalConv: '+str(total_conv),'KeptFC: '+str(kept_fc),'TotalFC: '+str(total_fc), 'TotalKept: '+str(kp), 'TestEnergy: '+str(valid_energy)])
    ts = int(time.time())
    test_name = './results/test_'+nnmodel+'_'+dataset+'_'+mode+'_'+str(ts)+'.txt'

    report.extend(['Number of trainable parameters: '+str(count_parameters(model))])
    np.savetxt(test_name, report, delimiter="\n", fmt="%s") # Test results [test loss, test accuracy in percentage]

    results = np.vstack((train_loss_collect,train_accuracy_collect,valid_loss_collect,valid_accuracy_collect,valid_accuracy_collect3,valid_accuracy_collect5,kp_collect,lr_collect)) # stack in order vertically
    results_evolutionary = np.vstack((collect_avg_state_cost_alll,collect_cost_best_state_alll,kept_rate_iter)) # stack in order vertically

    res_name = './results/loss_'+nnmodel+'_'+dataset+'_'+mode+'_'+str(ts)+'.txt'
    res_name_evol = './results/evolutionaryCost_'+nnmodel+'_'+dataset+'_'+mode+'_'+str(ts)+'.txt'
    print(test_name)
    if not os.path.exists('./results'):
        os.makedirs('./results')
    np.savetxt(res_name,results)
    np.savetxt(res_name_evol,results_evolutionary)
    ## Testing
    return None



def train(args,pretrained_name,pretrained_name_save,stopcounter, input_size, n_classes,s_input_channel,nnmodel,ts):

    # Pytorch Setup
    torch.manual_seed(args.seed)
    torch.cuda.is_available()
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"  
    global device
    device = torch.device(dev) 
    kwargs = {'num_workers': 20, 'pin_memory': True} if use_cuda else {}
    print(nnmodel,device,s_input_channel,n_classes)
    model = model_select(nnmodel,device,s_input_channel,n_classes)
    # weights_init = weights_initialization(model)
    print('################  Model Setup  ################')
    # load data and weights
    train_loader, valid_loader = dataloader.traindata(kwargs, args, input_size, valid_percentage, dataset)

    ## Optimizer
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr, rho=0.9, eps=1e-06, weight_decay=0.000125)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.000125,momentum=0.9,nesterov=True)
    # scheduler = ExponentialLR(optimizer, gamma=0.97,last_epoch=-1)

    # scheduler = MultiStepLR(optimizer, milestones=[2.,4.,6.], gamma=0.1, last_epoch=-1)
    scheduler = StepLR(optimizer, step_size=args.lr_stepsize, gamma=args.gamma,last_epoch=-1)

    ## Result collection
    train_loss_collect = []
    train_accuracy_collect = []
    valid_loss_collect = []
    valid_accuracy_collect = []
    valid_accuracy_collect3 = []
    valid_accuracy_collect5 = []
    lr_collect= []
    # Initialization Step
    etc = 0
    etime = 0
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        n_batches = len(train_loader)
        epoch_accuracy = 0
        epoch_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)


            optimizer.zero_grad()
            output, logits = model(data) 
            # Compute Loss
            criterion = nn.CrossEntropyLoss()
            loss = criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability       
            correct = pred.eq(target.view_as(pred)).sum().item()
            loss.backward()
            optimizer.step()

            acc = 100*(correct/np.float(args.batch_size))
            epoch_loss+=loss.item()
            epoch_accuracy+=acc
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
                print('Accuracy: ', acc)
                print('Epoch: ',epoch)
                print('Learning rate: ',scheduler.get_lr())
                print(10*'*')


            # Break
            if batch_idx==stopcounter:
                print('Broke at batch indx ',batch_idx)
                break
        epoch_loss = epoch_loss/(batch_idx+1)
        epoch_accuracy = epoch_accuracy/(batch_idx+1)

        etime+= time.time()-start_time 
        etc+=1
        print('time:',etime/etc)
        # Validation
        print('Validation at epoch %d is:'%(epoch))

        valid_loss, valid_accuracy,valid_accuracy3,valid_accuracy5 = test(args, model, device, valid_loader,criterion,isvalid=True)
        scheduler.step()

        ## Result collection
        train_loss_collect.append(epoch_loss)
        train_accuracy_collect.append(epoch_accuracy)
        valid_loss_collect.append(valid_loss)
        valid_accuracy_collect.append(valid_accuracy)
        valid_accuracy_collect3.append(valid_accuracy3)
        valid_accuracy_collect5.append(valid_accuracy5)
        lr_collect.append(scheduler.get_lr()[0])

        ## Early Stopping
        if epoch>5: # give time to collect valid results
        # #     # checkpoint weights
            if valid_loss_collect[-2]>valid_loss_collect[-1] and valid_accuracy_collect[-1]>valid_accuracy_collect[-2]:
                torch.save(model.state_dict(), pretrained_name_save)
        #     if np.mean(valid_loss_collect[-5:-1])<valid_loss and np.mean(valid_accuracy_collect[-5:-1])>valid_accuracy:
        #         print('early stopping',np.mean(valid_loss_collect[-5:-1]),valid_loss, np.mean(valid_accuracy_collect[-5:-1]),valid_accuracy)
        #         break



    ############## Test the model on test data ############## 
    # # Load checkpoint
    pretrained_weights = torch.load(pretrained_name_save)
    model.load_state_dict(pretrained_weights)
    ## Loading data
    test_loader = dataloader.testdata(kwargs,args,input_size,dataset)
    test_loss,test_accuracy,test_accuracy3,test_accuracy5 = test(args, model, device, test_loader,criterion,isvalid=False)
    ## Testing
    test_name = './results/test_'+nnmodel+'_'+dataset+'_'+mode+'_'+str(ts)+'.txt'
    # np.savetxt(test_name,[test_loss,test_accuracy]) # Test results [test loss, test accuracy in percentage]
    # print('Number of trainable parameters: ', count_parameters(model))
    report = ['TestLoss: '+str(test_loss),'TestAcc: '+ str(test_accuracy),'TestAcc3: '+str(test_accuracy3),'TestAcc5: '+str(test_accuracy5)]

    ## Saving results
    results = np.vstack((train_loss_collect,train_accuracy_collect,valid_loss_collect,valid_accuracy_collect,valid_accuracy_collect3,valid_accuracy_collect5,lr_collect)) # stack in order vertically
    res_name = './results/loss_'+nnmodel+'_'+dataset+'_'+mode+'_'+str(ts)+'.txt'
    if not os.path.exists('./results'):
        os.makedirs('./results')
    np.savetxt(res_name, results)

    report.extend(['Number of trainable parameters: '+str(count_parameters(model))])
    np.savetxt(test_name, report, delimiter="\n", fmt="%s") # Test results [test loss, test accuracy in percentage]
    print(test_name)
    ## Save model weights
    if args.save_model:
        torch.save(model.state_dict(), pretrained_name_save)

    return None


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N', help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1., metavar='LR', help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.1, metavar='M', help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--lr_stepsize', type=float, default=50, metavar='M', help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=20, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--save-ls', action='store_true', default=False, help='For Saving the current Model')
    parser.add_argument('--save_model', action='store_true', default=True, help='For Saving the current Model')
    parser.add_argument('--NS', default=32, help='pop size')
    parser.add_argument('--stopcounter', default=10, help='stopcounter')
    parser.add_argument('--threshold_early_state', default=5, help='threshold_early_state')
    parser.add_argument('--pre_trained_f', default=False, help='load pretrained weights')
    parser.add_argument('--scheduler_m', default='StepLR', help='lr scheduler')
    parser.add_argument('--optimizer_m', default="Adadelta", help='optimizer model')
    parser.add_argument('--nnmodel', default="resnet34", help='original neural net to prune')
    parser.add_argument('--dataset', default="flowers", help='dataset')


    args = parser.parse_args()
    argparse_dict = vars(args)
    loss_func = 'ce'

    global dataset, nnmodel
    global network_size
    global n_classes
    global s_input_channel

    # for ds in ['kuzushiji','flowers','cifar10','cifar100','fashion']:
    dataset = args.dataset #kuzushiji
    nnmodel = args.nnmodel
    mode = 'ising'
    keep_rate_user = 0.1
    # limiteddata = False # @1000 samples

    stopcounter = args.stopcounter #10#e10
    NS = args.NS # number of candidate states
    n_classes, s_input_channel = dataloader.dataset_specs(dataset)
    input_size = (s_input_channel,32,32)
    valid_percentage = 0.1 # out of 1

    use_cuda = False # not args.no_cuda and torch.cuda.is_available()
    ts = int(time.time())
    pretrained_name_save = "./weights/"+dataset+"_"+mode+"_"+nnmodel+"_"+str(ts)+".pt"
    pretrained_name = 'None' #./weights/cifar10_simple_resnet18_1587500809.pt"
        

    torch.cuda.set_device(0)

    if mode=='ising':
        threshold_early_state = args.threshold_early_state
        train_ising(args,pretrained_name,pretrained_name_save,stopcounter,input_size, n_classes,s_input_channel,nnmodel,threshold_early_state,ts,loss_func,keep_rate_user)
        print(mode,dataset,nnmodel,args.lr,args.epochs,args.gamma,args.batch_size,threshold_early_state, args.NS)
    elif mode=='simple':
        train(args,pretrained_name,pretrained_name_save,stopcounter,input_size, n_classes,s_input_channel,nnmodel,ts)
        print(mode,dataset,nnmodel,args.lr,args.epochs,args.gamma,args.batch_size)
    elif mode=='random':
        threshold_early_state = 0
        args.NS = 1
        train_ising(args,pretrained_name,pretrained_name_save,stopcounter,input_size, n_classes,s_input_channel,nnmodel,threshold_early_state,ts,loss_func)
        print(mode,dataset,nnmodel,args.lr,args.epochs,args.gamma,args.batch_size)

    ## Save args
    argparse_dict_name = './results/args_'+nnmodel+'_'+dataset+'_'+mode+'_'+str(ts)+'.json'
    with open(argparse_dict_name, 'w') as fp:
        json.dump(argparse_dict, fp)
