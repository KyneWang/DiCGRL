import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch import autograd
import numpy as np


def estimate_fisher(model, train_loader, num_iters_per_epoch, sample_size, batch_size=128):
    # Get loglikelihoods from data
    params = [param for param in model.parameters()]
    F_accum = []
    for v, _ in enumerate(params):
        F_accum.append(torch.Tensor(np.zeros(list(params[v].size()))).cuda())
    loglikelihoods = []

    for iters in range(num_iters_per_epoch):
        # print(x.size(), y.size())
        batch_triples, batch_labels = train_loader.get_iteration_batch(iters)

        batch_triples = Variable(torch.LongTensor(batch_triples)).cuda()
        batch_labels = Variable(torch.FloatTensor(batch_labels)).cuda()

        pred, _ = model.test(batch_triples)

        loglikelihoods.append(F.logsigmoid(pred))

        if len(loglikelihoods) >= sample_size // batch_size:
            break

        # loglikelihood = torch.cat(loglikelihoods).mean(0)
        loglikelihood = torch.cat(loglikelihoods).mean(0)

        loglikelihood_grads = autograd.grad(loglikelihood, model.parameters(), retain_graph=True)
        #print("FINISHED GRADING", len(loglikelihood_grads), loglikelihood_grads)
        #print(len(model.F_accum))
        for v in range(len(F_accum)):
            #print("v", v, F_accum[v], loglikelihood_grads[v])
            F_accum[v] = torch.add(F_accum[v], torch.pow(loglikelihood_grads[v], 2).data)
            #print("v_2", v, F_accum[v])

    for v in range(len(F_accum)):
        F_accum[v] /= sample_size
        #print(model.F_accum[v])

    parameter_names = [
        n.replace('.', '__') for n, p in model.named_parameters()
    ]
    # print("RETURNING", len(parameter_names))

    return {n: g for n, g in zip(parameter_names, F_accum)}


def consolidate(model, fisher):
    for n, p in model.named_parameters():
        n = n.replace('.', '__')
        #print("named_parameter:", n, p.data, fisher[n].data)
        model.register_buffer('{}_estimated_mean'.format(n), p.data.clone())
        # print(dir(fisher[n].data))
        model.register_buffer('{}_estimated_fisher'.format(n), fisher[n].data)

    return model

def cal_ewc_loss(model, lamda):
    try:
        losses = []
        for n, p in model.named_parameters():
            # retrieve the consolidated mean and fisher information.
            n = n.replace('.', '__')
            mean = getattr(model, '{}_estimated_mean'.format(n))
            fisher = getattr(model, '{}_estimated_fisher'.format(n))
            # wrap mean and fisher in Vs.
            mean = Variable(mean)
            fisher = Variable(fisher.data)
            # calculate a ewc loss. (assumes the parameter's prior as
            # gaussian distribution with the estimated mean and the
            # estimated cramer-rao lower bound variance, which is
            # equivalent to the inverse of fisher information)
            losses.append((fisher * (p - mean) ** 2).sum())
        #print("normal", (lamda / 2) * sum(losses))
        return (lamda / 2) * sum(losses)
    except AttributeError:
        # ewc loss is 0 if there's no consolidated parameters.
        return (
            Variable(torch.zeros(1)).cuda()
        )

