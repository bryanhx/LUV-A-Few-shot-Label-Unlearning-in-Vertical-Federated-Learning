import os
import sys
import time

import numpy as np
import torch


def _iterative_unlearn_impl(unlearn_iter_func):
    def _wrapped(data_loaders, bottom_model_A, bottom_model_B, top_model, criterion, args):
        decreasing_lr = [91, 136]
        bottom_model_A_optimizer = torch.optim.SGD(
                                    bottom_model_A.parameters(), 
                                    args.unlearn_lr, 
                                    momentum=args.momentum, 
                                    weight_decay=args.weight_decay)
            
        bottom_model_B_optimizer = torch.optim.SGD(
                                    bottom_model_B.parameters(),
                                    args.unlearn_lr, 
                                    momentum=args.momentum, 
                                    weight_decay=args.weight_decay)
            
        bottom_model_A_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                                    bottom_model_A_optimizer, 
                                    milestones=decreasing_lr, 
                                    gamma=0.1)
        
        
        bottom_model_B_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                                    bottom_model_B_optimizer, 
                                    milestones=decreasing_lr, 
                                    gamma=0.1) 
        
        top_model_optimizer = torch.optim.SGD(
                        top_model.parameters(),
                        args.unlearn_lr, 
                        momentum=args.momentum, 
                        weight_decay=args.weight_decay)
        top_model_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                        top_model_optimizer, 
                        milestones=decreasing_lr, 
                        gamma=0.1) 
        
        for epoch in range(args.unlearn_epochs):
            print(
                "Epoch #{}, Bottom model A Learning rate: {}".format(
                    epoch, bottom_model_A_optimizer.state_dict()["param_groups"][0]["lr"]
                )
            )
            print(
                "Epoch #{}, Bottom model B Learning rate: {}".format(
                    epoch, bottom_model_B_optimizer.state_dict()["param_groups"][0]["lr"]
                )
            )
            print(
                "Epoch #{}, Top model Learning rate: {}".format(
                    epoch, top_model_optimizer.state_dict()["param_groups"][0]["lr"]
                )
            )
            train_acc = unlearn_iter_func(
                data_loaders, bottom_model_A, bottom_model_B, top_model, criterion, bottom_model_A_optimizer, bottom_model_B_optimizer, top_model_optimizer, epoch, args)
            bottom_model_A_scheduler.step()
            bottom_model_B_scheduler.step()
            top_model_scheduler.step()

    return _wrapped


def iterative_unlearn(func):
    """usage:

    @iterative_unlearn

    def func(data_loaders, model, criterion, optimizer, epoch, args)"""
    return _iterative_unlearn_impl(func)