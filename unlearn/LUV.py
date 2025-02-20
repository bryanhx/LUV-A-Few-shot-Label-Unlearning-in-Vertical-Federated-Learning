import torch
from torch import nn
import math
import numpy as np
from .impl import iterative_unlearn

@iterative_unlearn
def LUV(LUV_trainloader, bottom_model_A, bottom_model_B, top_model, criterion, bottom_model_A_optimizer, bottom_model_B_optimizer, top_model_optimizer, epochs, args):
    print("unlearn method : ", args.unlearn_method)
    bottom_model_A.train()
    bottom_model_B.train()
    top_model.train()
    for i, (images, labels) in enumerate(LUV_trainloader):
        if args.data != 'yahoo':
            images = images.cuda()
            labels = labels.cuda()
        if args.data == 'cifar10':
            x_a = images[:, :, :, 0:16]
            x_b = images[:, :, :, 16:32]
        elif args.data == 'mnist':
            x_a = images[:, :, :, 0:14]
            x_b = images[:, :, :, 14:28]
        elif args.data == 'cifar100':
            x_a = images[:, :, :, 0:16]
            x_b = images[:, :, :, 16:32]
        elif args.data == 'mri':
            x_a = images[:, :, :, 0:112]
            x_b = images[:, :, :, 112:224]
        elif args.data == 'yahoo':
            for i in range(len(images)):
                images[i] = images[i].long().cuda()
            labels = labels[0].long().cuda()
            x_b = images[1]
            x_a = images[0]
        else:
            raise ValueError(f'No dataset named {args.data}!')
        top_model_optimizer.zero_grad()

        output_tensor_bottom_model_a = bottom_model_A(x_a)
        output_tensor_bottom_model_b = bottom_model_B(x_b)


        mixup = [0.25, 0.5, 0.75]
        for i in range(output_tensor_bottom_model_a.shape[0]):
            x1_bottom_a = output_tensor_bottom_model_a[i]
            x1_bottom_b = output_tensor_bottom_model_b[i]
            if i == output_tensor_bottom_model_a.shape[0]-2:
                mid_bottom_a = mixup[1]*x1_bottom_a + (1-mixup[1])*x2_bottom_a
                mid_bottom_b = mixup[1]*x1_bottom_b + (1-mixup[1])*x2_bottom_b
                front_mid_bottom_a = mixup[0]*x1_bottom_a + (1-mixup[0])*x2_bottom_a
                front_mid_bottom_b = mixup[0]*x1_bottom_b + (1-mixup[0])*x2_bottom_b
                back_mid_bottom_a = mixup[2]*x1_bottom_a + (1-mixup[2])*x2_bottom_a
                back_mid_bottom_b = mixup[2]*x1_bottom_b + (1-mixup[2])*x2_bottom_b
                temperory_bottom_a = torch.stack((front_mid_bottom_a,mid_bottom_a,back_mid_bottom_a))
                temperory_bottom_b = torch.stack((front_mid_bottom_b,mid_bottom_b,back_mid_bottom_b))
                bottom_a_embedding_exp = torch.cat((bottom_a_embedding_exp,temperory_bottom_a),0)
                bottom_b_embedding_exp = torch.cat((bottom_b_embedding_exp, temperory_bottom_b),0)

            else:
                for x in range(i+1, output_tensor_bottom_model_a.shape[0]):
                    x2_bottom_a = output_tensor_bottom_model_a[x]
                    x2_bottom_b = output_tensor_bottom_model_b[x]
                    mid_bottom_a = mixup[1]*x1_bottom_a + (1-mixup[1])*x2_bottom_a
                    mid_bottom_b = mixup[1]*x1_bottom_b + (1-mixup[1])*x2_bottom_b
                    front_mid_bottom_a = mixup[0]*x1_bottom_a + (1-mixup[0])*x2_bottom_a
                    front_mid_bottom_b = mixup[0]*x1_bottom_b + (1-mixup[0])*x2_bottom_b
                    back_mid_bottom_a = mixup[2]*x1_bottom_a + (1-mixup[2])*x2_bottom_a
                    back_mid_bottom_b = mixup[2]*x1_bottom_b + (1-mixup[2])*x2_bottom_b
                    if x == i+1:
                        if i == 0:
                            bottom_a_embedding_exp = torch.stack((front_mid_bottom_a,mid_bottom_a,back_mid_bottom_a))
                            bottom_b_embedding_exp = torch.stack((front_mid_bottom_b,mid_bottom_b,back_mid_bottom_b))
                        else:
                            temperory_bottom_a = torch.stack((front_mid_bottom_a,mid_bottom_a,back_mid_bottom_a))
                            temperory_bottom_b = torch.stack((front_mid_bottom_b,mid_bottom_b,back_mid_bottom_b))
                            bottom_a_embedding_exp = torch.cat((bottom_a_embedding_exp,temperory_bottom_a),0)
                            bottom_b_embedding_exp = torch.cat((bottom_b_embedding_exp, temperory_bottom_b),0)
                    else:
                        temperory_bottom_a = torch.stack((front_mid_bottom_a,mid_bottom_a,back_mid_bottom_a))
                        temperory_bottom_b = torch.stack((front_mid_bottom_b,mid_bottom_b,back_mid_bottom_b))
                        bottom_a_embedding_exp = torch.cat((bottom_a_embedding_exp,temperory_bottom_a),0)
                        bottom_b_embedding_exp = torch.cat((bottom_b_embedding_exp, temperory_bottom_b),0)

        input_tensor_top_model_a = torch.tensor([], requires_grad=True)
        input_tensor_top_model_a.data = bottom_a_embedding_exp.data
        input_tensor_top_model_b = torch.tensor([], requires_grad=True)
        input_tensor_top_model_b.data = bottom_b_embedding_exp.data

        forward_bottom_a = []
        forward_bottom_b = []

        for y in range(math.ceil(bottom_a_embedding_exp.shape[0]/args.batch_size)):
            if y == 0:
                forward_bottom_a.append(input_tensor_top_model_a[0:args.batch_size])
                forward_bottom_b.append(input_tensor_top_model_b[0:args.batch_size])
            else:
                forward_bottom_a.append(input_tensor_top_model_a[(y*args.batch_size):((y+1)*args.batch_size)])
                forward_bottom_b.append(input_tensor_top_model_b[(y*args.batch_size):((y+1)*args.batch_size)])


        grad_output_bottom_model_a = torch.tensor([])
        grad_output_bottom_model_a = grad_output_bottom_model_a.to(args.device)
        grad_output_bottom_model_b = torch.tensor([])
        grad_output_bottom_model_b = grad_output_bottom_model_b.to(args.device)


        for z in range(len(forward_bottom_a)):
            forward_bottom_a[z] = forward_bottom_a[z].detach()
            forward_bottom_a[z].requires_grad=True
            forward_bottom_b[z] = forward_bottom_b[z].detach()
            forward_bottom_b[z].requires_grad=True
            outputs = top_model(forward_bottom_a[z], forward_bottom_b[z])
            label = torch.full((outputs.shape[0],), args.unlearn_class, dtype=int)
            label = label.to('cuda')
            loss = -criterion(outputs, label)
            loss.backward()
            top_model_optimizer.step()

            grad_output_bottom_model_a = torch.cat((grad_output_bottom_model_a, forward_bottom_a[z].grad),0)
            grad_output_bottom_model_b = torch.cat((grad_output_bottom_model_b, forward_bottom_b[z].grad),0)


        n = math.ceil(grad_output_bottom_model_a.shape[0]/output_tensor_bottom_model_a.shape[0])
        grad_a = torch.tensor([])
        grad_a = grad_a.to(args.device)
        for m in range(output_tensor_bottom_model_a.shape[0]):
            if m == 0:
                grad_a = torch.cat((grad_a,(torch.mean(grad_output_bottom_model_a[m:n], axis=0)).unsqueeze(0)), 0)
            else:
                grad_a = torch.cat((grad_a,(torch.mean(grad_output_bottom_model_a[(m*n):((m+1)*n)], axis=0)).unsqueeze(0)), 0)
        
        loss_bottom_A = torch.sum(grad_a * output_tensor_bottom_model_a)
        bottom_model_A_optimizer.zero_grad()
        loss_bottom_A.backward()
        bottom_model_A_optimizer.step()

        p = math.ceil(grad_output_bottom_model_b.shape[0]/output_tensor_bottom_model_b.shape[0])
        grad_b = torch.tensor([])
        grad_b = grad_b.to(args.device)
        for q in range(output_tensor_bottom_model_b.shape[0]):
            if q == 0:
                grad_b = torch.cat((grad_b,(torch.mean(grad_output_bottom_model_b[q:p], axis=0)).unsqueeze(0)), 0)
            else:
                grad_b = torch.cat((grad_b,(torch.mean(grad_output_bottom_model_b[(q*p):((q+1)*p)], axis=0)).unsqueeze(0)), 0)
        loss_bottom_B = torch.sum(grad_b * output_tensor_bottom_model_b)
        bottom_model_B_optimizer.zero_grad()
        loss_bottom_B.backward()
        bottom_model_B_optimizer.step()