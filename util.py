import torch

def calculate_dice_coefficient(pred,target):
    
    if len(pred)==0:
        return []
    score = (torch.sum((pred==1)&(target==1),dim=(1,2))*2+ 1e-6) /(torch.sum(pred==1,dim=(1,2))+torch.sum(target==1,dim=(1,2))+ 1e-6)
    return score.detach().cpu().tolist()