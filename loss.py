import torch
import torch.nn as nn
import monai

class TaylorSoftmax(nn.Module):

    def __init__(self, dim=1, n=2):
        super(TaylorSoftmax, self).__init__()
        assert n % 2 == 0
        self.dim = dim
        self.n = n

    def forward(self, x):
        
        fn = torch.ones_like(x)
        denor = 1.
        for i in range(1, self.n+1):
            denor *= i
            fn = fn + x.pow(i) / denor
        out = fn / fn.sum(dim=self.dim, keepdims=True)
        return out

class LabelSmoothingLoss(nn.Module):

    def __init__(self, classes, smoothing=0.0, dim=-1): 
        super(LabelSmoothingLoss, self).__init__() 
        self.confidence = 1.0 - smoothing 
        self.smoothing = smoothing 
        self.cls = classes 
        self.dim = dim 
    def forward(self, pred, target): 
        """Taylor Softmax and log are already applied on the logits"""
        #pred = pred.log_softmax(dim=self.dim)
        return torch.sum(-target * pred, dim=self.dim)
    

class TaylorCrossEntropyLoss(nn.Module):

    def __init__(self, n=2, ignore_index=-1, reduction='none', smoothing=0.1):
        super(TaylorCrossEntropyLoss, self).__init__()
        assert n % 2 == 0
        self.taylor_softmax = TaylorSoftmax(dim=1, n=n)
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.lab_smooth = LabelSmoothingLoss(5, smoothing=smoothing)

    def forward(self, logits, labels):

        log_probs = self.taylor_softmax(logits).log()
        loss = self.lab_smooth(log_probs, labels)
        return loss

class FocalLoss(nn.Module):
    """Some Information about FocalLoss"""
    def __init__(self,label_smoothing=0.1):
        super(FocalLoss, self).__init__()
        self.gamma = 2
        self.alpha = 1
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none',label_smoothing=label_smoothing)

    def forward(self, outputs, targets):
        ce_loss = self.criterion(outputs, targets)
        pt = torch.exp(-ce_loss)
        loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return loss

class MultiTaskLossWrapper(nn.Module):
    def __init__(self):
        super(MultiTaskLossWrapper, self).__init__()
        self.task_num = 5

        self.task1_criterion = torch.nn.CrossEntropyLoss(reduction='none',label_smoothing=0.1)
        #FocalLoss(label_smoothing=0.1)
        self.task2_criterion = torch.nn.CrossEntropyLoss(reduction='none',label_smoothing=0.0)
        #FocalLoss(label_smoothing=0.0)
        self.task2_dice_criterion = monai.losses.DiceLoss(softmax=True, reduction='none')
        self.task3_criterion = torch.nn.L1Loss(reduction='none')
        self.log_vars = nn.Parameter(torch.ones((self.task_num)))

    def forward(self, outputs, targets_task1, targets_task2,targets_task3, task, lam):
        outputs_task1, outputs_task21, outputs_task22, outputs_task23,outputs_task3,_ = outputs
        targets_task2 = ((targets_task2 - 0.5) * 0.9 + 0.5)
        b = outputs_task1.shape[0]
        
        task1 = (task == 1).float().cuda()
        task21 = (task == 21).float().cuda()
        task22 = (task == 22).float().cuda()
        task23 = (task == 23).float().cuda()
        task3 = (task == 3).float().cuda()


        targets_task1[task==1] = targets_task1[task==1] * lam + targets_task1[task==1].flip(0) * (1. - lam)
        targets_task3[task==3] = targets_task3[task==3] * lam + targets_task3[task==3].flip(0) * (1. - lam)

        #targets_task2[task==21] = targets_task2[task==21] * lam + targets_task2[task==21].flip(0) * (1. - lam)
        #targets_task2[task==22] = targets_task2[task==22] * lam + targets_task2[task==22].flip(0) * (1. - lam)
        #targets_task2[task==23] = targets_task2[task==23] * lam + targets_task2[task==23].flip(0) * (1. - lam)
        loss_task1 = torch.mean(self.task1_criterion(outputs_task1, targets_task1) * task1)
        #loss_task1 = 0.5 / (self.log_vars[0] ** 2) * loss_task1 + torch.log(1 + self.log_vars[0] ** 2)

        loss_task21 = torch.mean((self.task2_criterion(outputs_task21, targets_task2).mean((1,2))+ self.task2_dice_criterion(outputs_task21, targets_task2).mean((1, 2, 3))) * task21)*0.5
        
        #loss_task21 = 0.5 / (self.log_vars[1] ** 2) * loss_task21 + torch.log(1 + self.log_vars[1] ** 2)
    
    
        loss_task22 = torch.mean((self.task2_criterion(outputs_task22, targets_task2).mean((1,2))+ self.task2_dice_criterion(outputs_task22, targets_task2).mean((1, 2, 3))) * task22)*0.5

        #loss_task22 = 0.5 / (self.log_vars[2] ** 2) * loss_task22 + torch.log(1 + self.log_vars[2] ** 2)

    
        loss_task23 = torch.mean((self.task2_criterion(outputs_task23, targets_task2).mean((1,2))+ self.task2_dice_criterion(outputs_task23, targets_task2).mean((1, 2, 3))) * task23)*0.5

        #loss_task23 = 0.5 / (self.log_vars[3] ** 2) * loss_task23 + torch.log(1 + self.log_vars[3] ** 2)

    
        loss_task3 = torch.mean(self.task3_criterion(outputs_task3, targets_task3) * task3)
        loss_task3 = 0
        #print(loss_task3, sum(task3))
        #loss_task3 = 0.5 / (self.log_vars[4] ** 2) * loss_task3 + torch.log(1 + self.log_vars[4] ** 2)

        return loss_task1, loss_task21, loss_task22, loss_task23, loss_task3


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

class SupConLoss_Buffer(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss_Buffer, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.K = 512
        self.register_buffer("queue_features", torch.randn(128, self.K))
        self.queue = nn.functional.normalize(self.queue_features, dim=0)
        self.register_buffer("queue_labels", torch.ones(self.K)*(-1))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, features,labels,tasks):
        # gather features before updating queue
        features = concat_all_gather(features)
        labels = concat_all_gather(labels)
        tasks = concat_all_gather(tasks)

        features = features[tasks==1]
        labels = labels[tasks==1]

        batch_size = features.shape[0]

        ptr = int(self.queue_ptr)
        
        features = features[:min(ptr + batch_size,self.K)-ptr]
        labels = labels[:min(ptr + batch_size,self.K)-ptr]

        
        # replace the features at ptr (dequeue and enqueue)
        self.queue_features[:, ptr : min(ptr + batch_size,self.K)] = features.T
        self.queue_labels[ptr : min(ptr + batch_size,self.K)] = labels

        ptr = min(ptr + batch_size,self.K) % self.K  # move pointer

        self.queue_ptr[0] = ptr
    
    def forward(self, features, labels, tasks):


        
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [batch size, n_views, features].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0]

        feature_buffer = self.queue_features.clone().detach()
        label_buffer = self.queue_labels.clone().detach().view(-1, 1)
        
        labels_ = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels_, label_buffer.T).float().to(device)

        mask_tasks = (tasks==1).float()

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features, feature_buffer),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # compute log_prob
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = torch.mean(torch.mean(mask * log_prob,dim=1)*mask_tasks)

        # loss
        loss = -mean_log_prob_pos
        if self.training:
            self._dequeue_and_enqueue(features,labels,tasks)
            print(int(self.queue_ptr))

        return loss

"""
    def forward(self, features, labels, tasks):

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0]

        feature_buffer = self.queue_features.clone().detach()
        label_buffer = self.queue_labels.clone().detach().view(-1, 1)
        
        labels_ = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels_, label_buffer.T).float().to(device)

        mask_tasks = (tasks==1).float()

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(features, feature_buffer),self.temperature)
        
        loss = torch.nn.functional.cross_entropy(anchor_dot_contrast, mask, reduction='none',label_smoothing=0.1)
        loss = torch.mean(loss*mask_tasks)

        if self.training:
            self._dequeue_and_enqueue(features.detach(),labels,tasks)
            print(int(self.queue_ptr))

        return loss
"""