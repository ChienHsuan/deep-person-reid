import torch

from torchreid import metrics
from torchreid.losses import AMSoftmaxLoss, ArcFaceLoss, TripletLoss

from ..engine import Engine


class ImageAngularLossEngine(Engine):
    def __init__(
        self,
        datamanager,
        model,
        optimizer,
        scheduler=None,
        use_gpu=True,
        label_smooth=True,
        type='amsoftmax',
        use_triplet=False,
        margin=0.3
    ):
        super(ImageAngularLossEngine, self).__init__(datamanager, use_gpu)

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.register_model('model', model, optimizer, scheduler)

        if type == 'amsoftmax':
            self.criterion = AMSoftmaxLoss(
                use_gpu=True,
                m=0.35,
                s=30.0,
                label_smooth=label_smooth,
                epsilon=0.1,
                conf_penalty=0.0,
                pr_product=False
            )
        elif type == 'arcface':
            self.criterion = ArcFaceLoss(
                use_gpu=True,
                m=0.3,
                s=40,
                label_smooth=label_smooth,
                epsilon=0.1
            )
        else:
            raise NameError(f'Unsupported loss type: {type}')

        self.use_triplet = use_triplet
        if use_triplet == True:
            self.criterion_t = TripletLoss(margin=margin)

    def forward_backward(self, data):
        imgs, pids = self.parse_data_for_train(data)

        if self.use_gpu:
            imgs = imgs.cuda()
            pids = pids.cuda()

        outputs, features = self.model(imgs)
        
        loss = 0
        loss_summary = {}

        loss_an = self.compute_loss(self.criterion, outputs, pids)
        loss += loss_an
        loss_summary['loss_an'] = loss_an.item()
        loss_summary['acc'] = metrics.accuracy(outputs, pids)[0].item()


        if self.use_triplet == True:
            loss_t = self.compute_loss(self.criterion_t, features, pids)
            loss += loss_t
            loss_summary['loss_t'] = loss_t.item()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss_summary