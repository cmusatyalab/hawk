import numpy as np
import torch
import torch.nn as nn


class FewShotModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.backbone_class == "ConvNet":
            from ..networks.convnet import ConvNet

            self.encoder = ConvNet()
        elif args.backbone_class == "Res12":
            # hdim = 640
            from ..networks.res12 import ResNet

            self.encoder = ResNet()
        elif args.backbone_class == "Res18":
            # hdim = 512
            from ..networks.res18 import ResNet

            self.encoder = ResNet()
        elif args.backbone_class == "WRN":
            # hdim = 640
            from ..networks.WRN28 import Wide_ResNet

            # we set the dropout=0.5 directly here, it may achieve better
            # results by tuning the dropout
            self.encoder = Wide_ResNet(28, 10, 0.5)
        else:
            raise ValueError("")

    def split_instances(self, data):
        args = self.args
        if self.training:
            return (
                torch.Tensor(np.arange(args.way * args.shot))
                .long()
                .view(1, args.shot, args.way),
                torch.Tensor(
                    np.arange(args.way * args.shot, args.way * (args.shot + args.query))
                )
                .long()
                .view(1, args.query, args.way),
            )
        else:
            return (
                torch.Tensor(np.arange(args.eval_way * args.eval_shot))
                .long()
                .view(1, args.eval_shot, args.eval_way),
                torch.Tensor(
                    np.arange(
                        args.eval_way * args.eval_shot,
                        args.eval_way * (args.eval_shot + args.eval_query),
                    )
                )
                .long()
                .view(1, args.eval_query, args.eval_way),
            )

    def forward(self, x, get_feature=False):
        if get_feature:
            # get feature with the provided embeddings
            return self.encoder(x)
        else:
            # feature extraction
            if len(x.shape) > 4:
                x = x.squeeze(0)
            instance_embs = self.encoder(x)
            # num_inst = instance_embs.shape[0]
            # split support query set for few-shot data
            # support_idx, query_idx = self.split_instances(x)

            self.probe_instance_embs = instance_embs
            # self.probe_support_idx = support_idx
            # self.probe_query_idx = query_idx
            output = self._forward(instance_embs)
            return output

    def _forward(self, x, support_idx, query_idx):
        raise NotImplementedError("Suppose to be implemented by subclass")
