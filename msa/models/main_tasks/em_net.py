from torch import nn
from .. import builder


class EMNet(nn.Module):

    def __init__(self,
                 embeder_appr,
                 embeder_cast,
                 embeder_action,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(EMNet, self).__init__()
        self.embeder_appr = builder.build_model(embeder_appr)
        self.embeder_cast = builder.build_model(embeder_cast)
        self.embeder_action = builder.build_model(embeder_action)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if pretrained is not None:
            self.init_weights(pretrained=pretrained)

        if pretrained is None:
            print('[EMNet] No pretrained weights given.')

    def init_weights(self, pretrained=None):
        assert isinstance(pretrained, dict)
        self.embeder_appr.init_weights(pretrained['appr'])
        self.embeder_action.init_weights(pretrained['action'])
        self.embeder_cast.init_weights()

    def forward(self,
                syn_appr,
                clip_appr,
                syn_action,
                clip_action,
                syn_cast,
                clip_cast,
                meta,
                return_loss=True,
                **kwargs):

        outputs = dict()
        meta_appr = [m['appr'] for m in meta]
        out = self.embeder_appr(syn_appr, clip_appr, meta_appr, key='appr')
        outputs.update(out)

        meta_action = [m['action'] for m in meta]
        out = self.embeder_action(
            syn_action, clip_action, meta_action, key='action')
        outputs.update(out)

        meta_cast = [m['cast'] for m in meta]
        out = self.embeder_cast(syn_cast, clip_cast, meta_cast, key='cast')
        outputs.update(out)

        return outputs
