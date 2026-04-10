import copy
import torch
from torch import nn

from convs.linears import TagFex_SimpleLinear
from convs.resnet import resnet18, resnet34, resnet50
from convs.resnet1d import resnet1d18


def get_convnet(args, pretrained=False):
    name = args["convnet_type"].lower()
    if name == "resnet1d18":
        return resnet1d18(args=args, pretrained=pretrained)
    if name == "resnet18":
        return resnet18(pretrained=pretrained, args=args)
    if name == "resnet34":
        return resnet34(pretrained=pretrained, args=args)
    if name == "resnet50":
        return resnet50(pretrained=pretrained, args=args)

    raise NotImplementedError(
        f"Standalone TagFex package supports resnet1d18/resnet18/resnet34/resnet50, got: {name}"
    )


class TagFexNet(nn.Module):
    def __init__(self, args, pretrained):
        super().__init__()
        self.convnet_type = args["convnet_type"]
        self.convnets = nn.ModuleList()
        self.pretrained = pretrained
        self.out_dim = None
        self.fc = None
        self.aux_fc = None
        self.task_sizes = []
        self.args = args

        self._device = args["device"][0]
        self.ta_net = get_convnet(args)
        self.ts_attn = None
        self.trans_classifier = None

        self.projector = nn.Sequential(
            TagFex_SimpleLinear(self.ta_feature_dim, self.args["proj_hidden_dim"]),
            nn.ReLU(True),
            TagFex_SimpleLinear(
                self.args["proj_hidden_dim"], self.args["proj_output_dim"]
            ),
        )
        self.predictor = None

    @property
    def ta_feature_dim(self):
        return self.ta_net.out_dim

    @property
    def feature_dim(self):
        if self.out_dim is None:
            return 0
        return self.out_dim * len(self.convnets)

    def extract_vector(self, x):
        features = [convnet(x)["features"] for convnet in self.convnets]
        return torch.cat(features, 1)

    def forward(self, x):
        ts_outs = [convnet(x) for convnet in self.convnets]
        features = torch.cat([ts_out["features"] for ts_out in ts_outs], 1)

        out = {
            "logits": self.fc(features),
            "aux_logits": self.aux_fc(features[:, -self.out_dim :]),
            "features": features,
        }

        ta_fmap = self.ta_net(x)["fmaps"][-1]
        ta_feature = ta_fmap.flatten(2).permute(0, 2, 1).mean(1)
        embedding = self.projector(ta_feature)
        out.update({"ta_feature": ta_feature, "embedding": embedding})

        if self.trans_classifier is not None:
            ts_feature = ts_outs[-1]["fmaps"][-1].flatten(2).permute(0, 2, 1)
            ta_features = ta_fmap.flatten(2).permute(0, 2, 1)
            merged_feature = self.ts_attn(ta_features.detach(), ts_feature).mean(1)
            out["trans_logits"] = self.trans_classifier(merged_feature)

        if self.predictor is not None:
            out["predicted_feature"] = self.predictor(ta_feature)

        return out

    def update_fc(self, nb_classes):
        if len(self.convnets) == 0:
            self.convnets.append(get_convnet(self.args))
        else:
            self.convnets.append(get_convnet(self.args))
            factor = self.args["init_interpolation_factor"]
            for ts_old, ts_new, ta_param in zip(
                self.convnets[-2].parameters(),
                self.convnets[-1].parameters(),
                self.ta_net.parameters(),
            ):
                ts_new.data = factor * ts_old.data + (1 - factor) * ta_param.data

        if self.out_dim is None:
            self.out_dim = self.convnets[-1].out_dim
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output, : self.feature_dim - self.out_dim] = weight
            fc.bias.data[:nb_output] = bias

        self.fc = fc

        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)

        self.aux_fc = self.generate_fc(self.out_dim, new_task_size + 1)

        if self.predictor is None:
            self.predictor = self.generate_fc(self.ta_feature_dim, self.ta_feature_dim)
        if self.ts_attn is None:
            self.ts_attn = TSAttention(self.out_dim, self.args["attn_num_heads"])
        else:
            self.ts_attn._reset_parameters()
        self.trans_classifier = self.generate_fc(self.ta_net.out_dim, new_task_size)

    def generate_fc(self, in_dim, out_dim):
        return TagFex_SimpleLinear(in_dim, out_dim)

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def get_freezed_copy_ta(self):
        ta_net_copy = copy.deepcopy(self.ta_net)
        for p in ta_net_copy.parameters():
            p.requires_grad_(False)
        return ta_net_copy.eval()

    def get_freezed_copy_projector(self):
        projector_copy = copy.deepcopy(self.projector)
        for p in projector_copy.parameters():
            p.requires_grad_(False)
        return projector_copy.eval()

    def freeze_conv(self):
        for param in self.convnets.parameters():
            param.requires_grad = False
        self.convnets.eval()

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        gamma = torch.mean(oldnorm) / torch.mean(newnorm)
        print("alignweights,gamma=", gamma)
        self.fc.weight.data[-increment:, :] *= gamma


class TSAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.norm_ts = nn.LayerNorm(embed_dim)
        self.norm_ta = nn.LayerNorm(embed_dim)

        self.weight_q = nn.Parameter(torch.empty((embed_dim, embed_dim)))
        self.weight_k_ts = nn.Parameter(torch.empty((embed_dim, embed_dim)))
        self.weight_k_ta = nn.Parameter(torch.empty((embed_dim, embed_dim)))
        self.weight_v_ts = nn.Parameter(torch.empty((embed_dim, embed_dim)))
        self.weight_v_ta = nn.Parameter(torch.empty((embed_dim, embed_dim)))

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_normal_(self.weight_q)
        nn.init.xavier_normal_(self.weight_k_ts)
        nn.init.xavier_normal_(self.weight_k_ta)
        nn.init.xavier_normal_(self.weight_v_ts)
        nn.init.xavier_normal_(self.weight_v_ta)
        self.norm_ta.reset_parameters()
        self.norm_ts.reset_parameters()

    def forward(self, ta_feats, ts_feats):
        bs, n_tokens, channels = ta_feats.shape
        ta_feats = self.norm_ta(ta_feats)
        ts_feats = self.norm_ts(ts_feats)

        q = (ts_feats @ self.weight_q).reshape(
            bs, n_tokens, self.num_heads, channels // self.num_heads
        ).transpose(1, 2)
        k_ts = (ts_feats @ self.weight_k_ts).reshape(
            bs, n_tokens, self.num_heads, channels // self.num_heads
        ).transpose(1, 2)
        k_ta = (ta_feats @ self.weight_k_ta).reshape(
            bs, n_tokens, self.num_heads, channels // self.num_heads
        ).transpose(1, 2)
        v_ts = (ts_feats @ self.weight_v_ts).reshape(
            bs, n_tokens, self.num_heads, channels // self.num_heads
        ).transpose(1, 2)
        v_ta = (ta_feats @ self.weight_v_ta).reshape(
            bs, n_tokens, self.num_heads, channels // self.num_heads
        ).transpose(1, 2)

        attn = q @ k_ts.transpose(-2, -1) + q @ k_ta.transpose(-2, -1)
        attn = attn / (channels // self.num_heads) ** 0.5
        attn = attn.softmax(dim=-1)

        merged = attn @ (v_ts + v_ta)
        return merged.transpose(1, 2).reshape(bs, n_tokens, channels)
