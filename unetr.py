from monai.networks.nets import SwinUNETR, UNETR
from utils import get_MSD_dataset_properties
from monai.losses import DiceCELoss


def SwinUNETR(args):
    properties = get_MSD_dataset_properties(args)
    n_class = len(properties["labels"])
    in_channels = len(properties["modality"])

    net = SwinUNETR(args.TRANSFORM.patch_size, in_channels=in_channels, out_channels=n_class,
                    depths=(2, 2, 2, 2), num_heads=(3, 6, 12, 24),
                    feature_size=24, norm_name='instance',
                    drop_rate=0.0, attn_drop_rate=0.0,
                    dropout_path_rate=0.0, normalize=True,
                    use_checkpoint=False, spatial_dims=3,
                    downsample='merging')

    return net


def UNETR(args):
    properties = get_MSD_dataset_properties(args)
    n_class = len(properties["labels"])
    in_channels = len(properties["modality"])

    net = UNETR(in_channels=in_channels, out_channels=n_class, img_size=args.TRANSFORM.patch_size,
                feature_size=16, hidden_size=768,
                mlp_dim=3072, num_heads=12, pos_embed='conv', norm_name='instance',
                conv_block=True, res_block=True, dropout_rate=0.0, spatial_dims=3, qkv_bias=False)

    return net


def loss_fn(pred, target):
    loss_fn = DiceCELoss(include_background=False, softmax=True, reduction='sum', to_onehot_y=True)
    loss = loss_fn(pred, target)
    return loss, pred
