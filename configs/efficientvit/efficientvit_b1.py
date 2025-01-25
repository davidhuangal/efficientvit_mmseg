# model settings
data_preprocessor = dict(
    type="SegDataPreProcessor",
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
)
model = dict(
    type="EncoderDecoder",
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type="EfficientViTBackbone",
        width_list=[16, 32, 64, 128, 256],
        depth_list=[1, 2, 3, 3, 4],
        dim=16,
    ),
    decode_head=dict(
        type="SegHead",
        fid_list=["stage4", "stage3", "stage2"],
        in_channels=[256, 128, 64],
        stride_list=[32, 16, 8],
        head_stride=8,
        head_width=64,
        head_depth=3,
        expand_ratio=4,
        middle_op="mbconv",
        final_expand=None,
        num_classes=19,
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)
