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
        type="EfficientViTLargeBackbone",
        width_list=[32, 64, 128, 256, 512],
        depth_list=[1, 2, 2, 8, 8],
    ),
    decode_head=dict(
        type="SegHead",
        fid_list=["stage4", "stage3", "stage2"],
        in_channels=[512, 256, 128],
        stride_list=[32, 16, 8],
        head_stride=8,
        head_width=128,
        head_depth=3,
        expand_ratio=4,
        middle_op="fmbconv",
        final_expand=8,
        num_classes=19,
        act_func="gelu",
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)
