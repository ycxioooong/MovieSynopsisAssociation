# model settings
key = 'action'
model = dict(
    type='CrossModalNet',
    key=key,
    pretrained=None,
    reduce_flag=True,
    video_embeder=dict(
        type='MLPEmbeder',
        indim=512,
        hidden_dim=256,
        outdim=256,
        with_bn=False,
        norm_opt='none',
        reduce_opt='reduce_norm'),
    syn_embeder=dict(
        type='MLPEmbeder',
        indim=300,
        hidden_dim=256,
        outdim=256,
        with_bn=False,
        norm_opt='none',
        reduce_opt='reduce_norm'))
loss = dict(type='BiPairwiseRankingLoss', margin=0.2, max_violation=False)
# model training and testing settings
train_cfg = None
test_cfg = None
# dataset settings
dataset_type = 'MSADataset'
ann_root = './data/anno/'
prefix = './data/feature'
data = dict(
    tasks_per_gpu=16,
    workers_per_gpu=16,
    train=dict(
        type=dataset_type,
        ann_file=ann_root + 'train.json',
        prefix=prefix,
        element='action',
        indims=(512, 300),
        test_mode=False),
    val=dict(
        type=dataset_type,
        ann_file=ann_root + 'val.json',
        prefix=prefix,
        element='action',
        indims=(512, 300),
        test_mode=False),
    test=dict(
        type=dataset_type,
        ann_file=ann_root + 'test.json',
        prefix=prefix,
        element='action',
        indims=(512, 300),
        test_mode=True))
# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=1.0 / 10,
    step=[15, 20])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
# yapf:enable
# runtime settings
total_epochs = 25

# train scrip opts
topk = (1, 5, 10)
bidirection = True
batch_processor_type = 'bp_basic'
eval_hook = dict(type='RetrievalEvalHook', args=dict(interval=1, key=key))

dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dir/null'
load_from = None
resume_from = None
workflow = [('train', 1), ('val', 1)]