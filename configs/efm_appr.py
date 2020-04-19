# model settings
model = dict(
    type='CrossModalNet',
    pretrained=None,
    reduce_flag=False,
    video_embeder=dict(
        type='MLPEmbeder',
        indim=2348,
        hidden_dim=1024,
        outdim=256,
        with_bn=False,
        norm_opt='none',
        reduce_opt='none'),
    syn_embeder=dict(
        type='MLPEmbeder',
        indim=300,
        hidden_dim=256,
        outdim=256,
        with_bn=False,
        norm_opt='none',
        reduce_opt='none'))
loss = dict(type='BiPairwiseRankingLoss', margin=0.2, max_violation=False)
# model training and testing settings
train_cfg = None
test_cfg = None
# dataset settings
dataset_type = 'MovieSeqDataset'
ann_root = '/mnt/SSD/movienext_meta/movieloc_meta/annotations/graph_328/'
prefix = dict(
    glob=dict(
        video='/mnt/SSD/movienext_meta/movieloc_data/movie330_video_sbtt_feat',
        syn='/mnt/SSD/movienext_meta/movieloc_data/w2v_embed_328/synopsis'))
data = dict(
    tasks_per_gpu=16,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        ann_file=ann_root + 'train.json',
        prefix=prefix,
        element='glob',
        max_len=None,
        data_aug_cfg=None,
        test_mode=False),
    val=dict(
        type=dataset_type,
        ann_file=ann_root + 'test.json',
        prefix=prefix,
        element='glob',
        max_len=None,
        data_aug_cfg=None,
        test_mode=False),
    test=dict(
        type=dataset_type,
        ann_file=ann_root + 'test.json',
        prefix=prefix,
        element='glob',
        max_len=None,
        data_aug_cfg=None,
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
    step=[25, 40])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 50

# train scrip opts
topk = (1, 5, 10)
only_syn2clip = False
batch_processor_type = 'dtw'
loss_weights = dict(
    hook=('LossWeightHook',
          dict(
              init_loss_weight=[0.1],
              final_loss_weight=[0.1],
              total_epoch=total_epochs)))
eval_hook = dict(type='RetrievalOptimEvalHook', args=dict(interval=1))

dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dir/null'
load_from = None
resume_from = None
workflow = [('train', 1), ('val', 1)]