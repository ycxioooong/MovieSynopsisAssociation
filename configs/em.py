# model settings
model = dict(
    type='EMNet',
    embeder_glob=dict(
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
            reduce_opt='none')),
    embeder_cast=dict(
        type='CrossModalNet',
        pretrained=None,
        reduce_flag=False,
        video_embeder=dict(
            type='FCEmbeder', indim=512, norm_opt='none', reduce_opt='none'),
        syn_embeder=dict(
            type='FCEmbeder', indim=512, norm_opt='none', reduce_opt='none')),
    embeder_action=dict(
        type='CrossModalNet',
        pretrained=None,
        reduce_flag=False,
        video_embeder=dict(
            type='MLPEmbeder',
            indim=512,
            hidden_dim=256,
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
            reduce_opt='none')),
    pretrained=dict(
        glob='./work_dir/dtw_glob/epoch_40.pth',
        action='./work_dir/dtw_action/epoch_25.pth'))
loss = dict(type='BiPairwiseRankingLoss', margin=0.2, max_violation=False)
# model training and testing settings
train_cfg = None
test_cfg = None
# dataset settings
dataset_type = 'MovieEMDataset'
ann_root = '/mnt/SSD/movienext_meta/movieloc_meta/annotations/graph_328/'
prefix = dict(
    glob=dict(
        video='/mnt/SSD/movienext_meta/movieloc_data/movie330_video_sbtt_feat',
        syn='/mnt/SSD/movienext_meta/movieloc_data/w2v_embed_328/synopsis'),
    action=dict(
        video='/mnt/SSD/ava/movie_feat',
        syn='/mnt/SSD/movienext_meta/movieloc_data/action_w2v_328_graph'),
    cast=dict(
        video='/mnt/SSD/movienext_meta/movieloc_data/cast_feat',
        syn='/mnt/SSD/movienext_meta/movieloc_data/cast_profile_328_graph'),
    graph=dict(
        video='/mnt/SSD/movienext_meta/movieloc_data/video_graph_328',
        syn='/mnt/SSD/movienext_meta/movieloc_data/syn_graph_328'))
data = dict(
    tasks_per_gpu=16,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        ann_file=ann_root + 'train.json',
        prefix=prefix,
        cast_unique=False,
        test_mode=False),
    val=dict(
        type=dataset_type,
        ann_file=ann_root + 'test.json',
        prefix=prefix,
        cast_unique=False,
        test_mode=False),
    test=dict(
        type=dataset_type,
        ann_file=ann_root + 'test.json',
        prefix=prefix,
        cast_unique=False,
        test_mode=True))
# optimizer
optimizer = dict(type='SGD', lr=0.0001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=1.0 / 10,
    step=[4])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 10

# train scrip opts
topk = (1, 5, 10)
bidirection = True
joint_refine = True
batch_processor_type = 'bp_em'
loss_weights = dict(
    hook=('LossWeightHook',
          dict(
              init_loss_weight=[0.1],
              final_loss_weight=[0.1],
              total_epoch=total_epochs)))
eval_hook = dict(type='RetrievalEvalHook', args=dict(interval=1))

dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dir/null'
load_from = None
resume_from = None
workflow = [('train', 1), ('val', 1)]