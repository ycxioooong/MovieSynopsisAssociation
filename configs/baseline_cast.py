# model settings
key = 'cast'
model = dict(type='Identity', pretrained=None)
# model training and testing settings
train_cfg = None
test_cfg = None
# dataset settings
dataset_type = 'MSADataset'
ann_root = './data/anno/'
prefix = './data/feature'
data = dict(
    tasks_per_gpu=16,
    workers_per_gpu=8,
    test=dict(
        type=dataset_type,
        ann_file=ann_root + 'test.json',
        prefix=prefix,
        element='cast',
        indims=(512, 512),
        test_mode=True))
# optimizer
topk = (1, 5, 10)
