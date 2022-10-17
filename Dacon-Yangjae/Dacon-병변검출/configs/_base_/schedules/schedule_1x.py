# optimizer
optimizer = dict(type='SGD', lr=1e-6, momentum=0.9, weight_decay=0.0001)
#optimizer = dict(type='Adam', lr=0.0002, weight_decay=0.0001)

optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    #step=[5, 13, 21, 29, 37, 45, 53])
    step=[3, 6, 9, 12])

#lr_config = dict(
#    policy='CosineAnnealing',
#    warmup='linear',
#    warmup_iters=1,
#    warmup_ratio=1.0 / 10,
#    min_lr_ratio=1e-7)

runner = dict(type='EpochBasedRunner', max_epochs=100)
