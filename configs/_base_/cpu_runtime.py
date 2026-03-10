# configs/_base_/cpu_runtime.py
# Use CPU
device = 'cpu'

# Disable cudnn
cudnn_benchmark = False
cudnn_deterministic = True

# Set multi-process start method
mp_start_method = 'spawn'

# Logging configuration
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
    ])

# Runtime settings
log_level = 'INFO'
log_file = None
load_from = None
resume_from = None
workflow = [('train', 1)]
dist_params = dict(backend='gloo')
work_dir = None