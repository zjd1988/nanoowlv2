from modelscope.hub.snapshot_download import snapshot_download

model_dir = snapshot_download('google/owlv2-large-patch14-ensemble', cache_dir='./models')
model_dir = snapshot_download('BAAI/AltCLIP', cache_dir='./models')
model_dir = snapshot_download('AI-ModelScope/chinese-clip-vit-large-patch14', cache_dir='./models')

