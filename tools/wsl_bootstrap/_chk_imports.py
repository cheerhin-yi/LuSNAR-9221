import ftfy, regex, torch
print(ftfy.__version__)
print(regex.__version__)
from mmseg.utils import register_all_modules
print('mmseg-ok')
print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())
