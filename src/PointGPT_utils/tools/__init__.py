import sys
sys.path.append('/home/nick/projects')
# from .runner import run_net
from src.PointGPT_utils.tools.runner import test_net
from src.PointGPT_utils.tools.runner_pretrain import run_net as pretrain_run_net
from src.PointGPT_utils.tools.runner_finetune import run_net as finetune_run_net
from src.PointGPT_utils.tools.runner_finetune import test_net as test_run_net