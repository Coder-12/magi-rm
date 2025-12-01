from src.setup_environment import setup_run, log_metrics, close_run
from src.env_setup import setup_mixed_precision
from src.utils.training_utils import save_checkpoint
import yaml, torch

cfg = yaml.safe_load(open("configs/train_config.yaml"))
state = setup_run(run_name=cfg["run_name"], cfg=cfg, seed=cfg["seed"], base_dir=cfg["logging"]["run_dir"], use_tensorboard=False)

dtype = setup_mixed_precision(cfg["precision"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on {device} with dtype={dtype}")
