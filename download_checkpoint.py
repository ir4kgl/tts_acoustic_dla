import gdown

checkpoint_dir = "./checkpoints/"

gdown.download("https://drive.google.com/uc?id=1BW-QKujOWjoKirk9ugA9yNoRjzwxK8MM",
               checkpoint_dir + "final_run/" + "checkpoint.pth", quiet=True)
