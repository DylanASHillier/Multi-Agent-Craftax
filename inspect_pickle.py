import pickle

ckpt_path = "wandb/run-20251117_181137-zhpsz0t7/files/model_iter_80.pickle"

with open(ckpt_path, "rb") as f:
    ckpt = pickle.load(f)   # ckpt is your top-level tuple

print(type(ckpt))
print("Tuple length:", len(ckpt))

for i, part in enumerate(ckpt):
    print(f"Index {i}: type={type(part)}")


