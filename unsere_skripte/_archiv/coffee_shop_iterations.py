"""Autoresearch iterations: systematically improve the coffee shop revenue model."""
import os, sys, json, subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "autoresearch_results")
os.makedirs(RESULTS_DIR, exist_ok=True)
ITERATIONS_FILE = os.path.join(SCRIPT_DIR, "iteration_results.json")

BASELINE_CONFIG = {
    "EPOCHS": 100, "BATCH_SIZE": 128, "LR": 0.001,
    "HIDDEN1": 50, "HIDDEN2": 30, "HIDDEN3": 30, "HIDDEN4": 20,
    "OPTIMIZER": "Adam", "DROPOUT": 0.0, "USE_BATCHNORM": False,
    "LR_SCHEDULER": None, "WEIGHT_DECAY": 0.0,
}

ITERATIONS = [
    {
        "name": "iter1_wider_network",
        "desc": "Increase hidden layer sizes for more capacity",
        "EPOCHS": 100, "BATCH_SIZE": 128, "LR": 0.001,
        "HIDDEN1": 128, "HIDDEN2": 64, "HIDDEN3": 32, "HIDDEN4": 16,
        "OPTIMIZER": "Adam", "DROPOUT": 0.0, "USE_BATCHNORM": False,
        "LR_SCHEDULER": None, "WEIGHT_DECAY": 0.0,
    },
    {
        "name": "iter2_dropout",
        "desc": "Add dropout + weight decay for regularization",
        "EPOCHS": 150, "BATCH_SIZE": 128, "LR": 0.001,
        "HIDDEN1": 64, "HIDDEN2": 32, "HIDDEN3": 32, "HIDDEN4": 16,
        "OPTIMIZER": "Adam", "DROPOUT": 0.2, "USE_BATCHNORM": False,
        "LR_SCHEDULER": None, "WEIGHT_DECAY": 1e-5,
    },
    {
        "name": "iter3_lr_scheduler",
        "desc": "ReduceLROnPlateau scheduler for adaptive LR",
        "EPOCHS": 200, "BATCH_SIZE": 64, "LR": 0.01,
        "HIDDEN1": 64, "HIDDEN2": 32, "HIDDEN3": 32, "HIDDEN4": 16,
        "OPTIMIZER": "Adam", "DROPOUT": 0.1, "USE_BATCHNORM": False,
        "LR_SCHEDULER": "ReduceLROnPlateau", "WEIGHT_DECAY": 1e-5,
    },
    {
        "name": "iter4_deeper_bn",
        "desc": "Deeper network (5 layers) + BatchNorm",
        "EPOCHS": 150, "BATCH_SIZE": 128, "LR": 0.001,
        "HIDDEN1": 128, "HIDDEN2": 64, "HIDDEN3": 64, "HIDDEN4": 32, "HIDDEN5": 16,
        "OPTIMIZER": "Adam", "DROPOUT": 0.1, "USE_BATCHNORM": True,
        "LR_SCHEDULER": None, "WEIGHT_DECAY": 1e-5,
    },
    {
        "name": "iter5_sgd_momentum",
        "desc": "SGD momentum + StepLR scheduler",
        "EPOCHS": 200, "BATCH_SIZE": 64, "LR": 0.01,
        "HIDDEN1": 64, "HIDDEN2": 32, "HIDDEN3": 32, "HIDDEN4": 16,
        "OPTIMIZER": "SGD", "MOMENTUM": 0.9, "DROPOUT": 0.1, "USE_BATCHNORM": False,
        "LR_SCHEDULER": "StepLR", "WEIGHT_DECAY": 1e-4,
    },
]


def gen_script(config, out_path):
    """Generate training script for given config."""
    hidden = []
    i = 1
    while f"HIDDEN{i}" in config:
        hidden.append(config[f"HIDDEN{i}"])
        i += 1
    n = len(hidden)

    # Init lines
    init_parts = []
    for j, h in enumerate(hidden):
        init_parts.append(f"        self.HIDDEN{j+1} = {h}")
    init_code = "\n".join(init_parts)

    # Layer defs
    layer_parts = []
    prev = "input_size"
    for j, h in enumerate(hidden):
        layer_parts.append(f"        self.linear_{j} = nn.Linear(in_features={prev}, out_features={h})")
        if config.get("USE_BATCHNORM"):
            layer_parts.append(f"        self.bn_{j} = nn.BatchNorm1d({h})")
        layer_parts.append(f"        self.relu_{j} = nn.ReLU()")
        if config.get("DROPOUT", 0) > 0:
            layer_parts.append(f"        self.dropout_{j} = nn.Dropout(p={config['DROPOUT']})")
        prev = h
    layer_parts.append(f"        self.linear_out = nn.Linear(in_features={prev}, out_features=output_size)")
    layer_code = "\n".join(layer_parts)

    # Forward
    fwd_parts = []
    for j in range(n):
        fwd_parts.append(f"        x = self.linear_{j}(x)")
        if config.get("USE_BATCHNORM"):
            fwd_parts.append(f"        x = self.bn_{j}(x)")
        fwd_parts.append(f"        x = self.relu_{j}(x)")
        if config.get("DROPOUT", 0) > 0:
            fwd_parts.append(f"        x = self.dropout_{j}(x)")
    fwd_parts.append("        x = self.linear_out(x)")
    fwd_code = "\n".join(fwd_parts)

    # Optimizer
    if config["OPTIMIZER"] == "Adam":
        opt_code = f"optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay={config['WEIGHT_DECAY']})"
    else:
        opt_code = f"optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum={config.get('MOMENTUM', 0.9)}, weight_decay={config['WEIGHT_DECAY']})"

    # Scheduler
    sched_init = ""
    sched_step = ""
    if config.get("LR_SCHEDULER") == "ReduceLROnPlateau":
        sched_init = "scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)"
        sched_step = "    scheduler.step(losses_val[-1])"
    elif config.get("LR_SCHEDULER") == "StepLR":
        sched_init = "scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)"
        sched_step = "    scheduler.step()"

    cfg_clean = {k: v for k, v in config.items() if k != "desc"}
    cfg_repr = repr(cfg_clean)

    res_path = out_path.replace(".py", "_results.json")

    lines = []
    lines.append('import torch, torch.nn as nn, numpy as np, pandas as pd, json')
    lines.append('import json as _json')
    lines.append('from torch.utils.data import Dataset, DataLoader')
    lines.append('from sklearn.model_selection import train_test_split')
    lines.append('from sklearn.preprocessing import StandardScaler')
    lines.append('')
    lines.append(f'EPOCHS = {config["EPOCHS"]}')
    lines.append(f'BATCH_SIZE = {config["BATCH_SIZE"]}')
    lines.append(f'LR = {config["LR"]}')
    lines.append('')
    lines.append('df = pd.read_csv("../data/coffee_shop_revenue.csv")')
    lines.append('X = df.drop(columns=["Daily_Revenue"])')
    lines.append('y = df["Daily_Revenue"]')
    lines.append('X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, random_state=42)')
    lines.append('X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1/0.9, random_state=42)')
    lines.append('scaler = StandardScaler()')
    lines.append('X_train_s = scaler.fit_transform(X_train)')
    lines.append('X_val_s = scaler.transform(X_val)')
    lines.append('X_test_s = scaler.transform(X_test)')
    lines.append('X_train_t = torch.tensor(X_train_s, dtype=torch.float32)')
    lines.append('X_val_t = torch.tensor(X_val_s, dtype=torch.float32)')
    lines.append('X_test_t = torch.tensor(X_test_s, dtype=torch.float32)')
    lines.append('y_train_t = torch.tensor(np.array(y_train).reshape(-1,1), dtype=torch.float32)')
    lines.append('y_val_t = torch.tensor(np.array(y_val).reshape(-1,1), dtype=torch.float32)')
    lines.append('y_test_t = torch.tensor(np.array(y_test).reshape(-1,1), dtype=torch.float32)')
    lines.append('')
    lines.append('class RegressionDataset(Dataset):')
    lines.append('    def __init__(self, X, y):')
    lines.append('        self.X, self.y = X, y')
    lines.append('    def __len__(self):')
    lines.append('        return self.X.shape[0]')
    lines.append('    def __getitem__(self, idx):')
    lines.append('        return self.X[idx], self.y[idx]')
    lines.append('')
    lines.append('train_ds = RegressionDataset(X_train_t, y_train_t)')
    lines.append('val_ds = RegressionDataset(X_val_t, y_val_t)')
    lines.append('test_ds = RegressionDataset(X_test_t, y_test_t)')
    lines.append('train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True)')
    lines.append('val_loader = DataLoader(val_ds, len(val_ds), shuffle=False)')
    lines.append('test_loader = DataLoader(test_ds, len(test_ds), shuffle=False)')
    lines.append('')
    lines.append('class RegressionTorch(nn.Module):')
    lines.append('    def __init__(self, input_size, output_size):')
    lines.append('        super().__init__()')
    lines.append(init_code)
    lines.append(layer_code)
    lines.append('')
    lines.append('    def forward(self, x):')
    lines.append(fwd_code)
    lines.append('        return x')
    lines.append('')
    lines.append('input_size = train_ds.X.shape[1]')
    lines.append('output_size = train_ds.y.shape[1]')
    lines.append('model = RegressionTorch(input_size, output_size)')
    lines.append('total_params = sum(p.numel() for p in model.parameters())')
    lines.append(f'config_dict = {cfg_repr}')
    lines.append('print("Config:", _json.dumps(config_dict))')
    lines.append(f'print(f"Params: {{total_params}}")')
    lines.append('')
    lines.append('loss_fn = nn.MSELoss()')
    lines.append(opt_code)
    lines.append(sched_init)
    lines.append('')
    lines.append('losses_train, losses_val = [], []')
    lines.append('for epoch in range(EPOCHS):')
    lines.append('    loss_epoch = 0')
    lines.append('    for Xb, yb in train_loader:')
    lines.append('        model.train()')
    lines.append('        optimizer.zero_grad()')
    lines.append('        yp = model(Xb)')
    lines.append('        loss = loss_fn(yp, yb)')
    lines.append('        loss_epoch += loss.item()')
    lines.append('        loss.backward()')
    lines.append('        optimizer.step()')
    lines.append('    losses_train.append(loss_epoch / len(train_loader))')
    lines.append('    model.eval()')
    lines.append('    val_loss = 0')
    lines.append('    with torch.no_grad():')
    lines.append('        for Xb, yb in val_loader:')
    lines.append('            yp = model(Xb)')
    lines.append('            val_loss += loss_fn(yp, yb).item()')
    lines.append('    losses_val.append(val_loss / len(val_loader))')
    if sched_step:
        lines.append(sched_step)
    lines.append('    if epoch % 20 == 0 or epoch == EPOCHS - 1:')
    lines.append('        print(f"E {epoch:3d} train: {losses_train[-1]:.2f} val: {losses_val[-1]:.2f}")')
    lines.append('')
    lines.append('model.eval()')
    lines.append('with torch.no_grad():')
    lines.append('    for Xb, yb in test_loader:')
    lines.append('        yp = model(Xb).numpy()')
    lines.append('        yt = yb.numpy()')
    lines.append('')
    lines.append('from sklearn.metrics import r2_score')
    lines.append('r2 = r2_score(y_pred=yp.flatten(), y_true=yt)')
    lines.append('print(f"\\nR2-Score: {r2:.4f}")')
    lines.append('')
    lines.append(f'res_path = r"{res_path}"')
    lines.append('res = {"r2": round(float(r2), 4), "total_params": total_params}')
    lines.append('with open(res_path, "w") as f:')
    lines.append('    json.dump(res, f, indent=2)')

    return "\n".join(lines)


def run_one(config, name, num):
    script_path = os.path.join(RESULTS_DIR, f"{name}.py")
    code = gen_script(config, script_path)
    with open(script_path, "w") as f:
        f.write(code)

    print(f"\n{'='*60}")
    print(f"ITERATION {num}: {name} - {config.get('desc', '')}")
    print(f"{'='*60}")

    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True, text=True, cwd=SCRIPT_DIR,
    )
    print(result.stdout)
    if result.stderr and ("Error" in result.stderr or "Traceback" in result.stderr):
        print(f"STDERR: {result.stderr[:2000]}")

    r2 = None
    for line in result.stdout.split("\n"):
        if "R2-Score:" in line:
            r2 = float(line.split(":")[1].strip())
    return r2


if __name__ == "__main__":
    all_results = {}

    # Baseline
    print(f"\n{'='*60}")
    print("BASELINE (original model)")
    print(f"{'='*60}")
    bl_result = subprocess.run(
        [sys.executable, os.path.join(SCRIPT_DIR, "coffee_shop_train_no_plot.py")],
        capture_output=True, text=True, cwd=SCRIPT_DIR,
    )
    print(bl_result.stdout)
    bl_r2 = None
    for line in bl_result.stdout.split("\n"):
        if "R2-Score" in line:
            bl_r2 = float(line.split(":")[1].strip())
    all_results["baseline"] = {"r2": bl_r2, "config": BASELINE_CONFIG}
    print(f"Baseline R^2: {bl_r2}")

    # 5 iterations
    for i, it in enumerate(ITERATIONS, 1):
        cfg = dict(it)
        name = cfg.pop("name")
        desc = cfg.pop("desc")
        cfg["desc"] = desc
        r2 = run_one(cfg, name, i)
        all_results[name] = {"r2": r2, "desc": desc, "config": cfg}
        print(f">>> {name}: R^2 = {r2} <<<")

    # Summary
    print(f"\n\n{'='*60}")
    print("SUMMARY OF 5 ITERATIONS")
    print(f"{'='*60}")
    print(f"{'Iteration':<28} {'R^2':<10} {'Delta':<14}")
    print("-" * 52)
    bl = bl_r2 or 0
    for name, data in all_results.items():
        r = data.get("r2", 0) or 0
        delta = r - bl
        print(f"{name:<28} {r:<10.4f} {delta:<+14.4f}")

    with open(ITERATIONS_FILE, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {ITERATIONS_FILE}")
