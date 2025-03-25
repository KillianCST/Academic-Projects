import copy
import torch
from tqdm import tqdm
from chamfer_distance import ChamferDistance
import emd_linear.emd_module as emd_module
from train_functions.training_utils import _slice_loader, _count_batches, compute_batch_f1
import os

def train_shapenet(model, device, train_loader, val_loader,
                   epochs=250, lr=1e-4, weight_decay=1e-4,
                   eval_mode=False, subsample=None, chamfer_weight=0.0):
    chamfer = ChamferDistance()
    EMD = emd_module.emdModule()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)

    num_train = _count_batches(train_loader, subsample)
    num_val   = _count_batches(val_loader, subsample)

    backup_dir = os.path.join(os.getcwd(), "train_functions", "temp_saves")
    os.makedirs(backup_dir, exist_ok=True)

    # Initial evaluation
    model.eval()
    sum_fwd = sum_bwd = sum_f1 = 0.0
    with torch.no_grad():
        for batch in tqdm(_slice_loader(val_loader, subsample), desc="Initial Eval", total=num_val, leave=False):
            pred   = model(batch['partial_noise'].to(device), batch['partial_enc'].to(device)).permute(0,2,1)
            target = batch['complete_cloud'].to(device)
            d1_sq, d2_sq, _, _ = chamfer(pred, target)
            sum_fwd += torch.sqrt(d1_sq).mean().item()
            sum_bwd += torch.sqrt(d2_sq).mean().item()
            sum_f1  += compute_batch_f1(pred, target, threshold=0.01)

    baseline_cd = ((sum_fwd + sum_bwd) / (2 * num_val)) * 1000
    baseline_f1= sum_f1 / num_val
    print(f"[Eval] mAvg. CD: {baseline_cd:.3f} | mAvg. F1@1%: {baseline_f1:.4f}")

    if eval_mode:
        return model

    best_cd = baseline_cd
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch in tqdm(_slice_loader(train_loader, subsample), desc=f"Train — LR: {optimizer.param_groups[0]['lr']:.2e}", total=num_train, leave=False):
            optimizer.zero_grad()
            pred = model(batch['partial_noise'].to(device), batch['partial_enc'].to(device)).permute(0,2,1)
            target = batch['complete_cloud'].to(device)
            dist_emd, _ = EMD(pred, target, 0.005, 50)
            # Compute the EMD loss as before
            emd_loss = torch.sqrt(dist_emd).mean(1).mean()
            # Compute chamfer loss
            d1_sq, d2_sq, _, _ = chamfer(pred, target)
            chamfer_loss = torch.sqrt(d1_sq).mean() + torch.sqrt(d2_sq).mean()
            # Combine the losses with chamfer loss scaled by chamfer_weight
            loss = emd_loss + chamfer_weight * chamfer_loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train = train_loss / num_train

        # Validation
        model.eval()
        sum_fwd = sum_bwd = sum_f1 = 0.0
        with torch.no_grad():
            for batch in tqdm(_slice_loader(val_loader, subsample), desc=f"Val   — LR: {optimizer.param_groups[0]['lr']:.2e}", total=num_val, leave=False):
                pred   = model(batch['partial_noise'].to(device), batch['partial_enc'].to(device)).permute(0,2,1)
                target = batch['complete_cloud'].to(device)
                d1_sq, d2_sq, _, _ = chamfer(pred, target)
                sum_fwd += torch.sqrt(d1_sq).mean().item()
                sum_bwd += torch.sqrt(d2_sq).mean().item()
                sum_f1  += compute_batch_f1(pred, target, threshold=0.01)

        metric_cd = ((sum_fwd + sum_bwd) / (2 * num_val)) * 1000
        metric_f1= sum_f1 / num_val
        scheduler.step(metric_cd)

        # Backup every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(backup_dir, f"completion_shapenet_backup_epoch_{epoch+1}.pth"))
            print(f"→ Backup saved: epoch {epoch+1}")

        if metric_cd < best_cd:
            best_cd = metric_cd
            best_weights = copy.deepcopy(model.state_dict())

        print(f"Epoch {epoch+1}/{epochs} — Train Loss: {avg_train:.4f} | mAvg. CD: {metric_cd:.3f} | mAvg. F1@1%: {metric_f1:.4f}")

    model.load_state_dict(best_weights)
    return model
