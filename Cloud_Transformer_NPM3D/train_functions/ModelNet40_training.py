import copy
import torch
from tqdm import tqdm
from train_functions.training_utils import _slice_loader, _count_batches

def train_modelnet40(model, device, train_loader, test_loader,
                     epochs=250, lr=1e-4, weight_decay=1e-4,
                     criterion=None, eval_mode=False, subsample=None,
                     return_loss_history=False):
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)

    num_train = _count_batches(train_loader, subsample)
    num_val = _count_batches(test_loader, subsample)

    # Lists for tracking loss history
    train_loss_history, val_loss_history = [], []

    # Baseline evaluation
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for batch in tqdm(_slice_loader(test_loader, subsample), desc="Eval — LR: N/A", total=num_val, leave=False):
            inputs = batch['pointcloud'].to(device).float()
            labels = batch['category'].to(device)
            logits, *_ = model(inputs)
            val_loss += criterion(logits, labels).item()
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    baseline_loss = val_loss / num_val
    baseline_acc = 100 * correct / total
    print(f"[Eval] Val Loss: {baseline_loss:.4f}, Acc: {baseline_acc:.2f}%")
    if eval_mode:
        return model

    best_val = baseline_loss
    best_weights = copy.deepcopy(model.state_dict())

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in tqdm(_slice_loader(train_loader, subsample), desc=f"Train — LR: {optimizer.param_groups[0]['lr']:.2e}", total=num_train, leave=False):
            inputs = batch['pointcloud'].to(device).float()
            labels = batch['category'].to(device)
            optimizer.zero_grad()
            logits, *_ = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train = train_loss / num_train
        train_loss_history.append(avg_train)

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for batch in tqdm(_slice_loader(test_loader, subsample), desc=f"Val   — LR: {optimizer.param_groups[0]['lr']:.2e}", total=num_val, leave=False):
                inputs = batch['pointcloud'].to(device).float()
                labels = batch['category'].to(device)
                logits, *_ = model(inputs)
                val_loss += criterion(logits, labels).item()
                correct += (logits.argmax(dim=1) == labels).sum().item()
                total += labels.size(0)
        avg_val = val_loss / num_val
        val_loss_history.append(avg_val)

        scheduler.step(avg_val)
        if avg_val < best_val:
            best_val = avg_val
            best_weights = copy.deepcopy(model.state_dict())

        print(f"Epoch {epoch+1}/{epochs} — Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}, Acc: {100*correct/total:.2f}%")

    model.load_state_dict(best_weights)

    if return_loss_history:
        return model, train_loss_history, val_loss_history

    return model