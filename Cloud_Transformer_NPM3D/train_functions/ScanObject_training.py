import copy
import torch
from tqdm import tqdm
from train_functions.training_utils import _slice_loader, _count_batches

def train_scanobject(model, device, train_loader, test_loader,
                     epochs=250, lr=1e-4, weight_decay=1e-4,
                     seg_weight=0.5, eval_mode=False, subsample=None):
    cls_criterion = torch.nn.CrossEntropyLoss()
    seg_criterion = torch.nn.BCEWithLogitsLoss()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)

    num_train = _count_batches(train_loader, subsample)
    num_val = _count_batches(test_loader, subsample)

    model.eval()
    total, correct_cls, correct_seg = 0, 0, 0
    test_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(_slice_loader(test_loader, subsample), desc="Eval — LR: N/A", total=num_val, leave=False):
            pc, labels, masks = batch['pointcloud'].to(device), batch['category'].to(device), batch['mask'].to(device)
            class_pred, mask_pred = model(pc)
            seg_pred = mask_pred[:, 0, 0]
            cls_loss = cls_criterion(class_pred, labels)
            seg_loss = seg_criterion(seg_pred, masks)
            test_loss += (1 - seg_weight) * cls_loss.item() + seg_weight * seg_loss.item()
            correct_cls += (class_pred.argmax(dim=1) == labels).sum().item()
            correct_seg += ((seg_pred.sigmoid() > 0.5) == masks).sum().item()
            total += labels.size(0)

    baseline_loss = test_loss / num_val
    baseline_cls_acc = 100 * correct_cls / total
    baseline_seg_acc = 100 * correct_seg / (total * masks.shape[1])
    print(f"[Eval] Loss: {baseline_loss:.4f}, Cls Acc: {baseline_cls_acc:.2f}%, Seg Acc: {baseline_seg_acc:.2f}%")
    if eval_mode:
        return model

    best_val = baseline_loss
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in tqdm(_slice_loader(train_loader, subsample), desc=f"Train — LR: {optimizer.param_groups[0]['lr']:.2e}", total=num_train, leave=False):
            pc, labels, masks = batch['pointcloud'].to(device), batch['category'].to(device), batch['mask'].to(device)
            optimizer.zero_grad()
            class_pred, mask_pred = model(pc)
            seg_pred = mask_pred[:, 0, 0]
            loss = (1 - seg_weight) * cls_criterion(class_pred, labels) + seg_weight * seg_criterion(seg_pred, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train = train_loss / num_train

        model.eval()
        val_loss, correct_cls, correct_seg, total = 0.0, 0, 0, 0
        with torch.no_grad():
            for batch in tqdm(_slice_loader(test_loader, subsample), desc=f"Val   — LR: {optimizer.param_groups[0]['lr']:.2e}", total=num_val, leave=False):
                pc, labels, masks = batch['pointcloud'].to(device), batch['category'].to(device), batch['mask'].to(device)
                class_pred, mask_pred = model(pc)
                seg_pred = mask_pred[:, 0, 0]
                cls_loss = cls_criterion(class_pred, labels)
                seg_loss = seg_criterion(seg_pred, masks)
                val_loss += (1 - seg_weight) * cls_loss.item() + seg_weight * seg_loss.item()
                correct_cls += (class_pred.argmax(dim=1) == labels).sum().item()
                correct_seg += ((seg_pred.sigmoid() > 0.5) == masks).sum().item()
                total += labels.size(0)
        avg_val = val_loss / num_val
        scheduler.step(avg_val)
        if avg_val < best_val:
            best_val = avg_val
            best_weights = copy.deepcopy(model.state_dict())
        print(f"Epoch {epoch+1}/{epochs} — Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}, Classif Acc: {100*correct_cls/total:.2f}%, Segm Acc: {100*correct_seg/(total*masks.shape[1]):.2f}%")

    model.load_state_dict(best_weights)
    return model
