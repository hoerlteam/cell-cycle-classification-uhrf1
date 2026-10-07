'''
3.3. train_utils.py
    - Implements model training loop with:
        • Early stopping
        • Learning rate scheduling (ReduceLROnPlateau)
        • Batch-level & epoch-level loss tracking
        • Test accuracy tracking
        • CSV logging
        • Precision-Recall Curve, Confusion Matrix, Loss/Accuracy Plotting
    - Supports manual continuation of training via interactive input after early stopping

-----
'''
import torch
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, average_precision_score

def train_model_input(model, train_loader, val_loader, test_loader, device, initial_epochs, step_epochs, lr, patience, save_path, csv_path):
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)   #Learning Rate Scheduler

    best_test_acc = 0.0
    no_improve_epochs = 0
    current_epoch = 0
    batch_loss_history, epoch_loss_history, val_loss_history, test_acc_history = [], [], [], []
    total_epochs_to_run = initial_epochs

    while True:
        for _ in range(total_epochs_to_run):
            model.train()
            current_epoch += 1
            batch_losses = []

            for Xs, ys in train_loader:
                Xs, ys = Xs.to(device), ys.to(device)
                optimizer.zero_grad()
                outputs = model(Xs)
                loss = criterion(outputs, ys)
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())
                batch_loss_history.append(loss.item())

            avg_epoch_loss = np.mean(batch_losses)
            epoch_loss_history.append(avg_epoch_loss)
            val_loss = evaluate_val_loss(model, val_loader, device, criterion)
            val_loss_history.append(val_loss)
            test_acc = evaluate_model(model, test_loader, device)
            test_acc_history.append(test_acc)

            print(f"Epoch {current_epoch} - Train Loss: {avg_epoch_loss:.4f} - Val Loss: {val_loss:.4f} - Test Acc: {test_acc:.2%}")
            log_epoch_to_csv(csv_path, current_epoch, avg_epoch_loss, val_loss, test_acc, optimizer.param_groups[0]['lr'])
            scheduler.step(val_loss)

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                no_improve_epochs = 0
                torch.save(model.state_dict(), save_path)
                print(f"New best model saved at {save_path} with Test Accuracy={test_acc:.2%}")
            else:
                no_improve_epochs += 1

            if no_improve_epochs >= patience:
                print(f"Early stopping triggered at epoch {current_epoch} (Best Test Acc={best_test_acc:.2%})")
                break

        user_input = input(f"Continue training for another {step_epochs} epochs? (y/n): ").strip().lower()
        if user_input == 'y':
            total_epochs_to_run = step_epochs
            no_improve_epochs = 0  # optional: reset
        else:
            print(f"Training stopped by user. Best Test Accuracy: {best_test_acc:.2%}")
            break

    plot_all_metrics(epoch_loss_history, batch_loss_history, test_acc_history, val_loss_history)
    all_preds, all_labels = detailed_report(model, test_loader, device)
    plot_precision_recall_curve(model, test_loader, device)
    with torch.no_grad():
        for Xs, ys in test_loader:
            Xs = Xs.to(device)
            preds = torch.argmax(model(Xs), dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(ys.numpy())

    return all_preds, all_labels

def log_epoch_to_csv(csv_path, epoch, train_loss, val_loss, test_acc, lr):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Epoch', 'Train Loss', 'Val Loss', 'Test Acc', 'LR'])
        writer.writerow([epoch, train_loss, val_loss, test_acc, lr])

def evaluate_model(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for Xs, ys in loader:
            Xs, ys = Xs.to(device), ys.to(device)
            preds = torch.argmax(model(Xs), dim=1)
            correct += (preds == ys).sum().item()
            total += ys.size(0)
    return correct / total

def evaluate_val_loss(model, val_loader, device, criterion):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for Xs, ys in val_loader:
            Xs, ys = Xs.to(device), ys.to(device)
            outputs = model(Xs)
            val_loss += criterion(outputs, ys).item()
    return val_loss / len(val_loader)

def plot_all_metrics(epoch_loss_history, batch_loss_history, test_acc_history, val_loss_history):
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.plot(epoch_loss_history, label="Train Loss")
    plt.plot(val_loss_history, label="Val Loss")
    plt.legend(); plt.grid(); plt.title("Loss per Epoch")

    plt.subplot(1, 3, 2)
    plt.plot(batch_loss_history, alpha=0.7)
    plt.grid(); plt.title("Loss per Batch")

    plt.subplot(1, 3, 3)
    plt.plot(test_acc_history)
    plt.grid(); plt.title("Test Accuracy")
    plt.tight_layout(); plt.show()


def detailed_report(model, test_loader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for Xs, ys in test_loader:
            Xs = Xs.to(device)
            preds = torch.argmax(model(Xs), dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(ys.numpy())

    # new labels order
    label_order = [1, 0, 2, 3, 4]
    class_names = ["G1G2", "early", "mid", "late", "ambiguous"]

    print(classification_report(
        all_labels,
        all_preds,
        labels=label_order,
        target_names=class_names
    ))

    cm = confusion_matrix(all_labels, all_preds, labels=label_order)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues")

    plt.title("Confusion Matrix (Reordered)")
    plt.grid(False)
    plt.show()
    return all_preds, all_labels


def plot_precision_recall_curve(model, test_loader, device):
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for Xs, ys in test_loader:
            Xs = Xs.to(device)
            probs = torch.nn.functional.softmax(model(Xs), dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(ys.numpy())
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    for i in range(all_probs.shape[1]):
        precision, recall, _ = precision_recall_curve(all_labels == i, all_probs[:, i])
        ap = average_precision_score(all_labels == i, all_probs[:, i])
        plt.plot(recall, precision, label=f"Class {i} (AP={ap:.2f})")
    plt.legend(); plt.grid(); plt.title("Precision-Recall Curve")
    plt.show()
    
