# -*- coding: utf-8 -*-
from ultralytics import YOLO
import time, os, csv, torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# ======== CONFIG ========
DATA_ROOT = "/home/agrilab/Documentos/Casanova/Arvores/Dataset/dados/Transversalv1"
IMGSZ     = 224
CSV_LOG   = os.path.join(os.getcwd(), "metrics_per_epoch.csv")  # opcional
# ========================

# --- Callback: calcula métricas no final de CADA época (após train + val) ---
def on_fit_epoch_end(trainer):
    """
    Usa o dataloader de validação do Ultralytics para obter preds/labels e
    calcular accuracy, precision/recall/F1 macro, sem interferir no grad.
    """
    # Pega o dataloader de validação
    val_loader = getattr(trainer.validator, "dataloader", None)
    if val_loader is None:
        print(f"\n[Epoch {trainer.epoch+1}] ⚠️ Sem dataloader de validação; pulando métricas extras.\n")
        return

    model_torch = trainer.model
    was_training = model_torch.training
    device = next(model_torch.parameters()).device

    preds, gts = [], []
    model_torch.eval()
    with torch.no_grad():
        for batch in val_loader:
            # Batches de classificação podem vir como dict ou (imgs, labels)
            if isinstance(batch, dict):
                imgs = batch.get("img", batch.get("imgs"))
                labels = batch.get("cls", batch.get("label", batch.get("labels")))
            else:
                # tuple/list: (imgs, labels, *extras)
                imgs, labels = batch[0], batch[1]

            imgs = imgs.to(device, non_blocking=True)
            logits = model_torch(imgs)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            pred_idx = logits.argmax(dim=1)

            preds.extend(pred_idx.cpu().tolist())
            if isinstance(labels, torch.Tensor):
                labels = labels.view(-1).cpu().tolist()
            else:
                labels = list(labels)
            gts.extend(labels)

    # Restaura o estado de treino para a próxima época
    model_torch.train(was_training)

    # Métricas macro
    acc = accuracy_score(gts, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(gts, preds, average="macro", zero_division=0)
    ep = trainer.epoch + 1
    print(f"\n[Epoch {ep}] acc={acc:.4f}  precision_macro={prec:.4f}  "
          f"recall_macro={rec:.4f}  f1_macro={f1:.4f}\n")

    # (opcional) registrar em CSV
    try:
        write_header = not os.path.exists(CSV_LOG)
        with open(CSV_LOG, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["epoch", "accuracy", "precision_macro", "recall_macro", "f1_macro"])
            w.writerow([ep, f"{acc:.6f}", f"{prec:.6f}", f"{rec:.6f}", f"{f1:.6f}"])
    except Exception as e:
        print(f"⚠️ Falha ao escrever CSV: {e}")

# --- Treino ---
model = YOLO('yolov8m-cls.pt')
initial_time = time.time()
model.info()

# registra o callback (❗️não passe 'callbacks=' no train)
model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

model.train(
    project='Cor Fotografia Microscópio Tangencial',
    name='Bases',
    data=DATA_ROOT,          # raiz com train/ val/ test/
    epochs=150,
    imgsz=IMGSZ,
    verbose=True,
    dropout=0.1,
    val=True,
    mixup=0.1,
    mosaic=0.0,              # não usado em classificação; força 0 para limpar log
    optimize=True
)

final_time = time.time()
print(f'Tempo de processamento: {final_time - initial_time:.2f} segundos')
print(f"CSV (métricas por época): {CSV_LOG}")
