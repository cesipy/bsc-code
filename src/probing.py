import os
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
import json

import vilbert
import datasets
import trainer
from config import *
from experiment_tracker import ExperimentTracker, ExperimentConfig
from logger import Logger
import analysis
import utils_plotting

PROBE_EPOCHS = 4
logger = Logger()

def get_metric(task:str):
    assert task in ["hateful_memes", "mm_imdb", "upmc_food"]

    if task == "hateful_memes":
        return "auc"
    elif task == "mm_imdb":
        return "f1_score_macro"
    elif task == "upmc_food":
        return "accuracy"

def get_loss_function(task: str):
    assert task in ["hateful_memes", "mm_imdb", "upmc_food"]
    if task in ["hateful_memes", "mm_imdb"]:
        return torch.nn.BCEWithLogitsLoss()
    else:
        return torch.nn.CrossEntropyLoss()


def get_rams(model, dl, layer_n,) -> dict:
    text_embeddings, vision_embeddings = analysis.get_alignment_data_layer_n(
        dataloader=dl,
        model=model,
        layer_n=layer_n,
    )
    rams = analysis.calculate_main_metrics(
        text_embeddings=text_embeddings, vision_embeddings=vision_embeddings,
        k=32
    )
    return rams



def eval_probe(model, prober, dl, layer, loss_fn, metric, num_classes):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    prober.eval()

    with torch.no_grad():
        all_preds = []
        all_labels = []
        num_batches = 0
        total_loss = 0

        for batch in dl:
            label = batch["label"].to(device)
            text = {k: v.squeeze(1).to(device) for k, v in batch["text"].items()}
            image = {k: v.squeeze(1).to(device) for k, v in batch["img"].items()}

            text_embedding, vision_embedding = model.forward_until_layer(
                layer_n=layer,
                text_input_ids=text["input_ids"],
                text_attention_mask=text["attention_mask"],
                text_token_type_ids=text.get("token_type_ids", None),
                image_pixel_values=image["pixel_values"],
                image_attention_mask=image.get("attention_mask", None),
            )
            combined = torch.cat((text_embedding, vision_embedding), 1)
            logits = prober(combined)

            if num_classes == 1:  # Binary classification
                logits = logits.squeeze()
                loss = loss_fn(logits, label.float())
                probs = torch.sigmoid(logits)
                all_preds.extend(probs.cpu().numpy())
                all_labels.extend(label.cpu().numpy())

            elif num_classes == MM_IMDB_NUM_GENRES:  # Multi-label
                loss = loss_fn(logits, label.float())
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                all_preds.append(preds.cpu().numpy())
                all_labels.append(label.cpu().numpy())

            else:
                label = torch.argmax(label, dim=1)
                loss = loss_fn(logits, label)
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(label.cpu().numpy())

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches

    if metric == "accuracy":
        if num_classes == 1:
            binary_preds = [1 if p > 0.5 else 0 for p in all_preds]
            acc = accuracy_score(all_labels, binary_preds)
        else:
            acc = accuracy_score(all_labels, all_preds)
        return avg_loss, acc

    elif metric == "f1_score_macro":
        import numpy as np
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        f1 = f1_score(all_labels, all_preds, average="macro")
        return avg_loss, f1

    elif metric == "auc":
        auc = roc_auc_score(all_labels, all_preds)
        return avg_loss, auc


def linear_probe(
    model: vilbert.ViLBERT,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    num_classes: int,
    task: str,
    layer: int,
    epochs: int,
    loss_fn: torch.nn.Module,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"


    output_dim = 1 if num_classes == 1 else num_classes
    prober = torch.nn.Linear(2*EMBEDDING_DIM, output_dim).to(device)

    optim = torch.optim.AdamW(
        params=prober.parameters(),
        lr=1e-3,
        weight_decay=1e-5,
    )
    model.eval()

    for epoch in range(epochs):
        prober.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            label = batch["label"].to(device)
            text = {k: v.squeeze(1).to(device) for k, v in batch["text"].items()}
            image = {k: v.squeeze(1).to(device) for k, v in batch["img"].items()}

            with torch.no_grad():
                text_embedding, vision_embedding = model.forward_until_layer(
                    layer_n=layer,
                    text_input_ids=text["input_ids"],
                    text_attention_mask=text["attention_mask"],
                    text_token_type_ids=text.get("token_type_ids", None),
                    image_pixel_values=image["pixel_values"],
                    image_attention_mask=image.get("attention_mask", None),
                )

            combined = torch.cat((text_embedding, vision_embedding), 1)
            logits = prober(combined)

            if task in ["hateful_memes", "mm_imdb"]:
                logits = logits.squeeze()
                label = label.float()
            else:       # upmc food
                # label = label.long()
                label = torch.argmax(label, dim=1)

            loss = loss_fn(logits, label)

            optim.zero_grad()
            loss.backward()
            optim.step()

            total_loss += loss.item()
            num_batches += 1

    # eval on testset
    loss, metric = eval_probe(
        model=model,
        prober=prober,
        dl=test_loader,
        metric=get_metric(task=task),
        layer=layer,
        loss_fn=loss_fn,
        num_classes=num_classes  # Added
    )





    return loss, metric



def probe_model(path: str, task:str):
    info_str = f"running for model: {path}, task: {task}"; print(info_str); logger.info(info_str)
    t = ExperimentTracker()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = vilbert.ViLBERT.load_model(load_path=path,
        device=device).to(device)

    conf = ExperimentConfig(
        t_biattention_ids=model.t_biattention_ids,
        v_biattention_ids=model.v_biattention_ids,
        use_contrastive_loss=False,
        epochs=10,  #placeholder
    )

    train_loader, val_loader = t.get_task_dataloader(task=task, config=conf)
    test_loader =datasets.get_task_test_dataset(
        task=task,
        batch_size=BATCH_SIZE_DOWNSTREAM,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        prefetch_factor=PREFETCH,
        persistent_workers=PERSISTENT_WORKERS,
        seed=5      # should be not important here
    )
    alignment_loader = t.get_task_alignment_dataloader(task, conf)

    if task == "hateful_memes":
        num_classes = 1
    elif task == "mm_imdb":
        num_classes = MM_IMDB_NUM_GENRES  #23
    elif task == "upmc_food":
        num_classes = 101
    else:
        assert False

    results = {}
    for i in range(1, 13):

        rams = get_rams(model=model,layer_n=i, dl=alignment_loader)
        loss, acc = linear_probe(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            num_classes=num_classes,
            task=task,
            layer=i,
            epochs=PROBE_EPOCHS,
            loss_fn=get_loss_function(task)
        )
        metric_name = get_metric(task)
        print(f"Layer {i}: Loss: {loss}, {metric_name}: {acc}")
        results[f"layer_{i}"] = {
            "loss"      : loss, "metric": acc,
            "mknn"      : rams["mknn"],
            "cka"       : rams["cka"],
            "svcca"     : rams["svcca"],
            "procrustes": rams["procrustes"],
            }

    for layer, metrics in results.items():
        infostr = f"{layer} - Loss: {metrics['loss']}, {metric_name} {metrics['metric']}"
        print(infostr); logger.info(infostr)


    pretrain_name = path.copy().split("/")[2]
    ft_name = path.split("/")[-1].replace(".pt", "")

    save_path_dir = os.path.join("plots_probing", pretrain_name)
    os.makedirs(save_path_dir, exist_ok=True)
    utils_plotting.plot_cka_vs_performance(results, metric_name, task,
        save_path=os.path.join(save_path_dir, f"cka_vs_{metric_name}_{task}.png")
    )
    utils_plotting.plot_all_metrics_vs_performance(results, metric_name, task,
        save_path=os.path.join(save_path_dir, f"all_metrics_vs_{metric_name}_{task}.png")
    )
    utils_plotting.plot_procrustes_vs_performance(results, metric_name, task,
        save_path=os.path.join(save_path_dir, f"procrustes_vs_{metric_name}_{task}.png")
    )

    save_path = os.path.join(save_path_dir, f"results_{task}.json")
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)



def main():
    path = "res/checkpoints/20251013-010227_pretrained_late_fusion/20251022-222808_finetuned_hateful_memes.pt"
    task = "hateful_memes"
    probe_model(path=path, task=task)

    path = "res/checkpoints/20251013-010227_pretrained_late_fusion/20251023-221117_finetuned_mm_imdb.pt"
    task = "mm_imdb"
    probe_model(path=path, task=task)

    path = "res/checkpoints/20251013-010227_pretrained_late_fusion/20251022-153419_finetuned_upmc_food.pt"
    task = "upmc_food"
    probe_model(path=path, task=task)

    path = "res/checkpoints/20251030-192145_pretrained_latefusion_cka/20251104-073222_finetuned_hateful_memes.pt"
    task = "hateful_memes"
    probe_model(path=path, task=task)



if __name__ == "__main__":
    main()