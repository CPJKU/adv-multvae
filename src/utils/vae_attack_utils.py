import os
import csv

import numpy as np
from tqdm import trange
from sklearn.metrics import balanced_accuracy_score

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter

from conf import EXP_SEED, ATK_MAX_EPOCHS
from src.utils.helper import json_dump, reproducible
from src.modules.polylinear import PolyLinear


def attack_single(log_dir, run_name, pretrained_model, attacker_config, attacker_opt,
                  device, tr_loader, vd_loader, te_loader,
                  log_batch_results_every, verbose=True):

    # Setting seed for reproducibility
    reproducible(EXP_SEED)

    log_dir = os.path.join(log_dir, "atk")
    writer_path = os.path.join(log_dir, run_name)
    os.makedirs(writer_path, exist_ok=True)
    writer = SummaryWriter(writer_path)

    pretrained_model.eval()

    n_epochs = e if (e := attacker_config.get("n_epochs")) else ATK_MAX_EPOCHS

    ce_loss_fn = CrossEntropyLoss()
    # Create attacker model that tries to predict the demographics (e.g., gender) of the users
    if attacker_config["activation"] not in ["relu", "tanh"]:
        raise AttributeError("Attacker currently only supports relu and tanh activations")

    activation_fn = torch.nn.ReLU() if attacker_config["activation"] == "relu" else torch.nn.Tanh()
    input_dropout = attacker_config.get("input_dropout")

    layer_config = [pretrained_model.latent] + attacker_config["size"]
    attacker_model = PolyLinear(layer_config=layer_config,
                                activation_fn=activation_fn,
                                input_dropout=input_dropout)
    attacker_model.to(device)

    opt = torch.optim.Adam(attacker_model.parameters(), **attacker_opt)

    batch_count = 0
    best_bal_acc = 0
    for epoch in trange(n_epochs, desc='epochs'):
        # --- Training --- #
        attacker_model.train()
        losses, bal_accs = [], []
        for batch_nr, (x, _, adv_targets) in enumerate(tr_loader):
            opt.zero_grad()
            x = x.to(device)
            adv_targets = adv_targets.to(device, dtype=torch.long)

            # we want to determine whether the latent space of the pretrained model
            # contains any information that may lead to the demographics of the users
            with torch.no_grad():
                x, _ = pretrained_model.encoder_forward(x)

            output = attacker_model(x)
            loss = ce_loss_fn(output, adv_targets[:, 0])
            losses.append(loss.item())

            # optimize model
            loss.backward()
            opt.step()

            # calculate the balanced accuracy score
            pred = torch.argmax(output, dim=1).detach().cpu().numpy()
            bal_acc = balanced_accuracy_score(y_true=adv_targets[:, 0].cpu().numpy(), y_pred=pred)
            bal_accs.append(bal_acc)

            if batch_count % log_batch_results_every == 0:
                writer.add_scalar('train/atk_batch_loss', loss, batch_count)
                writer.add_scalar('train/atk_batch_balanced_accuracy', bal_acc, batch_count)
            batch_count += 1

        writer.add_scalar('train/atk_avg_loss', np.mean(losses), epoch)
        writer.add_scalar('train/atk_avg_balanced_accuracy', np.mean(bal_accs), epoch)

        attacker_model.eval()
        losses, bal_accs = [], []
        with torch.no_grad():
            for batch_nr, (x, _, adv_targets) in enumerate(vd_loader):
                opt.zero_grad()
                x = x.to(device)
                adv_targets = adv_targets.to(device, dtype=torch.long)
                x, _ = pretrained_model.encoder_forward(x)

                output = attacker_model(x)
                loss = ce_loss_fn(output, adv_targets[:, 0])
                losses.append(loss.item())

                # calculate the balanced accuracy score
                pred = torch.argmax(output, dim=1).detach().cpu().numpy()
                bal_acc = balanced_accuracy_score(y_true=adv_targets[:, 0].cpu().numpy(), y_pred=pred)
                bal_accs.append(bal_acc)

        writer.add_scalar('val/atk_avg_loss', np.mean(losses), epoch)
        writer.add_scalar('val/atk_avg_balanced_accuracy', np.mean(bal_accs), epoch)

        # although we train to reduce the cross-entropy loss, we still want
        # to do early stopping on the balanced accuracy as this is what matters for us
        cur_bac = np.mean(bal_accs)
        if cur_bac >= best_bal_acc:
            if verbose:
                print("New best attacker")

            best_bal_acc = cur_bac
            json_dump(attacker_config, os.path.join(writer_path, 'atk_config.json'))
            torch.save(attacker_model.state_dict(), os.path.join(writer_path, 'atk_model.pt'))

    writer.flush()

    if verbose:
        # Training and validation is done, now determine loss on test data
        print("=" * 80)
        print("Running configuration on test set")
        print("=" * 80)

    # Load best attacker
    attacker_model.to(device)
    state_dict = torch.load(os.path.join(writer_path, 'atk_model.pt'))
    attacker_model.load_state_dict(state_dict)
    attacker_model.eval()

    # Finally, run the attacker network on the test set
    losses, bal_accs = [], []
    with torch.no_grad():
        for batch_nr, (x, _, adv_targets) in enumerate(te_loader):
            x = x.to(device)
            adv_targets = adv_targets.to(device, dtype=torch.long)
            x, _ = pretrained_model.encoder_forward(x)

            output = attacker_model(x)
            loss = ce_loss_fn(output, adv_targets[:, 0])
            losses.append(loss.item())

            # calculate the balanced accuracy score
            pred = torch.argmax(output, dim=1).detach().cpu().numpy()
            bal_acc = balanced_accuracy_score(y_true=adv_targets[:, 0].cpu().numpy(), y_pred=pred)
            bal_accs.append(bal_acc)

    avg_loss = np.mean(losses)
    avg_bal_acc = np.mean(bal_accs)
    if verbose:
        print(f"Loss on test set was {avg_loss:.4f}")
        print(f"Balanced accuracy on test set was {avg_bal_acc:.4f}")

    writer.flush()

    with open(os.path.join(writer_path, "atk_te_scores.csv"), "w", newline="") as fh:
        csv_writer = csv.writer(fh, delimiter=";", )
        csv_writer.writerow(["attacker_loss", "attacker_balanced_accuracy"])
        csv_writer.writerow([avg_loss, avg_bal_acc])

    return avg_loss, avg_bal_acc
