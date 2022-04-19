import os
from collections import defaultdict

import numpy as np
from tqdm import trange

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter

from conf import EXP_SEED, VAE_MAX_EPOCHS
from src.modules.losses import VAE_loss
from src.utils.helper import reproducible, json_dump
from src.utils.vae_training_utils import validate, eval_adversaries


def train(model, device, dataloader, vae_loss_fn, adv_loss_fn, writer, epoch,
          log_batch_results_every=10, use_adv_network=False, adv_loss_weight=1,
          vae_opt=None, normalize_gradients=False):
    model.train()
    losses, neg_lls, weighted_KLs, total_losses = [], [], [], []
    adv_losses, bal_accs = [], []

    batch_count = epoch * len(dataloader)
    for batch_nr, (x, y, adv_targets) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        adv_targets = adv_targets.to(device, dtype=torch.long)  # BCE loss requires data of type long

        vae_opt.zero_grad()

        logits, KL, adv_logits = model(x)
        loss, neg_ll, weighted_KL = vae_loss_fn(logits, KL, y)

        # Perform backpropagation only for reproduction loss (params for the other losses are fixed)
        neg_ll.backward()
        if normalize_gradients:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        vae_opt.step()

        total_loss = loss
        if use_adv_network:
            adv_loss, adv_bal_acc = eval_adversaries(adv_logits, adv_targets, adv_loss_fn)
            adv_losses.append(adv_loss.item())
            bal_accs.append(adv_bal_acc.item())
            total_loss = total_loss + adv_loss_weight * adv_loss
        total_losses.append(total_loss.item())

        if batch_count % log_batch_results_every == 0:
            # it may be easier to determine problems in the training phase by
            # considering performance changes during the epoch
            writer.add_scalar('train/batch_loss', loss, batch_count)
            writer.add_scalar('train/batch_neg_lss', neg_ll, batch_count)
            writer.add_scalar('train/batch_weighted_KL', weighted_KL, batch_count)

            if use_adv_network:
                writer.add_scalar('train/adv_batch_loss', adv_losses[-1], batch_count)
                writer.add_scalar('train/adv_batch_balanced_accuracy', bal_accs[-1], batch_count)
        batch_count += 1

        losses.append(loss.item())
        neg_lls.append(neg_ll.item())
        weighted_KLs.append(weighted_KL.item())

    writer.add_scalar('train/avg_loss', np.mean(losses), epoch)
    writer.add_scalar('train/avg_neg_lss', np.mean(neg_lls), epoch)
    writer.add_scalar('train/avg_weighted_KL', np.mean(weighted_KLs), epoch)

    if use_adv_network:
        writer.add_scalar('train/adv_avg_loss', np.mean(adv_losses), epoch)
        writer.add_scalar('train/adv_avg_balanced_accuracy', np.mean(bal_accs), epoch)


def retrain_single(run_dir, model, config, use_adv_network, device, user_groups_all_traits,
                   tr_loader, vd_loader, log_batch_results_every, log_val_every, log_val_metrics_every,
                   traits, val_levels, val_metrics, verbose=True, store_best_model=False):
    # Setting seed for reproducibility
    reproducible(EXP_SEED)

    # create own directory for results to keep folders nicely separated and clean
    result_dir = run_dir.replace(os.path.sep + "vae" + os.path.sep,
                                 os.path.sep + "retrain" + os.path.sep)

    os.makedirs(result_dir, exist_ok=True)
    writer = SummaryWriter(result_dir)

    # Store full configuration to easily re-run a single experiment later on
    json_dump(config, os.path.join(result_dir, "config.json"))

    model.to(device)

    vae_loss = VAE_loss(**config["loss"])
    ce_loss_fn = CrossEntropyLoss()

    general_config = config.get("general") or dict()
    normalize_gradients = bool(general_config.get("normalize_gradients"))
    n_epochs = e if (e := general_config.get("n_epochs")) else VAE_MAX_EPOCHS

    adv_loss_weight = lw if (lw := (config.get("adv") or dict()).get("loss_weight")) is not None else 1

    default_opt_config = config["opt"]
    decoder_opt_config = c if (c := config.get("dec_opt")) is not None else {}
    vae_opt = torch.optim.Adam([
        {"params": model.decoder.parameters(), **decoder_opt_config}
    ], **default_opt_config)

    best_scores = defaultdict(lambda: 0.)
    best_model_dicts = {}
    it = trange(n_epochs, desc="epochs") if verbose else range(n_epochs)
    vd_counter = 0
    for epoch in it:

        # --- Training --- #
        train(model, device, tr_loader, vae_loss, ce_loss_fn, writer, epoch,
              log_batch_results_every, use_adv_network, adv_loss_weight,
              vae_opt, normalize_gradients)

        # --- Validation --- #
        if epoch % log_val_every == 0:

            # we want to calculate the validation metrics less often as they require
            # more computations and therefore more time.
            calc_full_metrics = vd_counter % log_val_metrics_every == 0
            vd_counter += 1

            metrics, _ = validate(model, device, vd_loader, vae_loss, ce_loss_fn, writer, epoch,
                                  calc_full_metrics, metrics=val_metrics, metric_levels=val_levels,
                                  traits=traits, user_groups_all_traits=user_groups_all_traits,
                                  use_adv_network=use_adv_network)

            if calc_full_metrics:
                for metric in val_metrics:
                    score = metrics[f"{metric}_{val_levels[0]}"]

                    if score >= best_scores[metric]:
                        best_scores[metric] = score
                        best_model_dicts[metric] = {k: v.cpu() for k, v in model.state_dict().items()}

                        if store_best_model:
                            torch.save(best_model_dicts[metric],
                                       os.path.join(result_dir, f"best_model_{metric}_retrained.pt"))

        writer.flush()

    model.cpu()
    del model

    return dict(best_scores), dict(best_model_dicts)
