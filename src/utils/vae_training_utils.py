import os
from collections import defaultdict

import pandas as pd
import torch
import numpy as np

from sklearn.metrics import balanced_accuracy_score
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from conf import EXP_SEED, VAE_MAX_EPOCHS
from src.utils.eval import eval_recgap
from src.utils.helper import json_dump, reproducible
from src.modules.losses import VAE_loss
from src.modules.multi_vae import MultiVAEAdv

# uncomment for debugging
# torch.autograd.set_detect_anomaly(True)
# np.seterr(all='raise')

def eval_adversaries(logits, targets, loss_fn):
    device = targets.device if isinstance(targets, torch.Tensor) else None
    adv_loss = torch.tensor(0, dtype=torch.float64, device=device)
    bal_acc = torch.tensor(0, dtype=torch.float64, device=device)

    n_adversaries = len(logits)
    for k in range(n_adversaries):
        adv_loss += loss_fn(logits[k], targets[:, 0]) / n_adversaries

        # calculate the mean balanced accuracy score
        pred = torch.argmax(logits[k], dim=1).detach().cpu().numpy()
        bal_acc += balanced_accuracy_score(y_true=targets[:, 0].cpu().numpy(),
                                           y_pred=pred) / n_adversaries

    return adv_loss, bal_acc


def train(model, device, dataloader, vae_loss_fn, adv_loss_fn, writer, epoch,
          log_batch_results_every=10, use_adv_network=False, adv_loss_weight=1,
          vae_opt=None, adv_opt=None, perform_warmup=False, n_warmup_epochs=10,
          normalize_gradients=False):
    model.train()
    losses, neg_lls, weighted_KLs, total_losses = [], [], [], []
    adv_losses, bal_accs = [], []

    perform_warmup = use_adv_network and perform_warmup and epoch < n_warmup_epochs

    batch_count = epoch * len(dataloader)
    for batch_nr, (x, y, adv_targets) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        adv_targets = adv_targets.to(device, dtype=torch.long)  # BCE loss requires data of type long

        vae_opt.zero_grad()
        if use_adv_network:
            adv_opt.zero_grad()

        logits, KL, adv_logits = model(x)
        loss, neg_ll, weighted_KL = vae_loss_fn(logits, KL, y)

        # By including a warmup phase, we want to make the training more stable
        if perform_warmup:
            # only optimize encoder and decoder
            loss.backward()
            if normalize_gradients:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            vae_opt.step()

            adv_opt.zero_grad()
            _, _, adv_logits = model(x)
            adv_loss, adv_bal_acc = eval_adversaries(adv_logits, adv_targets, adv_loss_fn)
            adv_losses.append(adv_loss.item())
            bal_accs.append(adv_bal_acc.item())

            # optimize only the adversaries
            adv_loss.backward()
            if normalize_gradients:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            adv_opt.step()

        else:
            # optimize the whole model
            total_loss = loss
            if use_adv_network:
                adv_loss, adv_bal_acc = eval_adversaries(adv_logits, adv_targets, adv_loss_fn)
                adv_losses.append(adv_loss.item())
                bal_accs.append(adv_bal_acc.item())
                total_loss = total_loss + adv_loss_weight * adv_loss

            total_loss.backward()
            if normalize_gradients:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            vae_opt.step()
            if use_adv_network:
                # since the Adam optimizer keeps an internal state of the optimization,
                # we cannot just switch to the full optimizer after warmup phase
                adv_opt.step()
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


def validate(model, device, dataloader, vae_loss_fn, adv_loss_fn, writer, epoch, calc_full_metrics=False,
             metrics=("ndcg", "recall"), metric_levels=(20,), traits=("gender",), user_groups_all_traits=None,
             use_adv_network=False, ):
    if calc_full_metrics and user_groups_all_traits is None:
        raise AttributeError("'user_groups_all_traits' is required for full metrics")

    model.eval()
    with torch.no_grad():
        losses, neg_lls, weighted_KLs = [], [], []
        adv_losses, adv_bal_accs = [], []
        all_decoder_logits, all_decoder_targets = [], []
        for x, y, adv_targets in dataloader:
            x, y = x.to(device), y.to(device)

            logits, KL, adv_logits = model(x)
            loss, neg_ll, weighted_KL = vae_loss_fn(logits, KL, y)

            # === calc loss for adversarial ===
            if use_adv_network:
                # need datatype long for BCE loss
                adv_targets = adv_targets.to(device, dtype=torch.long)
                adv_loss, adv_bal_acc = eval_adversaries(adv_logits, adv_targets, adv_loss_fn)
                adv_losses.append(adv_loss.item())
                adv_bal_accs.append(adv_bal_acc.item())

            losses.append(loss.item())
            neg_lls.append(neg_ll.item())
            weighted_KLs.append(weighted_KL.item())

            if calc_full_metrics:
                # Removing items from training data
                logits[x.nonzero(as_tuple=True)] = .0

                # Fetching all predictions and ground_truth labels
                all_decoder_logits.append(logits.detach().cpu().numpy())
                all_decoder_targets.append(y.detach().cpu().numpy())

        writer.add_scalar('val/avg_loss', np.mean(losses), epoch)
        writer.add_scalar('val/avg_neg_lss', np.mean(neg_lls), epoch)
        writer.add_scalar('val/avg_weighted_KL', np.mean(weighted_KLs), epoch)

        avg_bal_acc = None
        if use_adv_network:
            avg_bal_acc = np.mean(adv_bal_accs)
            writer.add_scalar('val/adv_avg_loss', np.mean(adv_losses), epoch)
            writer.add_scalar('val/adv_avg_balanced_accuracy', avg_bal_acc, epoch)

        # calculating the metrics takes much more time than simply calculating
        # a loss. Therefore we want to calculate them much less often.
        # Note: Early-stopping and beta-annealing must still depend on the metrics, as they
        #       are what we actually want to improve on
        # Note2: For metrics that should be computed w.r.t. the user groups, e.g. for RecGap,
        #        the Dataloader must act deterministically (no resampling!)
        if calc_full_metrics:
            logits = np.concatenate(all_decoder_logits)
            targets = np.concatenate(all_decoder_targets)

            metric_scores = dict()
            for metric_lvl in metric_levels:
                for metric_name in metrics:
                    ug = user_groups_all_traits["gender"]
                    recgap, metric_val, metric_per_group = eval_recgap(ug, logits, targets, metric_name, metric_lvl,
                                                                       "val", return_avg_metric=True)

                    metric_str = f"{metric_name}_{metric_lvl}"
                    metric_scores[metric_str] = metric_val
                    metric_scores[f"recgap/{metric_str}"] = recgap

                    writer.add_scalar(f'val/{metric_str}', metric_val, epoch)
                    writer.add_scalar(f'val/recgap_{metric_str}', recgap, epoch)

                    for group_name, val in metric_per_group.items():
                        writer.add_scalar(f'val/{metric_str}_{group_name}', val, epoch)

            if use_adv_network:
                metric_scores.update({"adv_balanced_acc": avg_bal_acc})
            return metric_scores, avg_bal_acc

    return None, avg_bal_acc


def train_single(log_dir, run_name, config, use_adv_network, device, user_groups_all_traits,
                 tr_loader, vd_loader, log_batch_results_every, log_val_every, log_val_metrics_every,
                 traits, val_levels, val_metrics, verbose=True, store_best_model=False, store_model_every=0):
    # Setting seed for reproducibility
    reproducible(EXP_SEED)

    log_dir = os.path.join(log_dir, "vae")
    writer_path = os.path.join(log_dir, run_name)
    os.makedirs(writer_path, exist_ok=True)
    writer = SummaryWriter(writer_path)

    # Store full configuration to easily re-run a single experiment later on
    json_dump(config, os.path.join(writer_path, "config.json"))

    general_config = config.get("general") or dict()
    normalize_gradients = bool(general_config.get("normalize_gradients"))
    n_epochs = e if (e := general_config.get("n_epochs")) else VAE_MAX_EPOCHS

    adv_config = config.get("adv") or dict()
    adv_model_config, adv_loss_weight = None, None

    if use_adv_network:
        ld = ld if (ld := adv_config.get("latent_dropout")) else 0.5
        adv_model_config = {"grad_scaling": adv_config["grad_scaling"],
                            "latent_dropout": ld,
                            "adversaries": [adv_config["dims"]] * adv_config["n_adv"]}
        adv_loss_weight = adv_config["loss_weight"]

    earlystop_on_adv = bool(adv_config.get("earlystop_on_adv"))
    min_epochs_earlystopping = adv_config.get("min_epochs_earlystopping")
    if min_epochs_earlystopping is None:
        min_epochs_earlystopping = 10

    # Model definition, for simplicity of grid-search, let model itself decide which params to use
    model = MultiVAEAdv(**config["model"],
                        use_adv_network=use_adv_network,
                        adv_config=adv_model_config)

    if verbose:
        print("=" * 60)
        print("Model is ")
        print(model)
        print("=" * 60)

    model.to(device)

    vae_loss = VAE_loss(**config["loss"])
    ce_loss_fn = CrossEntropyLoss()

    default_opt_config = config["opt"]
    # enable possibility to have a different optimizer configuration for each part of the model
    # if not specified, it defaults to the general optimizer config
    encoder_opt_config = c if (c := config.get("enc_opt")) is not None else {}
    decoder_opt_config = c if (c := config.get("dec_opt")) is not None else {}
    vae_opt = torch.optim.Adam([
        {"params": model.encoder.parameters(), **encoder_opt_config},
        {"params": model.decoder.parameters(), **decoder_opt_config}
    ], **default_opt_config)

    adv_opt = None
    if use_adv_network:
        adv_opt_config = c if (c := config.get("adv_opt")) is not None else {}
        adv_opt = torch.optim.Adam([{
            "params": model.adversaries.parameters(), **adv_opt_config
        }], **default_opt_config)

    perform_warmup = adv_config.get("perform_warmup") or False

    we = adv_config.get("n_epochs_warmup")
    n_warmup_epochs = we if we is not None else 10

    store_last = bool(adv_config.get("store_last"))

    # Set seed again as especially for comparison of adv/no adv training, adv has more layers,
    # therefore the random initialization of them leads to a different random state at the current point
    reproducible(EXP_SEED)

    best_adv_score = 1
    best_scores = defaultdict(lambda: 0.)
    best_model_dicts = {}
    it = trange(n_epochs, desc="epochs") if verbose else range(n_epochs)
    vd_counter = 0
    for epoch in it:

        # --- Training --- #
        train(model, device, tr_loader, vae_loss, ce_loss_fn, writer, epoch,
              log_batch_results_every, use_adv_network, adv_loss_weight,
              vae_opt, adv_opt, perform_warmup, n_warmup_epochs=n_warmup_epochs,
              normalize_gradients=normalize_gradients)

        # --- Validation --- #
        if epoch % log_val_every == 0:

            # we want to calculate the validation metrics less often as they require
            # more computations and therefore more time.
            calc_full_metrics = vd_counter % log_val_metrics_every == 0
            vd_counter += 1

            metrics, adv_score = validate(model, device, vd_loader, vae_loss, ce_loss_fn, writer, epoch,
                                          calc_full_metrics, metrics=val_metrics, metric_levels=val_levels,
                                          traits=traits, user_groups_all_traits=user_groups_all_traits,
                                          use_adv_network=use_adv_network)

            # Add weights as arrays to tensorboard
            for i, (name, param) in enumerate(model.named_parameters()):
                if param.grad is not None:
                    writer.add_histogram(tag=f'val/param_{i}_{name}', values=param.cpu(),
                                         global_step=epoch)

            # Add gradients as arrays to tensorboard
            for i, (name, param) in enumerate(model.named_parameters()):
                if param.grad is not None:
                    writer.add_histogram(tag=f'val/gradients_{i}_{name}',
                                         values=param.grad.cpu(),
                                         global_step=epoch)

            if use_adv_network and earlystop_on_adv:
                # at the start of the training, the adversarial networks may perform randomly,
                # which is actually the lowest (best in our use-case) reasonable performance
                # we could achieve. Early stopping on these early epochs would however lead to
                # untrained models, and therefore a poor performance on the recommendation system
                # evaluation metric
                if min_epochs_earlystopping <= epoch:
                    # currently used adv evaluation metric is balanced accuracy, as we want to
                    # minimize it: the lower, the better
                    if adv_score <= best_adv_score:
                        best_adv_score = adv_score
                        model_params = {k: v.cpu() for k, v in model.state_dict().items()}
                        best_model_dicts["adv"] = model_params

                        if store_best_model:
                            torch.save(model_params, os.path.join(writer_path, f"best_model_adv.pt"))

            if calc_full_metrics:
                # for now we use ndcg for early stopping
                score = metrics[f"ndcg_{val_levels[0]}"]

                cur_beta = vae_loss.beta_step(score)
                writer.add_scalar("val/beta", cur_beta, epoch)

                for metric in val_metrics:
                    score = metrics[f"{metric}_{val_levels[0]}"]

                    if score >= best_scores[metric]:
                        best_scores[metric] = score
                        best_model_dicts[metric] = {k: v.cpu() for k, v in model.state_dict().items()}

                        if store_best_model:
                            torch.save(best_model_dicts[metric],
                                       os.path.join(writer_path, f"best_model_{metric}.pt"))

        writer.flush()

        if store_model_every > 0 and epoch % store_model_every == 0:
            torch.save(model.state_dict(), os.path.join(writer_path, f"adv_model_epoch_{epoch}.pt"))

    model.cpu()

    if store_last or store_model_every > 0:
        best_model_dicts["last"] = model.state_dict()
        torch.save(model.state_dict(), os.path.join(writer_path, f"adv_model_epoch_last.pt"))

    del model

    return dict(best_scores), dict(best_model_dicts)
