import copy
import json
import shutil
import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from utils import get_transforms, load_dataset, generate_images, compute_fid, weights_init, save_best_tuned_params
from sklearn.model_selection import ParameterSampler, ParameterGrid
from model_definitions.dcgan_model import DCGAN
from model_definitions.wgan_gp_model import WGAN_GP
from model_definitions.progan_model import ProGAN

# -------------------------------------------------------------------------------------------------------------------------------------------

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "real_vs_fake")

IMAGE_SIZE = 64
CHANNELS = 3
BATCH_SIZE = 128
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4
LATENT_DIM = 100


# -------------------------------------------------------------------------------------------------------------------------------------------
def tune_dcgan(train_loader, val_loader):
    params = {
        'lr': 2e-4,
        'adam_b1': 0.5,
        'adam_b2': 0.999,
        'batch_size': BATCH_SIZE,
        'num_epochs': 10,
        'feature_maps': 64,
        'image_size': 64,
    }

    model = DCGAN(
        latent_dim=LATENT_DIM,
        channels=CHANNELS,
        feature_maps=params['feature_maps']
    )

    return params, model


# -------------------------------------------------------------------------------------------------------------------------------------------
def tune_wgan_gp(train_loader, val_loader, img_size=IMAGE_SIZE, tuning=True):
    """Hyperparameter tuning function for WGAN-GP.

    Args:
        train_loader: Data loader for the training dataset used during tuning.
        val_loader: Data loader for the validation dataset used to evaluate candidate configurations.
        img_size (int, optional): Image size for the model/configuration to tune or load. Defaults to IMAGE_SIZE.
        tuning (bool, optional): If True, perform hyperparameter tuning; otherwise, load saved parameters from the WGAN-GP config file. Defaults to True.

    Returns:
        tuple: A tuple containing the selected hyperparameter dictionary and an initialized WGAN_GP model.
    """
    # NOTE: THINGS TO CONSIDER TUNING:
    # - Learning rate -> 1e-4, 2e-4, 5e-5
    # - n_critic -> 3, 5, 7
    # - feature maps -> 32, 64, 128

    if not tuning:
        with open(os.path.join("configs", "wgan_gp_config.json"), "r") as f:
            # params = json.load(f)
            all_configs = json.load(f)
            params = all_configs[f"img_size_{img_size}"]
        return params, WGAN_GP(img_size=img_size, latent_dim=LATENT_DIM,
                               channels=CHANNELS, feature_maps=params['feature_maps'])

    # reaches here if tuning is True
    if img_size == 64:
        search_params = {
            'lr': [1e-4, 2e-4, 5e-5], # we will see about how 5e-5 does, but it is pretty small so I expect to not want to try that out
            'n_critic': [3, 5, 7], # 7 isn't given a fair chance, but paper suggests 5, and 7 may be too expensive for our time constraints
            'feature_maps': [32, 64, 128]
        }
    else:
        # for 128 image size, use this: -> 18 configs to try x 40 epochs each -> then take the best one
        search_params = {
            'lr': [5e-5, 1e-4, 1.5e-4], # we will see about how 5e-5 does, but it is pretty small so I expect to not want to try that out
            'n_critic': [3, 5], # 7 isn't given a fair chance, but paper suggests 5, and 7 may be too expensive for our time constraints
            'feature_maps': [32, 64, 128]
        }

    fixed_params = {
        'adam_b1': 0.0,
        'adam_b2': 0.9,
        'batch_size': BATCH_SIZE
    }

    tune_epochs = 20  # short runs per config

    # param_configs = list(ParameterGrid(search_params))  # 27 combos - too many for now
    param_configs = list(ParameterSampler(search_params, n_iter=5, random_state=SEED))
    real_val_dir = os.path.join(DATA_ROOT, "valid", "real")

    best_checkpoint_path = ""
    best_fid = float('inf')
    best_params = None
    best_model = WGAN_GP(
        img_size=img_size,
        latent_dim=LATENT_DIM,
        channels=CHANNELS,
        feature_maps=64
    )

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("output/wgan_gp/tune_wgan_temp", exist_ok=True)

    for idx, config in enumerate(param_configs):
        params = {**fixed_params, **config, 'num_epochs': tune_epochs}
        print(f"\n--- Tuning config {idx + 1}/{len(param_configs)}: {config} ---")

        tune_model_path = f"checkpoints/wgan_gp_tune_{idx + 1}_{img_size}.pt"

        # build fresh model per config because of feature_maps
        model = WGAN_GP(img_size=img_size, latent_dim=LATENT_DIM,
                        channels=CHANNELS, feature_maps=config['feature_maps'])

        # train_wgan_gp handles .to(DEVICE), weights_init, .train() internally
        train_wgan_gp(train_loader, model, params, img_size=img_size, model_path=tune_model_path)

        # evaluate with FID on validation set
        model.eval()
        fake_dir = generate_images(model.generator, len(val_loader.dataset),
                                   "output/wgan_gp/tune_wgan_temp", BATCH_SIZE, LATENT_DIM, DEVICE)
        fid = compute_fid(real_val_dir, fake_dir, BATCH_SIZE, DEVICE)
        print(f"Config FID: {fid:.4f}")

        if fid < best_fid:
            best_fid = fid
            best_params = params
            best_model = model
            best_checkpoint_path = tune_model_path

    print(f"\nBest config: {best_params}, FID: {best_fid:.4f}")

    # rebuild fresh model with best feature_maps for full training
    if best_params is not None:
        best_params['num_epochs'] = 200  # set full training epochs
        best_model = WGAN_GP(img_size=img_size, latent_dim=LATENT_DIM,
                             channels=CHANNELS, feature_maps=best_params['feature_maps'])
        checkpoint = torch.load(best_checkpoint_path, map_location=DEVICE)
        best_model.generator.load_state_dict(checkpoint['generator_state_dict'])
        best_model.critic.load_state_dict(checkpoint['critic_state_dict'])
        best_params['critic_optimizer_state'] = checkpoint['critic_optimizer_state_dict']
        best_params['generator_optimizer_state'] = checkpoint['generator_optimizer_state_dict']

        # Save best config for future runs without tuning
        save_best_tuned_params(best_params, img_size, file_name="wgan_gp_config.json")

        best_params['start_epoch'] = tune_epochs

    # CLEANUP - remove the directories that were made during training
    if os.path.isdir("checkpoints"):
        shutil.rmtree("checkpoints")
    if os.path.isdir("output/wgan_gp/tune_wgan_temp"):
        shutil.rmtree("output/wgan_gp/tune_wgan_temp")

    return best_params, best_model


# -----------------------------------------------------------------------------------------------------------------------
def tune_progan(train_loader, val_loader, real_val_dir, img_size=IMAGE_SIZE, tuning=True):
    """Run a small amount of epochs on several different configs - save the best one and return to it in the tuning loop

    Args:
        train_loader (_type_): _description_
        val_loader (_type_): _description_

    Returns:
        dict: The best hyperparameter configuration found during tuning
        model: The model initialized with the best hyperparameter configuration

    NOTE: AI ASSISTED WITH THIS FUNCTION
    ProGAN paper uses these defaults
    """

    if not tuning:
        with open(os.path.join("configs", "progan_config.json"), "r") as f:
            all_configs = json.load(f)
            params = all_configs[f"img_size_{img_size}"]
        params['img_size'] = img_size
        return params, ProGAN(latent_dim=LATENT_DIM, channels=CHANNELS, feature_maps=params['feature_maps'])

    configs = [
        {'num_epochs_per_step': 8, 'learning_rate': 0.001, 'fade_in_epochs': 3, 'feature_maps': 512},
        {'num_epochs_per_step': 10, 'learning_rate': 0.0005, 'fade_in_epochs': 4, 'feature_maps': 256},
    ]

    fixed_params = {
        'beta1': 0.0,
        'beta2': 0.99,
        'batch_size': BATCH_SIZE,
    }

    best_fid = float('inf')
    best_params = None
    best_model = None

    for idx, cfg in enumerate(configs):
        params = {**fixed_params, **cfg}
        print(f"\n--- ProGAN tuning config {idx + 1}/{len(configs)}: {cfg} ---")

        progan = ProGAN(latent_dim=LATENT_DIM, channels=CHANNELS, feature_maps=cfg['feature_maps'])
        params['img_size'] = img_size
        train_progan(progan, train_loader, params)

        # generate fake images for FID
        progan.to(DEVICE)
        progan.eval()
        if os.path.isdir("output/progan/tune_temp"):
            shutil.rmtree("output/progan/tune_temp")
        os.makedirs("output/progan/tune_temp", exist_ok=True)

        max_step = int(np.log2(img_size)) - 2
        num_val = len(val_loader.dataset)
        fake_dir = generate_images(progan.gen, num_val, "output/progan/tune_temp",
                                   BATCH_SIZE, LATENT_DIM, DEVICE,
                                   step=max_step, alpha=1.0)

        fid_score = compute_fid(real_val_dir, fake_dir, BATCH_SIZE, DEVICE)
        print(f"Config FID: {fid_score:.4f}")

        # cleanup temp images
        shutil.rmtree("output/progan/tune_temp")
        os.makedirs("output/progan/tune_temp", exist_ok=True)

        if fid_score < best_fid:
            best_fid = fid_score
            best_params = params
            best_model = progan

    # final cleanup
    if os.path.isdir("output/progan/tune_temp"):
        shutil.rmtree("output/progan/tune_temp", ignore_errors=True)

    print(f"\nBest ProGAN config: {best_params}, FID: {best_fid:.4f}")

    if best_params is not None:
        save_best_tuned_params(best_params, img_size, file_name="progan_config.json")

    return best_params, best_model


# -------------------------------------------------------------------------------------------------------------------------------------------


def train_dcgan(train_loader, model, params, val_dir=None, num_val_samples=None, model_path=None):
    """Complete training on best configs - run however many epochs are specified in params until convergence."""
    if model_path is None:
        model_path = f"models/dcgan_model_{params.get('image_size', 64)}.pt"

    criterion = torch.nn.BCELoss()

    discriminator_optimizer = torch.optim.Adam(
        model.discriminator.parameters(),
        lr=params['lr'],
        betas=(params['adam_b1'], params['adam_b2'])
    )
    generator_optimizer = torch.optim.Adam(
        model.generator.parameters(),
        lr=params['lr'],
        betas=(params['adam_b1'], params['adam_b2'])
    )

    model.to(DEVICE)
    model.apply(weights_init)
    model.train()

    best_fid = float('inf')

    for epoch in range(params['num_epochs']):
        disc_loss_sum = 0.0
        disc_loss_count = 0
        gen_loss_sum = 0.0
        gen_loss_count = 0

        print(f"Epoch {epoch + 1}/{params['num_epochs']}")

        for real_data_batch in train_loader:
            x_real = real_data_batch[0].to(DEVICE)
            batch_size = x_real.size(0)

            real_labels = torch.ones(batch_size, 1, device=DEVICE)
            fake_labels = torch.zeros(batch_size, 1, device=DEVICE)

            # Train Discriminator
            z = torch.randn(batch_size, LATENT_DIM, 1, 1, device=DEVICE)
            with torch.no_grad():
                x_fake = model.generator(z)

            disc_out_real = model.discriminator(x_real)
            disc_out_fake = model.discriminator(x_fake)

            disc_real_loss = criterion(disc_out_real, real_labels)
            disc_fake_loss = criterion(disc_out_fake, fake_labels)
            disc_loss = disc_real_loss + disc_fake_loss

            discriminator_optimizer.zero_grad()
            disc_loss.backward()
            discriminator_optimizer.step()

            disc_loss_sum += disc_loss.item()
            disc_loss_count += 1

            # Train Generator
            for p in model.discriminator.parameters():
                p.requires_grad_(False)

            z = torch.randn(batch_size, LATENT_DIM, 1, 1, device=DEVICE)
            x_fake = model.generator(z)
            gen_out = model.discriminator(x_fake)

            gen_loss = criterion(gen_out, real_labels)

            generator_optimizer.zero_grad()
            gen_loss.backward()
            generator_optimizer.step()

            for p in model.discriminator.parameters():
                p.requires_grad_(True)

            gen_loss_sum += gen_loss.item()
            gen_loss_count += 1

        avg_disc_loss = disc_loss_sum / disc_loss_count if disc_loss_count > 0 else float('inf')
        avg_gen_loss = gen_loss_sum / gen_loss_count if gen_loss_count > 0 else float('inf')

        print(f"Discriminator loss: {avg_disc_loss:.4f} - Generator loss: {avg_gen_loss:.4f}")

        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': model.generator.state_dict(),
            'discriminator_state_dict': model.discriminator.state_dict(),
            'generator_optimizer_state_dict': generator_optimizer.state_dict(),
            'discriminator_optimizer_state_dict': discriminator_optimizer.state_dict(),
            'params': params,
            'discriminator_loss': avg_disc_loss,
            'generator_loss': avg_gen_loss,
        }

        use_val = (val_dir is not None) and (num_val_samples is not None)
        is_eval_epoch = use_val and ((epoch + 1) % 10 == 0 or epoch + 1 == params['num_epochs'])

        if not use_val:
            # fallback: save latest checkpoint if no validation data is provided
            torch.save(checkpoint, model_path)

        elif is_eval_epoch:
            model.eval()
            fake_dir = generate_images(
                model.generator,
                num_val_samples,
                "output/dcgan/fid_temp",
                BATCH_SIZE,
                LATENT_DIM,
                DEVICE
            )
            fid = compute_fid(val_dir, fake_dir, BATCH_SIZE, DEVICE)
            checkpoint['fid'] = fid

            if os.path.isdir(fake_dir):
                shutil.rmtree(fake_dir)

            print(f"Validation FID: {fid:.4f}")
            if fid < best_fid:
                best_fid = fid
                torch.save(checkpoint, model_path)
                print(f"New best model - {model_path}, with FID: {fid:.4f}")
            else:
                print(f"No FID improvement - {fid:.4f} over best {best_fid:.4f}")

            model.train()

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        model.generator.load_state_dict(checkpoint['generator_state_dict'])
        model.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])


# -------------------------------------------------------------------------------------------------------------------------------------------

def train_wgan_gp(train_loader, model: WGAN_GP, params, img_size=IMAGE_SIZE, val_dir=None, num_val_samples=None,
                  model_path=None):
    """Training loop for WGAN-GP

    Args:
        train_loader (DataLoader): DataLoader for training data
        model (WGAN_GP): WGAN-GP model instance
        params (dict): Hyperparameter configuration
        img_size (int, optional): Image resolution, used for checkpoint naming. Defaults to IMAGE_SIZE.
        val_dir (str, optional): Directory containing validation images for FID calculation. Defaults to None.
        num_val_samples (int, optional): Number of validation samples to generate for FID calculation. Defaults to None.
        model_path (str, optional): Path to save the model checkpoint. Defaults to None.
    """

    if model_path is None:
        model_path = f"models/wgan_gp_model_{img_size}.pt"

    critic_optimizer = torch.optim.Adam(model.critic.parameters(), lr=params['lr'],
                                        betas=(params['adam_b1'], params['adam_b2']))
    generator_optimizer = torch.optim.Adam(model.generator.parameters(), lr=params['lr'],
                                           betas=(params['adam_b1'], params['adam_b2']))

    model.to(DEVICE)

    if 'critic_optimizer_state' in params:
        critic_optimizer.load_state_dict(params.get('critic_optimizer_state'))
    if 'generator_optimizer_state' in params:
        generator_optimizer.load_state_dict(params.get('generator_optimizer_state'))
    
    # CosineAnnealingLR - smoothly decays lr over training, only during full training
    use_val = (val_dir is not None) and (num_val_samples is not None)
    if use_val:
        eta_min = params['lr'] * 0.25  # decay to 5% of initial lr
        critic_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            critic_optimizer, T_max=params['num_epochs'], eta_min=eta_min)
        gen_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            generator_optimizer, T_max=params['num_epochs'], eta_min=eta_min)
        # fast-forward schedulers if resuming from a checkpoint
        for _ in range(params.get('start_epoch', 0)):
            critic_scheduler.step()
            gen_scheduler.step()
    
    # if we are starting from scratch, load weight distribution recommended by paper
    # otherwise keep the weights from loaded model
    if params.get('start_epoch', 0) == 0:
        model.apply(weights_init)
    model.train()
    
    history = {
        'epoch': [],
        'critic_loss': [],
        'generator_loss': [],
        'fid': [],
        'lr': []
    }

    best_fid = float('inf')
    for epoch in range(params.get('start_epoch', 0), params['num_epochs']):
        gen_loss_sum = 0.0
        gen_loss_count = 0
        print(f"Epoch {epoch + 1}/{params['num_epochs']}")
        critic_loss = torch.tensor(0.0)  # initialize for print later
        generator_loss = torch.tensor(0.0)  # initialize for print later
        for i, real_data_batch in enumerate(train_loader):
            # CRITIC TRAINING
            # sample real data, latent var (z), and a random number
            x_real = real_data_batch[0].to(DEVICE)  # real data batch (x in WGAN-GP paper)
            z = torch.randn(x_real.size(0), LATENT_DIM, 1, 1, device=DEVICE)  # latent variable (z) as labeled by paper
            # Generate fake data with no gradient tracking to save memory and computation
            with torch.no_grad():
                x_fake: torch.Tensor = model.generator(z)

            # CRITIC SECTION
            critic_out_real: torch.Tensor = model.critic(x_real)
            critic_out_fake: torch.Tensor = model.critic(x_fake.detach())
            critic_loss = (torch.mean(critic_out_fake) - torch.mean(
                critic_out_real)) + model.calculate_gradient_penalty(x_fake, x_real)

            # UPDATE CRITIC WEIGHTS
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # NOTE: AI SUGGESTED THIS IF CONDITION TO REPLACE THE NESTED FOR LOOP
            if (i + 1) % params[
                'n_critic'] == 0:  # this block is only entered every n_critic steps -> this replaced the third loop
                for p in model.critic.parameters():
                    p.requires_grad_(False)

                # GENERATOR TRAINING
                z = torch.randn(params['batch_size'], LATENT_DIM, 1, 1, device=DEVICE)
                x_fake: torch.Tensor = model.generator(z)
                generator_loss = -torch.mean(model.critic(x_fake))

                gen_loss_sum += generator_loss.item()
                gen_loss_count += 1

                # UPDATE GENERATOR WEIGHTS
                generator_optimizer.zero_grad()
                generator_loss.backward()
                generator_optimizer.step()

                # Unfreeze critic params for next critic update
                for p in model.critic.parameters():
                    p.requires_grad_(True)

        avg_gen_loss = gen_loss_sum / gen_loss_count if gen_loss_count > 0 else float('inf')
        print(f"Critic loss: {critic_loss.item():.4f} - Generator loss: {avg_gen_loss:.4f}")
        
        history['epoch'].append(epoch)
        history['critic_loss'].append(critic_loss.item())
        history['generator_loss'].append(avg_gen_loss)
        history['lr'].append(critic_optimizer.param_groups[0]['lr'])

        # CHECKPOINTING - AI ASSISTED WITH THIS LOGIC
        use_val = (val_dir is not None) and (num_val_samples is not None)
        is_eval_epoch = use_val and ((epoch + 1) % 10 == 0 or epoch + 1 == params['num_epochs'])

        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': model.generator.state_dict(),
            'critic_state_dict': model.critic.state_dict(),
            'generator_optimizer_state_dict': generator_optimizer.state_dict(),
            'critic_optimizer_state_dict': critic_optimizer.state_dict(),
            'params': {k: v for k, v in params.items()
                       # AI TO FIX THIS LINE TO PREVENT ADDING MORE PARAMS THAN I MEANT TO
                       if k not in ('critic_optimizer_state', 'generator_optimizer_state')},
        }

        if not use_val:
            # tuning: always save latest weights so checkpoint matches in-memory model
            checkpoint['critic_loss'] = critic_loss.item()
            checkpoint['generator_loss'] = avg_gen_loss
            torch.save(checkpoint, model_path)
            history['fid'].append(None)

        elif is_eval_epoch:
            # full training: FID-based checkpointing every 10 epochs or on final epoch
            model.eval()
            fake_dir = generate_images(model.generator, num_val_samples,
                                       "output/wgan_gp/fid_temp", BATCH_SIZE, LATENT_DIM, DEVICE)
            fid = compute_fid(val_dir, fake_dir, BATCH_SIZE, DEVICE)
            checkpoint['fid'] = fid
            history['fid'].append(fid) 

            if os.path.isdir(fake_dir):
                shutil.rmtree(fake_dir)

            print(f"Validation FID: {fid:.4f}")
            if fid < best_fid:
                best_fid = fid
                torch.save(checkpoint, model_path)
                print(f"New best model - {model_path}, with FID: {fid:.4f}")
            else:
                print(f"No FID improvement - {fid:.4f} over best {best_fid:.4f}")
            model.train()
        else:
            # non-eval epoch with validation enabled — no FID computed
            history['fid'].append(None)
        
        # Reduce LR every epoch -> should happen slowly
        if use_val:
            critic_scheduler.step()
            gen_scheduler.step()
            if (epoch + 1) % 10 == 0:
                print(f"LR: {critic_optimizer.param_groups[0]['lr']:.6f}")
    return history
        


# -------------------------------------------------------------------------------------------------------------------------------------------

def train_progan(progan, train_loader, params, val_dir=None, num_val_samples=None):
    """Complete training on best configs - run however many epochs are specified in params until convergence

    Args:
        model (_type_): _description_
        params (_type_): _description_

    NOTE: AI ASSISTED WITH THIS FUNCTION
    """
    img_size = params.get('img_size', 64)
    default_params = {
        'learning_rate': 0.0002,
        'beta1': 0.5,
        'beta2': 0.999,
        'batch_size': 64,
        'num_epochs': 100,
        'num_epochs_per_step': 100,
        'fade_in_epochs': 0,
    }
    if params is None:
        params = {}

    params = {**default_params, **params}

    model = progan

    use_fid = (val_dir is not None) and (num_val_samples is not None)
    best_fid = float('inf')
    model_path = f'models/progan_model_{img_size}.pt'
    os.makedirs('models', exist_ok=True)

    gen_optimizer = torch.optim.Adam(
        model.gen.parameters(),
        lr=params['learning_rate'],
        betas=(params['beta1'], params['beta2'])
    )
    disc_optimizer = torch.optim.Adam(
        model.disc.parameters(),
        lr=params['learning_rate'],
        betas=(params['beta1'], params['beta2'])
    )
    model.to(DEVICE)
    model.train()
    model.gen.train()
    model.disc.train()

    resolutions = [4, 8, 16, 32, 64, 128]
    max_step = resolutions.index(img_size) + 1

    base_dataset = train_loader.dataset

    for step in range(max_step):
        res = resolutions[step]
        step_transform = get_transforms(res, CHANNELS)
        step_dataset = copy.copy(base_dataset)
        step_dataset.transform = step_transform
        step_loader = DataLoader(step_dataset, batch_size=params['batch_size'], shuffle=True,
                                 num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)

        for epoch in range(params['num_epochs_per_step']):
            # alpha: fade-in during first few epochs, then 1.0
            if epoch < params['fade_in_epochs'] and step > 0:
                alpha = epoch / params['fade_in_epochs']
            else:
                alpha = 1.0

            gen_loss_sum = 0.0
            disc_loss_sum = 0.0
            num_batches = 0

            for i, batch_data in enumerate(step_loader):
                x_real = batch_data[0].to(DEVICE)
                batch_size = x_real.size(0)

                # ---- Train Discriminator ----
                z = torch.randn(batch_size, LATENT_DIM, 1, 1, device=DEVICE)
                with torch.no_grad():
                    x_fake = model.gen(z, step=step, alpha=alpha)

                disc_real = model.disc(x_real, step=step, alpha=alpha)
                disc_fake = model.disc(x_fake.detach(), step=step, alpha=alpha)

                # WGAN-style loss (no sigmoid, raw scores)
                disc_loss = torch.mean(disc_fake) - torch.mean(disc_real)

                # gradient penalty (same idea as WGAN-GP)
                epsilon = torch.rand(batch_size, 1, 1, 1, device=DEVICE)
                x_interp = (epsilon * x_real + (1 - epsilon) * x_fake.detach()).requires_grad_(True)
                disc_interp = model.disc(x_interp, step=step, alpha=alpha)
                gradients = torch.autograd.grad(
                    outputs=disc_interp,
                    inputs=x_interp,
                    grad_outputs=torch.ones_like(disc_interp),
                    create_graph=True
                )[0]
                gp = 10 * ((gradients.reshape(batch_size, -1).norm(2, dim=1) - 1) ** 2).mean()
                disc_loss = disc_loss + gp

                disc_optimizer.zero_grad()
                disc_loss.backward()
                disc_optimizer.step()

                # ---- Train Generator ----
                z = torch.randn(batch_size, LATENT_DIM, 1, 1, device=DEVICE)
                x_fake = model.gen(z, step=step, alpha=alpha)
                gen_loss = -torch.mean(model.disc(x_fake, step=step, alpha=alpha))

                gen_optimizer.zero_grad()
                gen_loss.backward()
                gen_optimizer.step()

                gen_loss_sum += gen_loss.item()
                disc_loss_sum += disc_loss.item()
                num_batches += 1

            avg_gen = gen_loss_sum / num_batches
            avg_disc = disc_loss_sum / num_batches
            print(
                f"  Step {step + 1}/{max_step} | Epoch {epoch + 1}/{params['num_epochs_per_step']} | alpha: {alpha:.2f} | D loss: {avg_disc:.4f} | G loss: {avg_gen:.4f}")

            is_eval_epoch = use_fid and ((epoch + 1) % 10 == 0 or epoch + 1 == params['num_epochs_per_step'])
            is_final_step = (step == max_step - 1)

            checkpoint = {
                'step': step,
                'alpha': alpha,
                'epoch': epoch,
                'generator_state_dict': model.gen.state_dict(),
                'discriminator_state_dict': model.disc.state_dict(),
                'gen_optimizer_state_dict': gen_optimizer.state_dict(),
                'disc_optimizer_state_dict': disc_optimizer.state_dict(),
                'gen_loss': avg_gen,
                'disc_loss': avg_disc,
                'params': params,
            }

            if use_fid and is_eval_epoch and is_final_step:
                model.eval()
                fake_dir = generate_images(model.gen, num_val_samples,
                                           "output/progan/fid_temp",
                                           BATCH_SIZE, LATENT_DIM, DEVICE,
                                           step=step, alpha=alpha)
                fid = compute_fid(val_dir, fake_dir, BATCH_SIZE, DEVICE)
                checkpoint['fid'] = fid

                if os.path.isdir(fake_dir):
                    shutil.rmtree(fake_dir)

                print(f"  Validation FID: {fid:.4f}")
                if fid < best_fid:
                    best_fid = fid
                    torch.save(checkpoint, model_path)
                    print(f"  New best model saved - FID: {fid:.4f}")
                else:
                    print(f"  No FID improvement ({fid:.4f} vs best {best_fid:.4f})")
                model.train()

            elif not use_fid:
                torch.save(checkpoint, model_path)

        print(f"\nTraining complete! Best FID: {best_fid:.4f}" if use_fid else "\nTraining complete!")


# -----------------------------------------------------------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------------------------------------------------------
# main function
def main():
    # add argparses for model to run and image size
    parser = argparse.ArgumentParser(description="Train GAN models")
    parser.add_argument("--model", type=str, choices=["dcgan", "wgan_gp", "progan"], required=True,
                        help="Model to train")
    parser.add_argument("--size", type=int, choices=[64, 128], default=IMAGE_SIZE,
                        help="Image size for training")  # default to IMAGE_SIZE - 64x64
    args = parser.parse_args()

    # Check for incompatible model and image size combination (DCGAN only supports 64x64)
    if args.model == "dcgan" and args.size != 64:
        parser.error("DCGAN only supports --size 64")

    # when running training, the commandline tells us which model to do for now
    img_size = args.size
    model_choice = args.model

    # load data
    train_loader, train_dataset = load_dataset("train", DATA_ROOT, img_size, CHANNELS, BATCH_SIZE, NUM_WORKERS)
    val_loader, val_dataset = load_dataset("valid", DATA_ROOT, img_size, CHANNELS, BATCH_SIZE, NUM_WORKERS)
    test_loader, test_dataset = load_dataset("test", DATA_ROOT, img_size, CHANNELS, BATCH_SIZE, NUM_WORKERS)

    # define models as None until one is used
    dcgan = None
    wgan_gp = None
    progan = None

    os.makedirs("output", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    if model_choice == "dcgan":
        dc_params, dcgan = tune_dcgan(train_loader, val_loader)

        real_val_dir = os.path.join(DATA_ROOT, "valid", "real")
        os.makedirs("output/dcgan/fid_temp", exist_ok=True)

        train_dcgan(
            train_loader,
            dcgan,
            dc_params,
            val_dir=real_val_dir,
            num_val_samples=len(val_dataset)
        )

        if os.path.isdir("output/dcgan/fid_temp"):
            shutil.rmtree("output/dcgan/fid_temp")
    elif model_choice == "wgan_gp":
        wgan_params, wgan_gp = tune_wgan_gp(train_loader, val_loader, img_size=img_size, tuning=False)

        real_val_dir = os.path.join(DATA_ROOT, "valid", "real")
        os.makedirs("output/wgan_gp/fid_temp", exist_ok=True)

        train_wgan_gp(train_loader, wgan_gp, wgan_params, img_size=img_size,
                      val_dir=real_val_dir, num_val_samples=len(val_dataset))
        if os.path.isdir("output/wgan_gp/fid_temp"):
            shutil.rmtree("output/wgan_gp/fid_temp")
    elif model_choice == "progan":
        real_val_dir = os.path.join(DATA_ROOT, "valid", "real")

        progan_params, progan = tune_progan(
            train_loader,
            val_loader,
            real_val_dir,
            img_size=img_size,
            tuning=False)

        train_progan(progan, train_loader, progan_params, val_dir=real_val_dir, num_val_samples=len(val_dataset))

    # FOR LOADING EXISTING MODELS
    """
    # 1. Instantiate the model with the same architecture
    # model = WGAN_GP(img_size=64, latent_dim=LATENT_DIM, channels=CHANNELS, feature_maps=64)
    # model.to(DEVICE)

    # # 2. Load the checkpoint
    # checkpoint = torch.load("models/wgan_gp_model_64.pt", map_location=DEVICE)

    # # 3. Load state dicts
    # model.generator.load_state_dict(checkpoint['generator_state_dict'])
    # model.critic.load_state_dict(checkpoint['critic_state_dict'])

    # # 4. Set to eval mode for inference (or train mode to resume training)
    # model.eval()

    # TESTING BELOW -> until this is worked on more, commented out for now
    """

    # generate fake images from each model
    real_test_dir = os.path.join(DATA_ROOT, "test", "real")
    num_test = len(test_dataset)
    if dcgan is not None:
        dcgan_fake_dir = generate_images(dcgan.generator, num_test, "output/dcgan_fakes", BATCH_SIZE, LATENT_DIM,
                                         DEVICE)
        dcgan_fid = compute_fid(real_test_dir, dcgan_fake_dir, BATCH_SIZE, DEVICE)
        print(f"DCGAN FID: {dcgan_fid:.4f}")
    elif wgan_gp is not None:
        checkpoint = torch.load(f"models/wgan_gp_model_{img_size}.pt", map_location=DEVICE)
        wgan_gp.generator.load_state_dict(checkpoint['generator_state_dict'])
        wgan_gp.critic.load_state_dict(checkpoint['critic_state_dict'])

        wgan_gp.eval()
        wgan_gp_fake_dir = generate_images(wgan_gp.generator, num_test, f"output/wgan_gp_{img_size}", BATCH_SIZE,
                                           LATENT_DIM, DEVICE)
        wgan_gp_fid = compute_fid(real_test_dir, wgan_gp_fake_dir, BATCH_SIZE, DEVICE)
        print(f"WGAN-GP FID: {wgan_gp_fid:.4f}")
    elif progan is not None:
        # load best checkpoint
        progan.to(DEVICE)
        checkpoint = torch.load(f"models/progan_model_{img_size}.pt", map_location=DEVICE, weights_only=False)
        progan.gen.load_state_dict(checkpoint['generator_state_dict'])
        progan.disc.load_state_dict(checkpoint['discriminator_state_dict'])

        max_step = checkpoint.get('step', int(np.log2(img_size)) - 2)
        alpha = checkpoint.get('alpha', 1.0)
        progan.eval()

        # generate images at final resolution
        progan_fake_dir = generate_images(progan.gen, num_test, f"output/progan_{img_size}",
                                          BATCH_SIZE, LATENT_DIM, DEVICE,
                                          step=max_step, alpha=alpha)

        progan_fid = compute_fid(real_test_dir, progan_fake_dir, BATCH_SIZE, DEVICE)
        print(f"ProGAN FID: {progan_fid:.4f}")


if __name__ == "__main__":
    main()

