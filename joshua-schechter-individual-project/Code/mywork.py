# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# wgan_gp_model.py

import numpy as np
import torch
import torch.nn as nn


class Generator_WGAN_GP(nn.Module):
    """Generator class for WGAN-GP model.
    
    Args:
        img_size (int): Size of the output image (assumed to be square).
        latent_dim (int): Dimensionality of the input noise vector.
        channels (int): Number of channels in the output image.
        feature_maps (int, optional): Base number of feature maps. Defaults to 64.
    """
    def __init__(self, img_size, latent_dim, channels, feature_maps=64):
        
        super(Generator_WGAN_GP, self).__init__()

        n_stages = int(np.log2(img_size) - 2)

        self.network = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=latent_dim,
                out_channels=feature_maps * (2 ** (n_stages - 1)),
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(feature_maps * (2 ** (n_stages - 1))),
            
            nn.ReLU(True)
        )

        for i in range(n_stages - 1, 0, -1):
            self.network.append(
                nn.ConvTranspose2d(
                    in_channels=feature_maps * (2 ** i),
                    out_channels=feature_maps * (2 ** (i - 1)),
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False
                )
            )
            self.network.append(nn.BatchNorm2d(feature_maps * (2 ** (i - 1))))
            self.network.append(nn.ReLU(True))

        self.network.append(
            nn.ConvTranspose2d(
                in_channels=feature_maps,
                out_channels=channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            )
        )
        self.network.append(nn.Tanh())

    def forward(self, x):
        return self.network(x)


class Critic_WGAN_GP(nn.Module):
    """Critic class for WGAN-GP model.
    
    Args:
    img_size (int): Size of the input image (assumed to be square).
    channels (int): Number of channels in the input image.
    feature_maps (int, optional): Base number of feature maps. Defaults to 64.
    """
    def __init__(self, img_size, channels, feature_maps=64):
        super(Critic_WGAN_GP, self).__init__()

        n_stages = int(np.log2(img_size) - 2)

        self.network = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=feature_maps,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        for i in range(1, n_stages):
            self.network.append(
                nn.Conv2d(
                    in_channels=feature_maps * (2 ** (i - 1)),
                    out_channels=feature_maps * (2 ** i),
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False
                )
            )
            self.network.append(nn.GroupNorm(1, feature_maps * (2 ** i)))
            self.network.append(nn.LeakyReLU(0.2, inplace=True))

        self.network.append(
            nn.Conv2d(
                in_channels=feature_maps * (2 ** (n_stages - 1)),
                out_channels=1,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False
            )
        )

    def forward(self, x):
        output = self.network(x)
        return output.view(output.size(0), -1)


class WGAN_GP(nn.Module):
    """WGAN-GP architecture class.
    
    Args:
        img_size (int): Size of the input image (assumed to be square).
        latent_dim (int): Dimensionality of the input noise vector.
        channels (int): Number of channels in the input image.
        feature_maps (int, optional): Base number of feature maps. Defaults to 64.
    """
    def __init__(self, img_size, latent_dim, channels, feature_maps=64):
        super(WGAN_GP, self).__init__()
        self.generator = Generator_WGAN_GP(
            img_size=img_size,
            latent_dim=latent_dim,
            channels=channels,
            feature_maps=feature_maps
        )
        self.critic = Critic_WGAN_GP(
            img_size=img_size,
            channels=channels,
            feature_maps=feature_maps
        )

    def calculate_gradient_penalty(self, x_fake: torch.Tensor, x_real: torch.Tensor) -> torch.Tensor:
        """Custom gradient penalty method for wgan-gp. This is the principal componenet of the class

        Args:
            x_fake (torch.Tensor): _description_
            x_real (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        epsilon = torch.rand(
            x_real.size(0), 1, 1, 1,
            device=x_real.device,
            dtype=x_real.dtype
        )
        x_interpolated = epsilon * x_real + (1 - epsilon) * x_fake.detach()
        x_interpolated.requires_grad_(True)

        lamb = 10

        critic_interpolates = self.critic(x_interpolated)

        gradients = torch.autograd.grad(
            outputs=critic_interpolates,
            inputs=x_interpolated,
            grad_outputs=torch.ones_like(critic_interpolates),
            create_graph=True,
        )[0]

        gradients = gradients.reshape(x_real.size(0), -1)
        gradient_norm = gradients.norm(2, dim=1)
        penalty = lamb * ((gradient_norm - 1) ** 2).mean()

        return penalty


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# train.py


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
            'lr': [1e-4, 1.5e-4, 2e-4], # we will see about how 5e-5 does, but it is pretty small so I expect to not want to try that out
            'n_critic': [3, 5], # 7 isn't given a fair chance, but paper suggests 5, and 7 may be too expensive for our time constraints
            'feature_maps': [64, 128]
        }
    else:
        # for 128 image size, use this: -> 12 configs to try x 30 epochs each -> then take the best one
        search_params = {
            'lr': [1e-4, 1.5e-4, 2e-4], # trying out slightly variable lrs
            'n_critic': [3, 5], # 7 isn't given a fair chance, but paper suggests 5, and 7 may be too expensive for our time constraints
            'feature_maps': [64, 128] # 32 is not useful
        }

    fixed_params = {
        'adam_b1': 0.0,
        'adam_b2': 0.9,
        'batch_size': BATCH_SIZE
    }

    tune_epochs = 30  # short runs per config

    param_configs = list(ParameterGrid(search_params))  # 27 combos - too many for now
    # param_configs = list(ParameterSampler(search_params, n_iter=5, random_state=SEED))
    # real_val_dir = os.path.join(DATA_ROOT, "valid", "real")
    real_val_dir = export_real_images_for_fid(
        split="valid",
        data_root=DATA_ROOT,
        image_size=img_size,
        save_dir=os.path.join("output", "fid_cache", f"valid_real_{img_size}")
    )

    # best_checkpoint_path = ""
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
            # best_checkpoint_path = tune_model_path

    print(f"\nBest config: {best_params}, FID: {best_fid:.4f}")

    # rebuild fresh model with best feature_maps for full training
    if best_params is not None:
        best_params['num_epochs'] = 350  # set full training epochs; early stopping can end sooner
        best_model = WGAN_GP(img_size=img_size, latent_dim=LATENT_DIM,
                             channels=CHANNELS, feature_maps=best_params['feature_maps'])
        # checkpoint = torch.load(best_checkpoint_path, map_location=DEVICE)
        # best_model.generator.load_state_dict(checkpoint['generator_state_dict'])
        # best_model.critic.load_state_dict(checkpoint['critic_state_dict'])
        # best_params['critic_optimizer_state'] = checkpoint['critic_optimizer_state_dict']
        # best_params['generator_optimizer_state'] = checkpoint['generator_optimizer_state_dict']

        # Save best config for future runs without tuning
        save_best_tuned_params(best_params, img_size, file_name="wgan_gp_config.json")

        # best_params['start_epoch'] = tune_epochs

    # CLEANUP - remove the directories that were made during training
    if os.path.isdir("checkpoints"):
        shutil.rmtree("checkpoints")
    if os.path.isdir("output/wgan_gp/tune_wgan_temp"):
        shutil.rmtree("output/wgan_gp/tune_wgan_temp")

    return best_params, best_model


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
        eta_min = params['lr'] * 0.25  # decay to 25% of initial lr
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
    
    patience = 0

    best_fid = float('inf')
    for epoch in range(params.get('start_epoch', 0), params['num_epochs']):
        gen_loss_sum = 0.0
        gen_loss_count = 0
        critic_loss_sum = 0.0
        critic_loss_count = 0
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
            
            
            critic_loss_sum += critic_loss.item()
            critic_loss_count += 1

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
        avg_critic_loss = critic_loss_sum / critic_loss_count if critic_loss_count > 0 else float('inf')
        # print(f"Critic loss: {critic_loss.item():.4f} - Generator loss: {avg_gen_loss:.4f}")
        print(f"Critic loss: {avg_critic_loss:.4f} - Generator loss: {avg_gen_loss:.4f}")
        
        history['epoch'].append(epoch)
        # history['critic_loss'].append(critic_loss.item())
        history['critic_loss'].append(avg_critic_loss)
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
            # checkpoint['critic_loss'] = critic_loss.item()
            checkpoint['critic_loss'] = avg_critic_loss
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
                patience = 0
                best_fid = fid
                torch.save(checkpoint, model_path)
                print(f"New best model - {model_path}, with FID: {fid:.4f}")
            else:
                print(f"No FID improvement - {fid:.4f} over best {best_fid:.4f}")
                patience += 1
                
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
        
        
        if patience >= 3:
            print(f"Early stopping at epoch {epoch + 1} due to no FID improvement for 3 eval epochs.")
            return history
    return history
        



# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# generator.py

import os
import argparse
import numpy as np
import torch

from utils import export_real_images_for_fid, generate_images, compute_fid
from model_definitions.dcgan_model import DCGAN
from model_definitions.wgan_gp_model import WGAN_GP
from model_definitions.progan_model import ProGAN
from train import LATENT_DIM, DATA_ROOT, BATCH_SIZE, CHANNELS


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cleanup_directory(dir_path, keep_n=10):
    if not os.path.exists(dir_path):
        return

    files = [
        os.path.join(dir_path, f)
        for f in os.listdir(dir_path)
        if f.lower().endswith(".png")
    ]
    files.sort(key=os.path.getmtime)

    for f in files[:-keep_n]:
        os.remove(f)


def default_output_dir(model_name: str, img_size: int) -> str:
    if model_name == "dcgan":
        return "output/dcgan_fakes"
    if model_name == "wgan_gp":
        return f"output/wgan_gp_{img_size}"
    if model_name == "progan":
        return f"output/progan_{img_size}"
    raise ValueError(f"Unknown model: {model_name}")


def set_seed(seed: int | None):
    if seed is None:
        return
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dcgan(img_size: int):
    if img_size != 64:
        raise ValueError("DCGAN is designed for 64x64 images. Please choose --size 64.")

    checkpoint_path = "models/dcgan_model_64.pt"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    ckpt_params = checkpoint.get("params", {})

    model_latent_dim = int(ckpt_params.get("latent_dim", LATENT_DIM))
    model_channels = int(ckpt_params.get("channels", CHANNELS))
    model_feature_maps = int(ckpt_params.get("feature_maps", 128))

    dcgan = DCGAN(
        latent_dim=model_latent_dim,
        channels=model_channels,
        feature_maps=model_feature_maps,
    ).to(DEVICE)

    dcgan.generator.load_state_dict(checkpoint["generator_state_dict"])
    dcgan.eval()

    return dcgan, model_latent_dim


def load_wgan_gp(img_size: int):
    checkpoint_path = f"models/wgan_gp_model_{img_size}.pt"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    ckpt_params = checkpoint.get("params", {})

    model_img_size = int(ckpt_params.get("img_size", ckpt_params.get("image_size", img_size)))
    model_latent_dim = int(ckpt_params.get("latent_dim", LATENT_DIM))
    model_channels = int(ckpt_params.get("channels", CHANNELS))
    model_feature_maps = int(ckpt_params.get("feature_maps", 64))

    wgan_gp = WGAN_GP(
        img_size=model_img_size,
        latent_dim=model_latent_dim,
        channels=model_channels,
        feature_maps=model_feature_maps,
    ).to(DEVICE)

    wgan_gp.generator.load_state_dict(checkpoint["generator_state_dict"])
    wgan_gp.eval()

    return wgan_gp, model_latent_dim


def load_progan(img_size: int):
    checkpoint_path = f"models/progan_model_{img_size}.pt"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

    progan = ProGAN(latent_dim=LATENT_DIM, channels=CHANNELS).to(DEVICE)
    progan.gen.load_state_dict(checkpoint["generator_state_dict"])
    progan.disc.load_state_dict(checkpoint["discriminator_state_dict"])
    progan.eval()

    max_step = checkpoint.get("step", int(np.log2(img_size)) - 2)
    alpha = checkpoint.get("alpha", 1.0)

    return progan, LATENT_DIM, max_step, alpha


def maybe_prepare_real_test_dir(img_size: int, skip_fid: bool):
    if skip_fid:
        return None

    return export_real_images_for_fid(
        split="test",
        data_root=DATA_ROOT,
        image_size=img_size,
        save_dir=os.path.join("output", "fid_cache", f"test_real_{img_size}"),
    )


def main():
    parser = argparse.ArgumentParser(description="Generate images from trained GAN checkpoints")
    parser.add_argument(
        "--model",
        type=str,
        choices=["dcgan", "wgan_gp", "progan"],
        required=True,
        help="Model to generate from",
    )
    parser.add_argument(
        "--size",
        type=int,
        choices=[64, 128],
        default=64,
        help="Image size",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=10,
        help="Number of images to generate",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional custom output directory",
    )
    parser.add_argument(
        "--skip_fid",
        action="store_true",
        help="Skip FID computation (useful for live demo)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed",
    )
    parser.add_argument(
        "--keep_n",
        type=int,
        default=None,
        help="Optional cleanup: keep only the most recent N generated images in the output directory",
    )

    args = parser.parse_args()

    img_size = args.size
    model_name = args.model
    num_images = args.num_images
    output_dir = args.output_dir or default_output_dir(model_name, img_size)

    if num_images <= 0:
        raise ValueError("--num_images must be a positive integer.")

    set_seed(args.seed)
    os.makedirs(output_dir, exist_ok=True)

    real_test_dir = maybe_prepare_real_test_dir(img_size, args.skip_fid)

    if model_name == "dcgan":
        model, model_latent_dim = load_dcgan(img_size)

        fake_dir = generate_images(
            model.generator,
            num_images,
            output_dir,
            BATCH_SIZE,
            model_latent_dim,
            DEVICE,
        )

        print(f"Generated {num_images} DCGAN image(s) in: {fake_dir}")

        if real_test_dir is not None:
            dcgan_fid = compute_fid(real_test_dir, fake_dir, BATCH_SIZE, DEVICE)
            print(f"DCGAN FID: {dcgan_fid:.4f}")

    elif model_name == "wgan_gp":
        model, model_latent_dim = load_wgan_gp(img_size)

        fake_dir = generate_images(
            model.generator,
            num_images,
            output_dir,
            BATCH_SIZE,
            model_latent_dim,
            DEVICE,
        )

        print(f"Generated {num_images} WGAN-GP image(s) in: {fake_dir}")

        if real_test_dir is not None:
            wgan_gp_fid = compute_fid(real_test_dir, fake_dir, BATCH_SIZE, DEVICE)
            print(f"WGAN-GP FID: {wgan_gp_fid:.4f}")

    elif model_name == "progan":
        model, model_latent_dim, max_step, alpha = load_progan(img_size)

        fake_dir = generate_images(
            model.gen,
            num_images,
            output_dir,
            BATCH_SIZE,
            model_latent_dim,
            DEVICE,
            step=max_step,
            alpha=alpha,
        )

        print(f"Generated {num_images} ProGAN image(s) in: {fake_dir}")

        if real_test_dir is not None:
            progan_fid = compute_fid(real_test_dir, fake_dir, BATCH_SIZE, DEVICE)
            print(f"ProGAN FID: {progan_fid:.4f}")

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    if args.keep_n is not None and args.keep_n > 0:
        cleanup_directory(output_dir, keep_n=args.keep_n)
        print(f"Kept only the most recent {args.keep_n} image(s) in: {output_dir}")


if __name__ == "__main__":
    main()
    
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# utils.py

# NOTE: USED AI FOR THIS FUNCTION
def get_transforms(image_size, channels):
    """Define the image preprocessing."""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * channels, (0.5,) * channels),
    ])


# NOTE: USED AI FOR THIS FUNCTION
def load_dataset(split, data_root, image_size, channels, batch_size, num_workers):
    """Load real images for a dataset split (train, valid, or test) and return a DataLoader."""
    split_path = os.path.join(data_root, split)
    if not os.path.isdir(split_path):
        raise FileNotFoundError(f"Data split not found: {split_path}")

    dataset = ImageFolder(root=split_path, transform=get_transforms(image_size, channels))
    real_idx = dataset.class_to_idx["real"]
    dataset.samples = [(p, t) for p, t in dataset.samples if t == real_idx]
    dataset.targets = [t for t in dataset.targets if t == real_idx]

    shuffle = (split == "train")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"), # drops last to ensure all batches are full during training
    )
    print(f"[{split}] Loaded {len(dataset)} real images  |  batches: {len(loader)}")
    return loader, dataset


# NOTE: USED AI FOR THIS FUNCTION
def generate_images(generator, num_images, save_dir, batch_size=None, latent_dim=None, device=None, flatten_noise=False, step=None, alpha=None):
    """Generate fake images from a trained generator and save them to disk."""
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    defaults = Config()
    if batch_size is None:
        batch_size = defaults.batch_size
    if latent_dim is None:
        latent_dim = defaults.latent_dim
    if device is None:
        try:
            device = next(generator.parameters()).device
        except StopIteration:
            device = defaults.device
   
    generator.eval()
    count = 0
    with torch.no_grad():
        while count < num_images:
            batch = min(batch_size, num_images - count)
            
            if flatten_noise:
                noise = torch.randn(batch, latent_dim, device=device)
            else:
                noise = torch.randn(batch, latent_dim, 1, 1, device=device)
            if step is not None:
                fake_imgs = generator(noise, step=step, alpha=1.0 if alpha is None else alpha)
            elif alpha is not None:
                raise ValueError("alpha cannot be provided without step in generate_images().")
            else:
                try:
                    fake_imgs = generator(noise)
                except TypeError as exc:
                    raise TypeError(
                        "generate_images() called the generator without 'step', but this generator "
                        "appears to require progressive arguments. Pass 'step' (and optionally "
                        "'alpha') when using progressive models."
                    ) from exc
            
            fake_imgs = (fake_imgs + 1) / 2
            for img in fake_imgs:
                save_image(img, os.path.join(save_dir, f"{count:06d}.png"))
                count += 1
    generator.train()
    print(f"Generated {count} images -> {save_dir}")
    return save_dir


# NOTE: USED AI FOR THIS FUNCTION
def compute_fid(real_dir, fake_dir, batch_size, device):
    """Compute the Fréchet Inception Distance (FID) between real and generated images."""
    fid_value = fid_score.calculate_fid_given_paths(
        [real_dir, fake_dir],
        batch_size,
        device,
        dims=2048,
    )
    print(f"FID: {fid_value:.4f}")
    return fid_value




# NOTE: USED AI FOR THIS FUNCTION
def weights_init(m):
    """Initialize the weights of convolutional and normalization layers."""
    classname = m.__class__.__name__
    if 'Conv' in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif 'BatchNorm' in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif 'GroupNorm' in classname:
        if m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

# NOTE: AI ASSISTED WITH THIS FUNCTION
def save_best_tuned_params(best_params, img_size, file_name):
    # Save best config for future runs without tuning
    os.makedirs("configs", exist_ok=True)
    config_path = os.path.join("configs", file_name)
    save_params = {}
    for k, v in best_params.items():
        if k in ('critic_optimizer_state', 'generator_optimizer_state'):
            continue
        if isinstance(v, (np.integer,)):
            v = int(v)
        elif isinstance(v, (np.floating,)):
            v = float(v)
        save_params[k] = v

    # Load existing config to preserve other image size entries
    if os.path.isfile(config_path):
        with open(config_path, "r") as f:
            all_configs = json.load(f)
    else:
        all_configs = {}

    all_configs[f"img_size_{img_size}"] = save_params
    with open(config_path, "w") as f:
        json.dump(all_configs, f, indent=4)


# NOTE: AI ASSISTED WITH THIS FUNCTION
def export_real_images_for_fid(split, data_root, image_size, save_dir):
    
    
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# app.py
import streamlit as st
from components.sidebar.sidebar import sidebar
from pages.training_curves.training_curves import training_curves
from pages.overview.overview import overview
from pages.architectures.architectures import architectures
from pages.model_summary.model_summary import model_summary
from pages.image_gallery.image_gallery import image_gallery
from pages.live_demo.live_demo import live_demo
from pages.conclusions.conclusions import conclusions
from pages.references.reference import references


def main():
    st.set_page_config(page_title="GAN Dashboard", layout="wide")
    st.title("Generative Adversarial Network (GAN) Dashboard")

    page = sidebar()

    if page == "Overview":
        overview()
    elif page == "Architectures":
        architectures()
    elif page == "Training Curves":
        training_curves()
    elif page == "Model Summary":
        model_summary()
    elif page == "Image Gallery":
        image_gallery()
    elif page == "Live Generation":
        live_demo()
    elif page == "Conclusions and Findings":
        conclusions()
    elif page == "References":
        references()


if __name__ == "__main__":
    main()
    
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# pages/architectures/wgan_gp.py

import streamlit as st

def wgan_gp_section():
    st.header("Wasserstein GAN with Gradient Penalty (WGAN-GP)")
    st.write("""WGAN-GP is an improved version of the Wasserstein GAN that incorporates a gradient penalty to enforce the Lipschitz constraint. 
    This architecture helps stabilize training and allows for better convergence, resulting in higher quality generated images.
    """)
    
    st.subheader("How WGAN-GP Works")
    st.markdown(
        """
        1. The generator maps noise z to synthetic images G(z).
        2. The critic (formerly discriminator) scores real images x and generated images G(z) with scalar values.
        3. Training maximizes the critic gap between real and fake scores.
        4. The generator is trained to increase critic scores on fake samples.
        5. A gradient penalty term enforces a 1-Lipschitz critic, which stabilizes training.
        """
    )
    
    st.subheader("Wasserstein Objective")
    st.markdown(
        """
        The critic approximates the **Earth Mover (Wasserstein-1) distance**. WGAN uses the following loss functions:
        """
    )
    
    # NOTE: USED AI FOR THE LATEX FORMULAS -> REPLACING THE PNGS
    st.latex(
        r"\mathcal{L}_C = \mathbb{E}_{\tilde{x}\sim P_g}[D(\tilde{x})] - \mathbb{E}_{x\sim P_r}[D(x)]"
    )
    st.latex(
        r"\mathcal{L}_G = -\,\mathbb{E}_{z\sim p(z)}[D(G(z))]"
    )
    st.markdown(
        '***Note:*** above is the original WGAN loss without the gradient penalty. We will add the gradient penalty term next.'
    )
    st.markdown("#### Earth Mover Distance: Why It Helped Beyond DCGAN")
    st.markdown(
        """
        1. BCE loss:
            - The discriminator is training on a binary classification task (real or fake), and BCE returns a probability. When the probability distributions
            for the real and fake samples have little overlap, the discriminator can easily classify samples, and send back effectively useless information (gradients close to 0)
            to the generator.
                - Consequences are mode collapse, unstable training dynamics, and lack of ability to train on higher resolutions.
        2. Earth Mover Distance:
            - WGAN proposed using the Earth Mover (Wasserstein-1) distance as a loss function, which provides smoother gradients even when the real and fake distributions have little overlap.
            This allows for more stable training and better convergence.
                - The scalar output is a distance value, that basically tells **how far apart the real and fake distributions are**
            - The constraint is that the output must be **1-Lipschitz**
                - **1-Lipschitz** means that the output of the critic must increase at most the same as the input. This ensures that the critic is smooth and returns meaningful gradients.
                - Originally, the WGAN paper addressed this with weight clipping, but they said themselves this was a terrible idea and encouraged further research.
                
        #### Problem with Weight Clipping (bounds on weights):
        1. If the weights are clipped too small, the critic is too constrained (gradients vanish as nothing useful is being found)
        2. If weights are clipped too large, the critic has more leeway than needed, and can diverge (gradients explode as critic is finding too much)
        """
    )
    
    
    
    
    st.subheader("Gradient Penalty")
    st.markdown(
        "###### (A better approach than weight clipping to enforcing the 1-Lipschitz constraint)"
    )
    st.markdown(
        """
        #### How is gradient penalty computed?
        
        1. Interpolate between real and fake samples
        2. Compute the gradient of the critic output with respect to these interpolated samples.
        3. The gradient penalty is the squared difference between the norm of this gradient and 1, scaled by a penalty coefficient (lambda).
        
        #### What is the result?

        The gradient penalty enforces the Lipschitz constraint without weight clipping, which can lead to better convergence and higher quality images.
        The gradient penalty is computed as follows:
        """
    )
    
    # NOTE: USED AI FOR THE LATEX FORMULAS -> REPLACING THE PNGS
    st.latex(
        r"\hat{x} = \epsilon x + (1-\epsilon)\tilde{x}, \quad \epsilon \sim U(0,1)"
    )
    st.latex(
        r"\mathcal{L}_{GP} = \lambda \,\mathbb{E}_{\hat{x}}\left(\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1\right)^2"
    )
    st.markdown(
        '**Final WGAN-GP Critic Loss Function:**'
    )
    # NOTE: USED AI FOR THE LATEX FORMULAS -> REPLACING THE PNGS
    st.latex(
        r"\mathcal{L}_C = \mathbb{E}_{\tilde{x}\sim P_g}[D(\tilde{x})] - \mathbb{E}_{x\sim P_r}[D(x)] + \mathcal{L}_{GP}"
    )
    
    st.markdown('#### Gradient Penalty Benefits')
    st.markdown(
        """
        1. Enforces the Lipschitz constraint more effectively than weight clipping, leading to stable convergence.
        2. Allows training on higher resolution images without mode collapse or instability.
        3. All of this also prevents **mode collapse**, which is a common problem in GANs where the generator produces similar pictures frequently,
        resulting in a lack of diversity in the images it generates.
        """
    )
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# sidebar.py

import streamlit as st

def sidebar():
    st.sidebar.title("Table of Contents")

    page = st.sidebar.selectbox(
        "Go to",
        [
            "Overview",
            "Architectures",
            "Training Curves",
            "Model Summary",
            "Image Gallery",
            "Live Generation",
            "Conclusions and Findings",
            "References",
        ],
    )

    return page

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# loaders.py
import json
import pandas as pd


def load_wgan_data():
    with open('logs/wgan_gp_64_history.json', 'r') as f:
        data_64 = json.load(f)
    with open('logs/wgan_gp_128_history.json', 'r') as f:
        data_128 = json.load(f)
    return pd.DataFrame(data_64['train_history']), pd.DataFrame(data_128['train_history'])
    

def load_dcgan_data():
    with open('logs/dcgan_64_history.json', 'r') as f:
        data_64 = json.load(f)
    return pd.DataFrame(data_64['train_history']), None
  
def _reshape_progan(data):
    # NOTE: USED AI TO RESHAPE PROGAN DATA INTO A CORRECT DATA STRUCTURE
    th = data['train_history']
    n = len(th['gen_loss'])
    df = pd.DataFrame({
        'epoch': range(1, n + 1),
        'generator_loss': th['gen_loss'],
        'discriminator_loss': th['disc_loss'],
    })
    fid_dict = dict(zip(th['fid_epochs'], th['fid']))
    df['fid'] = df['epoch'].map(fid_dict)
    return df


def load_progan_data():
    with open('logs/progan_64_history.json', 'r') as f:
        data_64 = json.load(f)
    with open('logs/progan_128_history.json', 'r') as f:
        data_128 = json.load(f)
    return _reshape_progan(data_64), _reshape_progan(data_128)


def load_loss_data(model_type):
    if model_type == 'wgan':
        return load_wgan_data()
    elif model_type == 'dcgan':
        return load_dcgan_data()
    elif model_type == 'progan':
        return load_progan_data()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# architectures.py
import streamlit as st
from components.architectures.dcgan_section import dcgan_section
from components.architectures.wgan_gp_section import wgan_gp_section
from components.architectures.progan_section import progan_section


def _shared_setup_section():
    st.subheader("Shared Setup")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Latent Dim", "100")
    col2.metric("Channels", "3 (RGB)")
    col3.metric("Image Sizes", "64 / 128")
    col4.metric("Metric", "Fréchet Inception Distance (FID)")

    st.markdown(
        """
        - This page is intentionally top-to-bottom and scrollable.
        - Each model section follows: generator, discriminator or critic, then notes.
        """
    )
    

def _comparison_placeholder_section():
    st.subheader("Quick Comparison")
    st.table(
        {
            "Model": ["DCGAN", "WGAN-GP", "ProGAN"],
            "Core Idea": ["Introduced convolutional architectures to stabilize training in GAN models",
                          "WGAN Introduces Wasserstein distance as loss function but requires 1-Lipschitz functions (output has to increase at most the same as input). The GP stands for Gradient Penalty, which replaced weight clipping to enforce this constraint.",
                          "Train on smaller resolutions, and work way up until it reaches the target resolutions, allowing for more stable training progression. Combined with a Gradient Penalty, this should result in the best images of the three architectures"],
            "Problems": ["Mode collapse, training instability",
                         "Training on larger resolutions causes blurry generated images.",
                         "Complex training process and hyperparameter tuning."],
        }
    )


def architectures():
    st.header("Architectures")
    st.write(
        "Walkthrough of the GAN architectures used in this project. "
        "Scroll down for model-by-model details."
    )

    st.divider()
    _shared_setup_section()

    st.divider()
    dcgan_section()

    st.divider()
    wgan_gp_section()

    st.divider()
    progan_section()

    st.divider()
    _comparison_placeholder_section()

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# image_gallery.py

import os
import streamlit as st
from PIL import Image, UnidentifiedImageError



def image_gallery():
    st.header("Generated Images")
    st.write("View images generated by the GANs from generation script.")

    # Controls
    col1, col2 = st.columns(2)
    with col1:
        model = st.selectbox("Model", ["wgan_gp", "dcgan", "progan"])

    with col2:
        size = st.selectbox("Image Size", [64, 128])
        
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
    folder = os.path.join(BASE_DIR, f"output/{model}_{size}")
    
    try:
        image_files = sorted(
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(".png")
        )
    except FileNotFoundError:
        st.warning(f"Folder not found: {folder}")
        return
    
    image_files = sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith(".png")
    ])

    if not image_files:
        st.warning("No images found.")
        return

    cols_per_row = 5
    rows = [image_files[i:i+cols_per_row] for i in range(0, len(image_files), cols_per_row)]
    for row in rows:
        cols = st.columns(len(row))
        for col, img_path in zip(cols, row):
            try:
                img = Image.open(img_path)
                col.image(img, caption=os.path.basename(img_path), use_container_width=True)
            except (UnidentifiedImageError, OSError):
                # Issue with early images, adding this incase there is corruption
                col.warning(f"Corrupted:\n{os.path.basename(img_path)}")

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# model_summary.py
import json
import os
import pandas as pd
import streamlit as st
from PIL import Image


MODEL_CONFIG = {
    "wgan_gp": {
        "label": "WGAN-GP",
        "log_files": {
            64: "logs/wgan_gp_64_history.json",
            128: "logs/wgan_gp_128_history.json",
        },
        "image_dirs": {
            64: "output/wgan_gp_64",
            128: "output/wgan_gp_128",
        },
        "disc_key": "critic_loss",
    },
    "dcgan": {
        "label": "DCGAN",
        "log_files": {
            64: "logs/dcgan_64_history.json",
            128: "logs/dcgan_128_history.json",
        },
        "image_dirs": {
            64: "output/dcgan_64",
            128: "output/dcgan_128",
        },
        "disc_key": "discriminator_loss",
    },
    "progan": {
        "label": "ProGAN",
        "log_files": {
            64: "logs/progan_64_history.json",
            128: "logs/progan_128_history.json",
        },
        "image_dirs": {
            64: "output/progan_64",
            128: "output/progan_128",
        },
        "disc_key": "discriminator_loss",
    },
}


def load_history(model_key, image_size):
    path = MODEL_CONFIG[model_key]["log_files"][image_size]
    if not os.path.exists(path):
        return None

    with open(path, "r") as f:
        raw = json.load(f)

    history = raw.get("train_history", {})
    if not history:
        return None

    # ProGAN stores train_history as a dict of unequal-length lists
    if isinstance(history, dict) and "gen_loss" in history:
        n = len(history["gen_loss"])
        df = pd.DataFrame({
            "epoch": range(1, n + 1),
            "generator_loss": history["gen_loss"],
            "discriminator_loss": history["disc_loss"],
        })
        fid_dict = dict(zip(history["fid_epochs"], history["fid"]))
        df["fid"] = df["epoch"].map(fid_dict)
        return df

    return pd.DataFrame(history)


def latest_image(model_key, image_size):
    folder = MODEL_CONFIG[model_key]["image_dirs"][image_size]
    if not os.path.isdir(folder):
        return None

    images = sorted(
        os.path.join(folder, name)
        for name in os.listdir(folder)
        if name.endswith(".png")
    )
    return images[-1] if images else None


def build_summary_row(model_key, image_size):
    config = MODEL_CONFIG[model_key]
    df = load_history(model_key, image_size)
    image_path = latest_image(model_key, image_size)

    if df is None or df.empty:
        return {
            "Model": config["label"],
            "Status": "Missing training log",
            "Epochs": None,
            "Latest FID": None,
            "Best FID": None,
            "Final Generator Loss": None,
            "Final Disc/Critic Loss": None,
            "Sample Available": image_path is not None,
        }

    fid_series = df["fid"].dropna() if "fid" in df.columns else pd.Series(dtype=float)
    disc_key = config["disc_key"]

    return {
        "Model": config["label"],
        "Status": "Available",
        "Epochs": int(df["epoch"].max()) if "epoch" in df.columns else len(df),
        "Latest FID": float(fid_series.iloc[-1]) if not fid_series.empty else None,
        "Best FID": float(fid_series.min()) if not fid_series.empty else None,
        "Final Generator Loss": float(df["generator_loss"].dropna().iloc[-1]) if "generator_loss" in df.columns else None,
        "Final Disc/Critic Loss": float(df[disc_key].dropna().iloc[-1]) if disc_key in df.columns else None,
        "Sample Available": image_path is not None,
    }


def _metric_series(df, model_key, metric):
    disc_key = MODEL_CONFIG[model_key]["disc_key"]
    metric_map = {
        "FID": "fid",
        "Generator Loss": "generator_loss",
        "Discriminator/Critic Loss": disc_key,
    }
    column = metric_map[metric]
    if column not in df.columns or "epoch" not in df.columns:
        return None

    return df[["epoch", column]].dropna().rename(columns={column: MODEL_CONFIG[model_key]["label"]}).set_index("epoch")


def model_summary():
    st.header("Model Summaries and Comparison")
    st.write("Compare all GAN models using training metrics and generated samples.")

    control_col1, control_col2 = st.columns(2)
    with control_col1:
        image_size = st.selectbox("Image Size", [64, 128], index=0)
    with control_col2:
        metric = st.selectbox(
            "Comparison Metric",
            ["FID", "Generator Loss", "Discriminator/Critic Loss"],
            index=0,
        )

    selected_models = st.multiselect(
        "Models",
        options=list(MODEL_CONFIG.keys()),
        default=list(MODEL_CONFIG.keys()),
        format_func=lambda key: MODEL_CONFIG[key]["label"],
    )

    if not selected_models:
        st.info("Select at least one model to compare.")
        return

    st.subheader("Summary")
    summary_df = pd.DataFrame(
        [build_summary_row(model_key, image_size) for model_key in selected_models]
    )
    st.dataframe(summary_df, use_container_width=True)

    st.subheader(f"{metric} Across Models")
    comparison_frames = []
    for model_key in selected_models:
        df = load_history(model_key, image_size)
        if df is None:
            continue
        series = _metric_series(df, model_key, metric)
        if series is not None:
            comparison_frames.append(series)

    if comparison_frames:
        chart_df = pd.concat(comparison_frames, axis=1)
        st.line_chart(chart_df)
    else:
        st.info("No metric data available for the selected models.")

    st.subheader("Generated Sample Comparison")
    cols = st.columns(len(selected_models))
    for col, model_key in zip(cols, selected_models):
        with col:
            st.markdown(f"**{MODEL_CONFIG[model_key]['label']}**")
            image_path = latest_image(model_key, image_size)
            if image_path:
                col.image(Image.open(image_path), use_container_width=True)
            else:
                col.info("No generated samples available.")
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# references.py

import streamlit as st

def references():
    st.header("References")

    st.write(
        "References used for the dashboard, architecture explanations, model selection, training, and dataset context."
    )

    st.divider()
    st.subheader("Core GAN Papers")
    st.markdown("""
    - Goodfellow et al. (2014), *Generative Adversarial Nets*  
      https://arxiv.org/pdf/1406.2661

    - Radford et al. (2015), *Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks*  
      https://arxiv.org/pdf/1511.06434

    - Arjovsky et al. (2017), *Wasserstein GAN*  
      https://arxiv.org/abs/1701.07875
      
    - Gulrajani et al. (2017), *Improved Training of Wasserstein GANs*  
      https://arxiv.org/abs/1704.00028

    - Karras et al. (2017), *Progressive Growing of GANs for Improved Quality, Stability, and Variation*  
      https://arxiv.org/abs/1710.10196
      
    - Zeiler and Fergus (2010), *Deconvolutional Networks*  
      https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf
    """)

    st.divider()
    st.subheader("Code Reference")
    st.markdown("""
    - EmilienDupont, *wgan-gp* GitHub repository  
      Consulted for reference while implementing WGAN-GP ideas; code was not copied or directly reused.  
      https://github.com/EmilienDupont/wgan-gp
    """)

    st.divider()
    st.subheader("Dataset")
    st.markdown("""
    - 140k Real and Fake Faces dataset  
      https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces
    """)

    st.divider()
    st.subheader("Image Sources")
    st.markdown("""
    - `gan_arch_diag.png` — Custom diagram created for this project.

    - `dcgan_deconv.png` — Adapted from concepts in:  
      Radford et al. (2015), *Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks*  
      https://arxiv.org/pdf/1511.06434  
      
      Zeiler and Fergus (2010), *Deconvolutional Networks*  
      https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf
    """)

    st.divider()
    st.subheader("Additional Resources")
    st.markdown("""
    - ApX Machine Learning, *Fréchet Inception Distance (FID)*  
      https://apxml.com/courses/generative-adversarial-networks-gans/chapter-5-evaluation-of-gans/frechet-inception-distance-fid
    """)

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# training_curves.py

import streamlit as st
from components.loss_dashboard.loss_dashboard import loss_dashboard
from loaders.loaders import load_loss_data


def safe_load_model_data(model_key):
    try:
        return load_loss_data(model_key)
    except (FileNotFoundError, KeyError, ValueError):
        return None, None


def render_model_panel(col, title, model_key, image_size, data_64, data_128):
    with col:
        st.subheader(title)
        data = data_64 if image_size == 64 else data_128

        if data is None:
            st.info(f"Placeholder: {title} training logs for {image_size}x{image_size} are not available yet.")
            return

        st.caption("Showing available training curves.")
        loss_dashboard(data, model_key)


def training_curves():
    st.header("Training Curves")
    st.write("Visualize the training curves for the GANs, including loss curves for both the generator and discriminator.")
    
    st.subheader("Fréchet Inception Distance (FID)")
    st.write(
        "The FID score is a widely used metric for evaluating the quality of images generated by GANs. It is a distance measurement between the distribution of generated images and the distribution of real images. A lower FID score indicates that the generated images are more similar to the real images, while a higher FID score indicates that the generated images are less similar to the real images."
    )
    # NOTE: AI ASSISTED WITH LATEX TEXT
    st.latex(r"\text{FID} = \|\mu_r - \mu_g\|^2 + \text{Tr}\left(\Sigma_r + \Sigma_g - 2\left(\Sigma_r \Sigma_g\right)^{1/2}\right)")
    
    cols = st.columns(3)
    with cols[1]:
        # NOTE: AI ASSISTED WITH LATEX TEXT
        st.latex(r"""
        \text{Where:} \\
        \mu_r, \mu_g \ \text{— mean feature vectors of real and generated images} \\
        \Sigma_r, \Sigma_g \ \text{— covariance matrices of real and generated image features} \\
        \text{Tr} \ \text{— trace of a matrix}
        """
        )
    
    st.markdown(
    """
        Benefits of FID over Inception Score (IS):
    """
    )
    
    cols = st.columns(3)
    
    with cols[1]:
        st.table(
            {
                "Qualities":
                    [
                        "Evaluates real images",
                        "Accounts for mode collapse",
                        "Requires labels"
                    ],
                "IS" : [
                    "No",
                    "Yes",
                    "Yes"
                ],
                "FID" : [
                    "Yes",
                    "Yes",
                    "No"
                ]
            },
            width="content"
        )
    
    image_size = st.selectbox(
        "Select Image Size",
        [64, 128]
    )
    
    wgan_64, wgan_128 = safe_load_model_data('wgan')
    dcgan_64, _ = safe_load_model_data('dcgan')
    progan_64, progan_128 = safe_load_model_data('progan')
    
    col1, col2, col3 = st.columns(3)

    render_model_panel(col1, "DCGAN", "dcgan", image_size, dcgan_64, None)
    render_model_panel(col2, "WGAN-GP", "wgan", image_size, wgan_64, wgan_128)
    render_model_panel(col3, "ProGAN", "progan", image_size, progan_64, progan_128)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# loss_plots.py

import streamlit as st
import pandas as pd

def plot_fid(train_history):
    fid_df = (
        train_history[['fid']]
        .dropna(subset=['fid'])
        .reset_index(drop=True)
    )
    st.line_chart(fid_df)

def plot_generator_loss(train_history):
    gen_loss = train_history['generator_loss']
    epochs = train_history['epoch']
    
    gen_loss_df = pd.DataFrame({'epoch': epochs, 'generator_loss': gen_loss}).set_index('epoch')
    st.line_chart(gen_loss_df)
    
def plot_discriminator_loss(train_history, model):
  if model == 'wgan':
    loss = train_history['critic_loss']
  else:
    loss = train_history['discriminator_loss']
  
  epochs = train_history['epoch']
  disc_loss_df = pd.DataFrame({'epoch': epochs, 'discriminator_loss': loss}).set_index('epoch')
  st.line_chart(disc_loss_df)
  
def plot_combined_loss(train_history, model):
    epochs = train_history['epoch']
    gen_loss = train_history['generator_loss']
    disc_loss = train_history['critic_loss'] if model == 'wgan' else train_history['discriminator_loss']
    
    df = pd.DataFrame({
        'epoch': epochs,
        'generator_loss': gen_loss,
        'critic_loss' if model == 'wgan' else 'discriminator_loss': disc_loss
    }).set_index('epoch')
    st.line_chart(df)


def plot_loss_curves(train_history, model_type, metric='loss'):
  if metric == 'fid':
    plot_fid(train_history)
  else:
    plot_combined_loss(train_history, model_type)