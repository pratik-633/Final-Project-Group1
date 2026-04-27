#===ProGAN model definitions===============

import torch
import torch.nn as nn

"""
    ProGAN Generator - progressively grows from 4x4 to target resolution.
    Based on: https://arxiv.org/abs/1710.10196
    NOTE: AI ASSISTED WITH THIS ARCHITECTURE
"""


class EqualizedConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=bias)
        nn.init.normal_(self.conv.weight, 0, 1)
        if bias:
            nn.init.zeros_(self.conv.bias)
        fan_in = in_ch * kernel_size * kernel_size
        self.scale = (2 / fan_in) ** 0.5

    def forward(self, x):
        return self.conv(x * self.scale)


class EqualizedConvTranspose2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, padding, bias=bias)
        nn.init.normal_(self.conv.weight, 0, 1)
        if bias:
            nn.init.zeros_(self.conv.bias)
        fan_in = in_ch * kernel_size * kernel_size
        self.scale = (2 / fan_in) ** 0.5

    def forward(self, x):
        return self.conv(x * self.scale)


class EqualizedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        nn.init.normal_(self.linear.weight, 0, 1)
        nn.init.zeros_(self.linear.bias)
        self.scale = (2 / in_features) ** 0.5

    def forward(self, x):
        return self.linear(x * self.scale)


class PixelNorm(nn.Module):
    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)


class MinibatchStdDev(nn.Module):
    def forward(self, x):
        # x: (N, C, H, W)
        std = torch.std(x, dim=0, keepdim=True, unbiased=False)  # (1, C, H, W)
        mean_std = torch.mean(std, dim=[1, 2, 3], keepdim=True)  # (1, 1, 1, 1)
        repeated = mean_std.expand(x.size(0), 1, x.size(2), x.size(3))  # (N, 1, H, W)
        return torch.cat([x, repeated], dim=1)


class Generator_ProGAN(nn.Module):
    def __init__(self, latent_dim=100, channels=3, feature_maps=512):
        super(Generator_ProGAN, self).__init__()

        # initial block: latent_dim(length of noise vector) -> 4x4
        self.initial = nn.Sequential(
            EqualizedConvTranspose2d(latent_dim, feature_maps, 4, 1, 0),
            PixelNorm(),
            nn.LeakyReLU(0.2, inplace=True),
            EqualizedConv2d(feature_maps, feature_maps, 3, 1, 1),
            PixelNorm(),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # progressive blocks: each doubles resolution
        # 4->8, 8->16, 16->32, 32->64, (64-> 128)
        self.blocks = nn.ModuleList()
        self.to_rgb_layers = nn.ModuleList()

        # to_rgb for initial 4x4
        self.to_rgb_initial = EqualizedConv2d(feature_maps, channels, 1, 1, 0)

        in_ch = feature_maps

        # each step has the feature maps
        # # 4->8: 512->256, 8->16: 256->128, 16->32: 128->64, 32->64: 64->32, 64->128: 32->16
        for i in range(5):
            out_ch = in_ch // 2
            self.blocks.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                EqualizedConv2d(in_ch, out_ch, 3, 1, 1),
                PixelNorm(),
                nn.LeakyReLU(0.2, inplace=True),
                EqualizedConv2d(out_ch, out_ch, 3, 1, 1),
                PixelNorm(),
                nn.LeakyReLU(0.2, inplace=True),
            ))
            self.to_rgb_layers.append(EqualizedConv2d(out_ch, channels, 1, 1, 0))
            in_ch = out_ch

    def forward(self, x, step, alpha=1.0):
        """
        Args:
        x: noise tensor (batch, latent_dim, 1, 1)
        step: current growth step (0=4x4, 1=8x8, ..., 4=64x64, 5=128x128)
        alpha: fade-in factor (0.0 to 1.0) for smooth transition
        """
        out = self.initial(x)

        if step == 0:
            return torch.tanh(self.to_rgb_initial(out))
        for i in range(step):
            if i == step - 1:
                # keeps previous output for fade-in
                upsampled = nn.functional.interpolate(out, scale_factor=2, mode='nearest')
                if i == 0:
                    old_rgb = self.to_rgb_initial(upsampled)
                else:
                    old_rgb = self.to_rgb_layers[i - 1](upsampled)

            out = self.blocks[i](out)

        new_rgb = self.to_rgb_layers[step - 1](out)
        # alpha blend: smooth transition from old resolution to new
        return torch.tanh(alpha * new_rgb + (1 - alpha) * old_rgb)


class Discriminator_ProGAN(nn.Module):
    """
    ProGAN Discriminator - mirrors generator structure in reverse.
    Based on: https://arxiv.org/abs/1710.10196
    """

    def __init__(self, channels=3, feature_maps=512):
        super(Discriminator_ProGAN, self).__init__()
        # from_rgb layers: converts image to feature maps at each resolution
        self.from_rgb_layers = nn.ModuleList()
        self.blocks = nn.ModuleList()

        # build in reverse order(mirrors generator)
        ch_list = []
        temp = feature_maps
        for i in range(5):
            out_ch = temp // 2
            ch_list.append((out_ch, temp))
            temp = out_ch
        # reverse so index 0 = highest resolution block
        ch_list = ch_list[::-1]

        for (c_in, c_out) in ch_list:
            self.from_rgb_layers.append(nn.Sequential(
                EqualizedConv2d(channels, c_in, 1, 1, 0),
                nn.LeakyReLU(0.2, inplace=True),
            ))
            self.blocks.append(nn.Sequential(
                EqualizedConv2d(c_in, c_in, 3, 1, 1),
                nn.LeakyReLU(0.2, inplace=True),
                EqualizedConv2d(c_in, c_out, 3, 1, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.AvgPool2d(2)
            ))

        # from_rgb for the initial 4x4 resolution
        self.from_rgb_initial = nn.Sequential(
            EqualizedConv2d(channels, feature_maps, 1, 1, 0),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.final = nn.Sequential(
            MinibatchStdDev(),
            EqualizedConv2d(feature_maps + 1, feature_maps, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            EqualizedLinear(feature_maps, 1),
        )

    def forward(self, x, step, alpha=1.0):
        """
        Args:
        x: image tensor
        step: current growth step (0=4x4, 1=8x8, ..., 4=64x64, 5=128x128)
        alpha: fade-in factor for smooth transition
        """

        if step == 0:
            out = self.from_rgb_initial(x)
            return self.final(out)

        # Choose the current-resolution block from the growth step.
        # step=1 starts at the lowest progressive block (8x8 -> 4x4),
        # while larger steps start from progressively higher resolutions.
        block_idx = len(self.blocks) - step

        # new path: current resolution from_rgb -> current block
        out = self.from_rgb_layers[block_idx](x)
        out = self.blocks[block_idx](out)

        # old path: downsample once and convert using the previous resolution
        downsampled = nn.functional.avg_pool2d(x, 2)
        if block_idx + 1 < len(self.from_rgb_layers):
            old_out = self.from_rgb_layers[block_idx + 1](downsampled)
        else:
            old_out = self.from_rgb_initial(downsampled)

        # alpha blend between previous and current resolution paths
        out = alpha * out + (1 - alpha) * old_out

        # continue down through the remaining lower-resolution blocks
        for i in range(block_idx + 1, len(self.blocks)):
            out = self.blocks[i](out)

        return self.final(out)


class ProGAN(torch.nn.Module):
    def __init__(self, latent_dim=100, channels=3, feature_maps=512):
        super(ProGAN, self).__init__()

        self.gen = Generator_ProGAN(latent_dim, channels, feature_maps)
        self.disc = Discriminator_ProGAN(channels, feature_maps)

        # step/alpha tracked here for convenience
        self.step = 0
        self.alpha = 1.0

    # generator/discriminator properties for main() can call progan.generator

    @property
    def generator(self):
        return self.gen

    @property
    def discriminator(self):
        return self.disc

#======ProGAN_config=======
{
    "img_size_64": {
        "beta1": 0.0,
        "beta2": 0.99,
        "batch_size": 64,
        "feature_maps": 512,
        "learning_rate": 0.0005,
        "num_epochs_per_step": 300,
        "fade_in_epochs": 25,
        "n_critic": 1
    },
    "img_size_128": {
        "beta1": 0.0,
        "beta2": 0.99,
        "batch_size": 128,
        "feature_maps": 512,
        "learning_rate": 0.0005,
        "num_epochs_per_step": 300,
        "fade_in_epochs": 10,
        "n_critic": 2
    }
}

#===train.py ProGAN section=========
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

#--------------------------------------------

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
    best_gen_loss = float('inf')
    no_improve_count = 0
    history = {'gen_loss':[], 'disc_loss':[], 'fid':[], 'fid_epochs':[]}
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

    patience = params.get('patience', 3)
    no_improve_count = 0

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
                gp = 5 * ((gradients.reshape(batch_size, -1).norm(2, dim=1) - 1) ** 2).mean()
                disc_loss = disc_loss + gp

                disc_optimizer.zero_grad()
                disc_loss.backward()
                disc_optimizer.step()

                # ---- Train Generator ----
                if (i + 1) % params.get('n_critic', 1) == 0:
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

            history['gen_loss'].append(avg_gen)
            history['disc_loss'].append(avg_disc)

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
                history['fid'].append(fid)
                history['fid_epochs'].append(len(history['gen_loss']))
                checkpoint['fid'] = fid

                if os.path.isdir(fake_dir):
                    shutil.rmtree(fake_dir)

                print(f"  Validation FID: {fid:.4f}")
                if fid < best_fid:
                    best_fid = fid
                    best_gen_loss = avg_gen
                    no_improve_count = 0
                    torch.save(checkpoint, model_path)
                    print(f"  New best model saved - FID: {fid:.4f}")
                else:
                    no_improve_count += 1
                    print(f"  No FID improvement ({fid:.4f} vs best {best_fid:.4f})")
                    if no_improve_count >= patience:
                        print(f'Early stopping at step{step + 1}, epoch{epoch + 1}')
                        model.train()
                model.train()

            elif not use_fid:
                torch.save(checkpoint, model_path)

        os.makedirs('output', exist_ok=True)
        with open(f'output/progan_{img_size}_history.json','w') as f:
            json.dump(history, f)
        print(
            f"\nTraining complete! Best FID: {best_fid:.4f} | G loss at best FID: {best_gen_loss:.4f}" if use_fid else "\nTraining complete!")
#-------main ProGAN section========
   elif model_choice == "progan":
        real_val_dir = export_real_images_for_fid(
            split="valid",
            data_root=DATA_ROOT,
            image_size=img_size,
            save_dir=os.path.join("output", "fid_cache", f"valid_real_{img_size}")
        )

        progan_params, progan = tune_progan(
            train_loader,
            val_loader,
            real_val_dir,
            img_size=img_size,
            tuning=False)

        train_progan(progan, train_loader, progan_params, val_dir=real_val_dir, num_val_samples=len(val_dataset))
#======utils.py=============
@dataclass
class Config:
    image_size: list = None  # will be set to [64, 128] in __post_init__
    channels: int = 3
    batch_size: int = 64
    latent_dim: int =100
    num_workers: int = 4
    data_root: str = './data'
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __post_init__(self):
        if self.image_size is None:
            self.image_size = [64, 128]



#=====ProGAN dashboard architecture section=======
import streamlit as st

def progan_section():
    st.header("Progressive Growing of GANs (ProGAN)")
    st.write("""
    ProGAN is a GAN architecture that progressively grows the generator and discriminator networks during training. 
    This approach allows the model to learn coarse features first and then gradually refine them, leading to improved stability and higher quality generated images.
    """)

    st.subheader("How ProGAN Works")
    st.markdown(
        """
        1. Training starts at a very low resolution (4×4) for both the generator and discriminator.
        2. New layers are added progressively to double the resolution at each step (4×4 → 8×8 → ... → 64×64 or 128×128).
        3. Each new layer is introduced gradually using a **fade-in** mechanism controlled by an alpha parameter (0→1).
        4. The generator maps noise z to synthetic images, and the discriminator scores real vs. fake images at each resolution.
        5. A gradient penalty is applied to stabilize training.
        """
    )

    st.subheader("Progressive Growing Objective")
    st.markdown(
        """
        ProGAN adopts the **Wasserstein distance** as its training objective, combined with a gradient penalty to enforce the Lipschitz constraint:
        """
    )

    st.latex(
        r"\mathcal{L}_D = \mathbb{E}_{\tilde{x}\sim P_g}[D(\tilde{x})] - \mathbb{E}_{x\sim P_r}[D(x)] + \lambda\,\mathbb{E}_{\hat{x}}\left(\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1\right)^2"
    )
    st.latex(
        r"\mathcal{L}_G = -\,\mathbb{E}_{z\sim p(z)}[D(G(z))]"
    )

    st.subheader("Fade-In Mechanism")
    st.markdown(
        """
        When a new layer is added, it is blended in smoothly using an alpha parameter:
        """
    )
    st.latex(
        r"\text{output} = \alpha \cdot \text{new\_layer} + (1 - \alpha) \cdot \text{upsampled\_previous}"
    )
    st.markdown(
        """
        - **α = 0**: only the previous resolution output is used
        - **α = 1**: fully switched to the new higher resolution layer
        - This prevents sudden shocks to already-learned features when adding new layers.
        """
    )

    st.subheader("Key Techniques")
    st.markdown(
        """
        1. **Equalized Learning Rate**: weights are scaled at runtime so all layers train at the same effective speed, preventing any single layer from dominating.
        2. **Pixelwise Feature Normalization**: normalizes each pixel's feature vector to unit length in the generator, preventing signal magnitudes from spiraling out of control.
        3. **Minibatch Standard Deviation**: adds a statistic about diversity across the batch as an extra feature map in the discriminator, discouraging mode collapse.
        """
    )

    st.subheader("Why Progressive Training Helps")
    st.markdown(
        """
        - Starting from low resolution means the model first learns global structure (face shape, layout) before worrying about fine details (textures, hair).
        - The discriminator sees progressively harder tasks, keeping the training balanced.
        - This results in **more stable training** and **better image quality** compared to training at full resolution from scratch.
        - The downside is **higher implementation complexity** and the need for careful tuning of fade-in epochs and learning rate.
        """
    )