import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from utils import get_transforms, load_dataset, generate_images, compute_fid, weights_init

#-------------------------------------------------------------------------------------------------------------------------------------------

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "real_vs_fake")

# IMAGE_SIZE = (64, 64)
# IMAGE_SIZE = (128, 128) # this will be a later trial

IMAGE_SIZE = 64
# IMAGE_SIZE = 128
CHANNELS = 3
BATCH_SIZE = 128
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4
LATENT_DIM = 100

#-------------------------------------------------------------------------------------------------------------------------------------------


#-------------------------------------------------------------------------------------------------------------------------------------------
# TODO: PRATIK IMPLEMENT THIS - MODEL ARCHITECTURE
class Generator_DCGAN(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, channels=CHANNELS, feature_maps=64):
        super(Generator_DCGAN, self).__init__()

        self.network = nn.Sequential(
            # Input: (batch, latent_dim, 1, 1)

            # 1x1 -> 4x4
            nn.ConvTranspose2d(latent_dim, feature_maps * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True),

            # 4x4 -> 8x8
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),

            # 8x8 -> 16x16
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),

            # 16x16 -> 32x32
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),

            # 32x32 -> 64x64
            nn.ConvTranspose2d(feature_maps, channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.network(x)


class Discriminator_DCGAN(nn.Module):
    def __init__(self, channels=CHANNELS, feature_maps=64):
        super(Discriminator_DCGAN, self).__init__()

        self.network = nn.Sequential(
            # Input: (batch, channels, 64, 64)

            # 64x64 -> 32x32
            nn.Conv2d(channels, feature_maps, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # 32x32 -> 16x16
            nn.Conv2d(feature_maps, feature_maps * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # 16x16 -> 8x8
            nn.Conv2d(feature_maps * 2, feature_maps * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # 8x8 -> 4x4
            nn.Conv2d(feature_maps * 4, feature_maps * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # 4x4 -> 1x1
            nn.Conv2d(feature_maps * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x).view(-1, 1)


class DCGAN(torch.nn.Module):
    def __init__(self):
        super(DCGAN, self).__init__()
        self.generator = Generator_DCGAN()
        self.discriminator = Discriminator_DCGAN()

#-------------------------------------------------------------------------------------------------------------------------------------------
# TODO: JOSH IMPLEMENT THIS - MODEL ARCHITECTURE
class Generator_WGAN_GP(nn.Module):
    """Generator class for WGAN_GP model. Should basically mirror a simple DCGAN generator architecture.
    Args:
        nn (_type_): _description_
    """
    def __init__(self, img_size=IMAGE_SIZE, latent_dim=LATENT_DIM, channels=CHANNELS, feature_maps=64):
        """Generator_WGAN_GP constructor.

        Args:
            img_size (int, optional): Size of the input images. Defaults to IMAGE_SIZE.
            latent_dim (int, optional): Dimensionality of the latent vector. Defaults to LATENT_DIM.
            channels (int, optional): Number of output channels. Defaults to CHANNELS.
            feature_maps (int, optional): Number of feature maps in the first layer. Defaults to 64.
        """
        super(Generator_WGAN_GP, self).__init__()

        n_stages = int(np.log2(img_size) - 2) # relationship of image size to upsampling in stages

        # layers=[]

        # first layer: 4x4
        # inchannels is latent dims

        self.network = nn.Sequential(
            nn.ConvTranspose2d(in_channels=latent_dim,
                               out_channels=feature_maps * (2 ** (n_stages - 1)), # scale the output
                               kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=feature_maps * (2 ** (n_stages - 1))), # batch norm should be fine in generator, just not critic - WGAN-GP paper
            nn.ReLU(True)
        )

        # middle downsampling layers - this will get us. the exact number of layers needed based on image size (64 vs 128)
        # This for loop below was debugged with AI assistance -> originally was hardcoded for 64x64 img size, but wanted flexibility to allow for 128x128
        # self.network = nn.ModuleList([self.network])
        for i in range(n_stages - 1, 0, -1):
            self.network.append(nn.ConvTranspose2d(in_channels=feature_maps * (2 ** i),
                                                   out_channels=feature_maps * (2 ** (i - 1)),
                                                   kernel_size=4, stride=2, padding=1, bias=False))
            self.network.append(nn.BatchNorm2d(num_features=feature_maps * (2 ** (i - 1))))
            self.network.append(nn.ReLU(True))
            
        # final layer:
        self.network.append(nn.ConvTranspose2d(in_channels=feature_maps,
                                               out_channels=channels,
                                               kernel_size=4, stride=2, padding=1, bias=False))
        self.network.append(nn.Tanh()) # output function Tanh as per the DCGAN paper

    def forward(self, x):
        return self.network(x)

class Critic_WGAN_GP(nn.Module):
    """Critic class for WGAN_GP model. This is basically the WGAN version of the discriminator. Instead of outputting a probability,
    which is what DCGAN discriminators output, it outputs a scalar value representing the "realness" of the input. The critic's output
    is a continuous scalar value that estimates the Wasserstein distance between the real and generated data distributions.

    The Wasserstein distance, conceptually, is a measure of the difference between the real and generated data distributions. This was the key
    difference between the traditional GANs and WGANs.

    Args:
        nn (_type_): _description_
    """
    def __init__(self, img_size=IMAGE_SIZE, channels=CHANNELS, feature_maps=64):
        """Critic_WGAN_GP constructor.

        Args:
            img_size (int, optional): Size of the input images. Defaults to IMAGE_SIZE.
            channels (int, optional): Number of input channels. Defaults to CHANNELS.
            feature_maps (int, optional): Number of feature maps in the first layer. Defaults to 64.
        """
        super(Critic_WGAN_GP, self).__init__()

        n_stages = int(np.log2(img_size) - 2) # relationship of image size to upsampling in stages

        # first layer
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=channels,
                      out_channels=feature_maps,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # middle downsampling layers - this will get us. the exact number of layers needed based on image size (64 vs 128)
        # This for loop below was debugged with AI assistance -> originally was hardcoded for 64x64 img size, but wanted flexibility to allow for 128x128
        for i in range(1, n_stages):
            self.network.append(nn.Conv2d(in_channels=feature_maps * (2 ** (i - 1)),
                                          out_channels=feature_maps * (2 ** i), 
                                          kernel_size=4, stride=2, padding=1, bias=False))
            

            # NOTE: WGAN-GP recommends avoiding BatchNorm in the critic because normalization should not couple samples within a batch
            # PyTorch does provide nn.LayerNorm GroupNorm(1, num_channels) is used here as a LayerNorm-like choice for NCHW conv features
            self.network.append(nn.GroupNorm(1, feature_maps * (2 ** i)))

            self.network.append(nn.LeakyReLU(0.2, inplace=True))


        # final layer
        self.network.append(nn.Conv2d(in_channels=feature_maps * (2 ** (n_stages - 1)), # For WGAN, Output is scalar - continuous
                                      out_channels=1, kernel_size=4, stride=1, padding=0, bias=False))

    def forward(self, x):
        output = self.network(x)
        return output.view(output.size(0), -1) # flatten output to a single scalar per sample


class WGAN_GP(torch.nn.Module):
    """WGAN-GP architecture class. Note, that the key difference here between a standard WGAN and WGAN-GP is the use of a gradient penalty
    instead of weight clipping to enforce the Lipschitz constraint.

    The Lipschitz constraint is a key requirement for the critic in WGANs. In the standard WGAN, weight clipping (or bounding as I think of it) 
    was the method used to enforce this, which led to capacity underuse and exploding or vanishing gradients. WGAN-GP improved this by using a
    gradient penalty instead of weight clipping, which penalizes the norm of the gradient of the critic's output with respect to its input,
    encouraging the gradient norm to be close to 1. In short, gradient penalty simply penalizes the critic if its gradients norms deviate (are not equal to)
    the 1.

    The idea behind WGAN-GP is to provide a more stable and reliable training process for WGANs by ensuring that the critic satisfies the 
    Lipschitz constraint without the drawbacks of weight clipping.

    Args:
        latent_dim (int): Dimension of the input noise vector for the generator.
        channels (int): Number of input channels for the generator and critic.
        feature_maps (int): Number of feature maps for the generator and critic.
    """
    def __init__(self, img_size=IMAGE_SIZE, latent_dim=LATENT_DIM, channels=CHANNELS, feature_maps=64):
        super(WGAN_GP, self).__init__()
        # define generator and discriminator here
        self.generator = Generator_WGAN_GP(img_size=img_size, latent_dim=latent_dim, channels=channels, feature_maps=feature_maps)
        self.critic = Critic_WGAN_GP(img_size=img_size, channels=channels, feature_maps=feature_maps)
        

    def calculate_gradient_penalty(self, x_fake: torch.Tensor, x_real: torch.Tensor) -> torch.Tensor:
        """Calculate the gradient penalty for WGAN-GP.

        Args:
            x_fake (torch.Tensor): A batch of fake samples generated by the generator.
            x_real (torch.Tensor): A batch of real samples from the dataset.


        Returns:
            torch.Tensor: The gradient penalty.
        """
        # sample real data, latent var (z), and a random number
        epsilon = torch.rand(x_real.size(0), 1, 1, 1, device=x_real.device, dtype=x_real.dtype) # random number between 0 and 1, and it is used as the interpolation coefficient
        x_interpolated = epsilon * x_real + (1 - epsilon) * x_fake.detach() # interpolation between real and fake date - picking a point on the line between the 2 points
        x_interpolated.requires_grad_(True)

        lamb = 10 # gradient penalty coefficient - paper only used 10 for this, so will not parameterize
        
        critic_interpolates = self.critic(x_interpolated) # gets scores from critic by running interpolated points

        # compute gradients
        gradients = torch.autograd.grad(
            outputs=critic_interpolates,
            inputs=x_interpolated,
            grad_outputs=torch.ones_like(critic_interpolates),
            create_graph=True,
        )[0]

        gradients = gradients.reshape(x_real.size(0), -1) # flatten gradients to a single vector per sample
        gradient_norm = gradients.norm(2, dim=1) # norm of function
        penalty = lamb*((gradient_norm - 1) ** 2).mean() # gradient penalty -> gradient norm deviation

        return penalty


#-------------------------------------------------------------------------------------------------------------------------------------------
# TODO: JEONGWON IMPLEMENT THIS SECTION - MODEL ARCHITECTURE
class Generator_ProGAN(nn.Module):
    def __init__(self):
        super(Generator_ProGAN, self).__init__()
        # define generator layers here
        pass

    def forward(self, x):
        pass

class Discriminator_ProGAN(nn.Module):
    def __init__(self):
        super(Discriminator_ProGAN, self).__init__()
        # define discriminator layers here
        pass

    def forward(self, x):
        pass

class ProGAN(torch.nn.Module):
    def __init__(self):
        super(ProGAN, self).__init__()
        # define generator and discriminator here
        self.generator = Generator_ProGAN()
        self.discriminator = Discriminator_ProGAN()

    def generator(self, x):
        return self.generator(x)

    def discriminator(self, x):
        return self.discriminator(x)


#-------------------------------------------------------------------------------------------------------------------------------------------
def tune_dcgan(train_loader, val_loader):
    # TODO: PRATIK IMPLEMENT THIS
    return {}, DCGAN()

#-------------------------------------------------------------------------------------------------------------------------------------------
def tune_wgan_gp(train_loader, val_loader, img_size=IMAGE_SIZE):
    # NOTE: THINGS TO CONSIDER TUNING:
    # - Learning rate -> 1e-4, 2e-4, 5e-5
    # - n_critic -> 3, 5, 7
    # - feature maps -> 32, 64, 128

    params = {
            'num_epochs': 2,
            'lr': 0.0001,
            'adam_b1': 0.0,
            'adam_b2': 0.9,
            'batch_size': BATCH_SIZE,
            'n_critic': 5, # number of of times critic is trained for everytime generator is trained
            'feature_maps': 64
    }
    
    # NOTE: FOR TRAIN TESTING PURPOSES, THIS IS HARDCODED FOR NOW
    wgan_gp = WGAN_GP(img_size=img_size, latent_dim=LATENT_DIM, channels=CHANNELS, feature_maps=params['feature_maps'])

    return params, wgan_gp

def tune_progan(train_loader, val_loader):
    """Run a small amount of epochs on several different configs - save the best one and return to it in the tuning loop

    Args:
        train_loader (_type_): _description_
        val_loader (_type_): _description_

    Returns:
        dict: The best hyperparameter configuration found during tuning
        model: The model initialized with the best hyperparameter configuration
    """
    # TODO: JEONGWON IMPLEMENT THIS
    return {}, ProGAN()

#-------------------------------------------------------------------------------------------------------------------------------------------


def train_dcgan(train_loader, model, params):
    """Complete training on best configs - run however many epochs are specified in params until convergence

    Args:
        train_loader (_type_): _description_
        model (_type_): _description_
        params (_type_): _description_
    Returns:
        None
    """
    # TODO: PRATIK IMPLEMENT THIS

    # TODO: UPDATE PARAMS BASED ON WHAT MODEL NEEDS, AND WHAT TUNING SAYS IS BEST
    params = {'learning_rate': 0.0002,
              'beta1': 0.5,
              'beta2': 0.999,
              'batch_size': 64,
              'num_epochs': 100}
    pass

#-------------------------------------------------------------------------------------------------------------------------------------------

def train_wgan_gp(train_loader, model: WGAN_GP, params, img_size=IMAGE_SIZE):
    """Training loop for WGAN-GP

    Args:
        train_loader (DataLoader): DataLoader for training data
        model (WGAN_GP): WGAN-GP model instance
        params (dict): Hyperparameter configuration
        img_size (int, optional): Image resolution, used for checkpoint naming. Defaults to IMAGE_SIZE.
    """

    # checkpoint_path = f"wgan_gp_checkpoint_{img_size}.pt"
    model_path = f"models/wgan_gp_model_{img_size}.pt"

    critic_optimizer = torch.optim.Adam(model.critic.parameters(), lr=params['lr'],
                                        betas=(params['adam_b1'], params['adam_b2']))
    generator_optimizer = torch.optim.Adam(model.generator.parameters(), lr=params['lr'],
                                           betas=(params['adam_b1'], params['adam_b2']))
    
    best_gen_loss = float('inf')

    model.to(DEVICE)
    model.apply(weights_init)
    model.train()
    

    for epoch in range(params['num_epochs']):
        gen_loss_sum = 0.0
        gen_loss_count = 0
        print(f"Epoch {epoch+1}/{params['num_epochs']}")
        critic_loss = torch.tensor(0.0) # initialize for print later
        generator_loss = torch.tensor(0.0) # initialize for print later
        for i, real_data_batch in enumerate(train_loader):
            # CRITIC TRAINING
            # sample real data, latent var (z), and a random number
            x_real = real_data_batch[0].to(DEVICE) # real data batch (x in WGAN-GP paper)
            z = torch.randn(x_real.size(0), LATENT_DIM, 1, 1, device=DEVICE) # latent variable (z) as labeled by paper
            # Generate fake data with no gradient tracking to save memory and computation
            with torch.no_grad():
                x_fake: torch.Tensor = model.generator(z)

            # CRITIC SECTION
            critic_out_real: torch.Tensor = model.critic(x_real)
            critic_out_fake: torch.Tensor = model.critic(x_fake.detach())
            critic_loss = (torch.mean(critic_out_fake) - torch.mean(critic_out_real)) + model.calculate_gradient_penalty(x_fake, x_real)

            # UPDATE CRITIC WEIGHTS
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # NOTE: AI SUGGESTED THIS IF CONDITION TO REPLACE THE NESTED FOR LOOP
            if (i + 1) % params['n_critic'] == 0: # this block is only entered every n_critic steps -> this replaced the third loop
                # AI SUGGESTION: Freeze critic params to skip unnecessary gradient computation - saves time and computation
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

        print(f"Critic loss: {critic_loss.item():.4f} - Generator loss: {generator_loss.item():.4f}")

        # only save the best
        avg_gen_loss = gen_loss_sum / gen_loss_count if gen_loss_count > 0 else float('inf')
        if avg_gen_loss < best_gen_loss:
            best_gen_loss = avg_gen_loss
            torch.save({
                'epoch': epoch,
                'generator_state_dict': model.generator.state_dict(),
                'critic_state_dict': model.critic.state_dict(),
                'generator_optimizer_state_dict': generator_optimizer.state_dict(),
                'critic_optimizer_state_dict': critic_optimizer.state_dict(),
                'critic_loss': critic_loss.item(),
                'generator_loss': avg_gen_loss,
                'params': params,
                }, model_path)
            print(f"New best model - {model_path}, with generator loss: {avg_gen_loss:.4f}")
        else:
            print(f"No improvement in generator loss - {avg_gen_loss:.4f} over best {best_gen_loss:.4f}")

#-------------------------------------------------------------------------------------------------------------------------------------------

def train_progan(train_loader, model, params):
    """Complete training on best configs - run however many epochs are specified in params until convergence

    Args:
        train_loader (_type_): _description_
        model (_type_): _description_
        params (_type_): _description_
    """
    # TODO: JEONGWON IMPLEMENT THIS

    # TODO: UPDATE PARAMS BASED ON WHAT MODEL NEEDS, AND WHAT TUNING SAYS IS BEST
    params = {'learning_rate': 0.0002,
              'beta1': 0.5,
              'beta2': 0.999,
              'batch_size': 64,
              'num_epochs': 100}

    pass

#-----------------------------------------------------------------------------------------------------------------------------------------


#-----------------------------------------------------------------------------------------------------------------------------------------
# main function
def main():

    # add argparses for model to run and image size
    parser = argparse.ArgumentParser(description="Train GAN models")
    parser.add_argument("--model", type=str, choices=["dcgan", "wgan_gp", "progan"], required=True, help="Model to train")
    parser.add_argument("--size", type=int, choices=[64, 128], default=IMAGE_SIZE, help="Image size for training") # default to IMAGE_SIZE - 64x64
    args = parser.parse_args()

    # when running training, the commandline tells us which model to do for now
    img_size = args.size
    model_choice = args.model
    
    # load data
    train_loader, train_dataset = load_dataset("train",DATA_ROOT, img_size, CHANNELS, BATCH_SIZE, NUM_WORKERS)
    val_loader, val_dataset = load_dataset("valid",DATA_ROOT, img_size, CHANNELS, BATCH_SIZE, NUM_WORKERS)
    test_loader, test_dataset  = load_dataset("test",DATA_ROOT, img_size, CHANNELS, BATCH_SIZE, NUM_WORKERS)

    # define models as None until one is used
    dcgan = None
    wgan_gp = None
    progan = None

    os.makedirs("output", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    if model_choice == "dcgan":
        dcgan = DCGAN() # TODO: Pratik's model
        dc_params, dcgan = tune_dcgan(train_loader, val_loader) # TODO: Pratik's hyperparameter tuning function
        train_dcgan(train_loader, dcgan, dc_params) # TODO: Pratik's training function
    elif model_choice == "wgan_gp":
        # wgan_gp = WGAN_GP(img_size=img_size, latent_dim=LATENT_DIM, channels=CHANNELS, feature_maps=64)
        wgan_params, wgan_gp = tune_wgan_gp(train_loader, val_loader, img_size=img_size) # TODO: Josh's hyperparameter tuning function - current fixed params
        train_wgan_gp(train_loader, wgan_gp, wgan_params, img_size=img_size)
    elif model_choice == "progan":
        progan = ProGAN() # TODO: Jeongwon's model
        progan_params, progan = tune_progan(train_loader, val_loader) # TODO: Jeongwon's hyperparameter tuning function
        train_progan(train_loader, progan, progan_params) # TODO: Jeongwon's training function



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
        dcgan_fake_dir = generate_images(dcgan.generator, num_test, "output/dcgan_fakes", BATCH_SIZE, LATENT_DIM, DEVICE)
        dcgan_fid = compute_fid(real_test_dir, dcgan_fake_dir, BATCH_SIZE, DEVICE)
        print(f"DCGAN FID: {dcgan_fid:.4f}")
    elif wgan_gp is not None:
        wgan_gp.eval()
        wgan_gp_fake_dir = generate_images(wgan_gp.generator, num_test, "output/wgan_gp_fakes", BATCH_SIZE, LATENT_DIM, DEVICE)
        wgan_gp_fid = compute_fid(real_test_dir, wgan_gp_fake_dir, BATCH_SIZE, DEVICE)
        print(f"WGAN-GP FID: {wgan_gp_fid:.4f}")
    elif progan is not None:
        progan_fake_dir = generate_images(progan.generator, num_test, "output/progan_fakes", BATCH_SIZE, LATENT_DIM, DEVICE)
        progan_fid = compute_fid(real_test_dir, progan_fake_dir, BATCH_SIZE, DEVICE)
        print(f"ProGAN FID: {progan_fid:.4f}")


if __name__ == "__main__":
    main()



# NOTES:
#  - when you are tuning and training each model, be sure to save checkpoints throughout training
#  - all checkpoints saved during training should overwrite each other to allow for less memory usage
#  - in the end, we will all have 2 models each, one for 64 pixel resolution, another for 128 pixel resolution
#  - store all models and data locally, don't commit them, we will find another way to share them
#  - be sure to name your saved models in a way that clearly indicates the model architecture and resolution
#       - for example: dcgan_model_64.pt, progan_model_64.pt, wgan_gp_model_128.pt, etc.
#  - IMPORTANT: DO NOT CHANGE THE PREPROCESSING PIPELINE UNLESS YOU TALK TO US ALL FIRST
#       - the purpose of the experiment is to compare architectures when given the same input
#  - You can use any AI or existing tools, but make sure to do the following:
#       - 1. make sure you can explain what your code does in detail - he will ask
#       - 2. make sure to leave a comment next to sections of code you didn't write yourself
#           - this is just because he wants us to calculate what percentage of code we wrote on our own
