import torch
from tqdm import tqdm
import wandb

from src.gfn.gflownet import TBGFlowNet  # We use a GFlowNet with the Trajectory Balance (TB) loss
from src.gfn.gym import HyperGrid,DiscreteEBM  # We use the hyper grid environment
from src.gfn.modules import DiscretePolicyEstimator
from src.gfn.samplers import Sampler
from src.gfn.utils import NeuralNet  # NeuralNet is a simple multi-layer perceptron (MLP)

if __name__ == "__main__":

    # 1 - We define the environment

    # 8 - Define the Energy Function -- Cost Model
    from src.gfn.utils.edm_model import mlp_ebm
    # mlp_ebm is MLP 
    edm_model = mlp_ebm(28*28,256,1).cuda()
    
    env = DiscreteEBM(ndim=28*28,energy=edm_model,alpha=1, device_str="cuda")

    # env = HyperGrid(ndim=4, height=8, R0=0.01)  # Grid of size 8x8x8x8

    # 2 - We define the needed modules (neural networks)

    # The environment has a preprocessor attribute, which is used to preprocess the state before feeding it to the policy estimator
    # preprocessor is IdentityPreprocessor(), output_dim == ndim
    module_PF = NeuralNet(
        input_dim=env.preprocessor.output_dim,
        output_dim=env.n_actions
    )  # Neural network for the forward policy, with as many outputs as there are actions
    module_PB = NeuralNet(
        input_dim=env.preprocessor.output_dim,
        output_dim=env.n_actions - 1,
        torso=module_PF.torso  # We share all the parameters of P_F and P_B, except for the last layer
    )

    # 3 - We define the estimators

    pf_estimator = DiscretePolicyEstimator(module_PF, env.n_actions, is_backward=False, preprocessor=env.preprocessor)
    pb_estimator = DiscretePolicyEstimator(module_PB, env.n_actions, is_backward=True, preprocessor=env.preprocessor)

    # 4 - We define the GFlowNet

    gfn = TBGFlowNet(init_logZ=0., pf=pf_estimator, pb=pb_estimator)  # We initialize logZ to 0

    # 5 - We define the sampler and the optimizer

    f_sampler = Sampler(estimator=pf_estimator)  # We use an on-policy sampler, based on the forward policy
    
    b_sampler = Sampler(estimator=pb_estimator)  # We also need a backward policy sampler.

    # Policy parameters have their own LR.
    non_logz_params = [v for k, v in dict(gfn.named_parameters()).items() if k != "logZ"]
    optimizer = torch.optim.Adam(non_logz_params, lr=1e-3)

    # Log Z gets dedicated learning rate (typically higher).
    logz_params = [dict(gfn.named_parameters())["logZ"]]
    optimizer.add_param_group({"params": logz_params, "lr": 1e-1})
    optimizer.add_param_group({"params": edm_model.parameters(), "lr": 1e-3})

    # 6 - We train the GFlowNet for 1000 iterations, with 16 trajectories per iteration
    
    # 7 - Add the MNIST dataset
    
    from torchvision.datasets import MNIST
    from torchvision import transforms
    from torch.utils.data import DataLoader,Dataset
    dataset = MNIST(root="/root/share/dataset/mnist",
                    train=True,
                    download=True,
                    transform=transforms.Compose([
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.5],
                                                                        [0.5]),
                                                ]))
    
    dataloader = DataLoader(dataset,16,shuffle=True,num_workers=4)
    train_iter  = iter(dataloader)
    
    
    # 9. Define the visualization function

    import numpy as np
    import matplotlib.pyplot as plt

    def display_images(images,tag=""):
        # Check if the input is a valid numpy array
        if not isinstance(images, np.ndarray):
            images = images.clone().detach().cpu().numpy()
        
        # Check the dimension of the input
        if len(images.shape) != 4:
            images = images[:,None,...]

        batch_size, channels, height, width = images.shape

        # Decompose batch_size according to prime factorization
        factors = []
        i = 2
        while batch_size > 1:
            if batch_size % i == 0:
                factors.append(i)
                batch_size //= i
            else:
                i += 1

        # Get grid dimensions
        if len(factors) % 2 == 0:
            idx = len(factors) // 2
            grid_height = np.prod(factors[:idx])
            grid_width = np.prod(factors[idx:])
        else:
            idx = len(factors) // 2
            grid_height = np.prod(factors[:idx + 1])
            grid_width = np.prod(factors[idx + 1:])

        # Create a drawing board for the images
        board = np.zeros((grid_height * (height+2), grid_width * (width+2), channels), dtype=images.dtype)
        for i in range(grid_height):
            for j in range(grid_width):
                img_idx = i * grid_width + j
                if channels == 1:
                    board[i*(height+2):(i+1)*(height+2)-2, j*(width+2):(j+1)*(width+2)-2, 0] = images[img_idx, 0, :, :]
                else:
                    board[i*(height+2):(i+1)*(height+2)-2, j*(width+2):(j+1)*(width+2)-2, :] = images[img_idx].transpose(1, 2, 0)
        
        # Display the board
        if channels == 1:
            plt.imshow(board[:,:,0], cmap='gray')
        else:
            plt.imshow(board)
        plt.axis('off')
        plt.savefig(f"visualization_{tag}.png")
        
    # checkpoint = torch.load('checkpoint_365.pth')
    # # gfn.load_state_dict(checkpoint)
    # gfn.load_state_dict(checkpoint['gfn'])
    # edm_model.load_state_dict(checkpoint['edm'])
    
    epoch = 5e4

    wandb_project = "train discrete EBM GFlowNet"
    use_wandb = len(wandb_project) > 0
    if use_wandb:
        wandb.init(project=wandb_project)
        wandb.config.update({
            "learning_rate": 0.02,
            "architecture": "EBM GFlowNet",
            "dataset": "MNIST",
            "epochs": epoch,
        })

    import os,sys
    pbar = tqdm(range(0, int(epoch)))
    for i in (pbar):
            
        try:
            x, _ = next(train_iter)
        except:
            train_iter = iter(dataloader)
            x, _ = next(train_iter)
        
        x = (x.cuda()>0).long()
        x = x.view(x.shape[0],-1)
        
        
        states = env.States(x)
        f_trajectories = f_sampler.sample_trajectories(env=env, n_trajectories=16)
        b_trajectories = b_sampler.sample_trajectories(env=env, n_trajectories=16,states=states)
        
        real_y = edm_model(x.float())
        
        shuffle_ratio = 0.25
        fake_x = f_trajectories.states.tensor[-2,...].clone().detach().float()
        fake_x[:int(shuffle_ratio * fake_x.shape[0])] = (torch.randn_like(fake_x[:int(shuffle_ratio * fake_x.shape[0])]) > 0).float()
        fake_y = edm_model(fake_x)
        print("real_y",real_y.mean(),"fake_y",fake_y.mean())
        logit = torch.stack([real_y,fake_y],-1).softmax(-1)
        edm_loss = -torch.log(logit[...,0]).mean() * 0.5 + torch.log(logit[...,1]).mean() * 0.5
        optimizer.zero_grad()
        
        f_loss = gfn.loss(env, f_trajectories)
        b_loss = gfn.loss(env, b_trajectories)

        # if int(i / 100 ) & 1 or int(i/50) & 1 or int(i/25) & 1:
        loss = b_loss + f_loss
        # else:
        #     loss = b_loss + f_loss + edm_loss
        loss.backward()
        optimizer.step()
        
        if i % 20 == 0:
            reward = torch.exp(f_trajectories.log_rewards).mean().item()
            pbar.set_postfix({"b_loss": b_loss.item(),"f_loss": f_loss.item(), "edm_loss": edm_loss.item(), "reward":reward})
            
            # log metrics to wandb
            wandb.log({"b_loss": b_loss.item(),"f_loss": f_loss.item(), "edm_loss": edm_loss.item(), "reward":reward})

            f_states = f_trajectories.states
            # f_states.detach() # AttributeError: 'DiscreteEBMStates' object has no attribute 'detach'

            total_synthesize_images = []
            for ii in range(f_states.tensor.shape[1]):
                latest_state = f_states[-2,ii,...].tensor
                mnist_synthesized_result = latest_state.view(28,28)
                total_synthesize_images.append(mnist_synthesized_result)
            total_synthesize_images = torch.stack(total_synthesize_images,0)
            display_images(total_synthesize_images,i)

            checkpoint = {"gfn":gfn.state_dict(),"edm":edm_model.state_dict()}
            torch.save(checkpoint,f"checkpoint_{i}.pth")
            os.system(f"rm -rf checkpoint_{i-25}.pth")
            
            