import torch
from tqdm import tqdm

from gfn.gflownet import TBGFlowNet  # We use a GFlowNet with the Trajectory Balance (TB) loss
from gfn.gym import HyperGrid, MetaScheduleEnv  # We use the hyper grid environment
from gfn.modules import DiscretePolicyEstimator
from gfn.samplers import Sampler
from gfn.utils import NeuralNet  # NeuralNet is a simple multi-layer perceptron (MLP)
from mlc_dataset import gflownet_data_load


def normalize_score(score,
                    _mean = 0.003680646535107316, 
                    _val = 0.0012118761480652196,
                    _min = 2.8831801089918256e-06,
                    _max = 4.567233072666666,
                    _tau = 1):
    score = (score - _mean) / (_val ** (1/2))
    return torch.log(torch.sigmoid(score / _tau))
    
if __name__ == "__main__":

    # 1 - We define the environment

    # 1 - Define the Energy Function
    from gfn.utils.edm_model import mlp_ebm
    state_len = 15 * 1 + 15 * 96
    action_len = 15 * 10 + 15 * 96 * 2 + 1 # add the terminal state
    edm_model = mlp_ebm(state_len,256,1).cuda()
    env = MetaScheduleEnv(energy=edm_model,alpha=1,device_str="cuda")

    # 2 - We define the needed modules (neural networks)
    # The environment has a preprocessor attribute, which is used to preprocess the state before feeding it to the policy estimator
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
    
    # 7 - Add the Meta-Schedule dataset
    import os,sys
    mount_path = "/home/imagenet"
    root = os.path.join(mount_path,"share/dataset/gflownet_dataset")
    dataloader = gflownet_data_load(root,without_condition=True,num_workers=4,batch_size=16)
    
    train_iter  = iter(dataloader)

    # 8 - Begin training
    pbar = tqdm(range(0,500))
    for i in (pbar):
        try:
            x, score = next(train_iter)
        except:
            train_iter = iter(dataloader)
            x, score = next(train_iter)
        
        x = x.cuda(non_blocking=True).long()
        x = x.view(x.shape[0],-1)
        score = normalize_score(score.cuda(non_blocking=True))
        states = env.States(x)
        f_trajectories = f_sampler.sample_trajectories(env=env, n_trajectories=16)
        b_trajectories = b_sampler.sample_trajectories(env=env, n_trajectories=16,states=states)
        real_y = edm_model(x.float())

        # fake_x = f_trajectories.states.tensor[-2,...].clone().detach().float()
        # fake_y = edm_model(fake_x)
        edm_loss = ((real_y - score) ** 2).mean()
        optimizer.zero_grad()
        f_loss = gfn.loss(env, f_trajectories)
        b_loss = gfn.loss(env, b_trajectories)
        loss = b_loss + f_loss + edm_loss
        loss.backward()
        optimizer.step()
        
        if i % 1 == 0:
            reward = torch.exp(f_trajectories.log_rewards).mean().item()
            pbar.set_postfix({"b_loss": b_loss.item(),"f_loss": f_loss.item(), "edm_loss": edm_loss.item(), "reward":reward})  
        if i & 5 == 0:      
            checkpoint = {"gfn":gfn.state_dict(),"edm":edm_model.state_dict()}
            torch.save(checkpoint,f"gflownet_checkpoint_{i}.pth")
            os.system(f"rm -rf gflownet_checkpoint_{i-25}.pth")
            
            