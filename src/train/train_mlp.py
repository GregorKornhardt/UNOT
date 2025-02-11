import torch
from tqdm import tqdm
import wandb 

from ..networks.generator import Generator
from ..networks.mlp import Predictor
from ..ot.sinkhorn import sink_vec, sink_vec_dist
from ..loss import hilbert_projoection 

class Training():
    def __init__(self,
                generator : Generator,
                predictor : Predictor,
                lr : float,
                weight_decay_generator : float,
                weight_decay_predictor : float,
                multiplicative_factor_generator : float,
                multiplicative_factor_predictor : float,
                numbr_training_iterations : int,
                numbr_mini_loop_predictor : int,
                numbr_mini_loop_generator : int,
                sinkhorn_max_iterations : int,
                sinkhorn_epsilon : float,
                cost_matrix : torch.Tensor,
                batch_size : int,
                dim_prior : int,
                test_data : torch.Tensor = None,
                device : torch.device = torch.device('cpu'),
                ):
        super().__init__()

        self.generator = generator
        self.predictor = predictor
        self.lr = lr
        self.weight_decay_generator = weight_decay_generator
        self.weight_decay_predictor = weight_decay_predictor
        self.multiplicative_factor_generator = multiplicative_factor_generator
        self.multiplicative_factor_predictor = multiplicative_factor_predictor
        self.numbr_training_iterations = numbr_training_iterations
        self.numbr_mini_loop_predictor = numbr_mini_loop_predictor
        self.numbr_mini_loop_generator = numbr_mini_loop_generator
        self.sinkhorn_max_iterations = sinkhorn_max_iterations
        self.sinkhorn_epsilon = sinkhorn_epsilon
        self.cost_matrix = cost_matrix
        self.batch_size = batch_size
        self.dim_prior = dim_prior
        self.test_data = test_data
        self.device = device

        if test_data is not None:
            self.validation_loss = {}
            for key in self.test_data.keys():
                self.validation_loss[key] = []
        self.train_losses = {'generator': [], 'predictor': []}

        self.generator_optimizer = torch.optim.AdamW(self.generator.parameters(), lr=self.lr, 
                                                     weight_decay=self.weight_decay_generator)
        self.predictor_optimizer = torch.optim.AdamW(self.predictor.parameters(), lr=self.lr, 
                                                     weight_decay=self.weight_decay_predictor)

        # initializing learning rate schedulers
        self.generator_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.generator_optimizer, 
                                                                     gamma=self.multiplicative_factor_generator)
        self.predictor_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.predictor_optimizer, 
                                                                     gamma=self.multiplicative_factor_predictor)


    def train(self):
        for i in tqdm(range(self.numbr_training_iterations)):
            for _ in range(self.numbr_mini_loop_predictor):
                # sample from the prior
                z = torch.randn(self.batch_size, 2 * self.dim_prior, device=self.device)
                mu, nu = self.generator(z)
                
                self.mini_loop(mu, nu) 
                
                self.generator_scheduler.step()
                self.predictor_scheduler.step()
                
                wandb.log({'generator_loss': self.train_losses['generator'][-1], 
                           'predictor_loss': self.train_losses['predictor'][-1]})
            if i % 100 == 0:
                print(f'Generator loss: {self.train_losses["generator"][-1]}', f'Predictor loss: {self.train_losses["predictor"][-1]}')
                self.test_performance()

        return


    def mini_loop(
            self, 
            mu, 
            nu
        ):
        self.predictor_optimizer.zero_grad()

        # calculate the target from the sinkhorn algorithm
        g_true, nan_mask = self.target_sinkhorn(mu,nu)# self.target_sinkhorn(mu, nu)

        # remove the nan values
        g_calculated = g_true[nan_mask] - torch.unsqueeze(g_true[nan_mask].mean(1), 1)
        mu_no_nan, nu_no_nan = mu[nan_mask], nu[nan_mask]

        # condition to avoid nan values in a batch
        if torch.sum(nan_mask) <= 1:
            return

        # compute the loss
        g_predicted = self.predictor(mu_no_nan, nu_no_nan)
        predictor_loss = hilbert_projoection(g_predicted, g_calculated)
        self.train_losses['predictor'].append(predictor_loss.item())
        # backpropagate predictor
        predictor_loss.backward(retain_graph=True)
        self.predictor_optimizer.step()

        # update the generator
        self.generator_optimizer.zero_grad()

        # caculate the target from the sinkhorn algorithm
        g_true, nan_mask  = self.target_sinkhorn(mu,nu) #self.target_sinkhorn(mu, nu)

        # remove the nan values
        g_no_nan = g_true[nan_mask]
        g_calculated = g_no_nan - torch.unsqueeze(g_no_nan.mean(1), 1)
        mu_no_nan, nu_no_nan = mu[nan_mask], nu[nan_mask]
        
        if torch.sum(nan_mask) <= 1:
            return

        # compute the loss
        g_predicted = self.predictor(mu_no_nan, nu_no_nan)
        generator_loss =  - hilbert_projoection(g_predicted, g_calculated)
        self.train_losses['generator'].append(generator_loss.item())

        # backpropagate generator
        generator_loss.backward(retain_graph=True)
        self.generator_optimizer.step()

        return
    

    def target_sinkhorn(
            self, 
            mu, 
            nu
        ):
        with torch.no_grad():
            # Bootstrap the empirical measures
            nu0 = torch.exp(self.predictor(mu, nu))
            _, v = sink_vec(mu, nu, self.cost_matrix, self.sinkhorn_epsilon, nu0, self.sinkhorn_max_iterations)

            g = torch.log(v)

        return g, ~torch.isnan(v).any(dim=1).to(self.device)


    def test_performance(self):
        if self.test_data is None:
            return
            
        def test_loss(test_set):
            mu = test_set['mu'].to(self.device)
            nu = test_set['nu'].to(self.device)
            true_dist = test_set['dist']
            self.predictor.eval()
            nu0 = torch.exp(self.predictor(mu, nu))
            with torch.no_grad():
                _,_,dist1= sink_vec_dist(mu, nu, self.cost_matrix, self.sinkhorn_epsilon, nu0, 1)
                
            test_dist = dist1.cpu().detach()
            nan_mask = ~torch.isnan(test_dist)
            test_dist = test_dist[nan_mask]
            true_dist = true_dist[nan_mask]
            relative_error = (torch.abs(test_dist - true_dist) / true_dist).mean()
            self.predictor.train()
            return relative_error

        for key in self.test_data.keys():
            test_set = self.test_data[key]
            test_data_distance = test_loss(test_set)
            print(f'Relative error {key}: {test_data_distance}')
            self.validation_loss[key].append(test_data_distance)
            wandb.log({'Relative error '+key: test_data_distance})