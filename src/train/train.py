import torch
import copy
import wandb
import math
from tqdm import tqdm
import torch.nn.functional as F

import src.ot.sinkhorn as sinkhorn
import src.ot.cost_matrix as cost
import torch.nn.functional as F
import src.utils.gaussian_random_field as grf
import src.evaluation.plot_measures as plot_measures

class Training():
    def __init__(self,
                generator,
                predictor,
                lr_gen : float,
                lr_fno : float,
                weight_decay_generator : float,
                weight_decay_predictor : float,
                multiplicative_factor_generator : float,
                multiplicative_factor_predictor : float,
                numbr_training_iterations : int,
                numbr_mini_loop_predictor : int,
                numbr_mini_loop_generator : int,
                sinkhorn_max_iterations : int,
                dust_const : float,
                length : int,
                numbr_latend_samples : int,
                sinkhorn_epsilon : float,
                cost_matrix_28 : torch.Tensor,
                cost_matrix_64 : torch.Tensor,
                grid: bool,
                sobel: bool,
                which_gen: bool,
                use_data: bool,
                batch_size : int,
                dim_prior : int,
                test_data_28 : torch.Tensor = None,
                test_data_64 : torch.Tensor = None,
                data_set: torch.Tensor = None,
                name : str = 'None',
                path : str = '.',
                device : torch.device = torch.device('cpu'),
                ):
        super().__init__()

        self.generator = generator
        self.predictor = predictor
        self.lr_gen = lr_gen
        self.lr_fno = lr_fno
        self.weight_decay_generator = weight_decay_generator
        self.weight_decay_predictor = weight_decay_predictor
        self.multiplicative_factor_generator = multiplicative_factor_generator
        self.multiplicative_factor_predictor = multiplicative_factor_predictor
        self.numbr_training_iterations = numbr_training_iterations
        self.sinkhorn_max_iterations = sinkhorn_max_iterations
        self.numbr_mini_loop_predictor = numbr_mini_loop_predictor
        self.numbr_mini_loop_generator = numbr_mini_loop_generator
        self.sinkhorn_epsilon = sinkhorn_epsilon
        self.cost_matrix_28 = cost_matrix_28
        self.cost_matrix_64 = cost_matrix_64
        self.numbr_latend_samples = numbr_latend_samples
        self.batch_size = batch_size
        self.use_data = use_data
        self.dim_prior = dim_prior
        self.dust_const = dust_const
        self.test_data_64 = test_data_64
        self.test_data_28 = test_data_28
        self.data_set = data_set
        self.grid = grid
        self.sobel = sobel
        self.which_gen = which_gen
        self.device = device
        self.name = name
        self.path = path
        self.length = length
        
        self.cost_matrix = cost.fast_get_cost_matrix(self.length, device=self.device)
        self.best_model = None
        self.best_total_loss = 10000
        self.model_save = []

        self.z_test  = torch.randn(20, 2 * self.dim_prior, device=self.device)
        self.old_mu = []
        self.old_nu = []
        self.old_fix_mu = []
        self.old_fix_nu = []
 
        if test_data_28 is not None:
            self.validation_loss = {}
            for key in self.test_data_28.keys():
                self.validation_loss[key] = []
        self.train_losses = {'generator 28': [], 'generator 64':[], 'predictor_data': [], 'predictor_gen': []}
    
        self.generator_optimizer = torch.optim.Adam(generator.parameters(), lr=self.lr_gen, 
                                                    weight_decay=self.weight_decay_generator)
        self.predictor_optimizer = torch.optim.AdamW(predictor.parameters(), lr=self.lr_fno, 
                                                    weight_decay=self.weight_decay_predictor)
        self.generator_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.generator_optimizer, 
                                                                     gamma=self.multiplicative_factor_generator)
        self.predictor_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.predictor_optimizer, 
                                                                     gamma=self.multiplicative_factor_predictor)


    def train(self):
        for i in tqdm(range(self.numbr_training_iterations)):
            z = self.latend_samples()

            with torch.no_grad():
                mu, nu = self.apply_generator(z, self.generator)
            
            data_gen = torch.stack((mu, nu),1).detach()
            
            data_loader_pred = torch.utils.data.DataLoader(data_gen, batch_size=self.batch_size, shuffle=False)
            data_loader_gen = torch.utils.data.DataLoader(z, batch_size=self.batch_size, shuffle=True)
            
            if self.use_data:
                raise NotImplementedError
            else:
                for data in data_loader_pred:
                    dim = torch.randint(high=64-10, size=(1,)) +10
                    dim = dim[0]
                    self.mini_loop_predictor(data[:,0], data[:,1], dim)   
                    
            for data in data_loader_gen:
                self.mini_loop_generator(data)
            
            self.generator_scheduler.step()
            self.predictor_scheduler.step()
            
            self.checkpoint(i)

        self.save_measure()
        self.plot()

        return



    def mini_loop_predictor(
            self, 
            mu, 
            nu, 
            dim
        ):
        self.predictor_optimizer.zero_grad()

        # convert the images to the correct dimension
        mu, nu = self.convert_image(mu, dim), self.convert_image(nu, dim)
        cost_matrix = cost.fast_get_cost_matrix(dim, device=self.device)

        # calculate the target from the sinkhorn algorithm
        g_true, nan_mask = self.target_sinkhorn(mu, nu, cost_matrix)

        # remove the nan values
        g_calculated = g_true[nan_mask] - torch.unsqueeze(g_true[nan_mask].mean(1), 1)
        mu_no_nan, nu_no_nan = mu[nan_mask], nu[nan_mask]

        # condition to avoid nan values in a batch
        if torch.sum(nan_mask) <= 1:
            return

        # compute the loss
        g_predicted = self.apply_predictor(mu_no_nan, nu_no_nan)
        predictor_loss = torch.mean((g_predicted - g_calculated)**2)

        self.train_losses['predictor_gen'].append(predictor_loss.item())
        wandb.log({'predictor_loss_gen': predictor_loss.item()})

        # backpropagate predictor
        predictor_loss.backward(retain_graph=True)
        self.predictor_optimizer.step()

        return


    def mini_loop_generator(
            self, 
            z
        ):
        # update the generator
        self.generator_optimizer.zero_grad()
        mu, nu = self.apply_generator(z, self.generator)

        # caculate the target from the sinkhorn algorithm
        g_true, nan_mask  = self.target_sinkhorn(mu, nu, self.cost_matrix) 

        # remove the nan values
        g_no_nan = g_true[nan_mask]
        g_calculated = g_no_nan - torch.unsqueeze(g_no_nan.mean(1), 1)
        mu_no_nan, nu_no_nan = mu[nan_mask], nu[nan_mask]
        
        if torch.sum(nan_mask) <= 1:
            return

        # compute the loss
        g_predicted = self.apply_predictor(mu_no_nan, nu_no_nan)
        generator_loss =  - torch.mean((g_predicted - g_calculated)**2)

        self.train_losses['generator 64'].append(generator_loss.item())
        wandb.log({'generator_loss': generator_loss.item()})

        # backpropagate generator
        generator_loss.backward(retain_graph=True)
        self.generator_optimizer.step()

        return  


    def mini_loop_predictor_data(
            self, 
            mu, 
            nu, 
            true
        ):
        self.predictor_optimizer.zero_grad()

        pred = self.apply_predictor(mu, nu)
        loss = torch.mean((pred - true)**2)
        self.train_losses['predictor_data'].append(loss.item())

        # backpropagate predictor
        loss.backward(retain_graph=True)
        self.predictor_optimizer.step()
        
        return
    
    
    def convert_image(
            self, 
            img: torch.Tensor, 
            k: int
        ) -> torch.Tensor:
        # img-Form: (Batch, Channels, n, n)
        img = img.reshape(-1,1,self.length,self.length)
        img_int = F.interpolate(img, size=(k, k), mode='bilinear', align_corners=False).reshape(-1, k**2)
        # renormalize
        img_int -= img_int.min(1)[0].unsqueeze(1)
        img_int /= img_int.sum(dim=1, keepdim=True)
        img_int += self.dust_const
        img_int /= img_int.sum(dim=1, keepdim=True)

        return img_int


    def latend_samples(self):
        if self.which_gen == 'FNO':
            z = self.gaussian_random_samples(self.numbr_latend_samples, self.length)
        else:
            z = torch.randn(self.numbr_latend_samples, 2 * self.dim_prior, device=self.device)
        return z


    def save_measure(self):
        with torch.no_grad():
            z = torch.randn(20, 2 * self.dim_prior, device=self.device)
            
            mu, nu = self.apply_generator(z, self.generator)
            self.old_mu.append(mu.detach().cpu().reshape(-1, self.length, self.length))
            self.old_nu.append(nu.detach().cpu().reshape(-1, self.length, self.length))
            
            mu,nu = self.apply_generator(self.z_test, self.generator)
            self.old_fix_mu.append(mu.detach().cpu().reshape(-1, self.length, self.length))
            self.old_fix_nu.append(nu.detach().cpu().reshape(-1, self.length, self.length))
        return
    
    
    def plot(self):
        path = self.path + '/Images/'

        plot_measures.plot_complete_measures(self.old_fix_mu, self.old_fix_nu, self.name+'fix', path)
        plot_measures.plot_complete_measures(self.old_mu, self.old_nu, self.name, path)
        plot_measures.plot_measure_tight(self.old_mu, self.old_nu, self.name, path)
        plot_measures.plot_measure_tight(self.old_fix_mu, self.old_fix_nu, self.name+'fix', path)
        plot_measures.plot_measure_short(self.old_mu, self.old_nu, self.name, path)
        plot_measures.plot_measure_short(self.old_fix_mu, self.old_fix_nu, self.name+'fix', path)
        measure_dict = {'mu': self.old_mu, 'nu': self.old_nu, 'mu_fix': self.old_fix_mu, 'nu_fix': self.old_fix_nu}
        torch.save(measure_dict, self.path+'/'+self.name+'.pt')

        return

    
    def apply_sobel_filter(
            self,
            mu
        ):
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_x = sobel_x.to(mu.device)
        sobel_y = sobel_y.to(mu.device)
        
        mu_x = F.conv2d(mu.unsqueeze(1), sobel_x, padding=1)
        mu_y = F.conv2d(mu.unsqueeze(1), sobel_y, padding=1)
        
        sobel_filtered_mu = torch.cat((mu_x, mu_y), dim=1)        
        return sobel_filtered_mu
    
    
    def apply_predictor(
            self, 
            mu, 
            nu
        ):
        length = int(math.isqrt(mu.shape[-1]))
        if self.sobel:
            mu_s = self.apply_sobel_filter(mu.reshape(-1, length, length))
            nu_s = self.apply_sobel_filter(nu.reshape(-1, length, length))
            grid = self.get_grid(length)
            input = torch.cat((mu.reshape(-1,1,length,length), nu.reshape(-1,1,length,length), grid.repeat(mu.shape[0], 1,1,1), mu_s, nu_s), 1).reshape(-1, 8, length, length)
        elif self.grid:
            grid = self.get_grid(length)
            input = torch.cat((mu.reshape(-1,1,length,length), nu.reshape(-1,1,length,length), grid.repeat(mu.shape[0], 1,1,1), ), 1).reshape(-1, 4, length, length)
        else:
            input = torch.cat((mu.reshape(-1,1,length,length), nu.reshape(-1,1,length,length)), 1).reshape(-1, 2, length, length)
        return self.predictor(input.float()).reshape(-1, length*length)
    

    def get_grid(
            self, 
            length
        ):
        x = torch.linspace(0, 1, length)
        y = torch.linspace(0, 1, length)
        grid = torch.stack(torch.meshgrid(x, y)).to(self.device)
        return grid
    

    def apply_generator(
            self, 
            z, 
            generator
        ):
        if self.which_gen == 'FNO':
            length = int(math.isqrt(z.shape[-1]))
            z = z.reshape(-1, 2, length, length)

            out = generator(z)

            mu = out[:,0].reshape(-1, self.length**2)
            nu = out[:,1].reshape(-1, self.length**2)

            mu -= mu.min(1)[0].unsqueeze(1)
            nu -= nu.min(1)[0].unsqueeze(1)
            mu /= mu.sum(dim=1, keepdim=True)
            nu /= nu.sum(dim=1, keepdim=True)
            mu += self.dust_const
            nu += self.dust_const
            mu /= mu.sum(dim=1, keepdim=True)
            nu /= nu.sum(dim=1, keepdim=True)
        else:
            mu, nu = generator(z)
        return mu, nu


    def target_sinkhorn(
            self, 
            mu, 
            nu, 
            cost_matrix
        ):
        with torch.no_grad():
            # Bootstrap the empirical measures
            nu0 = torch.exp(self.apply_predictor(mu, nu))
            _, v = sinkhorn.sink_vec(mu, nu, cost_matrix, self.sinkhorn_epsilon, nu0, self.sinkhorn_max_iterations)
            
            g = torch.log(v)

        return g, ~(torch.isnan(v).any(dim=1)).to(self.device)


    def get_cost_matrix(
            self, 
            dim
        ):
        length = int(math.isqrt(dim))
        if length == 28:
            return self.cost_matrix_28
        else:
            return self.cost_matrix_64
        

    def checkpoint(
            self, 
            i
        ):
        if i % 10 == 0:
            self.test_performance()

        if i % 100 == 0:
            print(f'Generator 64 loss: {self.train_losses["generator 64"][-1]}',  f'Predictor Gen loss: {self.train_losses["predictor_gen"][-1]}')

        if i % (self.numbr_training_iterations // 10) == 0:
            self.save_measure()
            torch.save(self.predictor.state_dict(), f'{self.path}/Models/fno_model_save_{self.name}_{i}.pt')
                

    def test_performance(
            self, 
            print_output=False
        ):
        if self.test_data_28 is None:
            return            
        total_loss = []    

        def test_loss(test_set, cost_matrix, length):
            mu = test_set['mu'].to(self.device)
            nu = test_set['nu'].to(self.device)
            true_dist = test_set['dist']
            test_dist = []
            test_dist2 = []
            self.predictor.eval()
            nu0 = torch.exp(self.apply_predictor(mu, nu))
            with torch.no_grad():
                _,_,test_dist = sinkhorn.sink_vec_dist(mu, nu, cost_matrix, self.sinkhorn_epsilon, nu0, 1)
            test_dist = test_dist.cpu()
            cond = ~torch.isnan(test_dist)
            relative_error = (torch.abs(test_dist[cond]  - true_dist[cond]) / true_dist[cond]).mean()
            del mu, nu, nu0, test_dist
            self.predictor.train()
            return relative_error

        for key in self.test_data_28.keys():
            test_set = self.test_data_28[key]
            test_data_distance = test_loss(test_set, self.cost_matrix_28, 28)
            if print_output:
                print(f'28 dim Relative error {key}: {test_data_distance}')
            total_loss.append(test_data_distance)
            self.validation_loss[key].append(test_data_distance)
            wandb.log({'28 dim Relative error '+key: test_data_distance})
        
        for key in self.test_data_64.keys():
            test_set = self.test_data_64[key]
            test_data_distance = test_loss(test_set, self.cost_matrix_64, 64)
            if print_output:
                print(f'64 dim Relative error {key}: {test_data_distance}')

            total_loss.append(test_data_distance)
            self.validation_loss[key].append(test_data_distance)
            wandb.log({'64 dim Relative error '+key: test_data_distance})
        wandb.log({'Relative error total': torch.mean(torch.tensor(total_loss))})

        if torch.mean(torch.tensor(total_loss)) < self.best_total_loss:
            self.best_total_loss = torch.mean(torch.tensor(total_loss))
            self.best_model = copy.deepcopy(self.predictor.state_dict())