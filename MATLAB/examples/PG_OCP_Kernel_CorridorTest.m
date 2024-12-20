%clear;
%clc;
%close all;

% Specify seed (for reproducible results).
rng(5);

% Import src
addpath('..\src')

% Import CasADi - insert your path here.
addpath('<yourpath>/casadi-3.6.5-windows64-matlab2018b')
import casadi.*

K = 2300; % number of PG samples
k_d = 70; % number of samples to be skipped to decrease correlation (thinning)
K_b = 2000; % length of burn-in period
N = 30; % number of particles of the particle filter

n_x = 2; % number of states
n_u = 1; % number of control inputs
n_y = 1; % number of outputs

%% State-space prior
% Define basis functions - assumed to be known in this example.
% Make sure that phi(x,u) is defined in vectorized form, i.e., phi(zeros(n_x,N), zeros(n_u, N)) should return a matrix of dimension (n_phi, N).
% Scaling the basis functions facilitates the exploration of the posterior distribution and reduces the required thinning parameter k_d.
n_phi = 3; % number of basis functions
%phi = @(x, u) [0.1 * x(1, :); 0.1 * x(2, :); u(1, :); 0.01 * cos(3*x(1, :)) .* x(2, :); 0.1 * sin(2*x(2, :)) .* u(1, :)]; % basis functions
phi = @(x, u) [0.5 * x(1, :); 0.5 * x(2, :); 0.5 * u(1, :)]; % basis functions
%phi = @(x, u) [0.1 * x(1, :); 0.1 * x(2, :); u(1, :); x(1, :) .* x(2, :); x(1, :) .* u(1, :)]; % basis functions

% Prior for Q - inverse Wishart distribution
ell_Q = 10; % degrees of freedom
Lambda_Q = 100 * eye(n_x); % scale matrix

% Prior for A - matrix normal distribution (mean matrix = 0, right covariance matrix = Q (see above), left covariance matrix = V)
V = diag(10*ones(n_phi, 1)); % left covariance matrix

% Initial guess for model parameters
Q_init = Lambda_Q; % initial Q
A_init = zeros(n_x, n_phi); % initial A

% Normally distributed initial state
x_init_mean = [2; 2]; % mean
x_init_var = 1 * ones(n_x, 1); % variance

%% Measurement model
% Define measurement model - assumed to be known (without loss of generality).
% Make sure that g(x,u) is defined in vectorized form, i.e., g(zeros(n_x,N), zeros(n_u, N)) should return a matrix of dimension (n_y, N).
g = @(x, u) [1, 0] * x; % observation function
R = 0.1; % variance of zero-mean Gaussian measurement noise

%% Parameters for data generation
T = 2000; % number of steps for training
T_test = 500; % number of steps used for testing (via forward simulation - see below)
T_all = T + T_test;

%% Generate training data.
% Choose the actual system (to be learned) and generate input-output data of length T_all.
% The system is of the form
% x_t+1 = f_true(x_t, u_t) + N(0, Q_true),
% y_t = g_true(x_t, u_t) + N(0, R_true).

% Unknown system
%f_true = @(x, u) [0.8 * x(1, :) - 0.5 * x(2, :) + 0.1 * cos(3*x(1, :)) * x(2, :); 0.4 * x(1, :) + 0.5 * x(2, :) + (1 + 0.3 * sin(2*x(2, :))) * u(1, :)]; % true state transition function
%f_true = @(x, u) [x(1, :) + 0.5 * x(2, :) + 0.125 * u(1, :); 0.8 * x(2, :) + 0.5 * u(1, :)]; % true state transition function
f_true = @(x, u) [0.8 * x(1, :) - 0.5 * x(2, :) ; 0.4 * x(1, :) + 0.5 * x(2, :) + u(1, :)]; % true state transition function
Q_true = [0.03, -0.004; -0.004, 0.01]; % true process noise variance
g_true = g; % true measurement function
R_true = R; % true measurement noise variance

% Input trajectory used to generate training and test data
u_training = mvnrnd(0, 3, T)'; % training inputs
u_test = 3 * sin(2*pi*(1 / T_test)*((1:T_test) - 1)); % test inputs
u = [u_training, u_test]; %  training + test inputs

% Generate data by forward simulation.
x = zeros(n_x, T_all+1); % true latent state trajectory
x(:, 1) = normrnd(x_init_mean, x_init_var); % random initial state
y = zeros(n_y, T_all); % output trajectory (measured)
for t = 1:T_all
    x(:, t+1) = f_true(x(:, t), u(t)) + mvnrnd(zeros(n_x, 1), Q_true)';
    y(:, t) = g_true(x(:, t)) + mvnrnd(0, R_true, n_y);
end

% Split data into training and test data
u_training = u(:, 1:T);
x_training = x(:, 1:T+1);
y_training = y(:, 1:T);

u_test = u(:, T+1:end);
x_test = x(:, T+1:end);
y_test = y(:, T+1:end);

%% Learn models.
% Result: K models of the type
% x_t+1 = PG_samples{i}.A*phi(x_t,u_t) + N(0,PG_samples{i}.Q),
% where phi are the basis functions defined above.

% PG_samples = load("PG_Samples\PGibbs_Samples.mat");
% PG_samples = PG_samples.PG_samples;

PG_samples = load("PG_Samples\PGibbs_Samples_K2300_kd_70.mat");
PG_samples = PG_samples.PG_samples;

%PG_samples = particle_Gibbs(u_training, y_training, K, K_b, k_d, N, phi, Lambda_Q, ell_Q, Q_init, V, A_init, x_init_mean, x_init_var, g, R);


H=21;
t_tmp = 5:15;



y_max1 = [inf * ones(1, 5), -2.5 * ones(1, 11), inf * ones(1, 5)];
y_min1 = [-inf * ones(1, 5), -5 * ones(1, 11), -inf * ones(1, 5)];


y_max2 = [inf * ones(1, 5), -5-sqrt(25-(t_tmp-10).^2), inf * ones(1, 5)];
y_min2 = -inf * ones(1, 21);

R = 0.1;
alpha = 0.1:0.1:0.3;


%sigma_mult = [1.5 5 5 1];                      %Sigma1 Used for previous examples
%sigma_mult = [1.6875 0.6250 1.6406 1.6875];    %Sigma2 Generated using SigmaTuning
sigma_mult = [0.5716 1.4062 1.4062 0.2109];    %Sigma3 Generated using SigmaTuning2   
%sigma_mult = [0.5648 1.4214 1.5660 0.2824];    %Sigma4 Generated using SigmaTuning3 (Potentially the same as SigmaTuning2, just more converged)


K_opt = 80;
if K_opt > K
    K_opt = K;
end

N_S = 5;

y_true_scenario = zeros(N_S, H);
y_true_kernel = zeros(N_S, H, length(alpha));

cost_scenario = zeros(N_S, 1);
cost_kernel = zeros(N_S, length(alpha));

Accuracy_scenario  = zeros(N_S, 1);
Accuracy_kernel = zeros(N_S, length(alpha));

path_safe = zeros(length(alpha) + 1, 1);

for n_s = 1:N_S
    n_s
    PG_samples = PG_samples(randperm(K));

    x_vec_0 = zeros(n_x, 1, K);
    for k = 1:K
        % Get model.
        A = PG_samples{k}.A;
        Q = PG_samples{k}.Q;
        f = @(x, u) A * phi(x, u);
    
        % Sample state at t=-1.
        star = systematic_resampling(PG_samples{k}.w_m1, 1);
        x_m1 = PG_samples{k}.x_m1(:, star);
    
        % Propagate.
        x_vec_0(:, 1, k) = f(x_m1, PG_samples{k}.u_m1) + mvnrnd(zeros(1, n_x), Q)';
    end
    
    v_vec = zeros(n_x, H, K);
    v_vec_0 = mvnrnd(zeros(1, n_x), Q_true, H)';
    for k = 1:K
        Q = PG_samples{k}.Q;
        v_vec(:, :, k) = mvnrnd(zeros(1, n_x), Q, H)';
        %v_vec(:, :, k) = v_vec_0;
    end
    
    % Sample measurement noise array e_vec if not provided.
    e_vec = zeros(n_y, H, K);
    e_vec_0 = mvnrnd(zeros(1, n_y), R, H)';
    
    for k = 1:K
        e_vec(:, :, k) = mvnrnd(zeros(1, n_y), R, H)';
        %e_vec(:, :, k) = e_vec_0;
    end
    
    v_true = zeros(n_x, H);
    e_true = zeros(n_y, H);
    
    for t = 1:H
        v_true(:,t) = mvnrnd(zeros(n_x, 1), Q_true)';
        e_true(:,t) = mvnrnd(zeros(n_y, 1), R_true);
    end
    
    [U_scenario, X_scenario, Y_scenario] = Solve_OCP_Scenario_Constraints(PG_samples, x_vec_0, v_vec, e_vec, H, K_opt, phi, g, n_x, n_y, n_u, y_min1, y_max1);
    
    if anynan(U_scenario)
        [U_scenario, X_scenario, Y_scenario] = Solve_OCP_Scenario_Constraints(PG_samples, x_vec_0, v_vec, e_vec, H, K_opt, phi, g, n_x, n_y, n_u, y_min2, y_max2);
        path_safe(1) = path_safe(1) + 1;
    end
    
    cost_scenario(n_s) = sum(U_scenario.^2);

    x_true_scenario = zeros(n_x, H+1);

    x_true_scenario(:, 1) = x_training(:, end);
    for t = 1:H
        x_true_scenario(:, t+1) = f_true(x_true_scenario(:, t), U_scenario(t)) + v_true(:,t);
        y_true_scenario(n_s, t) = g_true(x_true_scenario(:, t), U_scenario(t)) + e_true(:,t);
    end

    C_upper1 = all(y_true_scenario(n_s, :) > y_min1);
    C_lower1 = all(y_true_scenario(n_s, :) < y_max1);

    C_upper2 = all(y_true_scenario(n_s, :) > y_min2);
    C_lower2 = all(y_true_scenario(n_s, :) < y_max2);
    
    Accuracy_scenario(n_s) = (C_upper1 & C_lower1) | (C_upper2 & C_lower2) ;


    for i = 1:length(alpha)
    
        [U_maxConstr, X_maxConstr, Y_maxConstr] = Solve_OCP_Kernel_maxConstraint(PG_samples, x_vec_0, v_vec, e_vec, H, K_opt, phi, g, n_x, n_y, n_u, y_min1, y_max1, alpha(i), sigma_mult);
        
        if anynan(U_maxConstr)
            [U_maxConstr, X_maxConstr, Y_maxConstr] = Solve_OCP_Kernel_maxConstraint(PG_samples, x_vec_0, v_vec, e_vec, H, K_opt, phi, g, n_x, n_y, n_u, y_min2, y_max2, alpha(i), sigma_mult);
            path_safe(i+1) = path_safe(i+1) + 1;
        end
        
        cost_kernel(n_s, i) = sum(U_maxConstr.^2);

        x_true_kernel = zeros(n_x, H+1);
    
        x_true_kernel(:, 1) = x_training(:, end);
        for t = 1:H
            x_true_kernel(:, t+1) = f_true(x_true_kernel(:, t), U_maxConstr(t)) + v_true(:,t);
            y_true_kernel(n_s, t, i) = g_true(x_true_kernel(:, t), U_maxConstr(t)) + e_true(:,t);
        end

        C_upper1 = all(y_true_kernel(n_s, :, i) > y_min1);
        C_lower1 = all(y_true_kernel(n_s, :, i) < y_max1);
    
        C_upper2 = all(y_true_kernel(n_s, :, i) > y_min2);
        C_lower2 = all(y_true_kernel(n_s, :, i) < y_max2);
        
        Accuracy_kernel(n_s, i) = (C_upper1 & C_lower1) | (C_upper2 & C_lower2) ;
    end 
end
Accuracy = [sum(Accuracy_scenario)/N_S sum(Accuracy_kernel)/N_S]

plot_predictions_corridor(y_true_scenario, Accuracy_scenario, 'y_max1', y_max1, 'y_min1', y_min1, 'y_max2', y_max2, 'title', 'True output (Scenario Approach)')
for i = 1:length(alpha)
    plot_predictions_corridor(y_true_kernel(:,:,i), Accuracy_kernel(:,i), 'y_max1', y_max1, 'y_min1', y_min1, 'y_max2', y_max2, 'title', ['True output (Kernel Approach with alpha = ' num2str(alpha(i)) ')'])
end
