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

%K = 2300; % number of PG samples
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
n_phi = 5; % number of basis functions
phi = @(x, u) [0.1 * x(1, :); 0.1 * x(2, :); u(1, :); 0.01 * cos(3*x(1, :)) .* x(2, :); 0.1 * sin(2*x(2, :)) .* u(1, :)]; % basis functions
%phi = @(x, u) [0.5 * x(1, :); 0.5 * x(2, :); 0.5 * u(1, :)]; % basis functions

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
f_true = @(x, u) [0.8 * x(1, :) - 0.5 * x(2, :) + 0.1 * cos(3*x(1, :)) * x(2, :); 0.4 * x(1, :) + 0.5 * x(2, :) + (1 + 0.3 * sin(2*x(2, :))) * u(1, :)]; % true state transition function
%f_true = @(x, u) [0.8 * x(1, :) - 0.5 * x(2, :) ; 0.4 * x(1, :) + 0.5 * x(2, :) + u(1, :)]; % true state transition function
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
% PG_samples = load("PGibbs_Samples.mat");
% PG_samples = PG_samples.PG_samples;

%PG_samples = load("PGibbs_Samples_K2300_kd_70.mat");
%PG_samples = PG_samples.PG_samples;

PG_samples = load('PGibbs_Samples_K2200_R1.mat');
PG_samples = PG_samples.PG_samples;

%PG_samples = particle_Gibbs(u_training, y_training, K, K_b, k_d, N, phi, Lambda_Q, ell_Q, Q_init, V, A_init, x_init_mean, x_init_var, g, R);

K = length(PG_samples);

PG_samples = PG_samples(randperm(K));
%PG_samples = PG_samples(randperm(K));



s = 1;


if s == 1
    H = 11;
    y_min = [-inf * ones(1, 7) 10 -inf * ones(1, 3)];
    y_max = inf * ones(1, 11); 

elseif s == 2
    H = 41;
    y_min = [-inf * ones(1, 30), 10 * ones(1, 11)];
    y_max = [inf * ones(1, 10), -10* ones(1, 11), inf * ones(1, 20)];

elseif s == 3  
    H = 41;
    y_min = [-inf * ones(1, 10), -6.5:0.5:6, -inf * ones(1, 5)];
    y_max = [inf * ones(1, 10), -2.5:0.5:10, inf * ones(1, 5)];

elseif s == 4
    H = 101;
    y_min = [-inf * ones(1, 41), 10 * ones(1, 10), -inf * ones(1, 40), 10 * ones(1, 10)];
    y_max = [inf * ones(1, 9), -10* ones(1, 11), inf * ones(1, 41),-10* ones(1, 19), inf * ones(1, 21)];

elseif s == 5
    H = 101;
    y_min = [-inf * ones(1, 96), zeros(1,5)];
    y_max = [inf * ones(1, 29), -3* ones(1, 10), -6 * ones(1, 10), -9 * ones(1, 10),  -12 * ones(1, 20), inf * ones(1, 22)];

elseif s == 6
    H = 101;
    y_min = [-inf * ones(1, 20), -5* ones(1, 20), 5* ones(1, 10), -5* ones(1, 20), -inf * ones(1, 21), 10* ones(1, 10)];
    y_max = [inf * ones(1, 20), 0* ones(1, 10), 10 * ones(1, 30), 0 * ones(1, 10), inf * ones(1, 31)];

elseif s == 7
    H = 101;
    y_min = [-inf * ones(1, 20), -10* ones(1, 20), -5 * ones(1, 10), 0 * ones(1, 10),  5 * ones(1, 10), -inf * ones(1, 31)];
    y_max = [inf * ones(1, 20), -5* ones(1, 10), 0 * ones(1, 10), 5 * ones(1, 10),  10 * ones(1, 20), inf * ones(1, 31)];

elseif s == 8
    H = 101;
    y_min = [-inf * ones(1, 20), -10:0.25:5, -inf * ones(1, 20)];
    y_max = [inf * ones(1, 20), -5:0.25:10, inf * ones(1, 20)];

end


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
    v_true(:,t) = mvnrnd(zeros(n_y, 1), R_true);
end

X_stacked = reshape(x_vec_0, [], K);
V_stacked = reshape(v_vec, [], K);
W_stacked = reshape(e_vec, [], K);



K2 = size(PG_samples{1}.A, 1) * size(PG_samples{1}.A, 2);
A_stacked = zeros(K2, K);
for n = 1:K
    A_stacked(:,n) = reshape(PG_samples{n}.A, K2, 1);
end


iter_max = 15;
sigma_cnt = 51;

%sigma_init = [1.5 5 5 1];
sigma_init =  [0.716 1.4062 1.4062 0.4109];
%sigma_init = ones(1,4);

K_train = 200;
K_test = K - K_train;

sigma = zeros(iter_max * 4 + 1, 4);
sigma(1,:) = sigma_init;

sigma_range = zeros(sigma_cnt, 4, iter_max * 4 + 1);
sigma_range(:,:,1) = repmat(sigma_init, sigma_cnt, 1);

kernel_mean = zeros(K_test, sigma_cnt, iter_max * 4 + 1);
kernel_mean_mean = zeros(iter_max * 4 + 1, sigma_cnt);

cnts = 1;

for iter = 1:iter_max
    %iter
    for ts = [1 4]
        sigma_range(:,ts,cnts) = linspace(0.9* sigma(cnts, ts), 1.1* sigma(cnts, ts), sigma_cnt)';
        for sigma_iter = 1:sigma_cnt
            for k_test = 1:K_test
                for k_train = 1:K_train
                    if ts == 1
                        X_diff = norm(X_stacked(:,k_train) - X_stacked(:,k_test + K_train))^2;
                        kernel_tmp = 1/(sqrt(2 * pi) * sigma_range(sigma_iter,1,cnts)) * exp(-X_diff/(2*sigma_range(sigma_iter,1,cnts)^2));
                    elseif ts == 2
                        V_diff = norm(V_stacked(:,k_train) - V_stacked(:,k_test + K_train))^2;
                        kernel_tmp = 1/(sqrt(2 * pi) * sigma_range(sigma_iter,2,cnts)) * exp(-V_diff/(2*sigma_range(sigma_iter,2,cnts)^2));
                    elseif ts == 3 
                        W_diff = norm(W_stacked(:,k_train) - W_stacked(:,k_test + K_train))^2;
                        kernel_tmp = 1/(sqrt(2 * pi) * sigma_range(sigma_iter,3,cnts)) * exp(-W_diff/(2*sigma_range(sigma_iter,3,cnts)^2));
                    else
                        A_diff = norm(A_stacked(:,k_train) - A_stacked(:,k_test + K_train))^2;
                        kernel_tmp = 1/(sqrt(2 * pi) * sigma_range(sigma_iter,4,cnts)) * exp(-A_diff/(2*sigma_range(sigma_iter,4,cnts)^2));
                    end
    
                    kernel_mean(k_test, sigma_iter, cnts) = kernel_mean(k_test, sigma_iter, cnts) + kernel_tmp;
                end
    
                kernel_mean(:, sigma_iter, cnts) = kernel_mean(:, sigma_iter, cnts) / K_train;
                kernel_mean_mean(cnts, sigma_iter) = sum(kernel_mean(:, sigma_iter, cnts)) / K_test;
            end
        end
        [~, sigma_opt_idx] = max(kernel_mean_mean(cnts, :));
    
        minIdx = find(kernel_mean_mean == kernel_mean_mean(sigma_opt_idx));
    
        [~, minIdx_idx] = min((minIdx - ceil(sigma_cnt/2)).^2);
    
        sigma_opt_idx = minIdx(minIdx_idx);
    
        sigma_range(:,:,cnts+1) = repmat(sigma_range(sigma_opt_idx, :, cnts), sigma_cnt, 1);
    
        sigma(cnts+1, :) = sigma_range(sigma_opt_idx, :, cnts);
    
        cnts = cnts + 1;
    end
end

sigma(cnts, :)
