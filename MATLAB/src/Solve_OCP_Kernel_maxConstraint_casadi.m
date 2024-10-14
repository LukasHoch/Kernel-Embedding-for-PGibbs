function [U_opt, X_opt, Y_opt] = Solve_OCP_Kernel_maxConstraint_casadi(PG_samples, x_vec_0, v_vec, e_vec, H, K, phi, g, n_x, n_y, n_u, y_min, y_max, alpha, sigma_mult, u_init, solver_opts, casadi_opts)

optimization_timer = tic;

X_init_0 = reshape(x_vec_0, [size(x_vec_0, 3) * n_x, 1]);
X_init_0 = X_init_0(1:(n_x*K));

X_init = [X_init_0, zeros(n_x*K, H)]; % initial guess for X
Y_init = zeros(n_y*K, H); % initial guess for Y
for k = 1:K
    % Get model.
    A = PG_samples{k}.A;
    f = @(x, u) A * phi(x, u);
    for t = 1:H
        X_init(n_x*(k - 1)+1:n_x*k, t+1) = f(X_init(n_x*(k - 1)+1:n_x*k, t), u_init(:, t)) + v_vec(:, t, k);
        Y_init(n_y*(k - 1)+1:n_y*k, t) = g(X_init(n_x*(k - 1)+1:n_x*k, t), u_init(:, t)) + e_vec(:, t, k);
    end
end

h_scenario = @(u, x, y) bounded_output(u, x, y, y_min, y_max);

%maxfunc = @(x) log(1 + exp(x/.1)) * .1;
maxfunc = @(x) (x + sqrt(x.^2 + 0.01))/2;


x_vec_0 = x_vec_0(:,:,1:K);
v_vec = v_vec(:,:,1:K);

Kernel = rbf_kernel(x_vec_0, v_vec, e_vec, PG_samples, K, sigma_mult);
K_chol = chol(Kernel + 1e-2 * eye(K));
%epsilon = (1 + sqrt(2 * log(1 / alpha))) * sqrt(1 / K);

epsilon = BootstrapAmbiguity(Kernel, 10000, 0.95);

% solver_opts = struct('linear_solver', 'ma57', 'max_iter', 40000, 'hessian_approximation', 'limited-memory','print_level', 0);
% 
% %solver_opts = struct('nlp_scaling_method', 'gradient-based', 'tol', 1e-5, 'acceptable_tol', 1e-4, 'constr_viol_tol', 1e-5,...
%     %'max_iter', 1000, 'max_soc', 100, 'alpha_red_factor', 0.5, 'warm_start_init_point', 'yes', 'warm_start_bound_push', 1e-6,...
%     %'hessian_approximation', 'limited-memory', 'acceptable_iter', 10, 'watchdog_shortened_iter_trigger', 1, 'mu_strategy', 'adaptive', 'print_level', 5, 'output_file', 'ipopt_log.txt'); 
% 
% casadi_opts = struct('expand', 1);


opti = casadi.Opti();
U = opti.variable(n_u, H);
X = opti.variable(n_x*K, H+1);
Y = opti.variable(n_y*K, H);

opti.set_initial(U, u_init);
opti.set_initial(X, X_init);
opti.set_initial(Y, Y_init);

% Set the initial state.
opti.subject_to(X(:, 1) == X_init_0);

gammaN = opti.variable(K, 1);
g0 = opti.variable(1);
tk = opti.variable(1);

opti.set_initial(gammaN, ones(K,1));

g_rkhs = Kernel * gammaN;
Eg_rkhs = sum(g_rkhs) / K;
g_norm = norm(K_chol * gammaN);

opti.subject_to(g0 + Eg_rkhs + epsilon * g_norm <= tk * alpha);

% Define objective.
opti.minimize(sum(U.^2)+ 1e-6 * (sum(gammaN) + g0 + tk));

% Add dynamic and additional constraints for all scenarios.
for k = 1:K
    % Get model.
    A = PG_samples{k}.A;
    f = @(x, u) A * phi(x, u);

    for t = 1:H
        % Add dynamic constraints.
        opti.subject_to(X(n_x*(k - 1)+1:n_x*k, t+1) == f(X(n_x*(k - 1)+1:n_x*k, t), U(:, t))+v_vec(:, t, k));
        opti.subject_to(Y(n_y*(k - 1)+1:n_y*k, t) == g(X(n_x*(k - 1)+1:n_x*k, t), U(:, t))+e_vec(:, t, k));
    end

    opti.subject_to(maxfunc(h_scenario(U, X(n_x*(k - 1)+1:n_x*k, :), Y(n_y*(k - 1)+1:n_y*k, :)) + tk) <= g0 + g_rkhs(k));
end



% Add constraints for the input.
opti.subject_to(U <= 10);
opti.subject_to(U >= -10);


% Set numerical backend.
opti.solver('ipopt', casadi_opts, solver_opts);
try 
    sol = opti.solve();

    U_opt = sol.value(U);
    X_opt = reshape(sol.value(X)', [n_x, H + 1, K]);
    Y_opt = reshape(sol.value(Y)', [n_y, H, K]);

    time_maxConstraint = toc(optimization_timer)
catch
    time_maxConstraint = toc(optimization_timer);
    warning('Optimization not sucessful!\nRuntime: %.2f s\n', time_maxConstraint);
    U_opt = [];
    X_opt = [];
    Y_opt = [];
end
end
