function [U_opt, X_opt, Y_opt] = Solve_OCP_Kernel_maxConstraint_casadi(PG_samples, x_vec_0, v_vec, e_vec, H, K, phi, g, n_x, n_y, n_u, y_min, y_max, alpha, sigma_mult)

optimization_timer = tic;

h_scenario = @(u, x, y) bounded_output(u, x, y, y_min, y_max);
maxfunc = @(x) log(1 + exp(x/0.1)) * 0.1;

x_vec_0 = x_vec_0(:,:,1:K);
v_vec = v_vec(:,:,1:K);

Kernel = rbf_kernel(x_vec_0, v_vec, e_vec, PG_samples, K, sigma_mult);
K_chol = chol(Kernel + 1e-1 * eye(K));
%epsilon = (1 + sqrt(2 * log(1 / alpha))) * sqrt(1 / K);

epsilon = BootstrapAmbiguity(Kernel, 10000, 0.95);

solver_opts = struct('linear_solver', 'ma57', 'max_iter', 5000, 'hessian_approximation', 'limited-memory');
casadi_opts = struct('expand', 1);


opti = casadi.Opti();
U = opti.variable(n_u, H);
X = opti.variable(n_x*K, H+1);
Y = opti.variable(n_y*K, H);

X_init = reshape(x_vec_0, [size(x_vec_0, 3) * n_x, 1]);

% Set the initial state.
opti.subject_to(X(:, 1) == X_init(1:(n_x*K)));

gammaN = opti.variable(K, 1);
g0 = opti.variable(1);
tk = opti.variable(1);

opti.set_initial(gammaN, ones(K,1));

g_rkhs = Kernel * gammaN;
Eg_rkhs = sum(g_rkhs) / K;
g_norm = norm(K_chol * gammaN);

opti.subject_to(g0 + Eg_rkhs + epsilon * g_norm <= tk * alpha);

% Define objective.
opti.minimize(sum(U.^2));

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

sol = opti.solve();

U_opt = sol.value(U);
X_opt = reshape(sol.value(X)', [n_x, H + 1, K]);
Y_opt = reshape(sol.value(Y)', [n_y, H, K]);

time_maxConstraint = toc(optimization_timer)

end