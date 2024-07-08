function [U_opt, X_opt, Y_opt] = Solve_OCP_Scenario_Constraints(PG_samples, x_vec_0, v_vec, e_vec, H, K, phi, g, n_x, n_y, n_u, y_min, y_max)

optimization_timer = tic;

cvx_begin quiet
    variable U(n_u, H)

    expression X(n_x, H+1, K)
    expression Y(n_y, H, K)

    X(:, 1, :) = x_vec_0;

    for k = 1:K
        A = PG_samples{k}.A;
        f = @(x, u) A * phi(x, u);

        for t = 1:H
            X(:, t+1, k) = f(X(:, t, k), U(:, t)) + v_vec(:, t, k);
            Y(:, t, k) = g(X(:, t, k), U(:, t)) + e_vec(:, t, k);
        end
    end

    minimize( sum(U.^2) )

    subject to
        U >= -10;
        U <= 10;

        for t = 1:H
            if y_min(t) ~= -inf
                Y(:, t, :) >= y_min(t) * ones(n_y, 1, K)
            end

            if y_max(t) ~= inf
                Y(:, t, :) <= y_max(t) * ones(n_y, 1, K)
            end
        end
cvx_end

time_scenario = toc(optimization_timer)

U_opt = U;
X_opt = X;
Y_opt = Y;

end