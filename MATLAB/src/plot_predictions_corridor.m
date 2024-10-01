function plot_predictions_corridor(y_test, Accuracy, varargin)
    %PLOT_PREDITIONS Plot the predictions and the test data.
    %
    %   Inputs:
    %       y_pred: matrix containing the output predictions
    %       y_test: test output trajectory
    %
    %   Variable-length input argument list:
    %       plot_percentiles: if set to true, percentiles are plotted
    %       y_min: min output to be plotted as constraint
    %       y_max: max output to be plotted as constraint
    
    % Default values
    plot_percentiles = false;
    y_min1 = [];
    y_max1 = [];
    y_max2 = [];
    Approach = 'predicted output vs. true output';
    % Read variable-length input argument list.
    for i = 1:2:length(varargin)
        if strcmp('plot_percentiles', varargin{i})
            plot_percentiles = varargin{i+1};
        elseif strcmp('y_max1', varargin{i})
            y_max1 = varargin{i+1};
        elseif strcmp('y_min1', varargin{i})
            y_min1 = varargin{i+1};
        elseif strcmp('y_max2', varargin{i})
            y_max2 = varargin{i+1};
        elseif strcmp('title', varargin{i})
            Approach = varargin{i+1};    
        end
    end
    
    % Get prediction horizon and number of outputs.
    [n_y, T_pred] = size(y_test);
    t_pred = 0:T_pred - 1;
    figure
    hold on
    % Plot the predictions and the test data for all output dimensions.

        % Plot constraints.

    minValue = min(min(y_test));
    maxValue = max(max(y_test));


    % if maxValue >= 0
    %     Constraint_1 = [5, 5];
    % else
    %     Constraint_1 = [5, 5];
    % end

    Constraint_1 = [5, 5];
    Constraint_1 = [Constraint_1, y_max1(y_max1 ~= inf)];
    h3 = fill([15,5, 5:15], Constraint_1, 0.3*[1, 1, 1], 'linestyle', 'none', 'FaceAlpha', 0.35, 'DisplayName', 'Constraints');


    Constraint_2 = [y_min1(y_min1 ~= -inf), y_max2(y_max2 ~= inf)];
    fill([flip(5:15), 5:15], Constraint_2, 0.3*[1, 1, 1], 'linestyle', 'none', 'FaceAlpha', 0.35, 'DisplayName', 'Constraints');
    
    for i = 1:n_y
        % Plot true output.
        if Accuracy(i) == 1
            h1 = plot(t_pred, y_test(i, :), 'b', 'linewidth', 0.1, 'DisplayName', 'True Output (Successful)');
        else
            h2 = plot(t_pred, y_test(i, :), 'r', 'linewidth', 0.1, 'DisplayName', 'True Output (Failed)');
        end
    end

    %legend([h1(1), h2(1), h3(1)], 'True Output (Successful)', 'True Output (Failed)', 'Constraints');

    % Add title, labels...

    title(Approach);
    ylabel('y');
    xlabel('t');
    %legend('Location', 'northwest');
    %ylim([min([min(Constraint_2)-1, minValue]), Constraint_1(1)]);
    ylim([-12, 4])
    grid on;
    hold off;
end