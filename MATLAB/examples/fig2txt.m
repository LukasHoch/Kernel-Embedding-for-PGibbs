close all

addpath('..\src')
addpath('..\figures')
addpath('..\figures\Corridor_Figures')
addpath('..\figures\Report_Figures')
addpath('..\figures\Plots_nonlinearCase')

%file_input = 'AlphaTest_K300_MaxConstraint_S2';
%file_output = '..\data\AlphaTest_K300_MaxConstraint_S2';

file_input = 'Kernel_alpha02_K200_S10_nonlinear';
file_output = '..\data\Kernel_alpha02_K200_S10_nonlinear';

% file_input = 'Scenario_K100_N50';
% file_output = '..\data\Scenario_K100_N50';

[A, varnames] = ExtractFigData(file_input);

opt.fname = file_output;
opt.var_names = varnames;

data2txt(opt, A(:,1), A(:,2), A(:,3), A(:,4), A(:,5)) 

%data2txt(opt, A(:,1), A(:,2), A(:,3), A(:,4), A(:,5), A(:,6), A(:,7), A(:,8), A(:,9), A(:,10), A(:,11), A(:,12), A(:,13), A(:,14), A(:,15), A(:,16), A(:,17), A(:,18), A(:,19), A(:,20), A(:,21), A(:,22), A(:,23), A(:,24), A(:,25), A(:,26), A(:,27), A(:,28), A(:,29), A(:,30), A(:,31), A(:,32), A(:,33), A(:,34), A(:,35), A(:,36), A(:,37), A(:,38), A(:,39), A(:,40), A(:,41), A(:,42), A(:,43), A(:,44), A(:,45), A(:,46), A(:,47), A(:,48), A(:,49), A(:,50), A(:,51)) 

%AlphaTest_K300_MaxConstraints_S2_Alpha04



function  [A, varnames] = ExtractFigData(fileinput) 
    fig = openfig(fileinput);
    
    axObjs = get(fig, 'Children');
    dataObjs = get(axObjs, 'Children');
    
    dataObjs = dataObjs{2};
    
    n_filtered = [];
    n_total = 1;
    
    for n = 1:length(dataObjs)
        n_filtered = [n_filtered n];
        n_total = n_total + 1;
    
        if dataObjs(n).DisplayName == "Upper bound constraints"
            n_filtered(end) = [];
            n_total = n_total - 1;
        end
    
        if dataObjs(n).DisplayName == "Lower bound constraints"
            n_filtered(end) = [];
            n_total = n_total - 1;
        end

        if dataObjs(n).DisplayName == "Constraints"
            n_filtered(end) = [];
            n_total = n_total - 1;
        end
    
        if dataObjs(n).DisplayName == "all predictions"
            n_total = n_total + 1;
        end
    end
    
    A = zeros(length(dataObjs(n_filtered(1)).XData), n_total);
    varnames = cell(n_total, 1);
    
    cnts = 1;
    cntf = 1;
    
    for n = 1:length(n_filtered)
        if dataObjs(n_filtered(n)).DisplayName == "all predictions"
            varnames{n+1} = 'y_opt_min';
            A(:,n+1) = dataObjs(n_filtered(n)).YData(1:(end/2));
    
            varnames{end} = 'y_opt_max';
            A(:,end) = flip(dataObjs(n_filtered(n)).YData(((end/2)+1):end));
        else
            varnames{n+1} = dataObjs(n_filtered(n)).DisplayName;
            if string(dataObjs(n_filtered(n)).DisplayName) == "True Output (Successful)"
                varnames{n+1} = ['Successful' num2str(cnts)];
                cnts = cnts + 1;
            end
            if string(dataObjs(n_filtered(n)).DisplayName) == "True Output (Failed)"
                varnames{n+1} = ['Failed' num2str(cntf)];
                cntf = cntf + 1;
            end

            A(:,n+1) = dataObjs(n_filtered(n)).YData;
    
            varnames{1} = 't';
            A(:,1) = dataObjs(n_filtered(n)).XData;
        end
    end                   
end