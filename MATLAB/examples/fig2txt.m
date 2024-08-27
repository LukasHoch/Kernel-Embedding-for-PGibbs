close all

addpath('..\src')
addpath('..\figures')

file_input = 'ComputationTimes_K250_S2';
file_output = '..\data\ComputationTimes_K250_S2';

[A, varnames] = ExtractFigData(file_input);

opt.fname = file_output;
opt.var_names = varnames;

data2txt(opt, A(:,1), A(:,2), A(:,3)) 


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
    
        if dataObjs(n).DisplayName == "all predictions"
            n_total = n_total + 1;
        end
    end
    
    A = zeros(length(dataObjs(n_filtered(1)).XData), n_total);
    varnames = cell(n_total, 1);
    
    
    for n = 1:length(n_filtered)
        if dataObjs(n_filtered(n)).DisplayName == "all predictions"
            varnames{n+1} = 'y_opt_min';
            A(:,n+1) = dataObjs(n_filtered(n)).YData(1:(end/2));
    
            varnames{end} = 'y_opt_max';
            A(:,end) = flip(dataObjs(n_filtered(n)).YData(((end/2)+1):end));
        else
            varnames{n+1} = dataObjs(n_filtered(n)).DisplayName;
            A(:,n+1) = dataObjs(n_filtered(n)).YData;
    
            varnames{1} = 't';
            A(:,1) = dataObjs(n_filtered(n)).XData;
        end
    end                   
end