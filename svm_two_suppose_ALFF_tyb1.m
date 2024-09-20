% ALFF limbic+DMN+salient   46 regions
% select 30 is better for Peking
% actually both peking and NYU are used with 30 typical features   %
% 2024-5-30

% notice!
% output: train_h0_label, train_h1_label:  1(True) ADHD, 0(flase) HC
% output: test_h0_label,  test_h1_label :  2       ADHD, 1        HC
% train_h0_data/train_h1_data are with energy_normalization.   
% test_h0_data/test_h1_data are without energy_normalization.
% 2024-6-2

% test_h0_data/test_h1_data are energy_normalization.
% where test_h0_data/test_h1_data are the last data of train_h0_data/train_h1_data

function [train_h0_data, train_h0_label, train_h0_score,...
          train_h1_data, train_h1_label, train_h1_score, rank_h0]...
          =svm_two_suppose_ALFF_tyb1(index, data_name)
      
% Include dependencies
warning('off');
addpath('./lib'); % dependencies
addpath('./methods'); % FS methods
addpath(genpath('./lib/drtoolbox'));

% data_name = 'PU_alff_ds';


listFS = {'ILFS','InfFS','ECFS','mrmr','relieff','mutinffs','fsv','laplacian','mcfs','rfe','L0','fisher','UDFS','llcfs','cfs'};

% [ methodID ] = readInput( listFS );
selection_method = listFS{10}; % Selected rfe

% Load the data and select features for classification
load([data_name '.mat'])


% for index=1:172
% index=172

% X_temp = inform.brain_conn_show;    % FC��
% X_temp = alff(:,[21:22, 29:42, 71:78, 91:116]);                 % ALFF limbic+cerebellum 
% X_temp = alff(:,[3:6, 19:42, 65:68, 71:78, 83:88]);                 % FC
X_temp = alff(:,[1:6, 17:42, 57:60, 63:78, 83:88]);                 % FC
Y_temp = tag;
Y_temp = nominal(ismember(Y_temp,0));
Y_temp = logical(double(Y_temp)-1);
X = X_temp;

Y = nominal(ismember(Y_temp,1)); 

% train_h0_data is without ordered and contain test_h0_data
[train_h0_data, train_h0_label, train_h0_score, rank_h0] = train_rfe( X, Y,  score);
train_h0_label = double(train_h0_label)-1;
[train_h0_data] = energy_normalization(train_h0_data, logical(train_h0_label));

train_h0_data = train_h0_data([1:index-1 index+1:end index],:);
train_h0_score = train_h0_score([1:index-1 index+1:end index],:);
train_h0_label = train_h0_label([1:index-1 index+1:end index],:);

if Y(index) == 'true'
    Y(index) = 'false';
else
    Y(index) = 'true';
end

% bbb=0

[train_h1_data,  train_h1_label, train_h1_score,rank_h1] = train_rfe(X, Y, score);
train_h1_label = double(train_h1_label)-1;
[train_h1_data] = energy_normalization(train_h1_data, logical(train_h1_label));

train_h1_data = train_h1_data([1:index-1 index+1:end index],:);
train_h1_score = train_h1_score([1:index-1 index+1:end index],:);
train_h1_label = train_h1_label([1:index-1 index+1:end index],:);

%  aaa(index)=length(intersect(rank_h0,rank_h1));
%   end
end

function [train_data_out] = energy_normalization(train_data, train_label)
    tmp = train_data';
    sample_energy_tmp = sqrt(sum(tmp.^2));
    
    agv_energy_1 = mean(sample_energy_tmp(train_label));
    avg_energy_0 = mean(sample_energy_tmp(~train_label));
    sizeoftmp = size(tmp);
    sample_energy_map = ones(1, sizeoftmp(2));
    sample_energy_map(train_label) = agv_energy_1;
    sample_energy_map(~train_label) = avg_energy_0;
    
    energy_map = ones(size(tmp,1),1) * sample_energy_map;
    train_data_out = (tmp ./ energy_map)';     
    return
end

function [train_h0_data, train_h0_label, train_h0_score, rank_h0] = train_rfe(X, Y, score)
    X_train = double(X);
    Y_train = (double(Y)-1)*2-1; % labels: neg_class -1, pos_class +1

    numF =35;   %tyb 2020-11-17

    % feature Selection on training data
    ranking = spider_wrapper(X_train,Y_train,numF,'rfe');
    k = numF; % select the first 55 features
   
    rank_h0 = ranking(1:k);
    train_h0_data = X_train(:,ranking(1:k));
    train_h0_data = mapminmax(train_h0_data', 0, 1)';   % clap within [0 1]
    train_h0_label = Y;
    train_h0_score = score;
    
    return
end
