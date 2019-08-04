%%%%%%%%%%%%%%%%%%%%%%Reservoir Computing Exercise%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%June 13, 2017%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%Generate Input Data%%

addpath('/u/rtan/Desktop/Teuscher/ESNToolbox');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


clear all;
clc;

%%%% generate the training data
% %
% sequenceLength = 1000; %%sequenceLength has been replaced by 'num_xo_total'
% 
% random_perm = randperm(sequenceLength);
% 
disp('Generating data ............');
% disp(sprintf('Sequence Length %g', sequenceLength ));

%%Don't really need systemOrder
systemOrder = 3 ; % set the order of the NARMA equation


%%%%%%%%%%%%%%%%%%%%%%%%ON LOAN FROM NN ASSIGNMENT%%%%%%%%%%%%%%%%%%%%%%%%%

num_x = 500;
num_o = 500;
num_xo_total = num_x + num_o;
percent_train = 0.5;
num_hidden_neurons = 1000;
do_reactions = true;

noise = true;

num_data_train  = round(num_xo_total * percent_train);
num_data_test = round(num_xo_total * (1 - percent_train));

poss_x_num_patterns = 4;
poss_o_num_patterns = 4;
poss_x = [1 0 1 0 0 1 0 0 1 0 1 0 0 0 0 0; 0 1 0 1 0 0 1 0 0 1 0 1 0 0 0 0; 0 0 0 0 1 0 1 0 0 1 0 0 1 0 1 0; 0 0 0 0 0 1 0 1 0 0 1 0 0 1 0 1];
poss_o = [1 1 1 0 1 0 1 0 1 1 1 0 0 0 0 0; 0 0 0 0 1 1 1 0 1 0 1 0 1 1 1 0; 0 1 1 1 0 1 0 1 0 1 1 1 0 0 0 0; 0 0 0 0 0 1 1 1 0 1 0 1 0 1 1 1];


rand_x = randi([1 poss_x_num_patterns], 1, num_x)';
rand_o = randi([1 poss_o_num_patterns],  1, num_o)';

for i = 1:num_x
    data_matrix_x(i,:) = poss_x(rand_x(i,:),:);
end
for i = 1:num_o
    data_matrix_o(i,:) = poss_o(rand_o(i,:),:);
end

data_matrix_concat = [data_matrix_x; data_matrix_o];
random_perm = randperm(num_x + num_o);
data_matrix_actual = data_matrix_concat(random_perm,:); %%Randomizes xo matrix

%%Generate outputSequence
result_matrix_concat = [ones(num_x,1);zeros(num_o,1)];  %%Produces results of xo matrix
result_matrix_actual = result_matrix_concat(random_perm,:); %%Randomizes xo matrix


data_train = data_matrix_actual(1:num_data_train,:); %%Splits xo matrix to training set
data_test = data_matrix_actual((num_data_train + 1):(num_data_train + num_data_test),:); %%Splits xo matrix to testing set

%%Generate noisy test data (noise = false if no noise)
%%%%

noise_vector = [2];

if noise == true
%%For levels 1 and 2: First 2 is 1 --> 0; Second 2 is 0 --> 1; Level 3 is random
poss_x_noisy_test = [1 0 1 0 0 1 0 0 1 0 1 0 0 0 0 0; 0 1 0 1 0 0 1 0 0 1 0 1 0 0 0 0; 0 0 0 0 1 0 1 0 0 1 0 0 1 0 1 0; 0 0 0 0 0 1 0 1 0 0 1 0 0 1 0 1]; %% Level 0 noise
% poss_x_noisy_test = [ 1 0 1 0 0 0 0 0 1 0 1 0 0 0 0 0; 0 1 0 1 0 0 1 0 0 1 0 0 0 0 0 0; 1 0 0 0 1 0 1 0 0 1 0 0 1 0 1 0; 0 0 0 0 0 1 0 1 0 1 1 0 0 1 0 1]; %% Level 1 noise
% poss_x_noisy_test = [ 1 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0; 0 0 0 1 0 0 1 0 0 0 0 1 0 0 0 0; 0 0 0 0 1 0 1 1 1 1 0 0 1 0 1 0; 1 0 0 0 0 1 0 1 0 0 1 0 1 1 0 1]; %% Level 2 noise
% poss_x_noisy_test = [ 0 0 1 0 1 1 0 0 1 0 1 0 0 1 0 0; 0 1 1 1 0 0 0 0 0 1 0 1 0 0 0 1; 0 0 0 0 1 0 0 0 1 0 0 0 1 0 1 0; 1 0 1 0 0 1 0 1 0 0 1 1 0 1 0 1];%% Level 3 noise
                 
                 
poss_o_noisy_test = [1 1 1 0 1 0 1 0 1 1 1 0 0 0 0 0; 0 0 0 0 1 1 1 0 1 0 1 0 1 1 1 0; 0 1 1 1 0 1 0 1 0 1 1 1 0 0 0 0; 0 0 0 0 0 1 1 1 0 1 0 1 0 1 1 1]; %% Level 0 noise
% poss_o_noisy_test = [ 1 1 1 1 1 0 1 0 1 1 1 0 0 0 0 0; 0 0 0 0 1 1 1 0 1 1 1 0 1 1 1 0; 0 1 1 1 0 1 0 1 0 1 1 1 0 0 0 1; 0 0 0 1 0 1 1 1 0 1 0 1 0 1 1 1]; %% Level 1 noise
% poss_o_noisy_test = [ 1 1 1 0 1 0 1 0 1 1 1 0 1 0 1 0; 0 0 0 0 1 1 1 1 1 1 1 0 1 1 1 0; 1 1 1 1 0 1 0 1 0 1 1 1 0 0 1 0; 0 1 0 0 0 1 1 1 0 1 0 1 1 1 1 1]; %% Level 2 noise
% poss_o_noisy_test = [ 0 1 1 0 1 0 1 0 1 1 0 1 0 1 0 0; 0 1 0 0 1 0 0 1 1 0 1 0 1 1 1 0; 1 1 0 1 0 1 0 1 0 1 1 1 0 1 1 0; 0 0 1 0 0 1 1 0 1 1 1 0 0 1 0 1];%% Level 3 noise
                 
result_matrix_noisy_test = [ones(poss_x_num_patterns*size(noise_vector,2),1);zeros(poss_o_num_patterns*size(noise_vector,2),1)];
                 
poss_concat_noisy_test = [poss_x_noisy_test; poss_o_noisy_test];

rand_xo_noisy_test = randi([1 (poss_x_num_patterns + poss_o_num_patterns)*size(noise_vector,2)], 1, num_data_test); 


%Will override previous data_test
data_test = poss_concat_noisy_test(rand_xo_noisy_test,:);

end

%%%%

result_matrix_actual_train = result_matrix_actual(1:num_data_train,:); %%Produces results for test
result_matrix_actual_test = result_matrix_actual((num_data_train + 1):(num_data_train + num_data_test),:); %%Produces results for test

if noise == true
result_matrix_actual_test = result_matrix_noisy_test(rand_xo_noisy_test);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Generate inputSequence
% data_matrix_x = repmat([1 0 1 0 1 0 1 0 1],sequenceLength/2,1);
% data_matrix_o = repmat([1 1 1 1 0 1 1 1 1],sequenceLength/2,1);
% data_matrix_concat = [data_matrix_x; data_matrix_o];
% random_perm = randperm(sequenceLength);
% data_matrix_actual = data_matrix_concat(random_perm,:); %%Generates atestError randomized matrix of x or o rows
% data_matrix_actual = [ones(sequenceLength,1),data_matrix_actual]; %%Include bias (1)
% 
% %%Generate outputSequence
% result_matrix_concat = [ones(sequenceLength/2,1);zeros(sequenceLength/2,1)];
% result_matrix_actual = result_matrix_concat(random_perm,:);
% 
% 
% %[inputSequence outputSequence] = generate_NARMA_sequence(sequenceLength , systemOrder) ; 
% 
% 
% %%%% split the data into train and test
% 
% train_fraction = 0.7 ; % use 50% in training and 50% in testing
% [trainInputSequence, testInputSequence] = ...
%     split_train_test(data_matrix_actual,train_fraction);
% [trainOutputSequence,testOutputSequence] = ...
%     split_train_test(result_matrix_actual,train_fraction);

%%%%%%

%%%%%%%%%%Change it up: convert my data to train and test
trainInputSequence = [ones(num_data_train,1) , data_train];
testInputSequence = [ones(num_data_test,1) , data_test];
trainOutputSequence = result_matrix_actual_train;
testOutputSequence = result_matrix_actual_test;

%%%%%%%%%%

disp('Generating ESN ............');


%%%% generate an esn 
nInputUnits = 17; nInternalUnits = num_hidden_neurons; nOutputUnits = 1; 
% 
esn = generate_esn(nInputUnits, nInternalUnits, nOutputUnits, ...
    'spectralRadius',0.3,'inputScaling',0.1*ones(nInputUnits,1),'inputShift',zeros(nInputUnits,1), ...
    'teacherScaling',[0.3],'teacherShift',[-0.2],'feedbackScaling', 0, ...
    'type', 'plain_esn'); 

esn.internalWeights = esn.spectralRadius * esn.internalWeights_UnitSR;

%%%% train the ESN
nForgetPoints = 100 ; % discard the first 100 points
[trainedEsn stateMatrix] = ...
    train_esn(trainInputSequence, trainOutputSequence, esn, nForgetPoints) ; 

%%%% save the trained ESN
save_esn(trainedEsn, 'esn_narma_demo_1'); 

%%%% plot the internal states of 4 units
nPoints = 200 ; 
% plot_states(stateMatrix,[1], nPoints, 1, 'traces of first 1 reservoir units') ; 

% compute the output of the trained ESN on the training and testing data,
% discarding the first nForgetPoints of each
nForgetPoints = 0 ; 
predictedTrainOutput = test_esn(trainInputSequence, trainedEsn, nForgetPoints);
predictedTestOutput = test_esn(testInputSequence,  trainedEsn, nForgetPoints) ; 

% create input-output plots
nPlotPoints = 100 ; 
% plot_sequence(trainOutputSequence(nForgetPoints+1:end,:), predictedTrainOutput, nPlotPoints,...
%     'training: teacher sequence (red) vs predicted sequence (blue)');
% plot_sequence(testOutputSequence(nForgetPoints+1:end,:), predictedTestOutput, nPlotPoints, ...
%     'testing: teacher sequence (red) vs predicted sequence (blue)') ; 

disp('Using ESN to predict ............');

%%Compute rounded predictions
predictedTrainOutputActual = round(predictedTrainOutput);

%%%Some values are below 0 or above one, we need to deal with that
for i = 1:size(predictedTestOutput,1)
    if predictedTestOutput(i) <= 1 && predictedTestOutput(i) >= 0
        predictedTestOutputActual(i) = round(predictedTestOutput(i));
    elseif predictedTestOutput(i) > 1
            predictedTestOutputActual(i) = 1;
    else
        predictedTestOutputActual(i) = 0;
    end
    
end
predictedTestOutputActual = predictedTestOutputActual';

diff_matrix_train = abs(predictedTrainOutputActual - trainOutputSequence);
train_error_sum = sum(diff_matrix_train);
diff_matrix_test = abs(predictedTestOutputActual - testOutputSequence);
test_error_sum = sum(diff_matrix_test);

train_error = train_error_sum / (num_xo_total * percent_train);
disp(sprintf('Percentage of correctly trained = %s', num2str(1 - train_error), ' percent'))

test_error = test_error_sum / (num_xo_total * (1 - percent_train));
disp(sprintf('Percentage of correctly tested = %s', num2str(1 - test_error), ' percent'))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('Computing error ............');

%%%%compute NRMSE training error
%%Changed NRMSE Code to produce RMSE instead
trainNRMSEError = compute_NRMSE(predictedTrainOutput, trainOutputSequence);
disp(sprintf('train RMSE = %s', num2str(trainNRMSEError)))

%%%%compute NRMSE testing error
testNRMSEError = compute_NRMSE(predictedTestOutput, testOutputSequence); 
disp(sprintf('test RMSE = %s', num2str(testNRMSEError)))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
