%%%%%%%%%%%%%%%%%%%%%%%%%%Neural Network Assignment%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%June 14 2017%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
clc;

%%Parameters:%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

epochs = 1000;
num_00 = 250;
num_01 = 250;
num_11 = 250;
num_10 = 250;
num_0_1_total = num_00 + num_01 + num_11 + num_10;
percent_train = 0.7;
learning_rate = 0.8;
momentum = 0.3; %0 if no momentum
noise = false;
num_unique_molecules = 6;
concentration_matrix_00 = [0.1790, 0.0030, 0.1080, 0.0000, 0.6800, 0.0300]; %a, b, c, d, e, p in that order
concentration_matrix_01 = [0.0600, 0.2100, 0.4480, 0.0000, 0.2820, 0.0000];
concentration_matrix_11 = [0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000];
concentration_matrix_10 = [0.0760, 0.2090, 0.4190, 0.0000, 0.2960, 0.0000];

concentration_total_matrix_pre = zeros(num_0_1_total,num_unique_molecules);
result_operation_matrix_pre = zeros(num_0_1_total,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%Data Generation

%For perceptron making purposes (AND) (result):
for i = 1:num_00
    result_operation_matrix_pre(i,1) = 0;
end
for i = num_00 + 1: num_00 + num_01
    result_operation_matrix_pre(i,1) = 0;
end
for i = num_00 + num_01 + 1:num_00 + num_01 + num_11
    result_operation_matrix_pre(i,1) = 1;
end
for i = num_00 + num_01 + num_11 + 1:num_0_1_total
    result_operation_matrix_pre(i,1) = 0;
end

%For perceptron making purposes (AND) (input):
for i = 1:num_00
    concentration_total_matrix_pre(i,:) = [0.1790, 0.0030, 0.1080, 0.0000, 0.6800, 0.0300];
end
for i = num_00 + 1: num_00 + num_01
    concentration_total_matrix_pre(i,:) = [0.0600, 0.2100, 0.4480, 0.0000, 0.2820, 0.0000];
end
for i = num_00 + num_01 + 1:num_00 + num_01 + num_11
    concentration_total_matrix_pre(i,:) = [0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000];
end
for i = num_00 + num_01 + num_11 + 1:num_0_1_total
    concentration_total_matrix_pre(i,:) = [0.0760, 0.2090, 0.4190, 0.0000, 0.2960, 0.0000];
end

num_data_train  = round(num_0_1_total * percent_train);
num_data_test = num_0_1_total - num_data_train;

rand_perm = randperm(num_0_1_total)';

concentration_total_matrix = concentration_total_matrix_pre(rand_perm,:);
result_operation_matrix = result_operation_matrix_pre(rand_perm,:);

%Altered "data_matrix_actual" to "test_concentration_matrix"
data_train = concentration_total_matrix(1:num_data_train,:); %%Splits xo matrix to training set
data_test = concentration_total_matrix((num_data_train + 1):(num_data_train + num_data_test),:); %%Splits xo matrix to testing set

%%Generate noisy test data (noise = false if no noise)

noise_vector = [0];

%%%%


%%%%

result_matrix_actual_train = result_operation_matrix(1:num_data_train,:); %%Produces results for test
result_matrix_actual_test = result_operation_matrix((num_data_train + 1):(num_data_train + num_data_test),:); %%Produces results for test


%%%%%%%%%%%%%%%%FIRST TIME: RANDOMIZED WEIGHT MATRIX%%%%%%%%%%%%%%%%%%%%%%%

tic

%%Generate Matrix of Input Neurons

data_each_shape = data_train(1,:);

input_neurons = zeros(1,size(data_each_shape,2));
for i = 1:size(data_each_shape,2) 
    input_neurons(i) = data_each_shape(i);
end

%%Define Output Vectors - Orientation
% output_vector = zeros(1,2);
% if isempty(strfind(fnames_train{1},'left')) == 0;
%     output_vector(1,:) = [1 0];
% end
% if isempty(strfind(fnames_train{1},'right')) == 0;
%     output_vector(1,:) = [0 1];
% end

output_vector = result_matrix_actual_train(1,:);

%%Generate Matrix of Weights
rng('shuffle')
weightmin = -0.5;
weightmax = 0.5;
weight_matrix_12 = weightmin + rand(1,size(data_train,2))*(weightmax - weightmin);
% weight_matrix_23 = weightmin + rand(1,num_hid_nrns+1)*(weightmax - weightmin);
%^^Weightrix23 is no longer...

%Single layer perceptron: No longer need this:
% %%Dot-Products to Produce Hidden Neurons
% hidden_neurons = ones(1,num_hid_nrns+1);
% hidden_neurons_before_act = ones(1,num_hid_nrns+1);
% for i = 1:num_hid_nrns
%     hidden_neurons_before_act(1,i) = dot(input_neurons(1,:),weight_matrix_12(i,:));
%     hidden_neurons(1,i) = sigmf(hidden_neurons_before_act(1,i),[1 0]);
% end
% num_hid_nrns = num_hid_nrns + 1;

%%Dot-Products to Produce Output Neurons

output_neurons_before_act = zeros(1, size(output_vector,2));
output_neurons = zeros(1,size(output_vector,2));
for i = 1:size(output_vector,2)
    output_neurons_before_act(1,i) = dot(input_neurons(1,:),weight_matrix_12(i,:));
    output_neurons(1,i) = sigmf(output_neurons_before_act(1,i),[1 0]);
end

%%Compute backpropagation error - Output Layer

pd_cost_output = zeros(1, size(output_vector,2));
sigma_deriv_output = zeros(1, size(output_vector,2));
for i = 1:size(output_vector,2)
    pd_cost_output(1,i) = (output_neurons(1,i)-output_vector(1,i));
    sigma_deriv_output(1,i) = sigmf(output_neurons_before_act(1,i),[1 0])*(1-sigmf(output_neurons_before_act(1,i),[1 0]));
end     
gammaL_output = pd_cost_output.*sigma_deriv_output;
gammaL_output_act = ((input_neurons')*gammaL_output)';

% %%Compute backpropagation error - Hidden layer
% 
% sigma_deriv_hidden = zeros(1,num_hid_nrns);
% for i = 1:num_hid_nrns
%     sigma_deriv_hidden(1,i) = sigmf(hidden_neurons_before_act(1,i),[1 0])*(1-sigmf(hidden_neurons_before_act(1,i),[1 0]));
% end
% gammaL_hidden = (gammaL_output*weight_matrix_23).*sigma_deriv_hidden;
% gammaL_hidden_act = ((input_neurons')*gammaL_hidden)';


%%Gradient Descent: Weights - learning rate*gammaL

if momentum ~= 0
% velocity_new_hidden = - learning_rate*gammaL_hidden_act;
velocity_new_output = - learning_rate*gammaL_output_act;
weight_matrix_12 = weight_matrix_12 + velocity_new_output;
% weight_matrix_23 = weight_matrix_23 + velocity_new_output;
end

if momentum == 0
weight_matrix_12 = weight_matrix_12 - learning_rate*gammaL_output_act;
% weight_matrix_23 = weight_matrix_23 - learning_rate*gammaL_output_act;
end

toc;
display('Epoch 1 done');




%%%%%%%%%%%%%%%%%%%%%%%SECOND TIME: TRAINED WEIGHT MATRIX%%%%%%%%%%%%%%%%%%

%%Define Loop for each Data Row

for n = 2:epochs
    
tic

for m = 2:num_data_train
    
%%Generate Matrix of Input Neurons/Letter Matrix

data_each_shape = data_train(m,:);

input_neurons = zeros(1,size(data_each_shape,2));
for i = 1:size(data_each_shape,2) 
    input_neurons(i) = data_each_shape(i);
end

%%Define Output Vectors - Orientation
% output_vector = zeros(1,2);
% if isempty(strfind(fnames_train{m},'left')) == 0;
%     output_vector(1,:) = [1 0];
% end
% if isempty(strfind(fnames_train{m},'right')) == 0;
%     output_vector(1,:) = [0 1];
% end
output_vector = result_matrix_actual_train(m,:);

%%Dot-Products to Produce Hidden Neurons
% num_hid_nrns = num_hid_nrns - 1;
% hidden_neurons = ones(1,num_hid_nrns+1);
% hidden_neurons_before_act = ones(1,num_hid_nrns+1);
% for i = 1:num_hid_nrns
%     hidden_neurons_before_act(1,i) = dot(input_neurons(1,:),weight_matrix_12(i,:));
%     hidden_neurons(1,i) = sigmf(hidden_neurons_before_act(1,i),[1 0]);
% end
% num_hid_nrns = num_hid_nrns + 1;

%%Dot-Products to Produce Output Neurons

output_neurons_before_act = zeros(1, size(output_vector,2));
output_neurons = zeros(1,size(output_vector,2));
for i = 1:size(output_vector,2)
    output_neurons_before_act(1,i) = dot(input_neurons(1,:),weight_matrix_12(i,:));
    output_neurons(1,i) = sigmf(output_neurons_before_act(1,i),[1 0]);
end

%%Compute backpropagation error - Output Layer

pd_cost_output = zeros(1, size(output_vector,2));
sigma_deriv_output = zeros(1, size(output_vector,2));
for i = 1:size(output_vector,2)
    pd_cost_output(1,i) = (output_neurons(1,i)-output_vector(1,i));
    sigma_deriv_output(1,i) = sigmf(output_neurons_before_act(1,i),[1 0])*(1-sigmf(output_neurons_before_act(1,i),[1 0]));
end     
gammaL_output = pd_cost_output.*sigma_deriv_output;
gammaL_output_act = ((input_neurons')*gammaL_output)';

% %%Compute backpropagation error - Hidden layer
% 
% sigma_deriv_hidden = zeros(1,num_hid_nrns);
% for i = 1:num_hid_nrns
%     sigma_deriv_hidden(1,i) = sigmf(hidden_neurons_before_act(1,i),[1 0])*(1-sigmf(hidden_neurons_before_act(1,i),[1 0]));
% end
% gammaL_hidden = (gammaL_output*weight_matrix_23).*sigma_deriv_hidden;
% gammaL_hidden_act = ((input_neurons')*gammaL_hidden)';


%%Gradient Descent: Weights - learning rate*gammaL

if momentum ~= 0
% velocity_new_hidden = - learning_rate*gammaL_hidden_act;
velocity_new_output = - learning_rate*gammaL_output_act;
weight_matrix_12 = weight_matrix_12 + velocity_new_output;
% weight_matrix_23 = weight_matrix_23 + velocity_new_output;
end

if momentum == 0
weight_matrix_12 = weight_matrix_12 - learning_rate*gammaL_output_act;
% weight_matrix_23 = weight_matrix_23 - learning_rate*gammaL_output_act;
end

end

toc;
display(['Epoch ' num2str(n) ' done']);

end



%%%%%%%%%%%%%%%%%%%%%TESTING PHASE%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

maximum_orientation = zeros(1,num_data_test);
actual_orientation = zeros(1,num_data_test);

for m = num_data_train + 1: num_data_train + num_data_test

%%Generate Matrix of Input Neurons/Letter Matrix

data_each_shape = data_test(m-num_data_train,:);

input_neurons = zeros(1,size(data_each_shape,2));
for i = 1:size(data_each_shape,2) 
    input_neurons(i) = data_each_shape(i);
end

% num_hid_nrns = num_hid_nrns - 1;
% 
% %%Dot-Products to Produce Hidden Neurons
% hidden_neurons = ones(1,num_hid_nrns+1);
% hidden_neurons_before_act = ones(1,num_hid_nrns+1);
% for i = 1:num_hid_nrns
%     hidden_neurons_before_act(1,i) = dot(input_neurons(1,:),weight_matrix_12(i,:));
%     hidden_neurons(1,i) = sigmf(hidden_neurons_before_act(1,i),[1 0]);
% end
% % hidden_neurons
% num_hid_nrns = num_hid_nrns + 1;

%%Dot-Products to Produce Output Neurons

output_neurons = zeros(1,size(output_vector,2));
for i = 1:size(output_vector,2)
    output_neurons_before_act(1,i) = dot(input_neurons(1,:),weight_matrix_12(i,:));
    output_neurons(1,i) = sigmf(output_neurons_before_act(1,i),[1 0])
end

% output_neurons
maximum_orientation(m-num_data_train) = round(output_neurons);

end



%%%%%%%%%%%%%%%%%%%%%%TOTAL ERRORS%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Define Actual Letter
% for i = 1:num_data_test
%     if isempty(strfind(fnames_test{i},'left')) == 0;
%         actual_orientation(i) = 1;
%     end
%     if isempty(strfind(fnames_test{i},'right')) == 0;
%         actual_orientation(i) = 2;
%     end
% end
% 
% correct_matrix = zeros(1,num_data_test);
% for i = 1:num_data_test
%     if actual_orientation(i) == maximum_orientation(i)
%         correct_matrix(i) = 1;
%     end
% end

wrong_matrix = abs(maximum_orientation' - result_matrix_actual_test);




total_wrong = sum(wrong_matrix);
total_correct = num_data_test - total_wrong;
predicted_correct_over_total_correct = total_correct/num_data_test*100;

% display(['For Number of Hidden Neurons of ' num2str(num_hid_nrns)]);
display(['Percent of Correct Predictions: ' num2str(predicted_correct_over_total_correct) '%']);



% confusion_matrix = zeros(2,2);
% LL = zeros(1,num_data_test);
% LR = zeros(1,num_data_test);
% RL = zeros(1,num_data_test);
% RR = zeros(1,num_data_test);
% for i = 1:num_data_test
%     if isempty(strfind(fnames_test{i},'left')) == 0;
%         if actual_orientation(i) == maximum_orientation(i)
%             LL(i) = 1;
%         end
%         if actual_orientation(i) ~= maximum_orientation(i)
%             LR(i) = 1;
%         end
%     end
%     if isempty(strfind(fnames_test{i},'right')) == 0;
%         if actual_orientation(i) == maximum_orientation(i)
%             RR(i) = 1;
%         end
%         if actual_orientation(i) ~= maximum_orientation(i)
%             RL(i) = 1;
%         end
%     end
% end
% 
% confusion_matrix(1,1) = sum(LL);
% confusion_matrix(1,2) = sum(LR);
% confusion_matrix(2,1) = sum(RL);
% confusion_matrix(2,2) = sum(RR);


%%%%%%%%%%%%%%%%%%%%%%%%%%%
% for i = 1:num_data_test
%     if isempty(strfind(fnames_test{i},'left')) == 0;
%         if actual_orientation(i) ~= maximum_orientation(i)
%             fnames_test{i}
%         end
%     end
%     if isempty(strfind(fnames_test{i},'right')) == 0;
%         if actual_orientation(i) ~= maximum_orientation(i)
%             fnames_test{i}
%         end
%     end
% end




fprintf('Perceptron Finished\n\n\n');