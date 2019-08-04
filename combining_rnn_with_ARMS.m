%%%%%%%%%%%%%%%%%%%%%%%%%%Combining RNN with ARMS%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%June 30 2017%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
clc;

%%Parameters:%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

epochs = 100;
num_x = 500;
num_o = 500;
num_xo_total = num_x + num_o;
percent_train = 0.7;
num_hid_nrns = 50;
learning_rate = 0.8;
momentum = 0.3; %0 if no momentum
noise = true;
random_reservoir = false;
do_reactions = true;
reaction_count = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%Data Generation

num_data_train  = round(num_xo_total * percent_train);
num_data_test = round(num_xo_total * (1 - percent_train));

poss_x_num_patterns = 4;
poss_o_num_patterns = 4;
poss_x = [1 0 1 0 0 1 0 0 1 0 1 0 0 0 0 0; 0 1 0 1 0 0 1 0 0 1 0 1 0 0 0 0; 0 0 0 0 1 0 1 0 0 1 0 0 1 0 1 0; 0 0 0 0 0 1 0 1 0 0 1 0 0 1 0 1];
poss_o = [1 1 1 0 1 0 1 0 1 1 1 0 0 0 0 0; 0 0 0 0 1 1 1 0 1 0 1 0 1 1 1 0; 0 1 1 1 0 1 0 1 0 1 1 1 0 0 0 0; 0 0 0 0 0 1 1 1 0 1 0 1 0 1 1 1];


rand_x = randi([1 poss_x_num_patterns], 1, num_x)';
rand_o = randi([1 poss_o_num_patterns], 1, num_o)';

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

noise_vector = [0];

%%%%

if noise == true
 poss_x_noisy_test = [%1 0 1 0 0 1 0 0 1 0 1 0 0 0 0 0; 0 1 0 1 0 0 1 0 0 1 0 1 0 0 0 0; 0 0 0 0 1 0 1 0 0 1 0 0 1 0 1 0; 0 0 0 0 0 1 0 1 0 0 1 0 0 1 0 1]; %% Level 0 noise
%                        1 0 1 0 0 0 0 0 1 0 1 0 0 0 0 0; 0 1 0 1 0 0 1 0 0 1 0 0 0 0 0 0; 1 0 0 0 1 0 1 0 0 1 0 0 1 0 1 0; 0 0 0 0 0 1 0 1 0 1 1 0 0 1 0 1]; %% Level 1 noise
                     1 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0; 0 0 0 1 0 0 1 0 0 0 0 1 0 0 0 0; 0 0 0 0 1 0 1 1 1 1 0 0 1 0 1 0; 1 0 0 0 0 1 0 1 0 0 1 0 1 1 0 1]; %% Level 2 noise
%                      0 0 1 0 1 1 0 0 1 0 1 0 0 1 0 0; 0 1 1 1 0 0 0 0 0 1 0 1 0 0 0 1; 0 0 0 0 1 0 0 0 1 0 0 0 1 0 1 0; 1 0 1 0 0 1 0 1 0 0 1 1 0 1 0 1];%% Level 3 noise

%%%Abort
% poss_x_noisy_test_actual = zeros(poss_x_num_patterns*size(noise_vector,2),size(poss_x_noisy_test_before,2));
% for i = 1 : poss_x_num_patterns * size(noise_vector,2)
%     i
%      for j = poss_x_num_patterns * (noise_vector(floor((i-1)/poss_x_num_patterns)+1)+1) : poss_x_num_patterns * noise_vector(floor(i/poss_x_num_patterns)+1)
%          j
%             poss_x_noisy_test_actual(i,:) = poss_x_noisy_test_before(j,:);
%      end
% end
%%%Abort

                 
                 
 poss_o_noisy_test = [%1 1 1 0 1 0 1 0 1 1 1 0 0 0 0 0; 0 0 0 0 1 1 1 0 1 0 1 0 1 1 1 0; 0 1 1 1 0 1 0 1 0 1 1 1 0 0 0 0; 0 0 0 0 0 1 1 1 0 1 0 1 0 1 1 1]; %% Level 0 noise
%                      1 1 1 1 1 0 1 0 1 1 1 0 0 0 0 0; 0 0 0 0 1 1 1 0 1 1 1 0 1 1 1 0; 0 1 1 1 0 1 0 1 0 1 1 1 0 0 0 1; 0 0 0 1 0 1 1 1 0 1 0 1 0 1 1 1]; %% Level 1 noise
                     1 1 1 0 1 0 1 0 1 1 1 0 1 0 1 0; 0 0 0 0 1 1 1 1 1 1 1 0 1 1 1 0; 1 1 1 1 0 1 0 1 0 1 1 1 0 0 1 0; 0 1 0 0 0 1 1 1 0 1 0 1 1 1 1 1]; %% Level 2 noise
%                      0 1 1 0 1 0 1 0 1 1 0 1 0 1 0 0; 0 1 0 0 1 0 0 1 1 0 1 0 1 1 1 0; 1 1 0 1 0 1 0 1 0 1 1 1 0 1 1 0; 0 0 1 0 0 1 1 0 1 1 1 0 0 1 0 1];%% Level 3 noise

                 
%%Define result vector
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
num_hid_nrns = num_hid_nrns - 1;
weight_matrix_12 = weightmin + rand(num_hid_nrns+1,size(data_train,2))*(weightmax - weightmin);
weight_matrix_23 = weightmin + rand(1,num_hid_nrns+1)*(weightmax - weightmin);

%%Dot-Products to Produce Hidden Neurons
hidden_neurons = ones(1,num_hid_nrns+1);
hidden_neurons_before_act = ones(1,num_hid_nrns+1);
for i = 1:num_hid_nrns
    hidden_neurons_before_act(1,i) = dot(input_neurons(1,:),weight_matrix_12(i,:));
    hidden_neurons(1,i) = sigmf(hidden_neurons_before_act(1,i),[1 0]);
end
num_hid_nrns = num_hid_nrns + 1;



%%%%Reservoir Implemetation%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if (random_reservoir == true)
    hidden_neurons = reaction_random(hidden_neurons,num_hid_nrns);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%Dot-Products to Produce Output Neurons

output_neurons = zeros(1,size(output_vector,2));
for i = 1:size(output_vector,2)
    output_neurons_before_act(1,i) = dot(hidden_neurons(1,:),weight_matrix_23(i,:));
    output_neurons(1,i) = sigmf(output_neurons_before_act(1,i),[1 0]);
end

%%Compute backpropagation error - Output Layer

for i = 1:size(output_vector,2)
    pd_cost_output(1,i) = (output_neurons(1,i)-output_vector(1,i));
    sigma_deriv_output(1,i) = sigmf(output_neurons_before_act(1,i),[1 0])*(1-sigmf(output_neurons_before_act(1,i),[1 0]));
end     
gammaL_output = pd_cost_output.*sigma_deriv_output;
gammaL_output_act = ((hidden_neurons')*gammaL_output)';

%%Compute backpropagation error - Hidden layer

for i = 1:num_hid_nrns
    sigma_deriv_hidden(1,i) = sigmf(hidden_neurons_before_act(1,i),[1 0])*(1-sigmf(hidden_neurons_before_act(1,i),[1 0]));
end
gammaL_hidden = (gammaL_output*weight_matrix_23).*sigma_deriv_hidden;
gammaL_hidden_act = ((input_neurons')*gammaL_hidden)';


%%Gradient Descent: Weights - learning rate*gammaL

if momentum ~= 0
velocity_new_hidden = - learning_rate*gammaL_hidden_act;
velocity_new_output = - learning_rate*gammaL_output_act;
weight_matrix_12 = weight_matrix_12 + velocity_new_hidden;
weight_matrix_23 = weight_matrix_23 + velocity_new_output;
end

if momentum == 0
weight_matrix_12 = weight_matrix_12 - learning_rate*gammaL_hidden_act;
weight_matrix_23 = weight_matrix_23 - learning_rate*gammaL_output_act;
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
num_hid_nrns = num_hid_nrns - 1;
hidden_neurons = ones(1,num_hid_nrns+1);
hidden_neurons_before_act = ones(1,num_hid_nrns+1);
for i = 1:num_hid_nrns
    hidden_neurons_before_act(1,i) = dot(input_neurons(1,:),weight_matrix_12(i,:));
    hidden_neurons(1,i) = sigmf(hidden_neurons_before_act(1,i),[1 0]);
end
num_hid_nrns = num_hid_nrns + 1;

if do_reactions == true
    if mod(reaction_count,4) == 1
        hidden_neurons = arms_reaction1(hidden_neurons, num_hid_nrns);
    end
    if mod(reaction_count,4) == 2
        hidden_neurons = arms_reaction2(hidden_neurons, num_hid_nrns);
    end
    if mod(reaction_count,4) == 3
        hidden_neurons = arms_reaction3(hidden_neurons, num_hid_nrns);
    end
    if mod(reaction_count,4) == 4
        hidden_neurons = arms_reaction4(hidden_neurons, num_hid_nrns);
    end
    reaction_count = reaction_count + 1;
end

%%Dot-Products to Produce Output Neurons

output_neurons = zeros(1,size(output_vector,2));
for i = 1:size(output_vector,2)
    output_neurons_before_act(1,i) = dot(hidden_neurons(1,:),weight_matrix_23(i,:));
    output_neurons(1,i) = sigmf(output_neurons_before_act(1,i),[1 0]);
end

%%Compute backpropagation error - Output Layer

for i = 1:size(output_vector,2)
    pd_cost_output(1,i) = (output_neurons(1,i)-output_vector(1,i));
    sigma_deriv_output(1,i) = sigmf(output_neurons_before_act(1,i),[1 0])*(1-sigmf(output_neurons_before_act(1,i),[1 0]));
end     
gammaL_output = pd_cost_output.*sigma_deriv_output;
gammaL_output_act = ((hidden_neurons')*gammaL_output)';
  
%%Compute backpropagation error - Hidden layer

for i = 1:num_hid_nrns
    sigma_deriv_hidden(1,i) = sigmf(hidden_neurons_before_act(1,i),[1 0])*(1-sigmf(hidden_neurons_before_act(1,i),[1 0]));
end
gammaL_hidden = (gammaL_output*weight_matrix_23).*sigma_deriv_hidden;
gammaL_hidden_act = ((input_neurons')*gammaL_hidden)';

%%Gradient Descent: Weights - learning rate*gammaL

if momentum ~= 0
velocity_new_hidden = momentum*velocity_new_hidden - learning_rate*gammaL_hidden_act;
velocity_new_output = momentum*velocity_new_output - learning_rate*gammaL_output_act;
weight_matrix_12 = weight_matrix_12 + velocity_new_hidden;
weight_matrix_23 = weight_matrix_23 + velocity_new_output;
end

if momentum == 0
weight_matrix_12 = weight_matrix_12 - learning_rate*gammaL_hidden_act;
weight_matrix_23 = weight_matrix_23 - learning_rate*gammaL_output_act;
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

num_hid_nrns = num_hid_nrns - 1;

%%Dot-Products to Produce Hidden Neurons
hidden_neurons = ones(1,num_hid_nrns+1);
hidden_neurons_before_act = ones(1,num_hid_nrns+1);
for i = 1:num_hid_nrns
    hidden_neurons_before_act(1,i) = dot(input_neurons(1,:),weight_matrix_12(i,:));
    hidden_neurons(1,i) = sigmf(hidden_neurons_before_act(1,i),[1 0]);
end
% hidden_neurons
num_hid_nrns = num_hid_nrns + 1;

%%Dot-Products to Produce Output Neurons

output_neurons = zeros(1,size(output_vector,2));
for i = 1:size(output_vector,2)
    output_neurons_before_act(1,i) = dot(hidden_neurons(1,:),weight_matrix_23(i,:));
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

display(['For Number of Hidden Neurons of ' num2str(num_hid_nrns)]);
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




display('done');