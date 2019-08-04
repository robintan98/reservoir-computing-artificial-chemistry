%%%%%%%%%%%%%%Combining Simple ARMS and Single Perceptron%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%NAND FUNCTION%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%August 14 2017%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
clc;

%%NAND FUNCTION

%%%Transcendence Prep%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('Beginning Main Simulation: NAND FUNCTION.')

tic

load('mol_coms_2.mat');
%Master Matrix: Stores percentage correct of all 30 different combinations
%in a single matrix that corresponds with the letter combinations in
%'mol_coms'
master_matrix = zeros(size(mol_coms,1),1); 

for c = 1:size(mol_coms,1) %Master loop

fprintf('\n\n');
disp(['Input 0: ' mol_coms(c,1) '.']);
disp(['Input 1: ' mol_coms(c,2) '.']);
fprintf('\n');
    
clearvars -except master_matrix mol_coms c
%%%%ARMS Parameters%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

A_initial_state = ['p', 'p', 'p', 'p', 'p', 'p', 'p', 'p', 'p', 'p']; %Creating initial state
num_ARMS_states = 5005; %Number of states/steps/generations
num_reactions = 6; %Number of possible reactions
num_unique_molecules = 6; %Number of unique molecules (i.e. a, b, c, d, e, p)
num_ARMS_trials = 100;
input_probability_threshold = 0.1; %Must be: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
rule_order_probability_threshold = 0.1;
input_letter_0 = mol_coms(c,1); %Must be: a, b, c, d, e, p %Will be altered later
input_letter_1 = mol_coms(c,2); %Must be: a, b, c, d, e, p %Will be altered later
rule_order = [1 2 3 4 5 6]; %Must be of length num_reactions
frequency_of_input = 5;

sorted_molecules = false;
view_alphabets = false;

%%%%

cardinality = length(A_initial_state);
alphabet_matrix = char(zeros(num_ARMS_states, cardinality));
concentration_matrix = zeros(1,6); %a, b, c, d, e, p in that order
terminated_state = 0;
number_of_terminated = 0;
new_rule_order_temp = zeros(5,1);
isTerminated = false;

a_count = 0;
b_count = 0;
c_count = 0;
d_count = 0;
e_count = 0;
p_count = 0;

%%%%Perceptron Parameters%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

epochs = 1000;
num_00 = 250;
num_01 = 250;
num_11 = 250;
num_10 = 250;
num_0_1_total = num_00 + num_01 + num_11 + num_10;
percent_train = 0.7;
learning_rate = 0.8;
momentum = 0.3; %0 if no momentum

%%%%
concentration_total_matrix_pre = zeros(num_0_1_total,num_unique_molecules);
result_operation_matrix_pre = zeros(num_0_1_total,1);

%For perceptron making purposes (NAND) (result):
for i = 1:num_00
    result_operation_matrix_pre(i,1) = 1;
end
for i = num_00 + 1: num_00 + num_01
    result_operation_matrix_pre(i,1) = 1;
end
for i = num_00 + num_01 + 1:num_00 + num_01 + num_11
    result_operation_matrix_pre(i,1) = 0;
end
for i = num_00 + num_01 + num_11 + 1:num_0_1_total
    result_operation_matrix_pre(i,1) = 1;
end

%%%%Simple ARMS%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Starting Main Chemistry\n\n')

for t = 1:num_0_1_total %Loop for each of four inputs for function

for trial = 1:num_ARMS_trials
   
    A = A_initial_state; %Reset to initial state

for i = 1:num_ARMS_states %Main iterating loop
    
    if t >= 1 && t <= num_00                                           %%00
        if (i ~= 1) %Add input for non-initial states
            if mod(i, 2*frequency_of_input) == 0 %Add a input_letter_A 
                A = add_input_0(A, 1, input_letter_0);
            end 
            if mod(i, 2*frequency_of_input) == frequency_of_input %Add a input_letter_A 
                A = add_input_0(A, 1, input_letter_0);
            end 
        end
    elseif t >= num_00 + 1 && t <= num_00 + num_01                     %%01
        if (i ~= 1) %Add input for non-initial states
            if mod(i, 2*frequency_of_input) == 0 %Add a input_letter_A 
                A = add_input_0(A, 1, input_letter_0);
            end 
            if mod(i, 2*frequency_of_input) == frequency_of_input %Add a input_letter_A 
                A = add_input_1(A, 1, input_letter_1);
            end 
        end
    elseif t >= num_00 + num_01 + 1 && t <= num_00 + num_01 + num_11   %%11
        if (i ~= 1) %Add input for non-initial states
            if mod(i, 2*frequency_of_input) == 0 %Add a input_letter_A 
                A = add_input_1(A, 1, input_letter_1);
            end 
            if mod(i, 2*frequency_of_input) == frequency_of_input %Add a input_letter_A 
                A = add_input_1(A, 1, input_letter_1);
            end 
        end
    else                                                               %%10
        if (i ~= 1) %Add input for non-initial states
            if mod(i, 2*frequency_of_input) == 0 %Add a input_letter_A 
                A = add_input_1(A, 1, input_letter_1);
            end 
            if mod(i, 2*frequency_of_input) == frequency_of_input %Add a input_letter_A 
                A = add_input_0(A, 1, input_letter_0);
            end 
        end
    end
    
    reaction_decision_temp = mod(i, num_reactions); %Determine which reaction to use based on log6
    if (reaction_decision_temp == 0)
        reaction_decision_temp = num_reactions; %If reaction_decision_temp is 0 due to modulus, then sets it to the maximum number of reactions
    end
    reaction_decision = rule_order(reaction_decision_temp);
    
    if (rand(1) >= (1 - rule_order_probability_threshold))
        new_rule_order_temp = rule_order;
        new_rule_order_temp(new_rule_order_temp == reaction_decision) = [];
        reaction_idx_temp = randi([1 length(new_rule_order_temp)]);
        reaction_decision = new_rule_order_temp(reaction_idx_temp);
    end
    
    if (view_alphabets == true)
        if (reaction_decision == 1)
            A = reaction1(A) %Choose Reaction 1
        elseif (reaction_decision == 2)
            A = reaction2(A) %Choose Reaction 2
        elseif (reaction_decision == 3)
            A = reaction3(A) %Choose Reaction 3
        elseif (reaction_decision == 4)
            A = reaction4(A) %Choose Reaction 4
        elseif (reaction_decision == 5)
            A = reaction5(A) %Choose Reaction 5
        else
            A = reaction6(A) %Choose Reaction 6
        end
    end
    if (view_alphabets == false)
        if (reaction_decision == 1)
            A = reaction1(A); %Choose Reaction 1
        elseif (reaction_decision == 2)
            A = reaction2(A); %Choose Reaction 2
        elseif (reaction_decision == 3)
            A = reaction3(A); %Choose Reaction 3
        elseif (reaction_decision == 4)
            A = reaction4(A); %Choose Reaction 4
        elseif (reaction_decision == 5)
            A = reaction5(A); %Choose Reaction 5
        else
            A = reaction6(A); %Choose Reaction 6
        end
    end
    
    if (sorted_molecules == true)
        A = sort_molecules(A);
    end
    
    alphabet_matrix(i,:) = A; %Stores alphabet at state 'i' in alphabet matrix
    
    isTerminated = check_if_terminated(A);
    if (isTerminated == true)
        terminated_state = i;
        break;
    end

end %End main ARMS loop (i)

alphabet_matrix = char(zeros(num_ARMS_states, cardinality));

end %End trial ARMS loop (trial)

%Stores A as final_alphabet as a reference to calculate number and lengths of cycles
final_alphabet = A;

%%Reset local variables
terminated_state = 0;
isTerminated = false;

for i = 1:cardinality
    if (final_alphabet(i) == 'a')
        a_count = a_count + 1;
    elseif (final_alphabet(i) == 'b')
        b_count = b_count + 1;
    elseif (final_alphabet(i) == 'c')
        c_count = c_count + 1;
    elseif (final_alphabet(i) == 'd')
        d_count = d_count + 1;
    elseif (final_alphabet(i) == 'e')
        e_count = e_count + 1;
    else
        p_count = p_count + 1;
    end
end

concentration_total = a_count + b_count + c_count + d_count + e_count + p_count;
a_concentration = a_count / concentration_total;
b_concentration = b_count / concentration_total;
c_concentration = c_count / concentration_total;
d_concentration = d_count / concentration_total;
e_concentration = e_count / concentration_total;
p_concentration = p_count / concentration_total;

concentration_matrix(1) = a_concentration;
concentration_matrix(2) = b_concentration;
concentration_matrix(3) = c_concentration;
concentration_matrix(4) = d_concentration;
concentration_matrix(5) = e_concentration;
concentration_matrix(6) = p_concentration;

%%Reset concentration counts
a_count = 0;
b_count = 0;
c_count = 0;
d_count = 0;
e_count = 0;
p_count = 0;

%%Add concentration matrix to concentration_total_matrix_pre

concentration_total_matrix_pre(t,:) = concentration_matrix;

display(['Input State ' num2str(t) ' done']);

end %%End input ARMS Loop (t)

%%%%Clean up and Finish%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear cardinality count_indices_outlier_matrix length_indices_outlier_matrix cycle_count cycle_length...
      i isTerminated reaction_decision reaction_decision_temp trial terminated_state;

fprintf('Finished Chemistry.\n\n')





%%%%Single Perceptron%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('Starting Main Perceptron\n\n')

%%Data Generation

num_data_train  = round(num_0_1_total * percent_train);
num_data_test = num_0_1_total - num_data_train;

rand_perm = randperm(num_0_1_total)';

concentration_total_matrix = concentration_total_matrix_pre(rand_perm,:);
result_operation_matrix = result_operation_matrix_pre(rand_perm,:);

%Altered "data_matrix_actual" to "test_concentration_matrix"
data_train = concentration_total_matrix(1:num_data_train,:); %%Splits xo matrix to training set
data_test = concentration_total_matrix((num_data_train + 1):(num_data_train + num_data_test),:); %%Splits xo matrix to testing set

result_matrix_actual_train = result_operation_matrix(1:num_data_train,:); %%Produces results for test
result_matrix_actual_test = result_operation_matrix((num_data_train + 1):(num_data_train + num_data_test),:); %%Produces results for test

%%%%%%%%%%%%%%%%FIRST TIME: RANDOMIZED WEIGHT MATRIX%%%%%%%%%%%%%%%%%%%%%%%

% tic

%%Generate Matrix of Input Neurons

data_each_shape = data_train(1,:);

input_neurons = zeros(1,size(data_each_shape,2));
for i = 1:size(data_each_shape,2) 
    input_neurons(i) = data_each_shape(i);
end

output_vector = result_matrix_actual_train(1,:);

%%Generate Matrix of Weights
rng('shuffle')
weightmin = -0.5;
weightmax = 0.5;
weight_matrix_12 = weightmin + rand(1,size(data_train,2))*(weightmax - weightmin);

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

%%Gradient Descent: Weights - learning rate*gammaL

if momentum ~= 0
velocity_new_output = - learning_rate*gammaL_output_act;
weight_matrix_12 = weight_matrix_12 + velocity_new_output;
end

if momentum == 0
weight_matrix_12 = weight_matrix_12 - learning_rate*gammaL_output_act;
end

% toc;
display('Epoch 1 done');

%%%%%%%%%%%%%%%%%%%%%%%SECOND TIME: TRAINED WEIGHT MATRIX%%%%%%%%%%%%%%%%%%

%%Define Loop for each Data Row

for n = 2:epochs
    
% tic

for m = 2:num_data_train
    
%%Generate Matrix of Input Neurons/Letter Matrix

data_each_shape = data_train(m,:);

input_neurons = zeros(1,size(data_each_shape,2));
for i = 1:size(data_each_shape,2) 
    input_neurons(i) = data_each_shape(i);
end

output_vector = result_matrix_actual_train(m,:);

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

%%Gradient Descent: Weights - learning rate*gammaL

if momentum ~= 0
velocity_new_output = - learning_rate*gammaL_output_act;
weight_matrix_12 = weight_matrix_12 + velocity_new_output;
end

if momentum == 0
weight_matrix_12 = weight_matrix_12 - learning_rate*gammaL_output_act;
end

end

% toc;
display(['Epoch ' num2str(n) ' done']);

end

%%%%%%%%%%%%%%%%%%%%%TESTING PHASE%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

maximum_orientation = zeros(1,num_data_test);

for m = num_data_train + 1: num_data_train + num_data_test

%%Generate Matrix of Input Neurons/Letter Matrix

data_each_shape = data_test(m-num_data_train,:);

input_neurons = zeros(1,size(data_each_shape,2));
for i = 1:size(data_each_shape,2) 
    input_neurons(i) = data_each_shape(i);
end

output_neurons = zeros(1,size(output_vector,2));
for i = 1:size(output_vector,2)
    output_neurons_before_act(1,i) = dot(input_neurons(1,:),weight_matrix_12(i,:));
    output_neurons(1,i) = sigmf(output_neurons_before_act(1,i),[1 0])
end

% output_neurons
maximum_orientation(m-num_data_train) = round(output_neurons);

end


%%%%%%%%%%%%%%%%%%%%%%TOTAL ERRORS%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


wrong_matrix = abs(maximum_orientation' - result_matrix_actual_test);

total_wrong = sum(wrong_matrix);
total_correct = num_data_test - total_wrong;
predicted_correct_over_total_correct = total_correct/num_data_test*100;

fprintf('\n');
display(['Input 0: ' input_letter_0 '.']); %Display input 0


display(['Input 1: ' input_letter_1 '.']); %Display input 1
fprintf('\n');

display(['Percent of Correct Predictions: ' num2str(predicted_correct_over_total_correct) '%']);

master_matrix(c) = predicted_correct_over_total_correct;

%%%%Clean up and Finish%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear gammaL_ouptput gammaL_output_act i num_00 num_01 num_10 num_11 output_vector pd_cost_output sigma_deriv_output;

fprintf('Finished Perceptron.\n')

end

toc
fprintf('\n');

fprintf('Finished Main Simulation: NAND FUNCTION.\n')