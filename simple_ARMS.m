%%%%%%%%%%%%%%%%%%%Using ARMS for DNA Reservoir Computing%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%August 8 2017%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
clc;

num_trials = 1;
terminate_count_matrix = zeros(num_trials, 1); %Matrix to store terminated states
cycle_length_matrix = zeros(num_trials, 1); %Length of cycles
cycle_count_matrix = zeros(num_trials, 1); %Number of cycles

%%%%Parameters%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

A_initial_state = ['p', 'p', 'p', 'p', 'p', 'p', 'p', 'p', 'p', 'p']; %Creating initial state
num_states = 5005; %Number of states/steps/generations
num_reactions = 6; %Number of possible reactions
num_unique_molecules = 6; %Number of unique molecules (i.e. a, b, c, d, e, p)
num_trials = 100;
input_probability_threshold = 0.1; %Must be: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
% input_probability_threshold = 0.4; %Must be: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
count_outlier_threshold = 1500;
% count_outlier_threshold = 200;
length_outlier_threshold = 100;
rule_order_probability_threshold = 0.1;
input_letter_0 = 'a'; %Must be: a, b, c, d, e, p
input_letter_1 = 'b'; %Must be: a, b, c, d, e, p
concentration_matrix = zeros(1,6); %a, b, c, d, e, p in that order

rule_order = [1 2 3 4 5 6]; %Must be of length num_reactions
% rule_order = [3 1 2 4 5 6]; %Must be of length num_reactions

sorted_molecules = false;
view_alphabets = false;

%%%%

cardinality = length(A_initial_state);
alphabet_matrix = char(zeros(num_states, cardinality));
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


%%%%Begin Main Code%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Starting Main Chemistry\n\n')

for trial = 1:num_trials
   
    A = A_initial_state; %Reset to initial state

for i = 1:num_states %Main iterating loop
    
    if (i ~= 1) %Add input for non-initial states
        if mod(i, 10) == 0 %Add a input_letter_A 
            A = add_input_1(A, 1, input_letter_1);
        end 
        if mod(i, 10) == 5 %Add a input_letter_A 
            A = add_input_0(A, 1, input_letter_0);
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

end %End main loop

%Stores A as final_alphabet as a reference to calculate number and lengths of cycles
final_alphabet = A;

%%Display results

if (isTerminated == true)
%     fprintf('Chemistry has terminated.\n') %Display if calculus has terminated)
%     fprintf('Terminated state: %d\n', terminated_state) %Print out terminated state for my viewing
%     terminate_count_matrix(trial) = terminated_state;
%     cycle_length_matrix(trial) = 0; %0 is default
%     cycle_count_matrix(trial) = 0;
%     fprintf('Final Alphabet: %s\n\n\n', final_alphabet) %Print out final alphabet for my viewing
else %Cycling occurred
%     fprintf('Cycling has occurred. \n')
%     terminate_count_matrix(trial) = 0; %0 is default
%     fprintf('Length of cycles: %d\n', cycle_length) %Print out length of cycles for my viewing
%     cycle_length_matrix(trial) = cycle_length;    
%     fprintf('Number of cycles: %d\n\n\n', cycle_count)  %Print out number of cycles for my viewing
%     cycle_count_matrix(trial) = cycle_count;
%     fprintf('Final Alphabet: %s\n\n\n', final_alphabet) %Print out final alphabet for my viewing
end

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

%Deleted: Counting and Plotting Outliers

%%%%Clean up and Finish%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear cardinality count_indices_outlier_matrix length_indices_outlier_matrix cycle_count cycle_length...
      i isTerminated reaction_decision reaction_decision_temp trial terminated_state;

fprintf('Finished Chemistry.\n')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%