%%%%%%%%%%%%%Exploring Chemical Rewriting System on Multisets%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%Part 2: Add more inputs at one time%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%June 27 2017%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
clc;

num_trials = 100;
terminate_count_matrix = zeros(num_trials, 1); %Matrix to store terminated states
cycle_length_matrix = zeros(num_trials, 1); %Length of cycles
cycle_count_matrix = zeros(num_trials, 1); %Number of cycles

%%%%Parameters%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

A_initial_state = ['p', 'p', 'p', 'p', 'p', 'p', 'p', 'p', 'p', 'p']; %Creating initial state
num_states = 5005; %Number of states/steps/generations
num_reactions = 6; %Number of possible reactions
num_unique_molecules = 6; %Number of unique molecules (i.e. a, b, c, d, e, p)
input_probability_threshold = 0.1; %Must be: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
% input_probability_threshold = 0.4; %Must be: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
count_outlier_threshold = 1500;
% count_outlier_threshold = 200;
length_outlier_threshold = 100;
rule_order_probability_threshold = 0.1;
num_of_repeat_input = 1;

test = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.26];

rule_order = [1 2 3 4 5 6]; %Must be of length num_reactions
% rule_order = [3 1 2 4 5 6]; %Must be of length num_reactions

sorted_molecules = false;
view_alphabets = false;

%%%%

cardinality = length(A_initial_state);
alphabet_matrix = char(zeros(num_states, cardinality));
cycle_length = 0;
cycle_count = 0;
terminated_state = 0;
number_of_terminated = 0;
new_rule_order_temp = zeros(5,1);
% desired_length_input_decision_matrix = input_probability_threshold * 10;
% Old^
% input_decision_matrix = []; %Old
isTerminated = false;

%%%%Begin Main Code%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Starting Main Chemistry\n\n')



for trial = 1:num_trials %%Begin Trial Loop
    
    A = A_initial_state; %Reset to initial state
    alphabet_matrix = char(zeros(num_states, cardinality)); %Reset alphabet_matrix
    
    %Old:
%     input_decision_matrix =
%     generate_input_decision_matrix(desired_length_input_decision_matrix);

for i = 1:num_states %Main iterating loop
    
    %Old:   
%     input_turn = mod(i, 10); %Determine which step of 10 this alphabet is part of based on log10; 10 is default (apparently)
%     if (input_turn == 0)
%         input_turn = 10; %If reaction_decision_temp is 0 due to modulus, then sets it to 10
%     end

    for j = 1:num_of_repeat_input
    
    if (i ~= 1)
        A = add_input(A, 0.1); %Add input for non-initial states
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
        
    %Old:
%     if (input_turn == 10)
%         input_decision_matrix = generate_input_decision_matrix(desired_length_input_decision_matrix); %Re-randomize input decision matrix
%     end
        
end %End main loop

%Stores A as final_alphabet as a reference to calculate number and lengths of cycles
final_alphabet = A;

%Calculate average length of cycles
cycle_length = count_length_cycles(final_alphabet, alphabet_matrix, num_unique_molecules);

%Calculate number of cycles
cycle_count = count_number_cycles(final_alphabet, alphabet_matrix, num_unique_molecules);

%%Display results

fprintf('For trial: %d\n', trial)

if (isTerminated == true)
%     fprintf('Chemistry has terminated.\n') %Display if calculus has terminated)
%     fprintf('Terminated state: %d\n\n\n', terminated_state) %Print out terminated state for my viewing
    terminate_count_matrix(trial) = terminated_state;
    cycle_length_matrix(trial) = 0; %0 is default
    cycle_count_matrix(trial) = 0;
else %Cycling occurred
%     fprintf('Cycling has occurred. \n')
    terminate_count_matrix(trial) = 0; %0 is default
%     fprintf('Length of cycles: %d\n', cycle_length) %Print out length of cycles for my viewing
    cycle_length_matrix(trial) = cycle_length;    
%     fprintf('Number of cycles: %d\n\n\n', cycle_count)  %Print out number of cycles for my viewing
    cycle_count_matrix(trial) = cycle_count;
end

%%Reset local variables
cycle_length = 0;
cycle_count = 0;
terminated_state = 0;
isTerminated = false;

end %End trial loop


%Calculate number of terminations
terminate_count_matrix(any(terminate_count_matrix==0,2),:) = [];
number_of_terminated = numel(terminate_count_matrix);

fprintf('Percent of trials resulting in terminations: %d out of %d trials, or %.2f%%.\n\n', number_of_terminated, num_trials, number_of_terminated/num_trials*100);

%%%%Plot Results%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

length_indices_outlier_matrix = NaN(length(cycle_length_matrix),1);
count_indices_outlier_matrix = NaN(length(cycle_count_matrix),1);

%%Get rid of outliers
for i = 1:length(cycle_length_matrix) %Cycle length matrix outliers
    if( (cycle_length_matrix(i,:) == 0) || (cycle_length_matrix(i,:) > length_outlier_threshold))
        length_indices_outlier_matrix(i) = i;
    end
end

for i = 1:length(cycle_count_matrix) %Cycle length matrix outliers
    if( (cycle_count_matrix(i,:) == 0) || (cycle_count_matrix(i,:) > count_outlier_threshold))
        count_indices_outlier_matrix(i) = i;
    end
end

length_indices_outlier_matrix(isnan(length_indices_outlier_matrix)) = [];
count_indices_outlier_matrix(isnan(count_indices_outlier_matrix)) = [];
total_indices_outlier_matrix = [length_indices_outlier_matrix; ...
                                    count_indices_outlier_matrix];
total_indices_outlier_matrix = unique(total_indices_outlier_matrix);

cycle_length_matrix(total_indices_outlier_matrix) = [];
cycle_count_matrix(total_indices_outlier_matrix) = [];

%%Plot Number of Cycles over Length of Cycles
figure %New figure
scatter(cycle_length_matrix, cycle_count_matrix, '.'); %Create scatter plot of cycle count over length
for i = 1:size(cycle_length_matrix,1)
    hold on
    line([cycle_length_matrix(i) cycle_length_matrix(i)], [0 cycle_count_matrix(i)],'LineWidth',2); %Make vertical lines for viewing purpose
end
title_string = sprintf('Frequency of Cycles vs. Number of Cycles for Input Probability %.1f', 0.1); %Format title of plot
title(title_string); 
xlabel('Length of Cycles') % x-axis label
ylabel('Number of Cycles') % y-axis label
set(gca,'XLim',[0 length_outlier_threshold],'YLim',[0 count_outlier_threshold]); %Set x and y axis limits for plot
hold off

fprintf('Average length of cycles: %f', mean(cycle_length_matrix));


%%%%Clean up and Finish%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear cardinality count_indices_outlier_matrix length_indices_outlier_matrix cycle_count cycle_length...
      i isTerminated reaction_decision reaction_decision_temp trial terminated_state;

fprintf('Finished Chemistry.\n')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%