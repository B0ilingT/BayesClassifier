classdef my_ClassificationNaiveBayes < handle
    %This class takes in training data and labels and uses them to create a
    %Naive Bayes object capable of generating predictions on inputted Test
    %Data
   
    properties
        % Note: we stick with the Matlab naming conventions from fitcnb      
        X % training examples
        Y % training labels
        ProbScore %stores the probability scores
        NumObservations % the total number of training examples
        ClassNames % each of the class labels in our problem
        Prior % the prior probabilities of each class, based on the training data
        DistributionParameters % the parameters of each Normal distribution (means and standard deviations)
        TestExamples           
        Verbose % are we printing out debug as we go?
    end
    
    methods        
        % constructor: implementing the fitting phase        
        function obj = my_ClassificationNaiveBayes(X, Y, Verbose)      
            obj.X = X;
            obj.Y = Y;
            obj.NumObservations = size(obj.Y, 1);
            obj.Verbose = Verbose;            
            obj.ClassNames = unique(obj.Y);
            obj.DistributionParameters = {};
            obj.Prior = [];  
            obj.ProbScore = [];
            for i = 1:length(obj.ClassNames)
                
                % grab the current class name:
				this_class = obj.ClassNames(i);
                examples_from_this_class = obj.X(obj.Y==this_class,:);
                obj.Prior(end+1) = size(examples_from_this_class,1) / obj.NumObservations;               
                if Verbose
                    fprintf('\nClass %s prior probability = %.2f\n\n', this_class, obj.Prior(end));
                end
                for j = 1:size(obj.X, 2)                   
                    obj.DistributionParameters{i,j} = [mean(examples_from_this_class(:,j)); std(examples_from_this_class(:,j))];                   
                    if Verbose
                        fprintf('Class %s, feature %d: mean=%.2f, standard deviation=%.2f\n', this_class, j, obj.DistributionParameters{i,j}(1), obj.DistributionParameters{i,j}(2));
                    end
                end                                
            end 

            if Verbose
                obj.Prior
                obj.DistributionParameters
            end
            
            
        end
        
        % the prediction phase:
        function [predictions, prob_scores] = predict(obj, test_examples) 
            % get ready to store our predicted class labels:
            obj.TestExamples = test_examples;
            predictions = categorical; 
            prob_scores = zeros(size(test_examples, 1), length(obj.ClassNames));
            if 1 
                % either...
                % write something to the last element you'll use:
                predictions(size(test_examples,1), 1) = obj.ClassNames(1);
                % or call zeros() (if it's a numerical array):
                posterior_ = zeros(1, length(obj.ClassNames));
            end
            for i=1:size(test_examples,1)
                if obj.Verbose
                    fprintf('classifying example example %i/%i: ', i, size(test_examples,1));
                end
                this_test_example = test_examples(i,:);
                for j=1:length(obj.ClassNames)
                    this_likelihood = 1;
                    for k=1:length(this_test_example)
                        this_likelihood = this_likelihood * obj.calculate_pd(this_test_example(k), obj.DistributionParameters{j,k}(1), obj.DistributionParameters{j,k}(2));
                    end                                        
                    this_prior = obj.Prior(j);                   
                    posterior_(j) = this_likelihood * this_prior;          
                end
                prob_scores(i,:) = posterior_/sum(posterior_);
                [~, winning_index] = max(posterior_);
                this_prediction = obj.ClassNames(winning_index);
                if obj.Verbose
                    fprintf('%s\n', this_prediction);
                end                
                predictions(i,1) = this_prediction;                
            end          
            if obj.Verbose
                predictions
            end   
            obj.ProbScore = prob_scores;
        end

        function pd = calculate_pd(obj, x, mu, sigma)
        
			first_bit = 1 / sqrt(2*pi*sigma^2);
            second_bit = - ( ((x-mu)^2) / (2*sigma^2) );
            pd = first_bit * exp(second_bit);
        
		end
        
    end
    
end