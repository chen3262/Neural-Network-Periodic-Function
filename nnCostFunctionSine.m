function [J grad] = nnCostFunctionSine(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer1_size, ...
                                   hidden_layer2_size, ...
                                   hidden_layer3_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
%Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
%                 hidden_layer_size, (input_layer_size + 1));
%
%Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
%                 num_labels, (hidden_layer_size + 1));
Theta1 = zeros(hidden_layer1_size,input_layer_size+1);
Theta2 = zeros(hidden_layer2_size,hidden_layer1_size+1);
Theta3 = zeros(hidden_layer3_size,hidden_layer2_size+1);
Theta4 = zeros(num_labels,hidden_layer3_size+1);

Theta1 = reshape(nn_params(1:numel(Theta1)), ...
                 size(Theta1,1), size(Theta1,2));
 
Theta2 = reshape(nn_params(1+numel(Theta1):numel(Theta1)+numel(Theta2)), ...
                 size(Theta2,1), size(Theta2,2));
             
Theta3 = reshape(nn_params(1+numel(Theta1)+numel(Theta2):...
         numel(Theta1)+numel(Theta2)+numel(Theta3)), ...
         size(Theta3,1), size(Theta3,2));
     
Theta4 = reshape(nn_params(1+numel(Theta1)+numel(Theta2)+numel(Theta3):end), ...
                 size(Theta4,1), size(Theta4,2));

%Theta3 = reshape(nn_params(1+numel(Theta1)+numel(Theta2):end), ...
%                 size(Theta3,1), size(Theta3,2));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
Theta3_grad = zeros(size(Theta3));
Theta4_grad = zeros(size(Theta4));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% Theta1 100x2
% Theta2 100x101
% Theta3 1*101
X = [ones(m, 1) X]; %1000x2
z2 = Theta1*X'; %100*1000
a2 = sigmoid(z2); %100*1000
a2 = [ones(1,m); a2]; %101*1000
z3 = Theta2*a2; %100*1000
a3 = sigmoid(z3); %100*1000
a3 = [ones(1,m); a3]; %101*1000
z4 = Theta3*a3; %1*1000
a4 = sigmoid(z4);
a4 = [ones(1,m); a4];
z5 = Theta4*a4;
h = z5;

J = (h-y')*(h'-y)/(2*m); 

 
% regularization
A = sum(sum(Theta1(:,2:size(Theta1,2)) .* Theta1(:,2:size(Theta1,2))));
B = sum(sum(Theta2(:,2:size(Theta2,2)) .* Theta2(:,2:size(Theta2,2))));
C = sum(sum(Theta3(:,2:size(Theta3,2)) .* Theta3(:,2:size(Theta3,2))));
D = sum(sum(Theta4(:,2:size(Theta4,2)) .* Theta4(:,2:size(Theta4,2))));
J = J + lambda/(2*m)*(A+B+C+D);
 
% compute gradients using backpropagation
 
delta_5 = h - y'; %1*1000
delta_4 = Theta4'*delta_5 .* sigmoidGradient([ones(1, m); z4]);
          %101*1  1*1000         101*1000
delta_3 = Theta3'*delta_4(2:end,:) .* sigmoidGradient([ones(1, m); z3]);
          %101*100  100*1000         %101*1000
delta_2 = Theta2'*delta_3(2:end,:) .* sigmoidGradient([ones(1, m); z2]);

Theta4_grad = 1/m * delta_5 * a4'; %1*101        
Theta3_grad = 1/m * delta_4(2:end,:) * a3'; % 100*101
Theta2_grad = 1/m * delta_3(2:end,:) * a2'; %100*2
Theta1_grad = 1/m * delta_2(2:end,:) * X;

% regularization

Theta4_grad = Theta4_grad + lambda/m*[zeros(size(Theta4,1),1) Theta4(:,2:end)];
Theta3_grad = Theta3_grad + lambda/m*[zeros(size(Theta3,1),1) Theta3(:,2:end)];
Theta2_grad = Theta2_grad + lambda/m*[zeros(size(Theta2,1),1) Theta2(:,2:end)];
Theta1_grad = Theta1_grad + lambda/m*[zeros(size(Theta1,1),1) Theta1(:,2:end)];


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:) ; Theta3_grad(:) ; Theta4_grad(:)];


end
