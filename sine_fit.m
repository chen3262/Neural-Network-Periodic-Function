%% Initialization
clear ; close all; clc

n = 4;
m = 200;
X = zeros(m,n);

x1 = linspace(0,4*pi,m);
for i = 1:m
    %X(i,1) = x1(1,i);
    for j = 1:n
        X(i,j) = mod(x1(i),j/n*2*pi);
    end
end

y = sin(x1')+sin(4*x1');

%% Setup the parameters you will use for this exercise
input_layer_size  = n;
hidden_layer1_size = 5;
hidden_layer2_size = 5;
hidden_layer3_size = 3;
num_labels = 1;
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer1_size);
initial_Theta2 = randInitializeWeights(hidden_layer1_size, hidden_layer2_size);
initial_Theta3 = randInitializeWeights(hidden_layer2_size, hidden_layer3_size);
initial_Theta4 = randInitializeWeights(hidden_layer3_size, num_labels);
Theta1 = initial_Theta1;
Theta2 = initial_Theta2;
Theta3 = initial_Theta3;
Theta4 = initial_Theta4;
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:) ;...
    initial_Theta3(:); ; initial_Theta4(:)];
nn_params = [Theta1(:) ; Theta2(:) ; Theta3(:) ; Theta4(:)];




options = optimset('MaxIter', 2000);
 
%  You should also try different values of lambda
lambda = 0;
 
% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunctionSine(p, ...
                                   input_layer_size, ...
                                   hidden_layer1_size, ...
			                       hidden_layer2_size, ...
                                   hidden_layer3_size, ...
                                   num_labels, X, y, lambda);
 
% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
nframe = 1;
paralist=zeros(size(nn_params,1),nframe);
for i = 1:nframe
    options = optimset('MaxIter', 2000/nframe*i);
    [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
    paralist(:,i) = nn_params
end 

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:numel(Theta1)), ...
                 size(Theta1,1), size(Theta1,2));
 
Theta2 = reshape(nn_params(1+numel(Theta1):numel(Theta1)+numel(Theta2)), ...
                 size(Theta2,1), size(Theta2,2));

Theta3 = reshape(nn_params(1+numel(Theta1)+numel(Theta2):...
         numel(Theta1)+numel(Theta2)+numel(Theta3)), ...
         size(Theta3,1), size(Theta3,2));
             
Theta4 = reshape(nn_params(1+numel(Theta1)+numel(Theta2)+numel(Theta3):end), ...
                 size(Theta4,1), size(Theta4,2));


h1 = sigmoid([ones(m, 1) X] * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2');
h3 = sigmoid([ones(m, 1) h2] * Theta3');
h4 = [ones(m, 1) h3] * Theta4';

% Test Extrapolation Stability
n2 = n;
m2 = 1.5*m;
X2 = zeros(m2,n2);
x2 = linspace(0,10*pi,m2);

for i = 1:m2
    %X2(i,1) = x2(1,i);
    for j = 1:n2
        X2(i,j) = mod(x2(i),j/n*2*pi);
    end
end

yy = sin(x2')+sin(4*x2');

h1 = sigmoid([ones(m2, 1) X2] * Theta1');
h2 = sigmoid([ones(m2, 1) h1] * Theta2');
h3 = sigmoid([ones(m2, 1) h2] * Theta3');
h42 = [ones(m2, 1) h3] * Theta4';

plot(x1,y,'b','LineWidth', 2);
hold on;
plot(x2,yy,'b');
hold on;
plot(x2,h42,'-.r','LineWidth',1);
hold off
legend({'sin(x)+sin(4x)',[ num2str(n) '-5-5-3-1 NN']},'FontSize',20)