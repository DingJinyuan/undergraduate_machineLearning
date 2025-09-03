function yuan

load IRIS_IN.csv;
load IRIS_OUT.csv;
input = IRIS_IN;

NumNeuron = 12;
target = zeros(150,3);

for i = 1:1:150
    if (IRIS_OUT(i) == 1)
        target(i, 1) = 1;
    elseif (IRIS_OUT(i) == 2)
        target(i, 2) = 1;
    else 
        target(i, 3) = 1;
    end
end

% initialize the weight matrix
outputmatrix=zeros(12,1);
for i=1:1:12
 for j=1:1:3
   outputmatrix(i,j)=rand;
 end
end

hiddenmatrix=zeros(4,12);
for i=1:1:4
 for j=1:1:12
   hiddenmatrix(i,j)=rand;
 end
end
% Training
for epoch = 1:1:100
    t = [];
    
    for iter = 1:1:75
        % Forward propagation
        SUMhid = input(iter, :) * hiddenmatrix;
        Ahid = logsig(SUMhid);
        
        SUMout = Ahid * outputmatrix;
        Aout = logsig(SUMout); 
        
        % Backpropagation
        DELTAout = target(iter, :) - Aout;
        error = target(iter, :) - Aout;
        t = [t; sum(error.^2)];
        
        DELTAhid = DELTAout.*dlogsig(Aout,logsig(SUMout))*outputmatrix';
        
        % Output weight (matrix) update
        for j =1:1:3
            outputmatrix(:, j) = outputmatrix(:, j) + 0.45  * DELTAout(j)*dlogsig(Aout(j),logsig(SUMout(j)))* Ahid';
        end
        % Hidden weight (matrix) update
        for i = 1:1:12
                hiddenmatrix(:, i) = hiddenmatrix(:, i) + 0.45 * DELTAhid(i)*dlogsig(Ahid(i),logsig(SUMhid(i)))* input(iter, :)';
            
        end
    end
    
    RMSE(epoch) = sqrt(sum(t) / 75);
    fprintf('Epoch %.0f: RMSE = %.3f\n', epoch, RMSE(epoch));
end

disp('--------------------------')

Tot_Correct = 0;

for i = 76:150
    SUMhid = input(i,:) * hiddenmatrix;
    Ahid = logsig(SUMhid);

    SUMout = Ahid * outputmatrix;
    Aout = purelin(SUMout);
    
    [~, max_index] = max(Aout);
    if (max_index == IRIS_OUT(i))
        Tot_Correct = Tot_Correct + 1;
    end
end

Test_Correct = Tot_Correct;
Tot_Percent = Tot_Correct / (length(input) - 75);
Test_correct_percent = Tot_Percent;

disp(['Number of Correct Predictions: ' num2str(Test_Correct)]);
disp(['Classification Accuracy: ' num2str(Test_correct_percent * 100) '%']);