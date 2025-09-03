function EBPTA2

load IRIS_IN.csv;
load IRIS_OUT.csv;
input = IRIS_IN;
target = zeros(100,3);
%设置目标值
for i=1:1:100
    if (IRIS_OUT(i)==1)
        target(i,1)=1;
    elseif (IRIS_OUT(i)==2)
        target(i,2)=1;
    else
        target(i,3)=1;
    end
end

% initialize the weight matrix
outputmatrix=zeros(12,3);
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

RMSE = zeros(100, 3);

% Training
for epoch = 1:1:100
    t = [];
    for iter = 1:1:75
        % Forward propagation
        SUMhid = input(iter, :) * hiddenmatrix;
        Ahid = logsig(SUMhid);
        SUMout = Ahid * outputmatrix;
        Aout = softmax_YN(SUMout);
        DELTA_dtransfer=[0,0,0];
        %计算误差
        for k = 1:3
            if (target(iter,k)==1)
                 DELTA_dtransfer(k) = Aout(k)-1;
            else
                 DELTA_dtransfer(k) = Aout(k);
            end
        end
        % Backpropagation
      
        error = target(iter,:) - Aout;
        t = [t; error.^2];
        DELAhid = DELTA_dtransfer * outputmatrix' * 1;


        % Output layer's weight update
        for i=1:1:3
            outputmatrix(:,i) = outputmatrix(:,i) - 0.45 * DELTA_dtransfer(i) * Ahid';
        end
        % Hidden weight update
        for j=1:1:12
            hiddenmatrix(:,j) = hiddenmatrix (:,j)- 0.45 * DELAhid(j) * dlogsig(Ahid(j),logsig(SUMhid(j))) * input(iter, :)';
        end
    end
end

%======================================================
fprintf('\nTotal number of epochs: %g\n', epoch);
plot(1:epoch, RMSE(1:epoch));

Tot_Correct = 0;

for i = 76:length(input)
    SUMhid = input(i, :) * hiddenmatrix;
    Ahid = sigmoid(SUMhid);
    SUMout = Ahid * outputmatrix;
    Aout = softmax_YN(SUMout);
    [~, max_index] = max(Aout);
    if (max_index==IRIS_OUT(i))
        Tot_Correct = Tot_Correct + 1;
    end
end

Tot_Percent= (Tot_Correct) / (length(input)-75);
Test_correct_percent=Tot_Percent;
fprintf('Test correct percent: %.2f%%\n', Test_correct_percent * 100);
end

