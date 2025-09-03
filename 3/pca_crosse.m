function pca_crosse( pca_Train_FACE_Max_Min_rescale,pca_Test_FACE_Max_Min_rescale)

train_input = pca_Train_FACE_Max_Min_rescale;
test_input = pca_Test_FACE_Max_Min_rescale;

num_of_train = 200;
num_of_test = 200;
orimatrix = (1:num_of_train)';
numofclass = 40;
group_size = 5;
input_dimension = 65;
num_of_neurons = 110;
lr=0.01;
epochs = 100;
target = zeros(size(orimatrix, 1), numofclass);


% 初始化权重矩阵
outputmatrix = rand(num_of_neurons, numofclass) * 2 - 1;
hiddenmatrix = rand(input_dimension, num_of_neurons) * 2 - 1;

for i = 1:numofclass
    staindex = (i - 1) * group_size + 1;
    endindex = i * group_size;
    target(staindex:endindex, i) = 1;
end


%qianchuan
for epoch=1:1:epochs
    for iter = 1:1:num_of_train
        hiddensigma = train_input(iter, :) * hiddenmatrix;
        hiddennet = logsig(hiddensigma);
        outputsigma = hiddennet * outputmatrix;
        outputnet = softmax_YN(outputsigma);
        
        deltasoftmax = outputnet - target(iter, :);
        DELTAhid = deltasoftmax * outputmatrix';
        
        % 更新输出层权重
        outputmatrix = outputmatrix - lr * hiddennet' * deltasoftmax;
         % 更新隐藏层权重
         for i=1:1:num_of_neurons
            hiddenmatrix(:, i) = hiddenmatrix(:, i) - lr * DELTAhid(i) * dlogsig(hiddennet(i), logsig(hiddensigma(i))) * train_input(iter, :)';
         end
    end
end
Tot_Correct = 0;
%=============================================
for iter = 1:1:num_of_test 
    %qianchuan
    hiddensigma = test_input(iter, :) * hiddenmatrix ;
    hiddennet = sigmoid(hiddensigma);
    outputsigma = hiddennet * outputmatrix;
    outputnet = softmax_YN(outputsigma);

    [~, predicted_class] = max(outputnet);
    [~, true_class] = max(target(iter, :));
    if predicted_class == true_class
        Tot_Correct = Tot_Correct + 1;
    end
end
Test_correct_percent = Tot_Correct / num_of_test;
disp(['Test_correct_percent: ' num2str(Test_correct_percent)]);





