%function [OriFACE, TotalMean,pca_Train_FACE,projectPCA,eigvector,prototypeFACE]=PCA_Train
function [Ori_Train_FACE,Ori_Test_FACE, TotalMean,pca_Train_FACE,pca_Test_FACE,projectPCA,lda_Train_FACE,lda_Test_FACE]=LDA1
 
minidTrain=0;
total=0;
all=200;

people=40;%资料集有40各类别

withinsample=5;%每个类别取五笔来做资料

principlenum=65;%维度降到65维

Ori_Train_FACE=[];%原始影像

    for k=1:1:people
        for m=1:2:10%1 3 5 7 9
            matchstring=['ORL3232' '\' num2str(k) '\' num2str(m) '.bmp'];
            matchX=imread(matchstring);%读字符串
            matchX=double(matchX);%影像格式转成数字模式
            %imshow(maychX)展示图片
            if(k==1 && m==1)%设定长宽
                [row,col]=size(matchX);
            end
        matchtempF=[];
        %arrange the image into a vector
            for n=1:row
                matchtempF=[matchtempF,matchX(n,:)];%1024
            end
        Ori_Train_FACE=[Ori_Train_FACE;matchtempF];%堆叠方式
        end
    end %end of k=1:1:people
    TotalMean=mean(Ori_Train_FACE);%平均值

    zeromeanTotalFACE=Ori_Train_FACE;
    %================================zero mean=======================
    for i=1:1:withinsample*people
        for j=1:1:(row)*(col)
            zeromeanTotalFACE(i,j)=zeromeanTotalFACE(i,j)-TotalMean(j);%正规化
        end
    end
    
    SST=zeromeanTotalFACE' * zeromeanTotalFACE;    %pcaSST=cov(zeromeanTotalFACE);
    [Evec,Eval]=eig(SST); %eig
    Eval=diag(Eval);    %对角线值
    [junk,index] =sort(Eval,'descend');
    PCA=Evec(:,index);
    Eval=Eval(index);
    projectPCA=PCA(:,1:principlenum);% extract the principle component
   

    pca_Train_FACE=[];
    pca_Train_FACE=zeromeanTotalFACE*projectPCA;

    %====================================================================
    Ori_Test_FACE=[];%原始影像

    for k=1:1:people
        for m=2:2:10%1 3 5 7 9
            matchstring=['ORL3232' '\' num2str(k) '\' num2str(m) '.bmp'];
            matchX=imread(matchstring);%读字符串
            matchX=double(matchX);%影像格式转成数字模式
            %imshow(maychX)展示图片
            if(k==1 && m==2)%设定长宽
                [row,col]=size(matchX);
            end
        matchtempF=[];
        %arrange the image into a vector
            for n=1:row
                matchtempF=[matchtempF,matchX(n,:)];%1024
            end
         Ori_Test_FACE=[ Ori_Test_FACE;matchtempF];%堆叠方式
        end
    end %end of k=1:1:people

    zeromeanTotalFACE=Ori_Test_FACE;
 
    %================================zero mean=======================
    for i=1:1:(withinsample*people)
        for j=1:1:(row)*(col)
            zeromeanTotalFACE(i,j)=zeromeanTotalFACE(i,j)-TotalMean(j);%正规化
        end
    end
    

    pca_Test_FACE=[];
    pca_Test_FACE=zeromeanTotalFACE*projectPCA;
    
%=========================LDA投影=========================
% 计算类内散度矩阵SW
for n = 1:withinsample:withinsample*people
    WithFACE = pca_Train_FACE(n:n+withinsample-1, :);

    if (n == 1)
        MeanFace = mean(WithFACE);
        classMEAN = MeanFace;
        WithFACE = WithFACE - MeanFace;
        SW = WithFACE' * WithFACE; % SW=cov(withinFACE) 
    end

    if (n > 1)
        MeanFace = mean(WithFACE);
        classMEAN = [classMEAN; MeanFace];
        WithFACE = WithFACE - MeanFace;
        SW = SW + WithFACE' * WithFACE; % SW=SW+cov(withinFACE);
    end
end

pca_Total_mean = mean(pca_Train_FACE);
classMEAN = classMEAN - pca_Total_mean;
SB = classMEAN' * classMEAN; % SB=cov(classMean) ;

[eigvector, eigvalue] = eig(inv(SW) * SB); % LDA特征向量和特征值
eigvalue = diag(eigvalue);

[~, index] = sort(eigvalue, 'descend');
eigvalue = eigvalue(index);
eigvector = eigvector(:, index);

projectLDA = eigvector;

lda_Train_FACE = pca_Train_FACE * projectLDA(:, 1:1:30); % LDA投影到低维空间
lda_Test_FACE = pca_Test_FACE * projectLDA(:, 1:1:30);

target = repelem(1:people, 5)';
DIST = zeros(all, all);
mind = zeros(all, 1);

% 计算正确率
for i = 1:all
    mindistance = Inf;
    minidTrain = -1;
    
    for j = 1:all
        distance = sqrt(sum((lda_Train_FACE(j, :) - lda_Test_FACE(i, :)).^2));
        if distance < mindistance
            mindistance = distance;
            minidTrain = j;
        end
    end
    
    if target(i) == target(minidTrain)
        total = total + 1;
    end
end

    accuracy = (total / all) * 100;
    fprintf('totalnumber:%g\n',total);
    fprintf('CorrectRate: %.2f%%\n\n',accuracy );
end


   

