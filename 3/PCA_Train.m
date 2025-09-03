%function [OriFACE, TotalMean,pca_Train_FACE,projectPCA,eigvector,prototypeFACE]=PCA_Train
function [Ori_Train_FACE,Ori_Test_FACE, TotalMean,pca_Train_FACE,pca_Test_FACE,projectPCA,pca_Train_FACE_Max_Min_rescale,pca_Test_FACE_Max_Min_rescale]=PCA_Train

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
    for i=1:1:withinsample*people
        for j=1:1:(row)*(col)
            zeromeanTotalFACE(i,j)=zeromeanTotalFACE(i,j)-TotalMean(j);%正规化
        end
    end
    

    pca_Test_FACE=[];
    pca_Test_FACE=zeromeanTotalFACE*projectPCA;
    pca_Train_FACE_Max_Min_rescale=[];
    pca_Test_FACE_Max_Min_rescale=[];
      %===================================================================
   for n=1:principlenum
        M_max=max(pca_Train_FACE(:,n));
        M_min=min(pca_Train_FACE(:,n));

        b_train=minus(pca_Train_FACE(:,n),M_min);
        c_train=b_train./(M_max-M_min);
        
        pca_Train_FACE_Max_Min_rescale=[pca_Train_FACE_Max_Min_rescale,c_train];
        

        b_test=minus(pca_Test_FACE(:,n),M_min);
        c_test=b_test./(M_max-M_min);
        
        pca_Test_FACE_Max_Min_rescale=[pca_Test_FACE_Max_Min_rescale,c_test];
   end

end %end of function