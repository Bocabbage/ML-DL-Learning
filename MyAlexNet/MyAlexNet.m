clc;clear all
%%超参数
k=2;n=5;alpha=10^(-4);beta=0.75;   
dropout_rate=0.5;             
A=normrnd(0,10,227,227,3);     %产生一个输入
Knum=[48,128,192,192,128];     %卷积层卷积核种类数（单GPU内）
Ksize=[11,5,3,3,3];            %卷积核大小
Stride=[4,1,1,1,1];            %各卷积层卷积步长 
pool_size=3; pool_stride=2;    
fc_num=[2048 2048 1000];       %全连接层神经元数量（单个GPU内）
%
%% Conv1 and Pool1

Kernal1_1=rand([Ksize(1) Ksize(1) size(A,3) Knum(1)]);
Kernal1_2=rand([Ksize(1) Ksize(1) size(A,3) Knum(1)]);

Conv1_1=Pool_3D(LRN(Conv_3D(A,Kernal1_1,Stride(1)),k,n,alpha,beta),...
        pool_size,pool_stride,'max');
Conv1_2=Pool_3D(LRN(Conv_3D(A,Kernal1_2,Stride(1)),k,n,alpha,beta),...
        pool_size,pool_stride,'max');

%% Conv2 and Pool2

Kernal2_1=rand([Ksize(2) Ksize(2) size(Conv1_1,3) Knum(2)]);
Kernal2_2=rand([Ksize(2) Ksize(2) size(Conv1_2,3) Knum(2)]);

Conv2_1=Pool_3D(LRN(Conv_3D(Conv1_1,Kernal2_1,Stride(2)),k,n,alpha,beta),...
        pool_size,pool_stride,'max');
Conv2_2=Pool_3D(LRN(Conv_3D(Conv1_2,Kernal2_2,Stride(2)),k,n,alpha,beta),...
        pool_size,pool_stride,'max');
%% Conv3
Kernal3_1=rand([Ksize(3) Ksize(3) size(Conv2_1,3)+size(Conv2_2,3) Knum(3)]);
Kernal3_2=rand([Ksize(3) Ksize(3) size(Conv2_1,3)+size(Conv2_2,3) Knum(3)]);

Cat=zeros(size(Conv2_1,1),size(Conv2_1,2),size(Conv2_1,3)+size(Conv2_2,3)); %将上一层输出进行拼接
Cat(:,:,1:size(Conv2_1,3))=Conv2_1;
Cat(:,:,size(Conv2_1,3)+1:end)=Conv2_2;

Conv3_1=LRN(Conv_3D(Cat,Kernal3_1,Stride(3)),k,n,alpha,beta);
Conv3_2=LRN(Conv_3D(Cat,Kernal3_2,Stride(3)),k,n,alpha,beta);

%% Conv4
Kernal4_1=rand([Ksize(4) Ksize(4) size(Conv3_1,3) Knum(4)]);
Kernal4_2=rand([Ksize(4) Ksize(4) size(Conv3_2,3) Knum(4)]);

Conv4_1=LRN(Conv_3D(Conv3_1,Kernal4_1,Stride(4)),k,n,alpha,beta);
Conv4_2=LRN(Conv_3D(Conv3_2,Kernal4_2,Stride(4)),k,n,alpha,beta);  

%% Conv5 and Pool3
Kernal5_1=rand([Ksize(5) Ksize(5) size(Conv4_1,3) Knum(5)]);
Kernal5_2=rand([Ksize(5) Ksize(5) size(Conv4_2,3) Knum(5)]);

Conv5_1=Pool_3D(LRN(Conv_3D(Conv4_1,Kernal5_1,Stride(5)),k,n,alpha,beta),...
        pool_size,pool_stride,'max');
Conv5_2=Pool_3D(LRN(Conv_3D(Conv4_2,Kernal5_2,Stride(5)),k,n,alpha,beta),...
        pool_size,pool_stride,'max');

%% Full Connect

Cat_5=zeros(size(Conv5_1,1),size(Conv5_1,2),...
    size(Conv5_1,3)*2);
Cat_5(:,:,1:size(Conv5_1,3))=Conv5_1;
Cat_5(:,:,size(Conv5_1,3)+1:end)=Conv5_2;


filt_size=[size(Cat_5) fc_num(1)];
Kernal6_1=rand(filt_size);
Kernal6_2=rand(filt_size);

FC6_1=dropout_run(Conv_3D(Cat_5,Kernal6_1,1),dropout_rate);
FC6_2=dropout_run(Conv_3D(Cat_5,Kernal6_2,1),dropout_rate);

Cat_6=[FC6_1;FC6_2];

FC7_1=zeros(fc_num(2),1);
FC7_2=zeros(fc_num(2),1);
Weight7_1=rand([ size(FC7_1,1) size(Cat_6,1)]);
Weight7_2=rand([ size(FC7_2,1 ) size(Cat_6,1)]);
Bias7_1=ones(size(FC7_1));
Bias7_2=ones(fc_num(2),1);

FC7_1=dropout_run(ReLU(Weight7_1*Cat_6+Bias7_1),dropout_rate);
FC7_2=dropout_run(ReLU(Weight7_2*Cat_6+Bias7_2),dropout_rate);

%%

Cat_7=[FC7_1;FC7_2];
FC8=zeros(fc_num(3),1);
Bias8=ones(size(FC8));
Weight8=rand([ size(FC8,1) size(Cat_7,1)]);
FC8=ReLU(Weight8*Cat_7+Bias8);

%%
Softmax=FC8./sum(FC8);