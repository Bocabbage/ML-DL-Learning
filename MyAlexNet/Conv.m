function [ Result ] = Conv(A,B,stride)

size_A=size(A);
size_B=size(B);
if (size(A)==size(B))
    Result=sum(sum(sum(A.*B)));
    return
end
        
%%   检验
if(size_A(3)~= size_B(3))
    error('厚度不等');
end

if(sum(size_A<size_B))
    error('卷积核太大');
end

if(stride>1)
    re=zeros(2,1);
    re(1)=rem(size_A(1)-size_B(1),stride);
    re(2)=rem(size_A(2)-size_B(2),stride);
    pad=[rem(stride-re(1),stride) rem(stride-re(2),stride)];
%% Padding（步长为1作为特殊情况处理）

    pa_A=zeros(size_A(1)+pad(1),...
        size_A(2)+pad(2),...
        size_A(3));
    a=ceil(pad(1)/2);
    b=ceil(pad(2)/2);
    pa_A(1+a:size_A(1)+a,1+b:size_A(2)+b,:)=A;
    
else
    if(stride==1)
        pad=[(size_B(1)-1)/2 (size_B(2)-1)/2];        
        pa_A=zeros(size_A(1)+2*pad(1),size_A(2)+2*pad(2),size_A(3));
        pa_A(pad(1)+1:size_A(1)+pad(1),pad(2)+1:size_A(2)+pad(2),:)=A;
    end
end

size_paA=size(pa_A);
%% Conv
Result=zeros(1+(size_paA(1)-size_B(1))/stride,...
    1+(size_paA(2)-size_B(2))/stride);
for i=1:size(Result,1)
    for j=1:size(Result,2)
        temp=pa_A(((i-1)*stride+1):((i-1)*stride+size_B(1)),...
            ((j-1)*stride+1):((j-1)*stride+size_B(2)),:);
        
        Result(i,j)=sum(sum(sum(temp.*B)));
    end
end

Result=ReLU(Result);    %激活

end
