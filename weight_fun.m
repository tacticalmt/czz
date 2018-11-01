function [L] = weight_fun( train_binary_data,train_binary_label,k )
%计算邻接矩阵的函数
%   Detailed explanation goes here
clear L;
[~,tempt_location] = unique(train_binary_label);%[a,b]=unique(A),a返回向量A中不重复的元素，每种一个；b返回第n个元素最大的位置，得到的n1为第一类样本的数量，标签是一个很长的标签
%向量，包含该类所有样本
n1=tempt_location(1);
n2=tempt_location(2)-tempt_location(1);%n1和n2分别是第一类和第二类的样本数


%计算类内关系样本间距离
%--------------------------
X=train_binary_data';%把样本矩阵转换成每列一个样本
n=size(X,2);%计算列数，即求样本的个数
A=diag(X'*X);%求样本的平方 x^2
B=A';%A为列向量，B为行向量
D_squ=A*ones(1,n)+ones(n,1)*B-2.*(X'*X); %样本两两之间距离平方 (a^2-b^2)
Dis_Sam=sqrt(D_squ);%距离
%--------------------------
%k近邻矩阵构造
[~,Loc_sam_1]=sort(Dis_Sam,2);%对距离进行排序，结果loc_sam存放的结果只用前k个，具体数值无意义,只关注数值的位置，取前k个
[~,Loc_sam_2]=sort(Loc_sam_1,2);%对loc_sam_1存放的结果求位置，所得到Loc_sam_2的结果数值为loc_sam中的位置，数值小于k的，表示为k近邻内的样本,Loc_sam_2的数值为原始样本的位置
                                %例如结果为[3 1 2]时，排序后的矩阵的第三个为原始样本第一个位置的值，意味着原始样本中第一个值是距离第3大的
c_sort=reshape(Dis_Sam,1,n*n);%排成一个行序列，新序列用于构造邻近矩阵
c_sort(1,find(Loc_sam_2>k))=0;%find用于找出矩阵中值的索引号，Loc_sam_2中的索引代表了原来样本的位置，如[3 1 2]，3>k而3的索引为1，则第一个样本3返回得到c_sort(1,1)，所以从c_sort(1,1)=0
c_sort(1,find(Loc_sam_2<=k+1))=1;
S_sorted_sample=reshape(c_sort,n,n);%生成k近邻矩阵
S_sorted_sample=S_sorted_sample+diag(-diag(S_sorted_sample));%把对角线元素置零
Ww_ij=S_sorted_sample;%类内关系
Wb_ij=S_sorted_sample;%类间关系

%构造类间类内k近邻矩阵
Ww_ij(1:n1,n1+1:n2)=0;  %只有1到n1的样本为第一类，所以大于n1的为第二类样本，则1到n1的样本跟n1+1到末尾的关系为类间关系，不是类内关系，所以全部置零
Ww_ij(n1+1:end,1:n1)=0;
Wb_ij(1:n1,1:n1)=0;
Wb_ij(n1+1:end,n1+1:end)=0;
L.w=Ww_ij;
L.b=Wb_ij;


end

