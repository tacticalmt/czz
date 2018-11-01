function [ L ] = Eweight_fun( train_binary_data,train_binary_label,k )
%EWEIGHT_FUN Summary of this function goes here
%   Detailed explanation goes here
%按熵排列做求2k个近邻里的k个熵最小的点为近邻
clear L;
[class_label,tempt_location] = unique(train_binary_label);%[a,b]=unique(A),a返回向量A中不重复的元素，每种一个；b返回第n个元素最大的位置，得到的n1为第一类样本的数量，标签是一个很长的标签
%向量，包含该类所有样本
n1=length(find(train_binary_label==class_label(1)));
n2=length(find(train_binary_label==class_label(2)));%n1和n2分别是第一类和第二类的样本数
%计算类内关系样本间距离
%--------------------------
X=train_binary_data';%把样本矩阵转换成每列一个样本
n=size(X,2);%计算列数，即求样本的个数
A=diag(X'*X);%求样本的平方 x^2
B=A';%A为列向量，B为行向量
D_squ=A*ones(1,n)+ones(n,1)*B-2.*(X'*X); %样本两两之间距离平方 (a^2-b^2)
Dis_Sam=sqrt(D_squ);%距离
[~,Loc_sam_1]=sort(Dis_Sam,2);%对距离进行每一行排序，结果loc_sam存放的结果只用前k个，具体数值无意义,只关注数值的位置，取前k个
[~,Loc_sam_2]=sort(Loc_sam_1,2);%对loc_sam_1存放的结果求位置，所得到Loc_sam_2的结果数值为loc_sam中的位置，数值小于k的，表示为k近邻内的样本,Loc_sam_2的数值为原始样本的位置
                                %例如结果为[3 1 2]时，排序后的矩阵的第三个为原始样本第一个位置的值，意味着原始样本中第一个值是距离第3大的
c_sort=reshape(Dis_Sam,1,n*n);%排成一个行序列，新序列用于构造邻近矩阵
c_sort(1,find(Loc_sam_2>k))=0;%find用于找出矩阵中值的索引号，Loc_sam_2中的索引代表了原来样本的位置，如[3 1 2]，3>k而3的索引为1，则第一个样本3返回得到c_sort(1,1)，所以从c_sort(1,1)=0
c_sort(1,find(Loc_sam_2<=k+1))=1;
S_sorted_sample=reshape(c_sort,n,n);%生成k近邻矩阵
S_sorted_sample=S_sorted_sample+diag(-diag(S_sorted_sample));%把对角线元素置零
%---------------2k近邻---
c_sort_e=reshape(Dis_Sam,1,n*n);%排成一个行序列，新序列用于构造邻近矩阵
c_sort_e(1,find(Loc_sam_2>2*k))=0;%find用于找出矩阵中值的索引号，Loc_sam_2中的索引代表了原来样本的位置，如[3 1 2]，3>k而3的索引为1，则第一个样本3返回得到c_sort(1,1)，所以从c_sort(1,1)=0
c_sort_e(1,find(Loc_sam_2<=2*k+1))=1;
S_sorted_sample_e=reshape(c_sort_e,n,n);%生成k近邻矩阵
S_sorted_sample_e=S_sorted_sample_e+diag(-diag(S_sorted_sample_e));%把对角线元素置零


%--------求熵近邻-----
entropy_sam=zeros(length(train_binary_label),1);%存每个样本的熵
for i_sam_e=1:length(train_binary_label)
    knn_sum=sum(S_sorted_sample(i_sam_e,:));%求该点总的近邻数
    knn_label=find(S_sorted_sample(i_sam_e,:)==1);%近邻的全部类标的样本号
    sam_num_1=sum(train_binary_label(knn_label)==class_label(1));%第一类样本的数量
    sam_num_2=sum(train_binary_label(knn_label)==class_label(2));%第二类样本的数量
    p_1=sam_num_1/knn_sum;
    p_2=sam_num_2/knn_sum;
    if p_1==0||p_2==0
        entropy_sam(i_sam_e)=0;
    else
        entropy_sam(i_sam_e)=-1*p_1*log(p_1)+(-1)*p_2*log(p_2);
    end
end
%------生成熵矩阵------
entro_matr=ones(length(train_binary_label),1)*entropy_sam';%熵矩阵
for i_ent=1:length(train_binary_label)
    entro_matr(i_ent,find(S_sorted_sample_e(i_ent,:)~=1))=10;%方便排序，把不相关的都设定为10
end
%------生成k熵近邻-----
[~,Loc_ent]=sort(entro_matr,2);
[~,Loc_ent_2]=sort(Loc_ent,2);
e_sort=reshape(entro_matr,1,n*n);
e_sort(1,find(Loc_ent_2>k))=0;
e_sort(1,find(Loc_sam_2<=k+1))=1;
W_sorted_sample=reshape(e_sort,n,n);
W_sorted_sample=W_sorted_sample+diag(-diag(W_sorted_sample));%把对角线元素置零
Ww_ij=W_sorted_sample;%类内关系
Wb_ij=W_sorted_sample;%类间关系
%构造类间类内k近邻矩阵
Ww_ij(1:n1,n1+1:n2)=0;  %只有1到n1的样本为第一类，所以大于n1的为第二类样本，则1到n1的样本跟n1+1到末尾的关系为类间关系，不是类内关系，所以全部置零
Ww_ij(n1+1:end,1:n1)=0;
Wb_ij(1:n1,1:n1)=0;
Wb_ij(n1+1:end,n1+1:end)=0;
L.w=Ww_ij;
L.b=Wb_ij;

end

