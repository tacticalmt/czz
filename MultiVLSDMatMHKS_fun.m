function [GLMatStruct,u_v_b,co_result] = MultiVLSDMatMHKS_fun(train_binary_data,train_binary_label,Sampled_maxtrix,InputPar,L,mat_sample_num,mat_sample_way)
%MULTIVLSDMATMHKS_FUN Summary of this function goes here
%   Detailed explanation goes here
%train_binary_data：训练数据集，每一行一个样本
%train_binary_label：训练数据集对应的类标号
%InputPar：参数，包含k、C、lam等
%M_row：当前矩阵化的行数
%M_col：当前矩阵化的列数
%L：ARLE产生的全局局部度量矩阵
%mat_sample_num为视角数量
%Matlab2010b
label_vector = unique(train_binary_label);%[a,b]=unique(A),a返回向量A中不重复的元素，每种一个；b返回第一个不同元素的位置

n1=length(find(train_binary_label==label_vector(1)));
n2=length(find(train_binary_label==label_vector(2)));%n1和n2分别是第一类和第二类的样本数

train_binary_label(1:n1) = 1;%转类标号为1和-1
train_binary_label(n1+1:n1+n2) = -1;

%初始化参数
total_iter = 100;
M_row=zeros(mat_sample_num,1);%每一列为一个视角的行列值
M_col=zeros(mat_sample_num,1);
del_var_2=0;
del_var_1=0;
cor_res=zeros(5,1);%记录每5轮的值
i_cov=1;
comp_co=0;%记录上5轮的第一个值
cov=0;%记录收敛值，每5次中最小值如果比cov大则判断不收敛
sa=0;
co_result='succeed';

for p_view=1:mat_sample_num  %p_view为视角标号
        M_row(p_view) = mat_sample_way(p_view,1);%当前矩阵化后矩阵的行数
        M_col(p_view) = mat_sample_way(p_view,2);%当前矩阵化后矩阵的列数
        u{p_view} = zeros(M_row(p_view) + 1,total_iter);
        v{p_view} = zeros(M_col(p_view) + 1,total_iter);
        b{p_view} = zeros(n1+n2,total_iter);
        e{p_view} = zeros(n1+n2,total_iter);
        u{p_view}(:,1) = [InputPar.u_u * ones(M_row(p_view),1);1];%u的最后一行为1
        b{p_view}(:,1) = InputPar.b_b * ones(n1+n2,1);
        S_1{p_view} = M_row(p_view)*eye(M_row(p_view));
        S_2{p_view} = M_col(p_view)*eye(M_col(p_view));
        S1{p_view} = [S_1{p_view} zeros(size(S_1{p_view},1),1);zeros(1,size(S_1{p_view},2)) 1];
        S2{p_view} = [S_2{p_view} zeros(size(S_2{p_view},1),1);zeros(1,size(S_2{p_view},2)) 1];
end%for


I = ones(n1+n2,1);%公式中用到的全1向量


rho = 0.99;%p
eta = 10^(-4);%判断终止的条件
k_iter = 1;
for i_mat=1:mat_sample_num  %i_mat为当前视角标号
u_v_b{i_mat}=zeros(5+M_row(i_mat)+1+M_col(i_mat)+1+n1+n2,total_iter);%每一次迭代存储u,v,b的值
end
%开始迭代训练
while (k_iter < total_iter)
    
    for p_view=1:mat_sample_num  %p_view为视角标号
 %       M_row = mat_sample_way(p_view,1);%当前矩阵化后矩阵的行数
%        M_col = mat_sample_way(p_view,2);%当前矩阵化后矩阵的列数
if p_view==InputPar.view1selected||p_view==InputPar.view2selected
    if p_view==InputPar.view1selected
    InputPar.curC=InputPar.C;
    else
        InputPar.curC=InputPar.C2;
    end
    v{p_view}(:,k_iter) = get_v(train_binary_label,Sampled_maxtrix,InputPar,S2{p_view},I,u{p_view}(:,k_iter),b{p_view}(:,k_iter),L,mat_sample_num,p_view,u,v,k_iter);
    
    e{p_view}(:,k_iter) = get_e(train_binary_label,Sampled_maxtrix,I,p_view,u{p_view}(:,k_iter),v{p_view}(:,k_iter),b{p_view}(:,k_iter));
    b{p_view}(:,k_iter+1) = (b{p_view}(:,k_iter) + rho*(e{p_view}(:,k_iter) + abs(e{p_view}(:,k_iter))));
 
%    stop_tag = norm(b{p_view}(:,k_iter+1) - b{p_view}(:,k_iter),2);
%     tag(k_iter)=stop_tag;
    u_v_b{p_view}(:,k_iter)=[InputPar.k;InputPar.C;InputPar.lam;M_row(p_view);M_col(p_view);u{p_view}(:,k_iter);v{p_view}(:,k_iter);b{p_view}(:,k_iter)];
    if k_iter==total_iter
        break;
    end
    u{p_view}(:,k_iter+1) = get_u(train_binary_label,Sampled_maxtrix,InputPar,M_row(p_view),S1{p_view},I,v{p_view}(:,k_iter),b{p_view}(:,k_iter),L,mat_sample_num,p_view,u,v,k_iter);
end%end if    
    
%    if stop_tag < eta% 待修改
%        if p_view==mat_sample_num
%            break;
%        end
%        continue;
%    else
%        u{p_view}(:,k_iter+1) = get_u(train_binary_data,train_binary_label,InputPar,M_row(p_view),M_col(p_view),S1{p_view},I,v{p_view}(:,k_iter),b{p_view}(:,k_iter),L,mat_sample_num,p_view,mat_sample_way,u,v,k_iter);
%    end%end_if
    
    end%for P_view
    if k_iter>1
        del_var_2=MultiVLSDLk(train_binary_data,train_binary_label,InputPar,S1,S2,I,u,v,b,k_iter,L,mat_sample_num,mat_sample_way);
        del_var_1=MultiVLSDLk(train_binary_data,train_binary_label,InputPar,S1,S2,I,u,v,b,k_iter-1,L,mat_sample_num,mat_sample_way);
        
    end
    op=norm(del_var_2-del_var_1,2);
    ab=op/norm(del_var_1,2);
    %------------------收敛判断------------
    if i_cov==1
        comp_co=ab;
    end
    
    cor_res(i_cov)=ab;
    if i_cov==5
        i_cov=1;
        cov=min(cor_res);
        sa=comp_co-cov;%cov应该小于comp_co才为收敛，不收敛时sa为负值
    else 
        i_cov=i_cov+1;
    end
    if sa<0
        co_result='reject';
    end
    %-------------------------------------
    %-------------------------------------
    if ab<eta||sa<0
        break;
    end
    
    k_iter = k_iter + 1;
    
    
end%while
for final_v=1:mat_sample_num
if k_iter == total_iter
    GLMatStruct.u{final_v} = u{final_v}(:,k_iter);
    GLMatStruct.v{final_v} = v{final_v}(:,k_iter-1);
else
    GLMatStruct.u{final_v} = u{final_v}(:,k_iter);
    GLMatStruct.v{final_v} = v{final_v}(:,k_iter);    
end%end_if
end%end for final_v
end

