function [k_sum] = MultiVLSDLk(train_binary_data,train_binary_label,InputPar,S1,S2,I,u_view,v_view,b,k_iter,L,mat_sample_num,mat_sample_way )
%MULTIVLSDLK Summary of this function goes here
%   Detailed explanation goes here
%求L的k+1次和k的总和




clear k_sum;
Y_view=cell(mat_sample_num,1);%%为每个视角存Y值
Y_result=zeros(1,mat_sample_num);
cur_view_sum=0;
[samp_size,~]=size(train_binary_label);
Y_all=zeros(samp_size,1);
view_s=[InputPar.view1selected InputPar.view2selected];

for all_p_v = 1:size(train_binary_label)
%     if p_v == 1
%         A = reshape(train_binary_data(p_v,:),M_row,M_col);
%         B=[A zeros(size(A,1),1);zeros(1,size(A,2)) 1];
%         y = (u'*B)';
%         Y = train_binary_label(p_v)*y;
%         Y_sw=y;
%         clear A;clear B;
%     else
%------------求每个视角的Yv-------------------------------------
        for c_v=1:length(view_s)
        M_row_view=mat_sample_way(view_s(c_v),1);
        M_col_view=mat_sample_way(view_s(c_v),2);
        A_view=reshape(train_binary_data(all_p_v,:),M_row_view,M_col_view);
        B_view=[A_view zeros(size(A_view,1),1);zeros(1,size(A_view,2)) 1];
        y_view=train_binary_label(all_p_v)*(u_view{view_s(c_v)}(:,k_iter)'*B_view)';%%%
        Y_view{view_s(c_v)}=[Y_view{view_s(c_v)},y_view];
        end%for c_v
%--------------------------------------------------------------

end%all_p_v
%_____________求视角总和_____________
for c_v=1:length(view_s)
    
    Y_view_temp=Y_view{view_s(c_v)}';
    Y_all=Y_all+(Y_view_temp*v_view{view_s(c_v)}(:,k_iter));
    
end

M_row=zeros(mat_sample_num,1);%每一列为一个视角的行列值
M_col=zeros(mat_sample_num,1);

for cur_view=1:mat_sample_num%算视角求和

Y=[];
Y_w=0;%zeros(1,size(S2{cur_view}));%%可能是数
Y_b=0;%zeros(1,size(S2{cur_view}));
if cur_view==InputPar.view1selected||cur_view==InputPar.view2selected
    if cur_view==InputPar.view1selected
    InputPar.curC=InputPar.C;
    else
        InputPar.curC=InputPar.C2;
    end
M_row(cur_view) = mat_sample_way(cur_view,1);%当前矩阵化后矩阵的行数
M_col(cur_view) = mat_sample_way(cur_view,2);%当前矩阵化后矩阵的列数
for p_v = 1:size(train_binary_label)        
        A = reshape(train_binary_data(p_v,:),M_row(cur_view),M_col(cur_view));
        B=[A zeros(size(A,1),1);zeros(1,size(A,2)) 1];
        
        y = train_binary_label(p_v)*(u_view{cur_view}(:,k_iter)'*B)';
        Y = [Y,y];
        
%         clear A;clear B;
index_w=find(L.w(p_v,:)==1);
index_b=find(L.b(p_v,:)==1);
for p_v2=1:length(index_w)
    A2 = reshape(train_binary_data(index_w(p_v2),:),M_row(cur_view),M_col(cur_view));
    B2=[A2 zeros(size(A2,1),1);zeros(1,size(A2,2)) 1];
    Y_w=Y_w+v_view{cur_view}(:,k_iter)'*(B-B2)'*u_view{cur_view}(:,k_iter)*u_view{cur_view}(:,k_iter)'*(B-B2)*v_view{cur_view}(:,k_iter)*(InputPar.lam*L.w(p_v,index_w(p_v2)));%计算正则化项R_local
end
for p_v2=1:length(index_b)
    A2 = reshape(train_binary_data(index_b(p_v2),:),M_row(cur_view),M_col(cur_view));
    B2=[A2 zeros(size(A2,1),1);zeros(1,size(A2,2)) 1];
    Y_b=Y_b+v_view{cur_view}(:,k_iter)'*(B-B2)'*u_view{cur_view}(:,k_iter)*u_view{cur_view}(:,k_iter)'*(B-B2)*v_view{cur_view}(:,k_iter)*((InputPar.lam-1)*L.b(p_v,index_b(p_v2)));%计算正则化项R_local
end
%     end%end_if
end%end_p_v 
Y = Y';
Y_wb=Y_w+Y_b;%当前视角的总LSD ,数值

Y_result(cur_view)=(Y*v_view{cur_view}(:,k_iter)-I-b{cur_view}(:,k_iter))'*(Y*v_view{cur_view}(:,k_iter)-I-b{cur_view}(:,k_iter))+InputPar.curC*(u_view{cur_view}(:,k_iter)'*S1{cur_view}*u_view{cur_view}(:,k_iter)+v_view{cur_view}(:,k_iter)'*S2{cur_view}*v_view{cur_view}(:,k_iter))+Y_wb;
%Y_result(cur_view)

cur_view_sum=cur_view_sum+InputPar.gamma*(Y*v_view{cur_view}(:,k_iter)-(1/mat_sample_num)*Y_all)'*(Y*v_view{cur_view}(:,k_iter)-(1/mat_sample_num)*Y_all);
end%end if
end%cur_view

%_____________求视角总和_____________
%for c_v=1:mat_sample_num
    %
  %  Y_view_temp=Y_view{c_v}';
   % Y_all=Y_all+(Y_view_temp*v_view{c_v}(:,k_iter));
    
%end
All_data_sum=0;
for all_sum=1:mat_sample_num
    All_data_sum=All_data_sum+Y_result(all_sum);%+InputPar.gamma*(Y*v_view{cur_view}(:,k_iter)-(1/mat_sample_num)*Y_all);
end
k_sum=All_data_sum+cur_view_sum;


end

