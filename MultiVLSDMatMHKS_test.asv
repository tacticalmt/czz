function [Group,dv] = MultiVLSDMatMHKS_test(LSDMatStruct,test_data_final,label_one,label_two,mat_sample_num,mat_sample_way,view_par)

%用于测试LSDMatMHKS的代码
%LSDMatStruct：含有用于当前测试的训练数据的结构体
%test_data_final：测试数据集
dv=[];
tag_all_view=zeros(mat_sample_num,1);
for p_test = 1:size(test_data_final,1)
    class_tag=0;
    for p_view=1:length(view_par)
        M_row=mat_sample_way(view_par(p_view),1);
        M_col=mat_sample_way(view_par(p_view),2);
        A = reshape(test_data_final(p_test,:),M_row,M_col);
        B=[A zeros(size(A,1),1);zeros(1,size(A,2)) 1];
        tag_all_view(view_par(p_view)) = LSDMatStruct.u{view_par(p_view)}'*B*LSDMatStruct.v{view_par(p_view)};
    end %for
    
 %   for cal=1:mat_sample_num
  %      class_tag=class_tag+tag_all_view(cal);
  %  end%求视角总和
     class_tag=sum(tag_all_view(1:end)); 
  dv=[dv;class_tag];
  
    if class_tag >= 0
        Group(p_test) = label_one;
    else
        Group(p_test) = label_two;
    end%end_if
end%end_for_p_test

Group = Group';

clear tag_all_view;
clear class_tag;

