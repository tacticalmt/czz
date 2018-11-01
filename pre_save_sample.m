function [matlized_sample] = pre_save_sample(train_binary_data,train_binary_label,mat_sample_num,mat_sample_way,InputPar)
%PRE_SAVE_SAMPLE Summary of this function goes here
%   Detailed explanation goes here
%转换样本为矩阵和多视角模式存储起来
view_s=[InputPar.view1selected InputPar.view2selected];

for c_sample=1:size(train_binary_label)
    for c_view_num=1:length(view_s)
        M_row_view=mat_sample_way(view_s(c_view_num),1);
        M_col_view=mat_sample_way(view_s(c_view_num),2);
        A_view=reshape(train_binary_data(c_sample,:),M_row_view,M_col_view);
        matlized_sample{view_s(c_view_num)}{c_sample}=A_view;
    end%end c_view
end%end c_sample


end

