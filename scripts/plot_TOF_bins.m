%%
%Plot TOF bins. 
% Nikos Efthimiou. 2018/11/01
% University of Hull 

%This scripts loads LOR files, exported by test_time_of_flight to the disk and plots them. 
%%
clc; clear all;
%Path to TOF files. 
path_name ='/home/nikos/Desktop/conv_LOR/'

pre_sort_files_in_path = dir(path_name)
nums = []
names = []

for i = 1: size(pre_sort_files_in_path)
    cur_file = pre_sort_files_in_path(i).name   
   if strfind (cur_file, 'glor')
       num = sscanf(cur_file,'glor_%d')
       % The following number can change accordingly.
       if ((mod(num,1)==0) || num == 500000000)
        nums{end+1} = int32(num);
        names{end+1} = cur_file;
      end
       
   end   
end

clear cur_file
sorted_filenames = cell(numel(nums),2);
[Sorted_A, Index_A] = sort(cell2mat(nums));
sorted_filenames(:,2) = names(Index_A);

% hold x values
x_values = [];
% hold the tof bins.
y_tf_values = [];
% hold the non tof LOR
y__ntf_values = [];

for i = 1 : size(sorted_filenames,1)
    cur_file = sorted_filenames{i,2};
    
    if strfind (cur_file, 'glor')
        
        if strfind(cur_file, '500000000')
            cur_full_path = fullfile(path_name, cur_file);
            
            A = importdata(cur_full_path);
            y_ntf_values = A(:,2);
        else
            cur_full_path = fullfile(path_name, cur_file);
            
            A = importdata(cur_full_path);
            
            if size(x_values) == 0
                x_values = A(:,1);
            end
            
            y_tf_values = [y_tf_values A(:,2)];
            
        end
    end
end

sum_of_all_bins = sum(y_tf_values,2);
x_v = x_values/0.299; 

%% Create Plot
plot(x_v,y_tf_values(:,:), x_v, sum_of_all_bins, x_v, y_ntf_values)
