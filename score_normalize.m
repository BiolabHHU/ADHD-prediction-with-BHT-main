function [train_score, bound]= score_normalize(index, data_name)
% index=12
% data_name = 'PU_alff_ds';
%    # if name_list_=='PU_alff_ds':
%     # y = tf.clip_by_value(y, 35, 68)
%     # if name_list_=='NYU_alff_ds':
%     # y = tf.clip_by_value(y, 55, 99)
%     
%     
%         # if name_list_=='PU_alff_ds':
%     # y = tf.clip_by_value(y, 18, 44)
%     # if name_list_=='NYU_alff_ds':
%     # y = tf.clip_by_value(y, 40, 62)
    
load([data_name '.mat'])
train_score = zeros(size(score));

if strcmp(data_name, 'NYU_alff_ds')
   tag = tag>0;
   [mm, ~]=find(tag==1);
   tmp_ = score(mm);
   MAX_AD = max(tmp_);
   MIN_AD = min(tmp_);
   train_score(mm) = (tmp_-MIN_AD)/(MAX_AD-MIN_AD);
   
   [mm, ~]=find(tag==0);
   tmp_ = score(mm);
%    MAX_HC = max(tmp_);
   MAX_HC = 62;
   MIN_HC = min(tmp_);
   train_score(mm) = (tmp_-MIN_HC)/(MAX_HC-MIN_HC);
    
elseif strcmp(data_name, 'PU_alff_ds')
   tag = tag>0;
   [mm, ~]=find(tag==1);
   tmp_ = score(mm);
   MAX_AD = max(tmp_);
   MIN_AD = min(tmp_);
   train_score(mm) = (tmp_-MIN_AD)/(MAX_AD-MIN_AD);
   
   [mm, ~]=find(tag==0);
   tmp_ = score(mm);
   MAX_HC = max(tmp_);
   MIN_HC = min(tmp_);
   train_score(mm) = (tmp_-MIN_HC)/(MAX_HC-MIN_HC);
  
  
  
else
    
end

bound = [MAX_AD MIN_AD MAX_HC MIN_HC];
tmp_ = train_score(index);
train_score(index)=[];
train_score = [train_score; tmp_];






      
      