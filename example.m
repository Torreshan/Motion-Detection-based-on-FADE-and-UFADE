%%---------------------------------------------------%%
% HDM05 actions classification with knn and 10-fold crossvalidation
%%---------------------------------------------------%%
close all;
clear;
warning('off','all');
load('allSegmentedDataset.mat');
tic;

%% 1. Select a dataset
allActionsIndx = 1:130;
% group classes with same number in to one class, they are too similar. 
class = [ 1 1 1 2 2 3 3 4 5 6 7 8 8 8 8 9 10 11 12 13 14 14 14 15 15 15 16 16 16 ...
          17 17 18 18 18 18 18 19 19 19 19 20 21 21 22 22 23 23 24 24 25 25 26 ...
          27 27 28 28 29 29 30 30 31 31 32 32 33 33 34 34 35 35 36 36 37 37 37 37 37 ...
          38 38 38 38 39 40 41 42 43 43 44 44 44 44 45 45 46 47 48 49 50 51 52 53 ...
          54 55 55 56 56 57 58 59 59 59 59 60 60 61 61 62 62 62 62 63 63 63 63 ...
          64 64 64 64 65 65];        
tab = tabulate(class);
action = cell(1,65);
action{1} = [jointAngles{1:3,1}];
for i = 2:65
    action{i} = [jointAngles{sum(tab(1:i-1,2))+1:sum(tab(1:i-1,2))+tab(i,2),1}];
end




   
%% 2. Compute FADE, UFADE descriptor for all actions
f_th = 10;  % Cut at 10Hz
f_s  = 120;  % Sampling frequency
K    = 500; % Desired dimensionality

v = zeros(59,2337);
vu = zeros(500*59,2337);
count = 1;
classn = zeros(1,2337);
for i = 1:65
    [~,n]= size(action{i});
    
    for j = 1:n
        v(:,count) = FADE(action{i}{j},K);
        vu(:,count) =  UFADE(action{i}{j},K);
        
        classn(count) = i;
        count = count + 1;
    end   
   
       % disp(count);
    
end


        
    


%% 3. EValuation (kNN/K fold cross validation)
numOfNN = 1;
Kfold = 10;
%mdl_fade = fitcknn(v',classn','Distance','cityblock','NumNeighbors',numOfNN);
%mdl_ufade = fitcknn(vu',classn','Distance','cityblock','NumNeighbors',numOfNN);
%CVMdl = crossval(mdl,'KFold',Kfold);  
%toc;
%kloss = kfoldLoss(CVMdl)
% % label = predict(mdl,Xnew)
% 
% 
accuracy_fade = zeros(10,1);
accuracy_ufade = zeros(10,1);
indices=crossvalind('Kfold',2337,Kfold);
%for numOfNN=1:10
for k=1:Kfold
        test = (indices == k); 
        train = ~test;
        train_data1=v(:,train);
        train_data2=vu(:,train);
        train_target=classn(train);
       
        mdl_fade = fitcknn(train_data1',train_target','Distance','cityblock','NumNeighbors',numOfNN,'Standardize',1);
        mdl_ufade = fitcknn(train_data2',train_target','Distance','cityblock','NumNeighbors',numOfNN,'Standardize',1);
        
        test_target=classn(test);
        test_data1=v(:,test);
        test_data2=vu(:,test);
        
        label1 = predict(mdl_fade,test_data1');
        label2 = predict(mdl_ufade,test_data2');
        
        accuracy_fade(k,1) = sum(label1' == test_target)/size(test_target,2);
        accuracy_ufade(k,1) = sum(label2' == test_target)/size(test_target,2);
        %disp(accuracy_fade);
        %disp(accuracy_ufade);
        c(1, k) = {confusionmat(label1, test_target')};
        c(2, k) = {confusionmat(label2, test_target')};
        
end
Mean_accuracy_fade(numOfNN) = sum(accuracy_fade)/Kfold;
Mean_accuracy_ufade(numOfNN) = sum(accuracy_ufade)/Kfold;
%end
%% plot the result
figure('Name', 'Confusion Plot', 'Position', [0, 0, 5000, 1000]);
%for idx = 1:Kfold
idx = 1;
    subplot(Kfold, 2, idx*2-1);
    imagesc(c{1, idx});
    title(['FADE, CV fold ' num2str(idx)]);
    subplot(Kfold, 2, idx*2);
    imagesc(c{2, idx});
    title(['U-FADE, CV fold ' num2str(idx)]);
%end
disp('Project finished!');
toc;
