% [data, info]=read(resizeTestImgs)
% tlL=tl.response
% tlL=categorical(tlL)
trueLabel=testImgs.Labels
result=classify(Runway_net_sgdm,resizeTestImgs)
figure(1)
confusionchart(trueLabel,result)

[YPred,probs] = classify(Runway_net_sgdm,resizeTestImgs);
accuracy = mean(YPred == testImgs.Labels)
n=9
idx = randperm(numel(testImgs.Files),n);
figure(2)
for i = 1:n
    subplot(3,3,i)
    I = readimage(testImgs,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label) + ", " + num2str(100*max(probs(idx(i),:)),3) + "%");
end

figure(3)
YValidation=ValiImgs.Labels
[YValPred,probsv] = classify(Runway_net_sgdm,resizeValiImgs);

figure('Units','normalized','Position',[0.2 0.2 0.4 0.4]);
cm = confusionchart(YValidation,YValPred);
cm.Title = 'Confusion Matrix for Validation Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';