T=read(resizeTestImgs)
ActualLabel=T(:,2)
ActualLabel=table2array(ActualLabel)
% 
testpreds = classify(Runway_net,resizeTestImgs)
 confusionchart(ActualLabel,testpreds)