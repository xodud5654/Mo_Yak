function [bboxes,scores,labels] = detector(data)

load("net_checkpoint__5230__2023_06_15__21_27_09.mat")
[bboxes,scores,labels] = detect(net,data);
data = insertObjectAnnotation(data,"rectangle",bboxes,scores);
imwrite(data,"testeee.png")
end
