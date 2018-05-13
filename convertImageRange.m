
for i=1:5830
    img = imread(['data/train/TrainImages/Image' num2str(i) '.jpg']);
   % img  = 2*(img/255.0)-1.0;
    imwrite(img,['data/train/TrainImagesSorted/Image' num2str(i+1000) '.jpg']);
end
for i=1:2298
    img = imread(['data/validation/ValidationImages/Image' num2str(i) '.jpg']);
%     img  = 2*(img/255.0)-1.0;
    imwrite(img,['data/validation/ValidationImagesSorted/Image' num2str(i+1000) '.jpg']);
end
for i=1:3460
    img = imread(['data/test/TestImages/Image' num2str(i) '.jpg']);
%     img  = 2*(img/255.0)-1.0;
    imwrite(img,['data/test/TestImagesSorted/Image' num2str(i+1000) '.jpg']);
end
