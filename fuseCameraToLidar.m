%% demo fuseCameraToLidar
% imPts = projectLidarPointsOnImage(p1,intrinsics0,tform);
dataPath = fullfile(toolboxdir('lidar'),'lidardata','lcc','sampleColoredPtCloud.mat');
gt = load(dataPath);
im = gt.im;
figure
imshow(im)
imwrite(im,'camera-view.png')
ptCloud = gt.ptCloud;
figure 
pcshow(ptCloud)
title('Original Point Cloud')

intrinsics = gt.camParams;
camToLidar = gt.tform;
ptCloudOut = fuseCameraToLidar(im,ptCloud,intrinsics,camToLidar);
pcshow(ptCloudOut)
title('Colored Point Cloud')

