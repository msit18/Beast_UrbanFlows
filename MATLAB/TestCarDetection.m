%This is the modified code.

tic; %starts recording processing time

foregroundDetector = vision.ForegroundDetector('NumGaussians', 5, ...
    'NumTrainingFrames', 100);


%vid1 = '/media/senseable-beast/beast-brain-1/Data/TrafficIntersectionVideos/slavePi2_RW1600_RH1200_TT900_FR15_06_13_2016_07_00_06_154714.avi';
%vid1 = '/media/senseable-beast/beast-brain-1/Data/TrafficIntersectionVideos/slavePi2_RW1600_RH1200_TT900_FR15_06_12_2016_17_34_39_833163.avi';
%vid1 = '/home/msit/Presentation/OpenCV/vehicle_detection_haarcascades-master/slavePi4_RW1600_RH1200_TT180_FR15_06_03_2016_13_53_54_254155.h264';
vid1 = '/media/senseable-beast/beast-brain-1/Data/TrafficIntersectionVideos/slavePi2_RW1600_RH1200_TT900_FR15_06_10_2016_18_11_00_604698.avi';
%vid1 = 'CloseDay.mp4';
total = 0; %increments for each car detection
record = 1; %used to pause processed video so I could record my screen
previousCentroids = zeros(1,1);

videoReader = vision.VideoFileReader(vid1); %input name of video file as string

for i = 1:150
    frame = step(videoReader); % read the next video frame
    foreground = step(foregroundDetector, frame);
end

%figure; imshow(frame); title('Video Frame'); %Figure 1

%figure; imshow(foreground); title('Foreground');

se = strel('square', 3); %second parameter (int) changes how much of the foreground to clean

filteredForeground = imopen(foreground, se);
%figure; imshow(filteredForeground); title('Clean Foreground'); %Figure 2

blobAnalysis = vision.BlobAnalysis('BoundingBoxOutputPort', true, ...
    'AreaOutputPort', false, 'CentroidOutputPort', true, ...
    'MinimumBlobArea', 3000); %last parameter sets minimum blob area
[centroids, bbox] = step(blobAnalysis, filteredForeground);

result = insertShape(frame, 'Rectangle', bbox, 'Color', 'green');

numCars = size(bbox, 1); %int, how many cars are in the frame
result = insertText(result, [10 10], numCars, 'BoxOpacity', 1, ...
    'FontSize', 30);
%figure; imshow(result); title('Detected Cars'); %Figure 4

videoPlayer = vision.VideoPlayer('Name', 'Detected Cars');
videoPlayer.Position(3:4) = [1600,1200];  % window size: [width, height]
se = strel('square', 8); % morphological filter for noise removal

while ~isDone(videoReader)
    frame = step(videoReader); % read the next video frame
    
    % Detect the foreground in the current video frame
    foreground = step(foregroundDetector, frame);

    % Use morphological opening to remove noise in the foreground
    filteredForeground = imopen(foreground, se);

    
    % Detect the connected components with the specified minimum area, and
    % compute their bounding boxes
    
    [centroids, bbox] = step(blobAnalysis, filteredForeground);
%     fprintf ('centroids');
%     disp (centroids);
%     disp(size(centroids));
%     disp(size(bbox));
    %     fprintf('oldCentroids');
%     disp (previousCentroids);
    
    %fprintf ('bbox');
    %disp (bbox);
    
%     for v = size(centroids):
%         disp(centroids(
    
%     if size(newCentroids) > 0
%         x = newCentroids(1,1); %reads x coordinate of blob centroid
%         %disp(x); %displays x coordinate
%         if (x < 60) && x > 50 %detects whether x coord is in certain range
%             total = total + 1;
%             %a better way would be to use ROIs
%         end
%     end
    
    % Draw bounding boxes around the detected cars
    result = insertShape(frame, 'Rectangle', bbox, 'Color', 'green');

    % Display the number of cars found in the video frame
    numCars = size(bbox, 1);
    %disp (size(bbox,1));
    %result = insertText(result, [10 10], total, 'BoxOpacity', 1, ...
    %    'FontSize', 30);

    step(videoPlayer, result);  % display the results
    
    %if statement is used to pause video so I could set up screen capture
    
%     if(record == 1)
%         %{pause(4); 
%         record = 0;
%     end
    
    
end
toc; %displays total processing time up to this point

release(videoReader); % close the video file