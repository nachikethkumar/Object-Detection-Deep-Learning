% 
% cam=webcam(2);
% view=preview(cam)
% vid=videoinput("winvideo","2","MJPG_1280x720")
% 
% set(vid, 'FramesPerTrigger', Inf);
% set(vid, 'ReturnedColorspace', 'rgb');
% for i=1:10
%     data = getsnapshot(vid);
%     figure(i);
%     imshow(data);
% end


% Create a webcam object
cam = webcam(2);

% Create a figure to display the webcam frames
figure;

% Main loop to continuously acquire and display frames
while ishandle(1) % Check if the figure is still open
    % Acquire a frame from the webcam
    frame = snapshot(cam);
    
    frame=imresize(frame,[500,500]);
    % Display the frame in the figure window
    frame=rgb2gray(frame);
    frame=edge(frame,"sobel");
    imshow(frame);
    
    % Add any additional processing or analysis here
    
   % drawnow; % Force the figure to update
end

% Clean up
clear cam;
