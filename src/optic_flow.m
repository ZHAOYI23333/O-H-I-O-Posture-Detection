% declare the sum of magnitude array
sum_flow = [];

% sum of optic flow (magnitude) 600 is the start/end point
video = VideoReader("../data/video/OHIO/3.mp4",'CurrentTime',0);
% video = VideoReader("../data/video/O/P3/1.mp4");

opticFlow = opticalFlowHS;
h = figure;
movegui(h);

hViewPanel = uipanel(h,'Position',[0 0 1 1],'Title','Plot of Optical Flow Vectors');
hPlot = axes(hViewPanel);

while hasFrame(video)
    
    % read and convert to grayscale
    frameRGB = readFrame(video);
    frameGray = rgb2gray(frameRGB);
    
    % estimate flow
    flow = estimateFlow(opticFlow,frameGray);
    imshow(frameRGB)
    
    % visualize
    hold on
    plot(flow,'DecimationFactor',[5 5],'ScaleFactor',100,'Parent',hPlot);
    hold off
    pause(10^-2)
    
    % update flow
    sum_flow = [sum_flow sum(flow.Magnitude, 'all')];
    
end

% discard first flow
sum_flow(1) = sum_flow(2);

% plot
plot(1:length(sum_flow), sum_flow, '.--b','MarkerSize', 20)
hold on
plot(1:length(sum_flow), ones(size(sum_flow)) * 600, '-r', 'LineWidth', 2)
title('Sum of Magnitude of Optic Flow vs. Frame Number', 'FontSize', 30)
xlabel('# Frame', 'FontSize', 18)
ylabel('Sum of Magnitude of Optic Flow', 'FontSize', 18)


