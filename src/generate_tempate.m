% generate_template.m

for posture = ["O", "H", "I"]
    path = sprintf("../data/video/%s/P1/", posture);
    generate_template(path, sprintf("../template/%s", posture), 4, 0.5);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Helper function: Calculating Similitude


function generate_template(path, output_name, T, forget_rate)

    if ~exist('forget_rate','var')
        forget_rate = 0.5;
    end
    
    folder_o = dir(path+"*.mp4");
 
    mhi = -1;
    
    for k = 1: length(folder_o)
        file_name = folder_o(k).name;
        temp = advanced_mhi(path + file_name, T, forget_rate);
        if mhi == -1
            mhi = temp;
        else
            mhi = mhi + temp;
        end
    end
    mhi = mhi / length(folder_o);
    imwrite(mhi, output_name+".png")
    
end


function dist = compare(x,y)
    x = similitudeMoments(x);
    y = similitudeMoments(y);
    dist = sqrt(sum((x-y).^2, 'all'));
end

function mhi = advanced_mhi(fileName, T, fg)
    % Get difference of im and im2 (absolute)
    % Threshold difference by T, coefficient of forget rate fg (0-1).
    
    % default forget rate is 0.5
    if ~exist('fg','var')
        fg = 0.5;
    end
    video = VideoReader(fileName);
    mhi = zeros(video.height, video.width);
    
    max_frame = floor(video.duration * video.frameRate);
    min_keep = ceil(max_frame * fg);
    timestamp = 0;
    prev_frame = -1;
    
    while hasFrame(video)
        % switch rgb to gray, smoothed by gaussian filter with sigma 
        frame = imgaussfilt(rgb2gray(readFrame(video)), 2.2);
        if prev_frame == -1
            prev_frame = frame;
            continue
        else
            dif = abs(frame - prev_frame);
            dif = medfilt2(dif);
            % re-normalized mhi. Forget rate = fg
            mhi(dif >= T) = max(0, (timestamp - min_keep + 1)/(max_frame - min_keep + 1));
        end
        % update previous frame and timestamp
        prev_frame = frame;
        timestamp = timestamp + 1;
    end
end


function Nvals = similitudeMoments(im)
    Nvals = [];
    
    % initialize matrix for row index, col index, x average and y average.
    xind = repmat(1:size(im,2),size(im,1),1); % col => x
    yind = repmat((1:size(im,1))', 1, size(im,2)); % row => y
   
    m00 = sum(im, 'all');
    m10 = sum(xind.*im, 'all');
    m01 = sum(yind.*im, 'all');

    xbar = ones(size(im)) * m10/m00;
    
    ybar = ones(size(im)) * m01/m00;
    % iteratively calculate 7 similitude moments
    for i = 0:3
        for j = max(0,(2-i)):(3-i)
            % 2 <= (i+j) <= 3
            nij = sum(((xind - xbar).^i).*((yind - ybar).^j).*im, 'all')/(m00.^((i+j)/2+1));
            Nvals = [Nvals, nij];
        end
    end
end