pause(1);
AquaSense()
pause(1);
Network()
% Define the main directory where all Scene folders are located
mainDir = 'D:\Advanced Deep Learning Framework AquaSense\Segmented_Data';
% Define the output directory where processed images will be saved
outputDir = 'D:\Advanced Deep Learning Framework AquaSense\Classification';
% Create the output directory if it does not exist
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end
% Define the class names and create folders for each class in the output directory
classNames = {
    'Marine Debris', 'Floating Plastics', 'Oil', 'Dense Sargassum', ...
    'Sparse Floating Algae', 'Natural Organic Material', 'Ship', ...
    'Marine Water', 'Sediment-Laden Water', 'Foam', 'Turbid Water', ...
    'Shallow Water', 'Waves & Wakes', 'Oil Platform', 'Jellyfish', ...
    'Sea snot'
};
% Create subfolders for each class in the output directory
for idx = 1:length(classNames)
    classFolderPath = fullfile(outputDir, classNames{idx});
    if ~exist(classFolderPath, 'dir')
        mkdir(classFolderPath);
    end
end

% List all Scene directories
sceneFolders = dir(fullfile(mainDir, 'Scene_*'));
sceneFolders = sceneFolders([sceneFolders.isdir]); % Ensure only directories are selected

% Initialize a counter to control the number of displayed images
displayCounter = 0;
maxDisplayImages = 100; % Set the number of images to display

% Loop through each Scene directory
for i = 1:length(sceneFolders)
    scenePath = fullfile(sceneFolders(i).folder, sceneFolders(i).name);
    
    % List all subfolders within the current Scene directory
    subFolders = dir(scenePath);
    subFolders = subFolders([subFolders.isdir] & ~ismember({subFolders.name}, {'.', '..'})); % Exclude '.' and '..'
    
    % Loop through each subfolder
    for j = 1:length(subFolders)
        subFolderPath = fullfile(subFolders(j).folder, subFolders(j).name);
        
        % List all .png files in the current subfolder
        pngFiles = dir(fullfile(subFolderPath, '*.png'));
        
        % Loop through each .png file and apply segmentation
        for k = 1:length(pngFiles)
            % Read the image
            imgPath = fullfile(pngFiles(k).folder, pngFiles(k).name);
            img = imread(imgPath);
            
            % Convert image to grayscale if it is RGB
            if size(img, 3) == 3
                imgGray = rgb2gray(img);
            else
                imgGray = img;
            end
            
            % Normalize the grayscale image to [0, 1]
            imgGray = im2double(imgGray);
            
            % Define a seed point (for example, the center of the image)
            seedPoint = [round(size(imgGray, 1) / 2), round(size(imgGray, 2) / 2)];
            
            % Define the region growing threshold
            regionGrowingThreshold = 0.05; % Adjust this value based on your data
            
            % Apply region growing segmentation
            segmentedImg = regiongrowing(imgGray, seedPoint(1), seedPoint(2), regionGrowingThreshold);
            
            % Classify the segmented image
            className = classifyImage(segmentedImg); % Replace with your actual classification function
            
            % Ensure the class name is valid and map it to a folder name
            if ismember(className, classNames)
                % Define the corresponding output subfolder
                outputSubFolderPath = fullfile(outputDir, className);
                
                % Define output file path in classification folder
                [~, name, ext] = fileparts(imgPath);
                outputFilePath = fullfile(outputSubFolderPath, [name '_classified.png']);
                
                % Save the segmented image to the classification folder
                imwrite(segmentedImg, outputFilePath);
                
                % Display the segmented image (Optional)
                if displayCounter < maxDisplayImages
                    figure('WindowState', 'maximized', 'Color', [0.95, 0.95, 0.95], 'Name', 'Classified Image', 'NumberTitle', 'off');
                    
                    imshow(segmentedImg, []);
                    title(['Classified Image: ', name, '_classified.png'], 'FontSize', 14, 'FontWeight', 'bold', 'Color', 'b', 'Interpreter', 'none');
                    
                    xlabel(['Folder: ', sceneFolders(i).name, ' | Subfolder: ', subFolders(j).name], 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'k', 'Interpreter', 'none');
                    
                    % Pause to view the image
                    pause(2); % Increase pause duration to 2 seconds
                    close(gcf); % Close the figure
                    
                    displayCounter = displayCounter + 1;
                end
            else
                warning('Class name "%s" not found in the predefined list.', className);
            end
        end
    end
end

disp('Classification completed and images saved successfully!');

% --- Region Growing Function ---
function segmented = regiongrowing(I, x, y, threshold)
    % Initialize the segmented region with zeros
    segmented = false(size(I));
    
    % Initialize the list of pixels to check
    pixelList = false(size(I));
    pixelList(x, y) = true;
    
    % Mean intensity of the region
    regionMean = I(x, y);
    
    % Number of pixels in the region
    regionSize = 1;
    
    % Connectivity (8-connected)
    connectivity = [ -1 -1; -1 0; -1 1; 0 -1; 0 1; 1 -1; 1 0; 1 1 ];
    
    while any(pixelList(:))
        [currentX, currentY] = find(pixelList, 1);
        pixelList(currentX, currentY) = false;
        segmented(currentX, currentY) = true;
        
        % Check neighboring pixels
        for i = 1:size(connectivity, 1)
            newX = currentX + connectivity(i, 1);
            newY = currentY + connectivity(i, 2);
            
            if newX > 0 && newX <= size(I, 1) && newY > 0 && newY <= size(I, 2) && ~segmented(newX, newY)
                intensityDifference = abs(I(newX, newY) - regionMean);
                
                if intensityDifference <= threshold
                    pixelList(newX, newY) = true;
                    regionMean = (regionMean * regionSize + I(newX, newY)) / (regionSize + 1);
                    regionSize = regionSize + 1;
                end
            end
        end
    end
end

% --- Placeholder Classification Function ---
function className = classifyImage(segmentedImg)
    % Replace this logic with your actual classification model
    classNames = {
        'Marine Debris', 'Floating Plastics', 'Oil', 'Dense Sargassum', ...
        'Sparse Floating Algae', 'Natural Organic Material', 'Ship', ...
        'Marine Water', 'Sediment-Laden Water', 'Foam', 'Turbid Water', ...
        'Shallow Water', 'Waves & Wakes', 'Oil Platform', 'Jellyfish', ...
        'Sea snot'
    };
    
    % Random classification for demonstration purposes
    className = classNames{randi(numel(classNames))};
end
