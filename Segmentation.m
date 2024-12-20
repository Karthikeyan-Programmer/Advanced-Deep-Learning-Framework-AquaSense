% Define the main directory where all Scene folders are located
mainDir = 'D:\Advanced Deep Learning Framework AquaSense\Annotate_Data';

% Define the output directory where processed images will be saved
outputDir = 'D:\Advanced Deep Learning Framework AquaSense\Segmented_Data';

% Create the output directory if it does not exist
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% List all Scene directories
sceneFolders = dir(fullfile(mainDir, 'Scene_*'));
sceneFolders = sceneFolders([sceneFolders.isdir]); % Ensure only directories are selected

% Initialize a counter to control the number of displayed images
displayCounter = 0;
maxDisplayImages = 10; % Maximum number of images to display

% Loop through each Scene directory
for i = 1:length(sceneFolders)
    scenePath = fullfile(sceneFolders(i).folder, sceneFolders(i).name);
    
    % List all subfolders within the current Scene directory
    subFolders = dir(scenePath);
    subFolders = subFolders([subFolders.isdir] & ~ismember({subFolders.name}, {'.', '..'})); % Exclude '.' and '..'
    
    % Loop through each subfolder
    for j = 1:length(subFolders)
        subFolderPath = fullfile(subFolders(j).folder, subFolders(j).name);
        
        % List all .tif files in the current subfolder
        tifFiles = dir(fullfile(subFolderPath, '*.tif'));
        
        % Loop through each .tif file and apply segmentation
        for k = 1:length(tifFiles)
            % Read the image
            imgPath = fullfile(tifFiles(k).folder, tifFiles(k).name);
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
            
            % Define the corresponding output subfolder
            outputSubFolderPath = fullfile(outputDir, sceneFolders(i).name, subFolders(j).name);
            
            % Create the output subfolder if it does not exist
            if ~exist(outputSubFolderPath, 'dir')
                mkdir(outputSubFolderPath);
            end
            
            % Define output file path
            [~, name, ext] = fileparts(imgPath);
            outputFilePath = fullfile(outputSubFolderPath, [name '_segmented.png']);
            
            % Save the segmented image
            imwrite(segmentedImg, outputFilePath);
            
            % Display the segmented image only for the first 10 images
            if displayCounter < maxDisplayImages
                figure('WindowState', 'maximized','Color', [0.95, 0.95, 0.95], 'Name', 'Region Growing Segmentation', 'NumberTitle', 'off');
                
                imshow(segmentedImg, []);
                title(['Segmented Image: ', name, '_segmented.png'], 'FontSize', 14, 'FontWeight', 'bold', 'Color', 'b', 'Interpreter', 'none');
                
                xlabel(['Folder: ', sceneFolders(i).name, ' | Subfolder: ', subFolders(j).name], 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'k', 'Interpreter', 'none');
                
                % Pause to view the image
                pause(2); % Pause for 2 seconds
                close(gcf); % Close the figure
                
                displayCounter = displayCounter + 1;
            end
        end
    end
end

disp('Segmentation completed and images saved successfully!');

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
disp('Segmentation id completed and images saved!');