% Define the main directory where all Scene folders are located
mainDir = 'D:\Advanced Deep Learning Framework AquaSense\Data_Preprocessing';

% Define the output directory where processed images will be saved
outputDir = 'D:\Advanced Deep Learning Framework AquaSense\Annotate_Data';

% Create the output directory if it does not exist
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% List all Scene directories
sceneFolders = dir(fullfile(mainDir, 'Scene_*'));

% Initialize a counter for the first 10 images
imageCounter = 0;
maxImages = 10; % Maximum number of images to display

% List to keep track of processed image paths for visualization
processedImages = {};

% Loop through each Scene directory
for i = 1:length(sceneFolders)
    scenePath = fullfile(sceneFolders(i).folder, sceneFolders(i).name);
    
    % List all subfolders (e.g., 10, 20, 60)
    subFolders = dir(fullfile(scenePath, '*'));
    subFolders = subFolders([subFolders.isdir]); % Only directories

    % Loop through each subfolder
    for j = 1:length(subFolders)
        subFolderPath = fullfile(subFolders(j).folder, subFolders(j).name);
        
        % List all .tif files in the current subfolder
        tifFiles = dir(fullfile(subFolderPath, '*.tif'));

        % Loop through each .tif file
        for k = 1:length(tifFiles)
            % Read the image
            imgPath = fullfile(tifFiles(k).folder, tifFiles(k).name);
            img = imread(imgPath);
            
            % Convert image to uint8 if it is not already
            if ~isa(img, 'uint8')
                img = im2uint8(img);
            end
            
            % Define the corresponding output subfolder
            outputSubFolderPath = fullfile(outputDir, sceneFolders(i).name, subFolders(j).name);

            % Create the output subfolder if it does not exist
            if ~exist(outputSubFolderPath, 'dir')
                mkdir(outputSubFolderPath);
            end
            
            % Define output file path
            [~, name, ext] = fileparts(imgPath);
            outputFilePath = fullfile(outputSubFolderPath, [name '_annotated' ext]);

            % Load and apply annotation mask if it exists
            annotationMaskPath = fullfile(subFolderPath, [name '_mask' ext]);
            if isfile(annotationMaskPath)
                annotationMask = imread(annotationMaskPath);
                % Ensure the mask is uint8
                if ~isa(annotationMask, 'uint8')
                    annotationMask = im2uint8(annotationMask);
                end
                % Apply the annotation mask
                img = img .* uint8(annotationMask); % Example application of mask
            end
            
            % Save the annotated image using Tiff class
            tiff = Tiff(outputFilePath, 'w');
            tagstruct.ImageLength = size(img, 1);
            tagstruct.ImageWidth = size(img, 2);
            tagstruct.Photometric = Tiff.Photometric.MinIsBlack;
            tagstruct.BitsPerSample = 8;
            tagstruct.SamplesPerPixel = 1;
            tagstruct.SampleFormat = Tiff.SampleFormat.UInt;
            tagstruct.Compression = Tiff.Compression.None;
            tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
            tiff.setTag(tagstruct);
            tiff.write(img);
            tiff.close();

            % Keep track of processed images for visualization
            imageCounter = imageCounter + 1;
            if imageCounter <= maxImages
                processedImages{end+1} = outputFilePath;
            end
        end
    end
end
for i = 1:min(maxImages, numel(processedImages))
    % Read the processed image
    img = imread(processedImages{i});
    
    % Display the image
    figure('WindowState', 'maximized', 'Color', [0.95, 0.95, 0.95]);
    imshow(img);
    title(['Processed Image ' num2str(i)], 'FontSize', 14, 'FontWeight', 'bold', 'Color', 'b'); % Blue title

    % Pause to view the image (adjust the pause duration as needed)
    pause(2); % Pause for 2 seconds before closing the figure
    close(gcf); % Close the figure
end

disp('Image annotation completed and images saved!');
