% Define the main directory where all Scene folders are located
mainDir = 'D:\Advanced Deep Learning Framework AquaSense\MADOS';

% Define the output directory where processed images will be saved
outputDir = 'D:\Advanced Deep Learning Framework AquaSense\Data_Preprocessing';

% Create the output directory if it does not exist
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% List all Scene directories (Scene_0 to Scene_173)
sceneFolders = dir(fullfile(mainDir, 'Scene_*'));

% Initialize a counter to display only 10 images
imageDisplayCounter = 0;
maxDisplayImages = 10; % Maximum number of images to display

% Loop through each Scene directory
for i = 1:length(sceneFolders)
    scenePath = fullfile(sceneFolders(i).folder, sceneFolders(i).name);
    
    % List all subfolders (e.g., 10, 20, 60) within the Scene directory
    subFolders = dir(fullfile(scenePath, '*'));
    subFolders = subFolders([subFolders.isdir]); % Only directories
    
    % Loop through each subfolder
    for j = 1:length(subFolders)
        % Ignore '.' and '..' directories
        if strcmp(subFolders(j).name, '.') || strcmp(subFolders(j).name, '..')
            continue;
        end
        
        subFolderPath = fullfile(subFolders(j).folder, subFolders(j).name);
        
        % List all .tif files in the current subfolder
        tifFiles = dir(fullfile(subFolderPath, '*.tif'));

        % Loop through each .tif file and apply pre-processing
        for k = 1:length(tifFiles)
            % Read the image
            imgPath = fullfile(tifFiles(k).folder, tifFiles(k).name);
            img = imread(imgPath);
            
            % Step 1: Normalization
            imgNorm = mat2gray(img);

            % Step 2: Histogram Equalization
            imgHistEq = histeq(imgNorm);

            % Step 3: Noise Reduction (e.g., using a median filter)
            imgNoiseRed = medfilt2(imgHistEq);

            % Step 4: Bilateral Filtering
            imgBilateral = imbilatfilt(imgNoiseRed);

            % Define the corresponding output subfolder
            outputSubFolderPath = fullfile(outputDir, sceneFolders(i).name, subFolders(j).name);

            % Create the output subfolder if it does not exist
            if ~exist(outputSubFolderPath, 'dir')
                mkdir(outputSubFolderPath);
            end
            
            % Define output file path
            [~, name, ext] = fileparts(imgPath);
            outputFilePath = fullfile(outputSubFolderPath, [name '_processed' ext]);
            
            % Save the processed image
            imwrite(imgBilateral, outputFilePath);

            % Display the first 10 images only
            if imageDisplayCounter < maxDisplayImages
                % Increment the image display counter
                imageDisplayCounter = imageDisplayCounter + 1;

                % Visualization: Display the original and preprocessed image side by side
                figure('WindowState', 'maximized', 'Color', [0.95, 0.95, 0.95]);
                
                % Original Image
                subplot(1, 2, 1);
                imshow(img);
                title('Original Image', 'FontSize', 14, 'FontWeight', 'bold', 'Color', 'b'); % Blue title

                % Add folder and image name on the original image
                annotation('textbox', [0.1, 0.9, 0.8, 0.05], 'String', ...
                           ['Folder: ' sceneFolders(i).name ' | Subfolder: ' subFolders(j).name ' | Image: ' tifFiles(k).name], ...
                           'FitBoxToText', 'on', 'BackgroundColor', 'w', 'FontSize', 12, 'FontWeight', 'bold', 'EdgeColor', 'b', 'Interpreter', 'none');

                % Preprocessed Image
                subplot(1, 2, 2);
                imshow(imgBilateral);
                title('Preprocessed Image', 'FontSize', 14, 'FontWeight', 'bold', 'Color', 'r'); % Red title

                % Add folder and image name on the preprocessed image
                annotation('textbox', [0.55, 0.9, 0.8, 0.05], 'String', ...
                           ['Folder: ' sceneFolders(i).name ' | Subfolder: ' subFolders(j).name ' | Image: ' [name '_processed' ext]], ...
                           'FitBoxToText', 'on', 'BackgroundColor', 'w', 'FontSize', 12, 'FontWeight', 'bold', 'EdgeColor', 'r', 'Interpreter', 'none');

                % Set attractive font and color for all figure elements
                set(gca, 'FontName', 'Helvetica', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'k');
                
                % Pause to view the images (adjust the pause duration as needed)
                pause(2); % Pause for 2 seconds before closing the figure
                close(gcf); % Close the figure
            end
        end
    end
end

disp('Pre-processing completed for all images, and the first 10 images were displayed!');
