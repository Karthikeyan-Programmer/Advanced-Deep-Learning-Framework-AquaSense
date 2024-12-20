% Define the path to your image data
imageFolderPath = 'D:\Advanced Deep Learning Framework AquaSense';

% Define valid class names
classNames = {
    'Marine_Debris', 'Floating_Plastics', 'Oil', 'Dense_Sargassum', ...
    'Sparse_Floating_Algae', 'Natural_Organic_Material', 'Ship', ...
    'Marine_Water', 'Sediment_Laden_Water', 'Foam', 'Turbid_Water', ...
    'Shallow_Water', 'Waves_and_Wakes', 'Oil_Platform', 'Jellyfish', ...
    'Sea_snot'
};

% Define the path to the image and label folders
imageFolder = fullfile(imageFolderPath, 'SyntheticImages'); % Assuming the images are in this folder
labelFolder = fullfile(imageFolderPath, 'SyntheticLabels'); % Assuming the labels are in this folder

% Create an imageDatastore for images
imds = imageDatastore(imageFolder, 'FileExtensions', '.png', 'LabelSource', 'foldernames');

% Define a mapping of class names to label IDs
% You need to match these IDs to the pixel values used in your label images
classNames = matlab.lang.makeValidName(classNames); % Ensure class names are valid MATLAB variable names
labelIDs = 1:length(classNames); % Assuming labels are numbered 1, 2, 3, ..., length(classNames)
labelIDs = uint8(labelIDs);


% Define the network architecture
lgraph = layerGraph([
    imageInputLayer([224 224 3], 'Name', 'input', 'Normalization', 'none')
    convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'conv1')
    reluLayer('Name', 'relu1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool1')
    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv2')
    reluLayer('Name', 'relu2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool2')
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv3')
    reluLayer('Name', 'relu3')
    averagePooling2dLayer(7, 'Stride', 1, 'Name', 'avgpool')
    fullyConnectedLayer(length(classNames), 'Name', 'fc', 'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10)
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'classification')
]);

% Display the network architecture
analyzeNetwork(lgraph);

% Define training options
options = trainingOptions('adam', ...
    'MaxEpochs', 4, ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 1e-3, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

% Define the training and validation datastores
% Split data into training and validation sets
[trainImds, valImds] = splitEachLabel(imds, 0.8, 'randomize');

% Create an augmented image datastore for training data
augmentedTrainImds = augmentedImageDatastore([224 224], trainImds, 'ColorPreprocessing', 'gray2rgb');
augmentedValImds = augmentedImageDatastore([224 224], valImds, 'ColorPreprocessing', 'gray2rgb');
