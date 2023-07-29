clear();
labelInfo           =       dir("/mnt/hdd2/pill_data/data/train/label/");
labelInfo           =       labelInfo(3:length(labelInfo)); % number of classes : 4023

train_image         =       strings(48276,1);
train_label         =       cell(48276,1);


valid_image         =       strings(16092,1);
valid_label         =       cell(16092,1);


test_image          =       strings(16092,1);
test_label          =       cell(16092,1);
train_index         =       0;
valid_index         =       0;
test_index          =       0;

for i = 1 : length(labelInfo)

    labelfiles              =           dir(fullfile(labelInfo(i).folder, labelInfo(i).name));
    labelfiles              =           labelfiles(3:22);

    for j = 1 : length(labelfiles)

        labelFile           =           fullfile(labelfiles(j).folder,labelfiles(j).name);
        txt                 =           fileread(labelFile);
        labelsJSON          =           jsondecode(txt);
        fields              =           [fieldnames(labelsJSON)];
        label               =           labelsJSON.(fields{1}).item_seq;


        if j <= 16

            train_index                     =       train_index     +       1;
            train_image(train_index)        =       "/mnt/hdd2/pill_data/data/train/data/"...
                                            +       labelInfo(i).name(1:length(labelInfo(i).name)-5)...
                                            +       "/"+labelfiles(j).name(1:length(labelfiles(j).name)-5)...
                                            +       ".png";
            train_label{train_index}        =       string(label);
        
        else

            valid_index                     =       valid_index     +       1;
            valid_image(valid_index)        =       "/mnt/hdd2/pill_data/data/train/data/"...
                                            +       labelInfo(i).name(1:length(labelInfo(i).name)-5)...
                                            +       "/"+labelfiles(j).name(1:length(labelfiles(j).name)-5)...
                                            +       ".png";
            valid_label{valid_index}        =       string(label);

   
        end
    end
end

train_images            =       imageDatastore(train_image(1:64368),LabelSource="foldernames");
train_images.ReadFcn    =       @customreader;

valid_images            =       imageDatastore(valid_image(1:16092),LabelSource="foldernames");
valid_images.ReadFcn    =       @customreader;


deepNetworkDesigner

function data = customreader(filename)
onState = warning('off', 'backtrace'); 
c = onCleanup(@() warning(onState)); 
data = imread(filename); % added lines: 
data = data(:,:,min(1:3, end)); 
data = imcrop(data,[188,340,600,600]);
data = imresize(data,[224 224]);
end


function data = augmentData(A)
    
    data = cell(size(A));
    for ii = 1:size(A,1)
        I = A{ii,1};
        labels = A{ii,2};
        sz = size(I);
        
        tform = Affine2d('YReflection', true, ...          
                                'Scale',[0.5 1.1], ...                           
                                'Rotation',[-20 20]);   
        rout = affineOutputView(sz,tform,'BoundsStyle', "centerOutput");
        I = imwarp(I,tform,'OutputView', rout);
        
        data(ii,:) = {I,labels};
    end
end
