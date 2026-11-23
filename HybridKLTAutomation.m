%%% HybridKLTAutomation.m
% Custom Video Labeler Automation Algorithm
% Hybrid Blob Detection + Kalman Filter + KLT Point Tracker
% Configurable for mission-specific UAV, surveillance, and ground tracking profiles
%
% Developed and Revised by Zarif Bin Akhtar

classdef HybridKLTAutomation < vision.labeler.AutomationAlgorithm & vision.labeler.mixin.Temporal

    %----------------------------------------------------------------------
    % STEP 1: ALGORITHM METADATA
    %----------------------------------------------------------------------
    properties(Constant)
        Name = 'Hybrid KLT Tracker';
        Description = 'Mission-Tunable Blob Detection + Kalman Filter + KLT Point Tracker | Zarif Bin Akhtar';
        
        UserDirections = {...
            '1. Create a Rectangle ROI label', ...
            '2. Draw a bounding box around the target in the first frame', ...
            '3. (Optional) Click Settings to select a mission preset or tune parameters', ...
            '4. Click Automate to run tracking'};
    end

    %----------------------------------------------------------------------
    % STEP 2: MISSION-TUNABLE ALGORITHM PROPERTIES
    %----------------------------------------------------------------------
    properties
        % --- Mission Preset Profile ---
        MissionProfile = 'Custom';
        
        % --- Morphological & Blob Detection Parameters ---
        filterSize = 3;                 % Morphological opening kernel size (noise removal)
        second_filterSize = 25;         % Morphological closing kernel size (gap bridging)
        minBlobArea = 200;              % Minimum blob area in pixels
        NumGaussians = 3;               % GMM background model components
        NumTrainingFrames = 40;         % Background model learning frames
        MinimumBackgroundRatio = 0.7;   % Background threshold ratio
        
        % --- Association Gate Parameter ---
        MaxAssociationDist = 200;       % Max Euclidean distance (px) between KF prediction & blob
        
        % --- vision.PointTracker (KLT) Parameters ---
        NumPyramidLevels = 3;           % Pyramid levels for multi-scale motion
        MaxBidirectionalError = 0.2;    % Forward-backward tracking error threshold (px)
        BlockSize = [31 31];            % Neighborhood block size [height width] (odd integers >= 5)
        MinFeatureQuality = 0.01;       % Minimum corner quality for detectMinEigenFeatures
        
        % --- Kalman Filter Parameters ---
        Qk = 1e-2;                      % Process noise covariance scalar (higher = agile maneuvers)
        Rk = 5e-2;                      % Measurement noise covariance scalar (higher = trust prediction more)
        
        % --- Tracking State (Internal) ---
        KalmanState = [];
        KalmanCov = [];
        BBoxSize = [];
        FrameCount = 0;
        LossCount = 0;
        Initialized = false;
        LabelName = '';
        PointTracker = [];
        TrackedPoints = [];
        InitialROI = [];
        
        % --- Blob Detection Objects ---
        fgDetector = [];
        blobStats = [];
    end

    %----------------------------------------------------------------------
    % STEP 3: SETUP & PRESET METHODS
    %----------------------------------------------------------------------
    methods
        function applyMissionPreset(algObj, profileName)
            % Configure tracker parameters for common operational scenarios
            algObj.MissionProfile = profileName;
            switch profileName
                case 'Small / High-Altitude UAV'
                    algObj.minBlobArea = 40;
                    algObj.filterSize = 2;
                    algObj.second_filterSize = 15;
                    algObj.MaxAssociationDist = 120;
                    algObj.NumPyramidLevels = 4;
                    algObj.MaxBidirectionalError = 0.5;
                    algObj.BlockSize = [21 21];
                    algObj.MinFeatureQuality = 0.005;
                    algObj.Qk = 5e-2;
                    algObj.Rk = 2e-2;
                    
                case 'Fast Agile / Low-Altitude Drone'
                    algObj.minBlobArea = 150;
                    algObj.filterSize = 3;
                    algObj.second_filterSize = 25;
                    algObj.MaxAssociationDist = 300;
                    algObj.NumPyramidLevels = 4;
                    algObj.MaxBidirectionalError = 1.0;
                    algObj.BlockSize = [41 41];
                    algObj.MinFeatureQuality = 0.01;
                    algObj.Qk = 1e-1;
                    algObj.Rk = 5e-2;
                    
                case 'Ground Vehicle / Pedestrian'
                    algObj.minBlobArea = 400;
                    algObj.filterSize = 5;
                    algObj.second_filterSize = 35;
                    algObj.MaxAssociationDist = 180;
                    algObj.NumPyramidLevels = 3;
                    algObj.MaxBidirectionalError = 0.2;
                    algObj.BlockSize = [31 31];
                    algObj.MinFeatureQuality = 0.01;
                    algObj.Qk = 1e-2;
                    algObj.Rk = 5e-2;
                    
                case 'Cluttered / Noisy Background'
                    algObj.minBlobArea = 250;
                    algObj.filterSize = 5;
                    algObj.second_filterSize = 30;
                    algObj.MaxAssociationDist = 150;
                    algObj.NumPyramidLevels = 3;
                    algObj.MaxBidirectionalError = 0.2;
                    algObj.BlockSize = [31 31];
                    algObj.MinFeatureQuality = 0.02;
                    algObj.Qk = 1e-2;
                    algObj.Rk = 1e-1;
            end
        end

        function isValid = checkLabelDefinition(~, labelDef)
            isValid = (labelDef.Type == labelType.Rectangle);
        end

        function isReady = checkSetup(algObj, labelsToAutomate)
            isReady = false;
            
            if isempty(labelsToAutomate)
                disp('[KLT] No labels found');
                return;
            end
            
            if ~istable(labelsToAutomate)
                disp('[KLT] labelsToAutomate is not a table');
                return;
            end
            
            varNames = labelsToAutomate.Properties.VariableNames;
            posColIdx = find(strcmp(varNames, 'Position'));
            
            if isempty(posColIdx)
                disp('[KLT] No "Position" column found');
                return;
            end
            
            try
                positionData = labelsToAutomate{1, posColIdx};
            catch
                disp('[KLT] Could not read position data');
                return;
            end
            
            if isnumeric(positionData) && numel(positionData) == 4
                algObj.InitialROI = positionData;
                isReady = true;
                
                nameColIdx = find(strcmp(varNames, 'Name'));
                if ~isempty(nameColIdx)
                    try
                        nameData = labelsToAutomate{1, nameColIdx};
                        if iscell(nameData)
                            algObj.LabelName = nameData{1};
                        else
                            algObj.LabelName = char(nameData);
                        end
                    catch
                        algObj.LabelName = 'ROI';
                    end
                else
                    algObj.LabelName = 'ROI';
                end
                
                algObj.BBoxSize = [positionData(3), positionData(4)];
                disp(['[KLT] Ready. Label: ' algObj.LabelName]);
                disp(['[KLT] Initial ROI: [' num2str(algObj.InitialROI) ']']);
                return;
            end
        end

        function settingsDialog(algObj)
            dlg = figure('Name', 'Hybrid KLT Mission Settings', ...
                        'Position', [100, 100, 480, 520], ...
                        'MenuBar', 'none', ...
                        'NumberTitle', 'off');
            
            presets = {'Custom', ...
                       'Small / High-Altitude UAV', ...
                       'Fast Agile / Low-Altitude Drone', ...
                       'Ground Vehicle / Pedestrian', ...
                       'Cluttered / Noisy Background'};
            presetIdx = find(strcmp(presets, algObj.MissionProfile));
            if isempty(presetIdx), presetIdx = 1; end
            
            y_pos = 475;
            uicontrol('Style', 'text', 'Position', [20, y_pos, 180, 22], ...
                     'String', 'Mission Preset:', 'FontWeight', 'bold', 'HorizontalAlignment', 'left');
            h_preset = uicontrol('Style', 'popupmenu', 'Position', [210, y_pos, 240, 25], ...
                                'String', presets, 'Value', presetIdx, ...
                                'Callback', @(src, ~) onPresetChange(src));
            
            y_pos = y_pos - 40;
            uicontrol('Style', 'text', 'Position', [20, y_pos, 180, 20], ...
                     'String', 'Min Blob Area (px):', 'HorizontalAlignment', 'left');
            h_min = uicontrol('Style', 'edit', 'Position', [210, y_pos, 100, 25], ...
                             'String', num2str(algObj.minBlobArea));
            
            y_pos = y_pos - 36;
            uicontrol('Style', 'text', 'Position', [20, y_pos, 180, 20], ...
                     'String', 'Open Filter Size (px):', 'HorizontalAlignment', 'left');
            h_filt = uicontrol('Style', 'edit', 'Position', [210, y_pos, 100, 25], ...
                             'String', num2str(algObj.filterSize));
            
            y_pos = y_pos - 36;
            uicontrol('Style', 'text', 'Position', [20, y_pos, 180, 20], ...
                     'String', 'Close Filter Size (px):', 'HorizontalAlignment', 'left');
            h_filt2 = uicontrol('Style', 'edit', 'Position', [210, y_pos, 100, 25], ...
                              'String', num2str(algObj.second_filterSize));
            
            y_pos = y_pos - 36;
            uicontrol('Style', 'text', 'Position', [20, y_pos, 180, 20], ...
                     'String', 'Max Association Dist (px):', 'HorizontalAlignment', 'left');
            h_dist = uicontrol('Style', 'edit', 'Position', [210, y_pos, 100, 25], ...
                             'String', num2str(algObj.MaxAssociationDist));
            
            y_pos = y_pos - 36;
            uicontrol('Style', 'text', 'Position', [20, y_pos, 180, 20], ...
                     'String', 'KLT Pyramid Levels:', 'HorizontalAlignment', 'left');
            h_pyr = uicontrol('Style', 'edit', 'Position', [210, y_pos, 100, 25], ...
                             'String', num2str(algObj.NumPyramidLevels));
            
            y_pos = y_pos - 36;
            uicontrol('Style', 'text', 'Position', [20, y_pos, 180, 20], ...
                     'String', 'Bidirectional Error (px):', 'HorizontalAlignment', 'left');
            h_bidir = uicontrol('Style', 'edit', 'Position', [210, y_pos, 100, 25], ...
                               'String', num2str(algObj.MaxBidirectionalError));
            
            y_pos = y_pos - 36;
            uicontrol('Style', 'text', 'Position', [20, y_pos, 180, 20], ...
                     'String', 'KLT Block Size (odd):', 'HorizontalAlignment', 'left');
            h_block = uicontrol('Style', 'edit', 'Position', [210, y_pos, 100, 25], ...
                               'String', num2str(algObj.BlockSize(1)));
            
            y_pos = y_pos - 36;
            uicontrol('Style', 'text', 'Position', [20, y_pos, 180, 20], ...
                     'String', 'Min Feature Quality:', 'HorizontalAlignment', 'left');
            h_qual = uicontrol('Style', 'edit', 'Position', [210, y_pos, 100, 25], ...
                             'String', num2str(algObj.MinFeatureQuality));
            
            y_pos = y_pos - 36;
            uicontrol('Style', 'text', 'Position', [20, y_pos, 180, 20], ...
                     'String', 'Process Noise (Q):', 'HorizontalAlignment', 'left');
            h_Q = uicontrol('Style', 'edit', 'Position', [210, y_pos, 100, 25], ...
                          'String', num2str(algObj.Qk));
            
            y_pos = y_pos - 36;
            uicontrol('Style', 'text', 'Position', [20, y_pos, 180, 20], ...
                     'String', 'Measurement Noise (R):', 'HorizontalAlignment', 'left');
            h_R = uicontrol('Style', 'edit', 'Position', [210, y_pos, 100, 25], ...
                          'String', num2str(algObj.Rk));
            
            uicontrol('Style', 'pushbutton', 'Position', [180, 25, 120, 32], ...
                     'String', 'Apply Settings', 'FontWeight', 'bold', ...
                     'Callback', @(~,~) applySettings);
            
            function onPresetChange(src)
                selected = presets{get(src, 'Value')};
                if ~strcmp(selected, 'Custom')
                    algObj.applyMissionPreset(selected);
                    set(h_min, 'String', num2str(algObj.minBlobArea));
                    set(h_filt, 'String', num2str(algObj.filterSize));
                    set(h_filt2, 'String', num2str(algObj.second_filterSize));
                    set(h_dist, 'String', num2str(algObj.MaxAssociationDist));
                    set(h_pyr, 'String', num2str(algObj.NumPyramidLevels));
                    set(h_bidir, 'String', num2str(algObj.MaxBidirectionalError));
                    set(h_block, 'String', num2str(algObj.BlockSize(1)));
                    set(h_qual, 'String', num2str(algObj.MinFeatureQuality));
                    set(h_Q, 'String', num2str(algObj.Qk));
                    set(h_R, 'String', num2str(algObj.Rk));
                end
            end
            
            function applySettings
                algObj.MissionProfile = presets{get(h_preset, 'Value')};
                algObj.minBlobArea = str2double(get(h_min, 'String'));
                algObj.filterSize = str2double(get(h_filt, 'String'));
                algObj.second_filterSize = str2double(get(h_filt2, 'String'));
                algObj.MaxAssociationDist = str2double(get(h_dist, 'String'));
                algObj.NumPyramidLevels = str2double(get(h_pyr, 'String'));
                algObj.MaxBidirectionalError = str2double(get(h_bidir, 'String'));
                blk = max(5, round(str2double(get(h_block, 'String'))));
                if mod(blk, 2) == 0, blk = blk + 1; end
                algObj.BlockSize = [blk, blk];
                algObj.MinFeatureQuality = str2double(get(h_qual, 'String'));
                algObj.Qk = str2double(get(h_Q, 'String'));
                algObj.Rk = str2double(get(h_R, 'String'));
                close(dlg);
            end
            
            uiwait(dlg);
        end
    end

    %----------------------------------------------------------------------
    % STEP 4: ALGORITHM EXECUTION
    %----------------------------------------------------------------------
    methods
        function initialize(algObj, I, ~)
            if isempty(algObj.InitialROI)
                error('[KLT] No initial ROI found');
            end
            
            bbox = algObj.InitialROI;
            algObj.BBoxSize = [bbox(3), bbox(4)];
            
            % --- Initialize Kalman Filter ---
            centerX = bbox(1) + bbox(3)/2;
            centerY = bbox(2) + bbox(4)/2;
            algObj.KalmanState = double([centerX; centerY; 0; 0]);
            algObj.KalmanCov = 1e-1 * eye(4);
            
            % --- Initialize Blob Detection ---
            algObj.fgDetector = vision.ForegroundDetector(...
                'NumGaussians', algObj.NumGaussians, ...
                'NumTrainingFrames', algObj.NumTrainingFrames, ...
                'MinimumBackgroundRatio', algObj.MinimumBackgroundRatio);
            algObj.blobStats = vision.BlobAnalysis(...
                'BoundingBoxOutputPort', true, ...
                'AreaOutputPort', true, ...
                'CentroidOutputPort', true, ...
                'MinimumBlobArea', algObj.minBlobArea);
            
            % --- Initialize PointTracker (Wired to user parameters) ---
            if size(I, 3) == 3
                I_gray = rgb2gray(I);
            else
                I_gray = I;
            end
            
            initialPoints = detectMinEigenFeatures(I_gray, 'ROI', bbox, ...
                'MinQuality', algObj.MinFeatureQuality, 'FilterSize', 3);
            if initialPoints.Count < 10
                initialPoints = detectMinEigenFeatures(I_gray, 'ROI', bbox, ...
                    'MinQuality', algObj.MinFeatureQuality * 0.5, 'FilterSize', 3);
            end
            algObj.TrackedPoints = initialPoints.Location;
            
            algObj.PointTracker = vision.PointTracker(...
                'NumPyramidLevels', algObj.NumPyramidLevels, ...
                'MaxBidirectionalError', algObj.MaxBidirectionalError, ...
                'BlockSize', algObj.BlockSize);
            initialize(algObj.PointTracker, algObj.TrackedPoints, I);
            
            algObj.FrameCount = 0;
            algObj.LossCount = 0;
            algObj.Initialized = true;
            
            disp(['[KLT] Initialized (' algObj.MissionProfile ') | Label: ' algObj.LabelName]);
            disp(['[KLT] Center: (' num2str(centerX) ', ' num2str(centerY) ')']);
            disp(['[KLT] Tracking ' num2str(size(algObj.TrackedPoints,1)) ' KLT points']);
        end

        function autoLabels = run(algObj, I)
            autoLabels = [];
            
            if ~algObj.Initialized
                return;
            end
            
            algObj.FrameCount = algObj.FrameCount + 1;
            if size(I, 3) == 3
                I_gray = rgb2gray(I);
            else
                I_gray = I;
            end
            
            % =============================================================
            % PART 1: Kalman Prediction
            % =============================================================
            dt = 1;
            F = [1 0 dt 0; 0 1 0 dt; 0 0 1 0; 0 0 0 1];
            H = [1 0 0 0; 0 1 0 0];
            Q = algObj.Qk * eye(4);
            R = algObj.Rk * eye(2);
            
            xPred = F * algObj.KalmanState;
            P = F * algObj.KalmanCov * F' + Q;
            predCent = xPred(1:2)';
            
            % =============================================================
            % PART 2: Blob Detection & Morphological Filtering
            % =============================================================
            fgMask = step(algObj.fgDetector, I_gray);
            fgMask = imopen(fgMask, strel('rectangle', [algObj.filterSize, algObj.filterSize]));
            fgMask = imclose(fgMask, strel('rectangle', [algObj.second_filterSize, algObj.second_filterSize]));
            fgMask = imfill(fgMask, 'holes');
            
            [~, centroids, bboxes] = step(algObj.blobStats, fgMask);
            
            detectedCent = [];
            valid = false;
            method = "Prediction";
            bboxCur = [];
            wB = algObj.BBoxSize(1);
            hB = algObj.BBoxSize(2);
            
            if ~isempty(centroids)
                centroids = double(centroids);
                bboxes = double(bboxes);
                dists = sqrt(sum((centroids - predCent).^2, 2));
                [minD, idxMin] = min(dists);
                
                if minD < algObj.MaxAssociationDist
                    detectedCent = centroids(idxMin, :);
                    bboxCur = bboxes(idxMin, :);
                    valid = true;
                    method = "Blob + KF";
                    wB = bboxCur(3);
                    hB = bboxCur(4);
                    
                    % Update KLT points on the detected blob
                    [~, isFound] = algObj.PointTracker(I);
                    if sum(isFound) < 5
                        pts_new = detectMinEigenFeatures(I_gray, 'ROI', bboxCur, ...
                            'MinQuality', algObj.MinFeatureQuality, 'FilterSize', 3);
                        if pts_new.Count > 10
                            release(algObj.PointTracker);
                            algObj.PointTracker = vision.PointTracker(...
                                'NumPyramidLevels', algObj.NumPyramidLevels, ...
                                'MaxBidirectionalError', algObj.MaxBidirectionalError, ...
                                'BlockSize', algObj.BlockSize);
                            initialize(algObj.PointTracker, pts_new.Location, I);
                        end
                    end
                end
            end
            
            % =============================================================
            % PART 3: Kalman Update
            % =============================================================
            if valid && ~isempty(detectedCent)
                z = double(detectedCent(:));
                yk = z - H * xPred;
                S = H * P * H' + R;
                K = P * H' / S;
                xHat = xPred + K * yk;
                P = (eye(4) - K * H) * P;
                algObj.LossCount = 0;
            else
                xHat = xPred;
                bboxCur = [xHat(1)-wB/2, xHat(2)-hB/2, wB, hB];
                algObj.LossCount = algObj.LossCount + 1;
            end
            
            algObj.KalmanState = xHat;
            algObj.KalmanCov = P;
            
            % =============================================================
            % PART 4: Generate Bounding Box
            % =============================================================
            centerX = xHat(1);
            centerY = xHat(2);
            
            if ~isempty(bboxCur) && valid
                w = bboxCur(3);
                h = bboxCur(4);
            else
                w = max(10, algObj.BBoxSize(1));
                h = max(10, algObj.BBoxSize(2));
            end
            
            bbox = [centerX - w/2, centerY - h/2, w, h];
            
            if isempty(algObj.LabelName)
                algObj.LabelName = 'ROI';
            end
            
            autoLabels = struct('Name', algObj.LabelName, ...
                               'Type', labelType.Rectangle, ...
                               'Position', bbox);
            
            % Debug output every 50 frames
            if mod(algObj.FrameCount, 50) == 0
                fprintf('[KLT] Frame %d: (%.1f, %.1f) | %s | Loss: %d
', ...
                    algObj.FrameCount, centerX, centerY, method, algObj.LossCount);
            end
        end

        function terminate(algObj)
            if ~isempty(algObj.PointTracker)
                release(algObj.PointTracker);
            end
            algObj.fgDetector = [];
            algObj.blobStats = [];
            disp('[KLT] Tracker terminated');
        end
    end
end
