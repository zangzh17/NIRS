function normalizedImg = adjust_image(img)
% adjust_image Enhanced slice viewer, supporting 2D/3D images
%   normalizedImg = adjust_image(img) displays an interactive image viewer,
%   supporting slice selection, contrast adjustment, gamma correction, etc.
%
%   Input:
%       img - 2D or 3D image array, for 3D images, 3rd dimension is slices
%
%   Output:
%       normalizedImg - Normalized image after current clip and gamma correction,
%                       range [0,1]

% Check input
if ~isnumeric(img)
    error('Input must be a numeric array');
end

% Determine image dimensions
[imgHeight, imgWidth, numSlices] = size(img);
is3D = (numSlices > 1);

if ~is3D
    numSlices = 1; % Treat 2D image as single slice
end

% Convert to double for processing
img = double(img);

% Calculate image min, max, and initial range
imgMin = min(img(:));
imgMax = max(img(:));
imgRange = imgMax - imgMin;

% Initialize parameters
currentSlice = 1;
clipLow = 0; % Normalized lower limit (0-1)
clipHigh = 1; % Normalized upper limit (0-1)
gammaValue = 1; % Initial gamma value
logScale = false; % Initialize logarithmic scale for histogram to off

% Create main figure window
mainFig = figure('Name', 'Enhanced Slice Viewer', ...
                 'NumberTitle', 'off', ...
                 'Position', [100, 100, 900, 700], ... 
                 'CloseRequestFcn', @closeFigure);

% Create panel layout
mainPanel = uipanel('Parent', mainFig, ...
                   'Units', 'normalized', ...
                   'Position', [0, 0, 1, 1]);
               
% Create image display area
imgPanel = uipanel('Parent', mainPanel, ...
                  'Units', 'normalized', ...
                  'Position', [0.05, 0.2, 0.6, 0.75]);
              
imgAxes = axes('Parent', imgPanel);

% Create histogram panel
histPanel = uipanel('Parent', mainPanel, ...
                   'Units', 'normalized', ...
                   'Position', [0.05, 0.05, 0.9, 0.15],...
                   'Title', 'Histogram');
                   
histAxes = axes('Parent', histPanel);

% FIXED APPROACH: Create controls using grid layout
% Create control panel with fixed layout positions 
ctrlPanel = uipanel('Parent', mainPanel, ...
                   'Units', 'normalized', ...
                   'Position', [0.7, 0.2, 0.25, 0.75]);

% Clear the control panel to start fresh - this ensures no hidden elements
delete(get(ctrlPanel, 'Children'));

% Create controls using a grid-based layout
% Create a vertical grid of positions
nRows = 22; % More rows for better spacing and new controls
rowHeight = 1/nRows;

% Function to calculate row position (from top)
rowPos = @(row) 1 - row*rowHeight;

% Row counter - start from row 1 (top)
row = 1;

% Panel title
uicontrol('Parent', ctrlPanel, ...
         'Style', 'text', ...
         'Units', 'normalized', ...
         'Position', [0.05, rowPos(row), 0.9, rowHeight], ...
         'String', 'Control Panel', ...
         'FontWeight', 'bold');
row = row + 1;

% Create slice slider (if 3D image)
if is3D
    % Section header
    row = row + 0.5; % Add some space
    uicontrol('Parent', ctrlPanel, ...
             'Style', 'text', ...
             'Units', 'normalized', ...
             'Position', [0.05, rowPos(row), 0.9, rowHeight], ...
             'String', 'Slice Selector');
    row = row + 1;
    
    % Slider
    sliceSlider = uicontrol('Parent', ctrlPanel, ...
                           'Style', 'slider', ...
                           'Units', 'normalized', ...
                           'Position', [0.1, rowPos(row), 0.8, rowHeight], ...
                           'Min', 1, 'Max', numSlices, ...
                           'Value', currentSlice, ...
                           'SliderStep', [1/(max(numSlices-1,1)), min(10/numSlices, 0.5)]);
    
    % Set up real-time response
    set(sliceSlider, 'Callback', @updateSlice);
    addlistener(sliceSlider, 'ContinuousValueChange', @updateSlice);
    row = row + 1;
    
    % Slice text
    sliceText = uicontrol('Parent', ctrlPanel, ...
                         'Style', 'text', ...
                         'Units', 'normalized', ...
                         'Position', [0.1, rowPos(row), 0.8, rowHeight], ...
                         'String', ['Slice: 1/' num2str(numSlices)]);
    row = row + 1.25; % Extra space after section
end

% Contrast Range section
uicontrol('Parent', ctrlPanel, ...
         'Style', 'text', ...
         'Units', 'normalized', ...
         'Position', [0.05, rowPos(row), 0.9, rowHeight], ...
         'String', 'Contrast Range');
row = row + 1;

% Min label and slider
uicontrol('Parent', ctrlPanel, ...
         'Style', 'text', ...
         'Units', 'normalized', ...
         'Position', [0.1, rowPos(row), 0.2, rowHeight], ...
         'String', 'Min:');

clipLowSlider = uicontrol('Parent', ctrlPanel, ...
                         'Style', 'slider', ...
                         'Units', 'normalized', ...
                         'Position', [0.3, rowPos(row), 0.6, rowHeight], ...
                         'Min', 0, 'Max', 1, ...
                         'Value', clipLow, ...
                         'SliderStep', [0.001, 0.01]);
set(clipLowSlider, 'Callback', @updateClipLow);
addlistener(clipLowSlider, 'ContinuousValueChange', @updateClipLow);
row = row + 1;

% Max label and slider
uicontrol('Parent', ctrlPanel, ...
         'Style', 'text', ...
         'Units', 'normalized', ...
         'Position', [0.1, rowPos(row), 0.2, rowHeight], ...
         'String', 'Max:');

clipHighSlider = uicontrol('Parent', ctrlPanel, ...
                          'Style', 'slider', ...
                          'Units', 'normalized', ...
                          'Position', [0.3, rowPos(row), 0.6, rowHeight], ...
                          'Min', 0, 'Max', 1, ...
                          'Value', clipHigh, ...
                          'SliderStep', [0.001, 0.01]);
set(clipHighSlider, 'Callback', @updateClipHigh);
addlistener(clipHighSlider, 'ContinuousValueChange', @updateClipHigh);
row = row + 1.5;

% Range display text
clipText = uicontrol('Parent', ctrlPanel, ...
                    'Style', 'text', ...
                    'Units', 'normalized', ...
                    'Position', [0.1, rowPos(row), 0.8, rowHeight*1.5], ...
                    'String', sprintf('Range: [%.2f, %.2f]', clipLow, clipHigh));
row = row + 1; % Extra space for multiline text

% Auto Contrast section
uicontrol('Parent', ctrlPanel, ...
         'Style', 'text', ...
         'Units', 'normalized', ...
         'Position', [0.05, rowPos(row), 0.9, rowHeight], ...
         'String', 'Auto Contrast');
row = row + 1;

% Percentile controls
uicontrol('Parent', ctrlPanel, ...
         'Style', 'text', ...
         'Units', 'normalized', ...
         'Position', [0.1, rowPos(row), 0.3, rowHeight], ...
         'String', 'Percentile:');


percentileEdit = uicontrol('Parent', ctrlPanel, ...
                         'Style', 'edit', ...
                         'Units', 'normalized', ...
                         'Position', [0.45, rowPos(row), 0.45, rowHeight], ...
                         'String', '0.01', ...
                         'TooltipString', 'Set percentile');
row = row + 1.5;

% Apply button
autoClipButton = uicontrol('Parent', ctrlPanel, ...
                          'Style', 'pushbutton', ...
                          'Units', 'normalized', ...
                          'Position', [0.3, rowPos(row), 0.6, rowHeight*1.2], ...
                          'String', 'Apply', ...
                          'Callback', @autoClip);
row = row + 2; % Extra space after button

% Gamma section - FIXED POSITIONING
uicontrol('Parent', ctrlPanel, ...
         'Style', 'text', ...
         'Units', 'normalized', ...
         'Position', [0.05, rowPos(row), 0.9, rowHeight], ...
         'String', 'Gamma Correction'); % Make it visible with background
row = row + 1.5;

% Gamma slider
gammaSlider = uicontrol('Parent', ctrlPanel, ...
                       'Style', 'slider', ...
                       'Units', 'normalized', ...
                       'Position', [0.1, rowPos(row), 0.8, rowHeight], ...
                       'Min', 0.1, 'Max', 3, ...
                       'Value', gammaValue, ...
                       'SliderStep', [0.001, 0.01]);
set(gammaSlider, 'Callback', @updateGamma);
addlistener(gammaSlider, 'ContinuousValueChange', @updateGamma);
row = row + 1;

% Gamma value text
gammaText = uicontrol('Parent', ctrlPanel, ...
                     'Style', 'text', ...
                     'Units', 'normalized', ...
                     'Position', [0.1, rowPos(row), 0.8, rowHeight], ...
                     'String', sprintf('Gamma: %.2f', gammaValue));
row = row + 1; % Extra space after gamma section

% Add histogram options section
uicontrol('Parent', ctrlPanel, ...
         'Style', 'text', ...
         'Units', 'normalized', ...
         'Position', [0.05, rowPos(row), 0.9, rowHeight], ...
         'String', 'Histogram Options');
row = row + 0.75;

% Add log scale checkbox
logScaleCheckbox = uicontrol('Parent', ctrlPanel, ...
                            'Style', 'checkbox', ...
                            'Units', 'normalized', ...
                            'Position', [0.1, rowPos(row), 0.8, rowHeight], ...
                            'String', 'Logarithmic Y-Axis', ...
                            'Value', logScale, ...
                            'Callback', @toggleLogScale);
row = row + 1.5; % Extra space after checkbox

% Add reset button
resetButton = uicontrol('Parent', ctrlPanel, ...
                       'Style', 'pushbutton', ...
                       'Units', 'normalized', ...
                       'Position', [0.3, rowPos(row), 0.6, rowHeight*1.2], ...
                       'String', 'Reset All', ...
                       'Callback', @resetAll);

row = row + 1.5; 
% Add save button
saveButton = uicontrol('Parent', ctrlPanel, ...
                       'Style', 'pushbutton', ...
                       'Units', 'normalized', ...
                       'Position', [0.3, rowPos(row), 0.6, rowHeight*1.5], ...
                       'String', 'Apply Changes & Exit', ...
                       'Callback', @saveAll);
% Initialize display
updateDisplay();

% Wait for figure to close, then return the processed image
uiwait(mainFig);

% Callback and helper functions
    function updateSlice(~, ~)
        if is3D
            currentSlice = round(get(sliceSlider, 'Value'));
            set(sliceText, 'String', ['Slice: ' num2str(currentSlice) '/' num2str(numSlices)]);
            updateDisplay();
        end
    end

    function updateClipLow(~, ~)
        clipLow = get(clipLowSlider, 'Value');
        % Ensure low value is less than high value
        if clipLow >= clipHigh
            clipLow = clipHigh - 0.01;
            set(clipLowSlider, 'Value', clipLow);
        end
        updateClipText();
        updateDisplay();
    end

    function updateClipHigh(~, ~)
        clipHigh = get(clipHighSlider, 'Value');
        % Ensure high value is greater than low value
        if clipHigh <= clipLow
            clipHigh = clipLow + 0.01;
            set(clipHighSlider, 'Value', clipHigh);
        end
        updateClipText();
        updateDisplay();
    end

    function updateClipText()
        % Calculate actual pixel value range
        lowVal = clipLow * imgRange + imgMin;
        highVal = clipHigh * imgRange + imgMin;
        set(clipText, 'String', sprintf('Range: [%.2f, %.2f]\nActual values: [%.2f, %.2f]', ...
            clipLow, clipHigh, lowVal, highVal));
    end

    function autoClip(~, ~)
        % Auto-adjust contrast using percentiles
        try
            percentile = str2double(get(percentileEdit, 'String'));
            if isnan(percentile) || percentile < 0 || percentile > 49
                errordlg('Percentile must be a value between 0 and 49', 'Input Error');
                return;
            end
        catch
            errordlg('Invalid percentile input', 'Input Error');
            return;
        end
        
        percentile = percentile / 100; % Convert to 0-1 range
        
        % Calculate percentiles
        if is3D
            currentImgSlice = img(:,:,currentSlice);
        else
            currentImgSlice = img;
        end
        
        values = sort(currentImgSlice(:));
        clipLow = (values(max(1, round(numel(values) * percentile))) - imgMin) / imgRange;
        clipHigh = (values(min(numel(values), round(numel(values) * (1-percentile)))) - imgMin) / imgRange;
        
        % Ensure clipLow < clipHigh
        if clipLow >= clipHigh
            clipLow = max(0, clipHigh - 0.1);
        end
        
        % Update sliders and display
        set(clipLowSlider, 'Value', clipLow);
        set(clipHighSlider, 'Value', clipHigh);
        updateClipText();
        
        updateDisplay();
    end

    function updateGamma(~, ~)
        gammaValue = get(gammaSlider, 'Value');
        set(gammaText, 'String', sprintf('Gamma: %.2f', gammaValue));
        updateDisplay();
    end

    function toggleLogScale(~, ~)
        logScale = get(logScaleCheckbox, 'Value');
        updateDisplay();
    end

    function resetAll(~, ~)
        % Reset all settings to initial values
        clipLow = 0;
        clipHigh = 1;
        gammaValue = 1;
        logScale = false;
        
        % Update UI elements
        set(clipLowSlider, 'Value', clipLow);
        set(clipHighSlider, 'Value', clipHigh);
        set(gammaSlider, 'Value', gammaValue);
        set(logScaleCheckbox, 'Value', logScale);
        
        % If 3D, reset slice too
        if is3D
            currentSlice = 1;
            set(sliceSlider, 'Value', currentSlice);
            set(sliceText, 'String', ['Slice: 1/' num2str(numSlices)]);
        end
        
        % Update display
        updateClipText();
        updateDisplay();
    end

    function updateDisplay()
        % Get current slice
        if is3D
            currentImgSlice = img(:,:,currentSlice);
        else
            currentImgSlice = img;
        end
        
        % Normalize
        normalizedSlice = (currentImgSlice - imgMin) / imgRange;
        
        % Apply clip
        displaySlice = normalizedSlice;
        displaySlice(displaySlice < clipLow) = clipLow;
        displaySlice(displaySlice > clipHigh) = clipHigh;
        displaySlice = (displaySlice - clipLow) / max(clipHigh - clipLow, eps);
        
        % Apply gamma correction
        displaySlice = displaySlice .^ (1/gammaValue);
        
        % Display image
        axes(imgAxes);
        imshow(displaySlice);
        title('Current Slice');
        
        % Get value under cursor
        set(imgAxes, 'ButtonDownFcn', @imageClick);
        
        % Prepare gamma corrected histogram data
        gammaCorrectedSlice = normalizedSlice .^ (1/gammaValue);
        
        % Update histogram with higher resolution (100 bins)
        axes(histAxes);
        [counts, edges] = histcounts(normalizedSlice(:), 500); % Higher resolution histogram
        edges = edges(1:end-1) + diff(edges)/2; % Get bin centers
        
        % Get histogram of gamma corrected values
        [gammaHistCounts, ~] = histcounts(gammaCorrectedSlice(:), 500);
        
        % Determine y-axis scale based on checkbox
        if logScale
            % Use log scale for Y-axis, adding 1 to avoid log(0)
            bar(edges, log(counts+1), 'BarWidth', 1);
            maxCount = log(max(counts) + 1);
            maxGammaCount = log(max(gammaHistCounts) + 1);
            % Add gamma corrected histogram as dashed line
            hold on;
            plot(edges, log(gammaHistCounts+1), '--', 'LineWidth', 1.5, 'Color', 'b');
        else
            % Use linear scale
            bar(edges, counts, 'BarWidth', 1);
            maxCount = max(counts);
            maxGammaCount = max(gammaHistCounts);
            % Add gamma corrected histogram as dashed line
            hold on;
            plot(edges, gammaHistCounts, '--', 'LineWidth', 1.5, 'Color', 'b');
        end
        
        % Use the maximum of both histograms for the y limit
        maxYLimit = max(maxCount, maxGammaCount) * 1.05;
        
        % Draw clip lines
        plot([clipLow clipLow], [0 maxYLimit], 'r-', 'LineWidth', 2);
        plot([clipHigh clipHigh], [0 maxYLimit], 'r-', 'LineWidth', 2);
        
        % Fill clip region
        x = [clipLow, clipHigh, clipHigh, clipLow];
        y = [0, 0, maxYLimit, maxYLimit];
        patch(x, y, 'r', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
        
        % Add legend
        legend('Original', 'Gamma', 'Location', 'northeast');
        
        hold off;
        
        % Set title and labels based on scale
        if logScale
            title('Histogram and Clip Range (Log Scale)');
            ylabel('Log(Frequency)');
        else
            title('Histogram and Clip Range');
            ylabel('Frequency');
        end
        
        xlabel('Pixel Value (Normalized)');
        xlim([0 1]);
        ylim([0 maxYLimit]);
        grid on;
    end

    function imageClick(~, eventdata)
        % Get click position
        pt = eventdata.IntersectionPoint(1:2);
        x = round(pt(1));
        y = round(pt(2));
        
        % Check if within image bounds
        if x >= 1 && x <= imgWidth && y >= 1 && y <= imgHeight
            % Get original, normalized, and display values
            if is3D
                originalValue = img(y, x, currentSlice);
            else
                originalValue = img(y, x);
            end
            
            normalizedValue = (originalValue - imgMin) / imgRange;
            
            % Apply clip and gamma
            displayValue = normalizedValue;
            if displayValue < clipLow
                displayValue = clipLow;
            elseif displayValue > clipHigh
                displayValue = clipHigh;
            end
            displayValue = (displayValue - clipLow) / (clipHigh - clipLow);
            displayValue = displayValue ^ (1/gammaValue);
            
        end
    end

    function saveAll(~, ~)
        % Process full image and return
        normalizedImg = zeros(size(img));
        
        for i = 1:numSlices
            if is3D
                slice = img(:,:,i);
            else
                slice = img;
            end
            
            % Apply same processing as display
            normSlice = (slice - imgMin) / imgRange;
            normSlice(normSlice < clipLow) = clipLow;
            normSlice(normSlice > clipHigh) = clipHigh;
            normSlice = (normSlice - clipLow) / (clipHigh - clipLow);
            normSlice = normSlice .^ (1/gammaValue);
            
            if is3D
                normalizedImg(:,:,i) = normSlice;
            else
                normalizedImg = normSlice;
            end
        end
        
        % Close figure and return
        delete(mainFig);
    end

    function closeFigure(src, event)
        % 弹出一个确认对话框，让用户选择 Save/Discard/Cancel
        choice = questdlg('Save?',...
                          'Confirm',...
                          'Save','Discard','Cancel',...
                          'Save');
        
        switch choice
            case 'Save'
                saveAll();
            case 'Discard'
                normalizedImg = rescale(img);
                delete(mainFig);
            case 'Cancel'
                return;
            otherwise
                % 万一对话框意外返回其它情况，也不关闭
                return;
        end
    end
end