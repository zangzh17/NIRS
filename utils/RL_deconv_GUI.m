function RL_deconv_GUI(ssResult, warpedPSFs)
% RL_deconv_GUI(ssResult, warpedPSFs)
%
% Interactive GUI for tuning parameters of Lucy-Richardson deconvolution.
%
% Input parameters:
%   ssResult   - 3D image data (multi-slice image)
%   warpedPSFs - PSF data (assumed to be a 4D array; later processed by raising to a power and averaging)
%
% Notes:
%   1. On first run, RL deconvolution is not executed; instead, the middle slice of ssResult is displayed.
%   2. The ROI selection window displays a falsecolor overlay (using imshowpair) combining the first and last slices
%      to help determine the region of interest.
%   3. All parameters except for the slice slider (used for display) are set via edit boxes.
%   4. A checkbox is provided to choose whether to apply the ROI mask. If checked, the mask is converted to double
%      before passing to deconvlucy.
%

%% Default parameter settings
nIter_default   = 12;
dampar_default  = 0.01;
readout_default = 0.005;
n_power_default = 4;

%% Create the main figure and image display area
f = figure('Position',[100,100,1100,700], 'Name','RL Deconvolution Parameter Tuning');
% Create a single axes (positioned high enough to avoid overlapping with lower controls)
ax = axes('Parent',f, 'Units','normalized', 'Position',[0.05, 0.35, 0.9, 0.6]);

% Initially display the middle slice of ssResult (RL deconvolution is not executed at startup)
midSlice = round(size(ssResult,3)/2);
imshow(ssResult(:,:,midSlice), [], 'Parent', ax);
title(ax, sprintf('ssResult: Slice %d', midSlice));

%% Create parameter input controls (edit boxes) and buttons

% Parameter nIter
uicontrol('Style','text','Parent',f, 'Units','normalized',...
    'Position',[0.05, 0.30, 0.1, 0.04],'String','nIter');
hIter = uicontrol('Style','edit','Parent',f, 'Units','normalized',...
    'Position',[0.15, 0.30, 0.2, 0.04],'String', num2str(nIter_default));

% Parameter dampar
uicontrol('Style','text','Parent',f, 'Units','normalized',...
    'Position',[0.40, 0.30, 0.1, 0.04],'String','dampar');
hDampar = uicontrol('Style','edit','Parent',f, 'Units','normalized',...
    'Position',[0.50, 0.30, 0.2, 0.04],'String', num2str(dampar_default));

% Parameter readout
uicontrol('Style','text','Parent',f, 'Units','normalized',...
    'Position',[0.05, 0.25, 0.1, 0.04],'String','readout');
hReadout = uicontrol('Style','edit','Parent',f, 'Units','normalized',...
    'Position',[0.15, 0.25, 0.2, 0.04],'String', num2str(readout_default));

% Parameter n_power
uicontrol('Style','text','Parent',f, 'Units','normalized',...
    'Position',[0.40, 0.25, 0.1, 0.04],'String','n_power');
hNPower = uicontrol('Style','edit','Parent',f, 'Units','normalized',...
    'Position',[0.50, 0.25, 0.2, 0.04],'String', num2str(n_power_default));

% ROI selection button: calls interactive ROI tool
hROI = uicontrol('Style','pushbutton','Parent',f, 'Units','normalized',...
    'Position',[0.75, 0.25, 0.12, 0.06],'String','Select ROI','Callback',@selectROI);

% Checkbox to choose whether to apply the ROI mask
hApplyMask = uicontrol('Style','checkbox','Parent',f, 'Units','normalized',...
    'Position',[0.75, 0.20, 0.12, 0.04],'String','Apply mask','Value',1);

% Update button: when pressed, RL deconvolution is executed
hUpdate = uicontrol('Style','pushbutton','Parent',f, 'Units','normalized',...
    'Position',[0.75, 0.30, 0.12, 0.06],'String','Update','Callback',@updateCallback);

% Slice display slider (only used to show slices of the deconvolved result)
uicontrol('Style','text','Parent',f, 'Units','normalized',...
    'Position',[0.05, 0.10, 0.15, 0.04],'String','Display Slice');
hSlice = uicontrol('Style','slider','Parent',f, 'Units','normalized',...
    'Position',[0.20, 0.11, 0.70, 0.04],'Min',1, 'Max', size(ssResult,3),...
    'Value', midSlice, 'Callback',@sliceSliderCallback);

%% Save controls and data in guidata
handles.ax = ax;
handles.ssResult = ssResult;
handles.warpedPSFs = warpedPSFs;
handles.mask = [];      % ROI mask (initially empty)
handles.hIter = hIter;
handles.hDampar = hDampar;
handles.hReadout = hReadout;
handles.hNPower = hNPower;
handles.hSlice = hSlice;
handles.hApplyMask = hApplyMask; % save checkbox handle
handles.deconvResult = [];  % RL deconvolution result (initially empty)
guidata(f, handles);

% Note: Do not call updateCallback at startup to avoid a lengthy RL computation.

%% --- Callback function: ROI selection ---
function selectROI(~, ~)
    handles = guidata(f);
    % Create a new figure for ROI selection. Use a falsecolor overlay:
    % imshowpair is used to combine the first slice (displayed in red) and the last slice (displayed in cyan).
    hFigROI = figure('Name','Select ROI');
    imshowpair(handles.ssResult(:,:,1), handles.ssResult(:,:,end), 'falsecolor');
    title('Falsecolor Overlay: First Slice vs. Last Slice');
    
    % Use interactive drawing of an elliptical ROI (supported in newer MATLAB versions)
    hEllipse = drawellipse();
    wait(hEllipse);  % Wait for the user to finish drawing the ROI
    mask = createMask(hEllipse);
    handles.mask = mask;
    guidata(f, handles);
    close(hFigROI);
end

%% --- Callback function: Update (run RL deconvolution) ---
function updateCallback(~, ~)
    handles = guidata(f);
    % Read parameters from the edit boxes and convert them to numeric values
    nIter   = round(str2double(get(handles.hIter, 'String')));
    dampar  = str2double(get(handles.hDampar, 'String'));
    readout = str2double(get(handles.hReadout, 'String'));
    n_power = str2double(get(handles.hNPower, 'String'));
    
    % Check the checkbox state to determine whether to use the mask
    useMask = get(handles.hApplyMask, 'Value');
    if useMask && ~isempty(handles.mask)
        % Convert the mask to double (deconvlucy generally expects a numeric array)
        weight = repmat(double(handles.mask),[1,1,size(handles.ssResult,3)]);
    else
        weight = [];
    end
    
    % Compute the PSF: raise warpedPSFs to the n_power and take the mean along the 4th dimension
    warpedPSF = mean(handles.warpedPSFs.^n_power, 4);
    
    % Display a wait message to the user
    hMsg = msgbox('Calculating RL... Please wait', 'Wait', 'modal');
    drawnow;
    
    % Execute Lucy-Richardson deconvolution (this might take a while)
    deconvResult = deconvlucy(handles.ssResult, warpedPSF, nIter, dampar, weight, readout);
    
    if ishandle(hMsg)
        close(hMsg);
    end
    handles.deconvResult = deconvResult;
    guidata(f, handles);
    
    % Update the slice slider range based on the number of slices in the deconvolved result
    numSlices = size(deconvResult, 3);
    set(handles.hSlice, 'Min', 1, 'Max', numSlices, 'Value', round(numSlices/2));
    
    % By default, display the middle slice of the deconvolved result
    midSlice_rl = round(numSlices/2);
    imshow(deconvResult(:,:,midSlice_rl), [], 'Parent', handles.ax);
    title(handles.ax, sprintf('RL deconv: Slice %d', midSlice_rl));
end

%% --- Callback function: Slice slider ---
function sliceSliderCallback(~, ~)
    handles = guidata(f);
    % If RL deconvolution has been run, display its result; otherwise, display the original ssResult
    if isempty(handles.deconvResult)
        imageData = handles.ssResult;
    else
        imageData = handles.deconvResult;
    end
    sliceIdx = round(get(handles.hSlice, 'Value'));
    sliceIdx = max(1, min(sliceIdx, size(imageData, 3)));  % ensure the index is valid
    imshow(imageData(:,:,sliceIdx), [], 'Parent', handles.ax);
    if isempty(handles.deconvResult)
        title(handles.ax, sprintf('ssResult: Slice %d', sliceIdx));
    else
        title(handles.ax, sprintf('RL deconv: Slice %d', sliceIdx));
    end
end

end
