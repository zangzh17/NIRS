function segmentation_data = keyframeSegTool(img_3d)
% Interactive keyframe segmentation tool (fixed version)
% Input: img_3d - 3D image data in XYT format
% Output: segmentation_data - structure containing segmentation info

    % Initialize variables
    [height, width, num_frames] = size(img_3d);
    current_frame = 1;
    segmented_frames = false(num_frames, 1);  % Record which frames are segmented
    masks = cell(num_frames, 1);  % Store mask for each frame
    
    % Create main window
    fig = figure('Name', 'Key frame tool', 'NumberTitle', 'off', ...
                 'Position', [100, 100, 800, 700], ...
                 'KeyPressFcn', @keyPressCallback, ...
                 'WindowScrollWheelFcn', @scrollWheelCallback, ...
                 'ResizeFcn', @resizeCallback, ...
                 'Units', 'normalized');
    
    % Create image display area (using normalized coordinates)
    ax_img = axes('Parent', fig, 'Position', [0.1, 0.35, 0.8, 0.6], ...
                  'Units', 'normalized');
    img_handle = imshow(img_3d(:,:,1), []);
    title(sprintf('Frame %d/%d', current_frame, num_frames));
    
    % Create timeline slider
    slider_time = uicontrol('Parent', fig, 'Style', 'slider', ...
                           'Min', 1, 'Max', num_frames, 'Value', 1, ...
                           'Units', 'normalized', ...
                           'Position', [0.1, 0.25, 0.8, 0.03], ...
                           'Callback', @sliderCallback);
    
    % Create progress bar showing segmented frames
    ax_progress = axes('Parent', fig, 'Position', [0.1, 0.15, 0.8, 0.08], ...
                       'Units', 'normalized');
    
    % Initialize progress bar data
    progress_data = double(segmented_frames');  % Convert to double type
    progress_handle = imagesc(progress_data);
    colormap(ax_progress, [0.9 0.9 0.9; 0 0.8 0]);  % Gray = unsegmented, Green = segmented
    set(ax_progress, 'YTick', [], 'CLim', [0 1], ...  % Set color range
        'XTick', [1, max(1, round(num_frames/4)), max(1, round(num_frames/2)), ...
                  max(1, round(3*num_frames/4)), num_frames], ...
        'XTickLabel', {sprintf('%d', 1), sprintf('%d', round(num_frames/4)), ...
                       sprintf('%d', round(num_frames/2)), sprintf('%d', round(3*num_frames/4)), ...
                       sprintf('%d', num_frames)});
    ylabel(ax_progress, 'Progress', 'FontSize', 8);
    
    % Create indicator for current frame position
    hold(ax_progress, 'on');
    current_frame_line = plot(ax_progress, [current_frame, current_frame], [0.5, 1.5], ...
                             'r-', 'LineWidth', 3);
    hold(ax_progress, 'off');
    
    % Create control buttons
    btn_segment = uicontrol('Parent', fig, 'Style', 'pushbutton', 'String', 'Start seg', ...
                           'Units', 'normalized', 'Position', [0.1, 0.05, 0.12, 0.06], ...
                           'Callback', @startSegmentation);
    
    btn_clear = uicontrol('Parent', fig, 'Style', 'pushbutton', 'String', 'Clear current', ...
                         'Units', 'normalized', 'Position', [0.25, 0.05, 0.12, 0.06], ...
                         'Callback', @clearCurrent);
    
    btn_finish = uicontrol('Parent', fig, 'Style', 'pushbutton', 'String', 'Finish', ...
                          'Units', 'normalized', 'Position', [0.4, 0.05, 0.12, 0.06], ...
                          'Callback', @finishSegmentation);
    
    % Add statistics display
    text_stats = uicontrol('Parent', fig, 'Style', 'text', ...
                          'String', sprintf('Already processed: 0/%d Frames', num_frames), ...
                          'Units', 'normalized', 'Position', [0.55, 0.05, 0.2, 0.06], ...
                          'BackgroundColor', get(fig, 'Color'));

    % Initialize display
    updateProgressBar();
    
    % Show usage instructions
    fprintf('=== Segmentation Tool ===\n');
    fprintf('Instruction:\n');
    fprintf('- Use slider or left/right arrow to switch between frames\n');
    fprintf('- Click "Start seg" or press Space to segment current frame\n');
    fprintf('- Click "Finish" button to end and retrieve results\n');
    fprintf('- Press ESC or close window to also finish\n');
    fprintf('====================\n');
    
    % Wait for user action - use more reliable wait mechanism
    waitfor(fig);  % Replace uiwait with waitfor
    
    % Check if completed normally (window might have been closed accidentally)
    if exist('segmented_frames', 'var') && any(segmented_frames)
        segmentation_data.segmented_frames = find(segmented_frames);
        segmentation_data.masks = masks(segmented_frames);
        segmentation_data.frame_indices = find(segmented_frames);
        segmentation_data.total_frames = num_frames;
        segmentation_data.num_segmented = sum(segmented_frames);
        fprintf('Segmentation finished! Total %d key frames segmented\n', sum(segmented_frames));
    else
        segmentation_data = struct('segmented_frames', [], 'masks', {{}}, ...
                                  'frame_indices', [], 'total_frames', num_frames, ...
                                  'num_segmented', 0);
        fprintf('No segmentation performed or window was closed unexpectedly\n');
    end
    
    %% Callback functions
    function sliderCallback(~, ~)
        new_frame = round(get(slider_time, 'Value'));
        updateFrame(new_frame);
    end
    
    function keyPressCallback(~, event)
        switch event.Key
            case 'leftarrow'
                updateFrame(max(1, current_frame - 1));
            case 'rightarrow'
                updateFrame(min(num_frames, current_frame + 1));
            case 'space'  % Add space key for quick segmentation
                startSegmentation();
            case 'escape'  % ESC to exit
                finishSegmentation();
        end
    end
    
    function scrollWheelCallback(~, event)
        delta = -15*event.VerticalScrollCount;
        updateFrame(max(1, min(num_frames, current_frame + delta)));
    end
    
    function resizeCallback(~, ~)
        % Callback when window size changes
        drawnow;
    end
    
    function updateFrame(new_frame)
        current_frame = new_frame;
        set(slider_time, 'Value', current_frame);
        
        % Show current frame
        set(img_handle, 'CData', img_3d(:,:,current_frame));
        if segmented_frames(current_frame)
            status_str = '(segmented)';
        else
            status_str = '';
        end
        title(ax_img, sprintf('Frame %d/%d %s', current_frame, num_frames, status_str));

        % If current frame is segmented, display mask
        axes(ax_img);  % Ensure operations happen on the correct axes
        hold on;
        delete(findobj(ax_img, 'Type', 'line'));  % Clear previous contours
        if segmented_frames(current_frame) && ~isempty(masks{current_frame})
            % Display mask contour (transparent dashed line)
            try
                contours = contourc(double(masks{current_frame}), [0.5, 0.5]);
                if ~isempty(contours)
                    i = 1;
                    while i < size(contours, 2)
                        level = contours(1, i);
                        n_points = contours(2, i);
                        if n_points > 0 && i + n_points <= size(contours, 2)
                            x_coords = contours(1, i+1:i+n_points);
                            y_coords = contours(2, i+1:i+n_points);
                            % Display contour with transparent dashed line
                            h_line = plot(x_coords, y_coords, '--', 'Color', [1, 0.2, 0.2, 0.7], ...
                                         'LineWidth', 1.5);
                        end
                        i = i + n_points + 1;
                    end
                end
            catch
                % If contour extraction fails, show mask region (transparent dots)
                [r, c] = find(masks{current_frame});
                if ~isempty(r)
                    h_dots = plot(c, r, '.', 'Color', [1, 0.2, 0.2, 0.5], 'MarkerSize', 2);
                    % Compatibility handling
                    if verLessThan('matlab', '9.4')
                        set(h_dots, 'Color', [1, 0.4, 0.4], 'MarkerSize', 2);
                    end
                end
            end
        end
        hold off;
        
        % Update current frame indicator in progress bar
        updateCurrentFrameIndicator();
    end
    
    function startSegmentation(~, ~)
        % Use polygon selection tool
        axes(ax_img);
        try
            fprintf('Draw a polygon on the image to segment current frame %d\n', current_frame);
            
            % Temporarily disable other controls to prevent interference
            set(btn_segment, 'Enable', 'off');
            set(btn_clear, 'Enable', 'off');
            set(btn_finish, 'Enable', 'off');
            
            h_poly = drawpolygon('Color', 'yellow', 'LineWidth', 2);
            
            % Re-enable controls
            set(btn_segment, 'Enable', 'on');
            set(btn_clear, 'Enable', 'on');
            set(btn_finish, 'Enable', 'on');
            
            if isvalid(h_poly) && ~isempty(h_poly.Position) && size(h_poly.Position, 1) >= 3
                % Create mask
                mask = createMask(h_poly, img_handle);
                masks{current_frame} = mask;
                segmented_frames(current_frame) = true;
                
                % Delete polygon object
                delete(h_poly);
                
                % Update display
                updateFrame(current_frame);
                updateProgressBar();
                
                fprintf('Frame %d segmentation complete\n', current_frame);
            else
                if isvalid(h_poly)
                    delete(h_poly);
                end
                fprintf('Segmentation cancelled or polygon invalid\n');
            end
        catch ME
            % Re-enable controls
            set(btn_segment, 'Enable', 'on');
            set(btn_clear, 'Enable', 'on');
            set(btn_finish, 'Enable', 'on');
            fprintf('Error during segmentation: %s\n', ME.message);
        end
    end
    
    function clearCurrent(~, ~)
        masks{current_frame} = [];
        segmented_frames(current_frame) = false;
        updateFrame(current_frame);
        updateProgressBar();
        fprintf('Frame %d segmentation cleared\n', current_frame);
    end
    
    function updateProgressBar()
        % Update progress bar data
        progress_data = double(segmented_frames');  % Ensure type is double
        set(progress_handle, 'CData', progress_data);
        
        % Force color refresh
        caxis(ax_progress, [0 1]);
        
        % Update statistics info
        num_segmented = sum(segmented_frames);
        set(text_stats, 'String', sprintf('Segmented: %d/%d Frames', num_segmented, num_frames));
        
        % Refresh display
        drawnow;
    end
    
    function updateCurrentFrameIndicator()
        % Update current frame indicator
        if isvalid(current_frame_line)
            set(current_frame_line, 'XData', [current_frame, current_frame], ...
                'YData', [0.5, 1.5]);
        else
            % If indicator was deleted, recreate
            axes(ax_progress);
            hold on;
            current_frame_line = plot([current_frame, current_frame], [0.5, 1.5], ...
                                     'r-', 'LineWidth', 3);
            hold off;
        end
    end
    
    function finishSegmentation(~, ~)
        num_segmented = sum(segmented_frames);
        fprintf('Segmentation finished! Total %d key frames segmented\n', num_segmented);
        delete(fig);  % Directly delete window, triggers waitfor end
    end
end
