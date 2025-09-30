function segmentation_data = keyframeSegToolDualLabel(img_3d)
% Interactive keyframe segmentation tool with dual-label support
% Input: img_3d - 3D image data in XYT format
% Output: segmentation_data - structure containing segmentation info for two labels

    % Initialize variables
    [height, width, num_frames] = size(img_3d);
    current_frame = 1;
    current_label = 1;  % Current label (1 or 2)
    label_colors = {[1, 0.2, 0.2], [0.2, 0.2, 1]};  % Red for label 1, Blue for label 2
    label_names = {'Label 1', 'Label 2'};
    
    % Store segmentation data for each label
    segmented_frames = cell(2, 1);
    masks = cell(2, 1);
    for i = 1:2
        segmented_frames{i} = false(num_frames, 1);
        masks{i} = cell(num_frames, 1);
    end
    
    % Create main window
    fig = figure('Name', 'Dual-Label Keyframe Tool', 'NumberTitle', 'off', ...
                 'Position', [100, 100, 900, 700], ...
                 'KeyPressFcn', @keyPressCallback, ...
                 'WindowScrollWheelFcn', @scrollWheelCallback, ...
                 'ResizeFcn', @resizeCallback, ...
                 'Units', 'normalized');
    
    % Create image display area
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
    
    % Create progress bars for both labels
    ax_progress1 = axes('Parent', fig, 'Position', [0.1, 0.18, 0.8, 0.03], ...
                        'Units', 'normalized');
    progress_data1 = double(segmented_frames{1}');
    progress_handle1 = imagesc(progress_data1);
    colormap(ax_progress1, [0.9 0.9 0.9; label_colors{1}]);
    set(ax_progress1, 'YTick', [], 'CLim', [0 1]);
    ylabel(ax_progress1, 'L1', 'FontSize', 8);
    
    ax_progress2 = axes('Parent', fig, 'Position', [0.1, 0.14, 0.8, 0.03], ...
                        'Units', 'normalized');
    progress_data2 = double(segmented_frames{2}');
    progress_handle2 = imagesc(progress_data2);
    colormap(ax_progress2, [0.9 0.9 0.9; label_colors{2}]);
    set(ax_progress2, 'YTick', [], 'CLim', [0 1]);
    ylabel(ax_progress2, 'L2', 'FontSize', 8);
    
    % Current frame indicators
    hold(ax_progress1, 'on');
    current_frame_line1 = plot(ax_progress1, [current_frame, current_frame], [0.5, 1.5], ...
                               'k-', 'LineWidth', 2);
    hold(ax_progress1, 'off');
    
    hold(ax_progress2, 'on');
    current_frame_line2 = plot(ax_progress2, [current_frame, current_frame], [0.5, 1.5], ...
                               'k-', 'LineWidth', 2);
    hold(ax_progress2, 'off');
    
    % Label selection button
    btn_label = uicontrol('Parent', fig, 'Style', 'pushbutton', ...
                          'String', sprintf('Current: %s', label_names{current_label}), ...
                          'Units', 'normalized', 'Position', [0.1, 0.05, 0.15, 0.06], ...
                          'BackgroundColor', label_colors{current_label}, ...
                          'ForegroundColor', 'white', ...
                          'FontWeight', 'bold', ...
                          'Callback', @switchLabel);
    
    % Control buttons
    btn_segment = uicontrol('Parent', fig, 'Style', 'pushbutton', 'String', 'Start seg', ...
                           'Units', 'normalized', 'Position', [0.27, 0.05, 0.12, 0.06], ...
                           'Callback', @startSegmentation);
    
    btn_clear = uicontrol('Parent', fig, 'Style', 'pushbutton', 'String', 'Clear current', ...
                         'Units', 'normalized', 'Position', [0.41, 0.05, 0.12, 0.06], ...
                         'Callback', @clearCurrent);
    
    btn_finish = uicontrol('Parent', fig, 'Style', 'pushbutton', 'String', 'Finish', ...
                          'Units', 'normalized', 'Position', [0.55, 0.05, 0.12, 0.06], ...
                          'Callback', @finishSegmentation);
    
    % Statistics display
    text_stats = uicontrol('Parent', fig, 'Style', 'text', ...
                          'String', sprintf('L1: 0/%d | L2: 0/%d', num_frames, num_frames), ...
                          'Units', 'normalized', 'Position', [0.69, 0.05, 0.2, 0.06], ...
                          'BackgroundColor', get(fig, 'Color'));

    % Initialize display
    updateProgressBar();
    
    % Show instructions
    fprintf('=== Dual-Label Segmentation Tool ===\n');
    fprintf('Instructions:\n');
    fprintf('- Click label button or press q to switch between labels\n');
    fprintf('- Use slider or arrow keys to navigate frames\n');
    fprintf('- Press Space or click "Start seg" to segment\n');
    fprintf('- Current label color shows which mask you are editing\n');
    fprintf('=====================================\n');
    
    % Wait for user action
    waitfor(fig);
    
    % Prepare output data
    if exist('segmented_frames', 'var')
        segmentation_data.label1.segmented_frames = find(segmented_frames{1});
        segmentation_data.label1.masks = masks{1}(segmented_frames{1});
        segmentation_data.label1.frame_indices = find(segmented_frames{1});
        segmentation_data.label1.num_segmented = sum(segmented_frames{1});
        
        segmentation_data.label2.segmented_frames = find(segmented_frames{2});
        segmentation_data.label2.masks = masks{2}(segmented_frames{2});
        segmentation_data.label2.frame_indices = find(segmented_frames{2});
        segmentation_data.label2.num_segmented = sum(segmented_frames{2});
        
        segmentation_data.total_frames = num_frames;
        fprintf('Segmentation finished! L1: %d frames, L2: %d frames\n', ...
                sum(segmented_frames{1}), sum(segmented_frames{2}));
    else
        segmentation_data = struct();
        segmentation_data.label1 = struct('segmented_frames', [], 'masks', {{}}, ...
                                         'frame_indices', [], 'num_segmented', 0);
        segmentation_data.label2 = struct('segmented_frames', [], 'masks', {{}}, ...
                                         'frame_indices', [], 'num_segmented', 0);
        segmentation_data.total_frames = num_frames;
    end
    
    %% Callback functions
    function switchLabel(~, ~)
        current_label = 3 - current_label;  % Toggle between 1 and 2
        set(btn_label, 'String', sprintf('Current: %s', label_names{current_label}), ...
            'BackgroundColor', label_colors{current_label});
        updateFrame(current_frame);
        fprintf('Switched to %s\n', label_names{current_label});
    end
    
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
            case 'space'
                startSegmentation();
            case 'q'
                switchLabel();
            case 'escape'
                finishSegmentation();
        end
    end
    
    function scrollWheelCallback(~, event)
        delta = -15*event.VerticalScrollCount;
        updateFrame(max(1, min(num_frames, current_frame + delta)));
    end
    
    function resizeCallback(~, ~)
        drawnow;
    end
    
    function updateFrame(new_frame)
        current_frame = new_frame;
        set(slider_time, 'Value', current_frame);
        
        % Show current frame
        set(img_handle, 'CData', img_3d(:,:,current_frame));
        
        % Update title with segmentation status
        status_str = '';
        if segmented_frames{1}(current_frame) && segmented_frames{2}(current_frame)
            status_str = '(L1+L2)';
        elseif segmented_frames{1}(current_frame)
            status_str = '(L1)';
        elseif segmented_frames{2}(current_frame)
            status_str = '(L2)';
        end
        title(ax_img, sprintf('Frame %d/%d %s', current_frame, num_frames, status_str));

        % Display masks for both labels
        axes(ax_img);
        hold on;
        delete(findobj(ax_img, 'Type', 'line'));
        
        % Show both masks with different colors
        for label_idx = 1:2
            if segmented_frames{label_idx}(current_frame) && ~isempty(masks{label_idx}{current_frame})
                try
                    contours = contourc(double(masks{label_idx}{current_frame}), [0.5, 0.5]);
                    if ~isempty(contours)
                        i = 1;
                        while i < size(contours, 2)
                            n_points = contours(2, i);
                            if n_points > 0 && i + n_points <= size(contours, 2)
                                x_coords = contours(1, i+1:i+n_points);
                                y_coords = contours(2, i+1:i+n_points);
                                % Different line style for current label
                                if label_idx == current_label
                                    line_style = '-';
                                    line_width = 2;
                                else
                                    line_style = '--';
                                    line_width = 1;
                                end
                                plot(x_coords, y_coords, line_style, ...
                                    'Color', [label_colors{label_idx}, 0.7], ...
                                    'LineWidth', line_width);
                            end
                            i = i + n_points + 1;
                        end
                    end
                catch
                    % Fallback display
                    [r, c] = find(masks{label_idx}{current_frame});
                    if ~isempty(r)
                        plot(c, r, '.', 'Color', [label_colors{label_idx}, 0.5], ...
                            'MarkerSize', 2);
                    end
                end
            end
        end
        hold off;
        
        % Update frame indicators
        updateCurrentFrameIndicator();
    end
    
    function startSegmentation(~, ~)
        axes(ax_img);
        try
            fprintf('Draw polygon for %s on frame %d\n', label_names{current_label}, current_frame);
            
            % Disable controls
            set([btn_segment, btn_clear, btn_finish, btn_label], 'Enable', 'off');
            
            % Draw polygon with current label color
            h_poly = drawpolygon('Color', label_colors{current_label}, 'LineWidth', 2);
            
            % Re-enable controls
            set([btn_segment, btn_clear, btn_finish, btn_label], 'Enable', 'on');
            
            if isvalid(h_poly) && ~isempty(h_poly.Position) && size(h_poly.Position, 1) >= 3
                % Create mask
                mask = createMask(h_poly, img_handle);
                masks{current_label}{current_frame} = mask;
                segmented_frames{current_label}(current_frame) = true;
                
                delete(h_poly);
                updateFrame(current_frame);
                updateProgressBar();
                
                fprintf('%s segmentation complete for frame %d\n', ...
                        label_names{current_label}, current_frame);
            else
                if isvalid(h_poly)
                    delete(h_poly);
                end
                fprintf('Segmentation cancelled\n');
            end
        catch ME
            set([btn_segment, btn_clear, btn_finish, btn_label], 'Enable', 'on');
            fprintf('Error: %s\n', ME.message);
        end
    end
    
    function clearCurrent(~, ~)
        masks{current_label}{current_frame} = [];
        segmented_frames{current_label}(current_frame) = false;
        updateFrame(current_frame);
        updateProgressBar();
        fprintf('%s cleared for frame %d\n', label_names{current_label}, current_frame);
    end
    
    function updateProgressBar()
        % Update both progress bars
        set(progress_handle1, 'CData', double(segmented_frames{1}'));
        set(progress_handle2, 'CData', double(segmented_frames{2}'));
        
        % Update statistics
        num_seg1 = sum(segmented_frames{1});
        num_seg2 = sum(segmented_frames{2});
        set(text_stats, 'String', sprintf('L1: %d/%d | L2: %d/%d', ...
                                          num_seg1, num_frames, num_seg2, num_frames));
        drawnow;
    end
    
    function updateCurrentFrameIndicator()
        % Update indicators for both progress bars
        if isvalid(current_frame_line1)
            set(current_frame_line1, 'XData', [current_frame, current_frame]);
        end
        if isvalid(current_frame_line2)
            set(current_frame_line2, 'XData', [current_frame, current_frame]);
        end
    end
    
    function finishSegmentation(~, ~)
        fprintf('Finished! L1: %d frames, L2: %d frames\n', ...
                sum(segmented_frames{1}), sum(segmented_frames{2}));
        delete(fig);
    end
end