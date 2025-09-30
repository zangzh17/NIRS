function segmentation_data = keyframeSegToolDualLabel_XYZT(img_4d)
% Interactive keyframe segmentation tool with dual-label support for XYZT data
% Input: img_4d - 4D image data in XYZT format
% Output: segmentation_data - structure containing segmentation info for two labels across all layers

    % Initialize variables
    [height, width, num_layers, num_frames] = size(img_4d);
    current_frame = 1;
    current_layer = ceil(num_layers/2);  % Start at middle layer
    current_label = 1;  % Current label (1 or 2)
    label_colors = {[1, 0.2, 0.2], [0.2, 0.2, 1]};  % Red for label 1, Blue for label 2
    label_names = {'Label 1', 'Label 2'};
    
    % Store segmentation data for each label - now 4D
    segmented_frames = cell(2, 1);
    masks = cell(2, 1);
    for i = 1:2
        segmented_frames{i} = false(num_layers, num_frames);
        masks{i} = cell(num_layers, num_frames);
    end
    
    % Create main window
    fig = figure('Name', 'Dual-Label XYZT Keyframe Tool', 'NumberTitle', 'off', ...
                 'Position', [100, 100, 900, 750], ...
                 'KeyPressFcn', @keyPressCallback, ...
                 'WindowScrollWheelFcn', @scrollWheelCallback, ...
                 'ResizeFcn', @resizeCallback, ...
                 'Units', 'normalized');
    
    % Create image display area
    ax_img = axes('Parent', fig, 'Position', [0.1, 0.4, 0.8, 0.55], ...
                  'Units', 'normalized');
    img_handle = imshow(img_4d(:,:,current_layer,current_frame), []);
    title(sprintf('Layer %d/%d, Frame %d/%d', current_layer, num_layers, current_frame, num_frames));
    
    % Create timeline slider for frames
    slider_time = uicontrol('Parent', fig, 'Style', 'slider', ...
                           'Min', 1, 'Max', num_frames, 'Value', 1, ...
                           'Units', 'normalized', ...
                           'Position', [0.1, 0.32, 0.8, 0.03], ...
                           'Callback', @sliderTimeCallback);
    
    % Add layer slider
    slider_layer = uicontrol('Parent', fig, 'Style', 'slider', ...
                            'Min', 1, 'Max', num_layers, 'Value', current_layer, ...
                            'Units', 'normalized', ...
                            'Position', [0.1, 0.27, 0.8, 0.03], ...
                            'Callback', @sliderLayerCallback);
    
    % Layer indicator text
    text_layer = uicontrol('Parent', fig, 'Style', 'text', ...
                          'String', sprintf('Z-Layer: %d/%d', current_layer, num_layers), ...
                          'Units', 'normalized', 'Position', [0.02, 0.265, 0.08, 0.04], ...
                          'BackgroundColor', get(fig, 'Color'), ...
                          'FontWeight', 'bold');
    
    % Create progress bars for both labels (showing current layer)
    ax_progress1 = axes('Parent', fig, 'Position', [0.1, 0.20, 0.8, 0.03], ...
                        'Units', 'normalized');
    progress_data1 = double(squeeze(segmented_frames{1}(current_layer,:)));
    progress_handle1 = imagesc(progress_data1);
    colormap(ax_progress1, [0.9 0.9 0.9; label_colors{1}]);
    set(ax_progress1, 'YTick', [], 'CLim', [0 1]);
    ylabel(ax_progress1, 'L1', 'FontSize', 8);
    
    ax_progress2 = axes('Parent', fig, 'Position', [0.1, 0.16, 0.8, 0.03], ...
                        'Units', 'normalized');
    progress_data2 = double(squeeze(segmented_frames{2}(current_layer,:)));
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
                          'Units', 'normalized', 'Position', [0.1, 0.08, 0.15, 0.06], ...
                          'BackgroundColor', label_colors{current_label}, ...
                          'ForegroundColor', 'white', ...
                          'FontWeight', 'bold', ...
                          'Callback', @switchLabel);
    
    % Control buttons
    btn_segment = uicontrol('Parent', fig, 'Style', 'pushbutton', 'String', 'Start seg', ...
                           'Units', 'normalized', 'Position', [0.27, 0.08, 0.12, 0.06], ...
                           'Callback', @startSegmentation);
    
    btn_clear = uicontrol('Parent', fig, 'Style', 'pushbutton', 'String', 'Clear current', ...
                         'Units', 'normalized', 'Position', [0.41, 0.08, 0.12, 0.06], ...
                         'Callback', @clearCurrent);
    
    btn_finish = uicontrol('Parent', fig, 'Style', 'pushbutton', 'String', 'Finish', ...
                          'Units', 'normalized', 'Position', [0.55, 0.08, 0.12, 0.06], ...
                          'Callback', @finishSegmentation);
    
    % Statistics display - enhanced for layers
    text_stats = uicontrol('Parent', fig, 'Style', 'text', ...
                          'String', updateStatsString(), ...
                          'Units', 'normalized', 'Position', [0.69, 0.08, 0.25, 0.06], ...
                          'BackgroundColor', get(fig, 'Color'));
    
    % Layer navigation help text
    text_help = uicontrol('Parent', fig, 'Style', 'text', ...
                         'String', 'Layer Nav: q/w(±1) a/s(±2) z/x(±4) h(mid)', ...
                         'Units', 'normalized', 'Position', [0.1, 0.02, 0.35, 0.04], ...
                         'BackgroundColor', get(fig, 'Color'), ...
                         'FontSize', 8);

    % Initialize display
    updateProgressBar();
    
    % Show instructions
    fprintf('=== Dual-Label XYZT Segmentation Tool ===\n');
    fprintf('Instructions:\n');
    fprintf('- Layer Navigation:\n');
    fprintf('  q/w: previous/next layer (±1)\n');
    fprintf('  a/s: jump layers (±2)\n');
    fprintf('  z/x: large jump (±4)\n');
    fprintf('  h: return to middle layer\n');
    fprintf('- Frame Navigation:\n');
    fprintf('  Left/Right arrows or slider\n');
    fprintf('- Segmentation:\n');
    fprintf('  f: switch between labels\n');
    fprintf('  Space: start segmentation\n');
    fprintf('  Escape: finish and save\n');
    fprintf('=========================================\n');
    
    % Wait for user action
    waitfor(fig);
    
    % Prepare output data for XYZT format
    if exist('segmented_frames', 'var')
        segmentation_data.label1.segmented_frames = segmented_frames{1};
        segmentation_data.label1.masks = masks{1};
        segmentation_data.label1.num_segmented_per_layer = sum(segmented_frames{1}, 2);
        segmentation_data.label1.total_segmented = sum(segmented_frames{1}(:));
        
        segmentation_data.label2.segmented_frames = segmented_frames{2};
        segmentation_data.label2.masks = masks{2};
        segmentation_data.label2.num_segmented_per_layer = sum(segmented_frames{2}, 2);
        segmentation_data.label2.total_segmented = sum(segmented_frames{2}(:));
        
        segmentation_data.num_layers = num_layers;
        segmentation_data.num_frames = num_frames;
        segmentation_data.total_possible = num_layers * num_frames;
        
        fprintf('Segmentation finished!\n');
        fprintf('Label 1: %d total segments across %d layers\n', ...
                segmentation_data.label1.total_segmented, num_layers);
        fprintf('Label 2: %d total segments across %d layers\n', ...
                segmentation_data.label2.total_segmented, num_layers);
    else
        segmentation_data = struct();
        segmentation_data.label1 = struct('segmented_frames', [], 'masks', {{}}, ...
                                         'num_segmented_per_layer', [], 'total_segmented', 0);
        segmentation_data.label2 = struct('segmented_frames', [], 'masks', {{}}, ...
                                         'num_segmented_per_layer', [], 'total_segmented', 0);
        segmentation_data.num_layers = num_layers;
        segmentation_data.num_frames = num_frames;
        segmentation_data.total_possible = num_layers * num_frames;
    end
    
    %% Callback functions
    function stats_str = updateStatsString()
        num_seg1_layer = sum(segmented_frames{1}(current_layer,:));
        num_seg2_layer = sum(segmented_frames{2}(current_layer,:));
        num_seg1_total = sum(segmented_frames{1}(:));
        num_seg2_total = sum(segmented_frames{2}(:));
        stats_str = sprintf('Layer: L1:%d/%d L2:%d/%d | Total: L1:%d L2:%d', ...
                           num_seg1_layer, num_frames, num_seg2_layer, num_frames, ...
                           num_seg1_total, num_seg2_total);
    end
    
    function switchLabel(~, ~)
        current_label = 3 - current_label;  % Toggle between 1 and 2
        set(btn_label, 'String', sprintf('Current: %s', label_names{current_label}), ...
            'BackgroundColor', label_colors{current_label});
        updateFrame(current_layer, current_frame);
        fprintf('Switched to %s\n', label_names{current_label});
    end
    
    function sliderTimeCallback(~, ~)
        new_frame = round(get(slider_time, 'Value'));
        updateFrame(current_layer, new_frame);
    end
    
    function sliderLayerCallback(~, ~)
        new_layer = round(get(slider_layer, 'Value'));
        updateFrame(new_layer, current_frame);
    end
    
    function keyPressCallback(~, event)
        switch event.Key
            % Frame navigation
            case 'leftarrow'
                updateFrame(current_layer, max(1, current_frame - 1));
            case 'rightarrow'
                updateFrame(current_layer, min(num_frames, current_frame + 1));
            
            % Layer navigation
            case 'q'  % Previous layer (-1)
                updateFrame(max(1, current_layer - 1), current_frame);
            case 'w'  % Next layer (+1)
                updateFrame(min(num_layers, current_layer + 1), current_frame);
            case 'a'  % Jump back (-2)
                updateFrame(max(1, current_layer - 2), current_frame);
            case 's'  % Jump forward (+2)
                updateFrame(min(num_layers, current_layer + 2), current_frame);
            case 'z'  % Large jump back (-4)
                updateFrame(max(1, current_layer - 4), current_frame);
            case 'x'  % Large jump forward (+4)
                updateFrame(min(num_layers, current_layer + 4), current_frame);
            case 'h'  % Return to middle layer
                updateFrame(ceil(num_layers/2), current_frame);
            
            % Segmentation controls
            case 'space'
                startSegmentation();
            case 'f'  % Changed from 'q' to 'f'
                switchLabel();
            case 'escape'
                finishSegmentation();
        end
    end
    
    function scrollWheelCallback(~, event)
        delta = -15*event.VerticalScrollCount;
        updateFrame(current_layer, max(1, min(num_frames, current_frame + delta)));
    end
    
    function resizeCallback(~, ~)
        drawnow;
    end
    
    function updateFrame(new_layer, new_frame)
        current_layer = new_layer;
        current_frame = new_frame;
        set(slider_time, 'Value', current_frame);
        set(slider_layer, 'Value', current_layer);
        
        % Update layer indicator
        set(text_layer, 'String', sprintf('Z-Layer: %d/%d', current_layer, num_layers));
        
        % Show current frame
        set(img_handle, 'CData', img_4d(:,:,current_layer,current_frame));
        
        % Update title with segmentation status
        status_str = '';
        if segmented_frames{1}(current_layer,current_frame) && segmented_frames{2}(current_layer,current_frame)
            status_str = '(L1+L2)';
        elseif segmented_frames{1}(current_layer,current_frame)
            status_str = '(L1)';
        elseif segmented_frames{2}(current_layer,current_frame)
            status_str = '(L2)';
        end
        title(ax_img, sprintf('Layer %d/%d, Frame %d/%d %s', ...
              current_layer, num_layers, current_frame, num_frames, status_str));

        % Display masks for both labels
        axes(ax_img);
        hold on;
        delete(findobj(ax_img, 'Type', 'line'));
        
        % Show both masks with different colors
        for label_idx = 1:2
            if segmented_frames{label_idx}(current_layer,current_frame) && ...
               ~isempty(masks{label_idx}{current_layer,current_frame})
                try
                    contours = contourc(double(masks{label_idx}{current_layer,current_frame}), [0.5, 0.5]);
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
                    [r, c] = find(masks{label_idx}{current_layer,current_frame});
                    if ~isempty(r)
                        plot(c, r, '.', 'Color', [label_colors{label_idx}, 0.5], ...
                            'MarkerSize', 2);
                    end
                end
            end
        end
        hold off;
        
        % Update progress bars and indicators
        updateProgressBar();
        updateCurrentFrameIndicator();
    end
    
    function startSegmentation(~, ~)
        axes(ax_img);
        try
            fprintf('Draw polygon for %s on Layer %d, Frame %d\n', ...
                    label_names{current_label}, current_layer, current_frame);
            
            % Disable controls
            set([btn_segment, btn_clear, btn_finish, btn_label], 'Enable', 'off');
            
            % Draw polygon with current label color
            h_poly = drawpolygon('Color', label_colors{current_label}, 'LineWidth', 2);
            
            % Re-enable controls
            set([btn_segment, btn_clear, btn_finish, btn_label], 'Enable', 'on');
            
            if isvalid(h_poly) && ~isempty(h_poly.Position) && size(h_poly.Position, 1) >= 3
                % Create mask
                mask = createMask(h_poly, img_handle);
                masks{current_label}{current_layer,current_frame} = mask;
                segmented_frames{current_label}(current_layer,current_frame) = true;
                
                delete(h_poly);
                updateFrame(current_layer, current_frame);
                updateProgressBar();
                
                fprintf('%s segmentation complete for Layer %d, Frame %d\n', ...
                        label_names{current_label}, current_layer, current_frame);
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
        masks{current_label}{current_layer,current_frame} = [];
        segmented_frames{current_label}(current_layer,current_frame) = false;
        updateFrame(current_layer, current_frame);
        updateProgressBar();
        fprintf('%s cleared for Layer %d, Frame %d\n', ...
                label_names{current_label}, current_layer, current_frame);
    end
    
    function updateProgressBar()
        % Update progress bars for current layer
        set(progress_handle1, 'CData', double(squeeze(segmented_frames{1}(current_layer,:))));
        set(progress_handle2, 'CData', double(squeeze(segmented_frames{2}(current_layer,:))));
        
        % Update statistics
        set(text_stats, 'String', updateStatsString());
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
        fprintf('Finished! Total segments - L1: %d, L2: %d\n', ...
                sum(segmented_frames{1}(:)), sum(segmented_frames{2}(:)));
        delete(fig);
    end
end