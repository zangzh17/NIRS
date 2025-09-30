function visualizeTransformedSquares(backward_rel_list)
    [nViews, nDepths] = size(backward_rel_list);
    % Visualize transformation based on squares
    square1 = [220 220; 420 220; 420 420; 220 420; 220 220]; % a closed square
    square2 = [270 270; 370 270; 370 370; 270 370; 270 270]; % a closed square
    figure;
    for i = 1:nViews
        subplot(2,3,i)
        hold on; grid on;
        xlabel('X');
        ylabel('Y');
        zlabel('Depth');
        title(['View ', num2str(i)]);
        colors = abyss(nDepths);
        for depth = 1:nDepths
            T = backward_rel_list(i, depth);
            [xTrans, yTrans] = transformPointsForward(T, square1(:,1), square1(:,2));
            zCoord = depth * ones(size(xTrans));
            plot3(xTrans, yTrans, zCoord, 'Color', colors(depth,:), 'LineWidth', 2);
        end
        colors = copper(nDepths);
        for depth = 1:nDepths
            T = backward_rel_list(i, depth);
            [xTrans, yTrans] = transformPointsForward(T, square2(:,1), square2(:,2));
            zCoord = depth * ones(size(xTrans));
            plot3(xTrans, yTrans, zCoord, 'Color', colors(depth,:), 'LineWidth', 2);
        end
        view(2)
        hold off;
    end
end