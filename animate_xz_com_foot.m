function animate_xz_com_foot(x_pos, foot_pos, z0)
% x_pos, foot_pos: Nx1 vectors (1D x positions)
% z0: CoM height (scalar)

    x_pos    = x_pos(:);
    foot_pos = foot_pos(:);

    if length(x_pos) ~= length(foot_pos)
        error('x_pos and foot_pos must have the same length.');
    end
    if nargin < 3 || isempty(z0)
        z0 = 1;
    end

    N = length(x_pos);

    % --- tuning ---
    framesPerSegment = 30;
    basePause = 0.02;
    speedupWithDistance = true;

    % --- figure setup ---
    figure; hold on; grid on;
    axis equal;
    xlabel('x');
    ylabel('z');
    title('CoM + Foot (x–z animation)');

    allx = [x_pos; foot_pos];
    xmin = min(allx); xmax = max(allx);
    pad  = 0.1 * max(xmax - xmin, 1e-6);
    xlim([xmin - pad, xmax + pad]);
    ylim([-0.2*z0, 1.2*z0]);

    % optional: faint guide lines
    plot([xmin - pad, xmax + pad], [0, 0], '-');     % ground
    plot(x_pos,  z0*ones(N,1), ':');                 % CoM path (top)
    plot(foot_pos, zeros(N,1), ':');                 % foot path (ground)

    % --- graphics objects ---
    hCom  = plot(x_pos(1),  z0,  'o', 'MarkerSize', 8, 'MarkerFaceColor','auto');
    hFoot = plot(foot_pos(1), 0, 's', 'MarkerSize', 8, 'MarkerFaceColor','auto');
    hLeg  = plot([foot_pos(1), x_pos(1)], [0, z0], '-', 'LineWidth', 2);

    % --- animation loop ---
    for i = 1:(N-1)
        foot = foot_pos(i);   % fixed during this segment
        x0 = x_pos(i);
        x1 = x_pos(i+1);

        dist = abs(x1 - x0);

        if speedupWithDistance
            pausePerFrame = basePause / max(dist, 0.05);      % farther => faster
            pausePerFrame = min(max(pausePerFrame, 0.002), 0.05);
        else
            pausePerFrame = basePause;
        end

        for k = 0:framesPerSegment
            t = k / framesPerSegment;
            com = (1 - t)*x0 + t*x1;

            set(hCom,  'XData', com,  'YData', z0);
            set(hFoot, 'XData', foot, 'YData', 0);
            set(hLeg,  'XData', [foot, com], 'YData', [0, z0]);

            drawnow limitrate;
            pause(pausePerFrame);
        end

        % update foot at node
        foot = foot_pos(i+1);
        set(hFoot, 'XData', foot, 'YData', 0);
        set(hLeg,  'XData', [foot, x1], 'YData', [0, z0]);
        drawnow;
    end
end
