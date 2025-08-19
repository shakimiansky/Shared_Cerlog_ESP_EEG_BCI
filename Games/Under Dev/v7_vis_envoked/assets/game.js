// This file should be saved as 'pong.js' inside an 'assets' folder.

if (!window.dash_clientside) {
    window.dash_clientside = {};
}

window.dash_clientside.renderPong = function(canvasId, gameState, appStatus, n_intervals, interval_ms, freqLeft, freqRight) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || !gameState || !appStatus) {
        return;
    }
    const ctx = canvas.getContext('2d');
    const W = canvas.width;
    const H = canvas.height;

    // --- Define constants for easy access ---
    const PADDLE_WIDTH = 150;
    const PADDLE_HEIGHT = 20;
    const BALL_RADIUS = 10;
    const FLICKER_ZONE_WIDTH = W * 0.25;

    const COLOR_BACKGROUND = '#1a1a1a';
    const COLOR_CENTER_COURT = '#111111';
    const COLOR_LEFT_OFF = '#003333';
    const COLOR_LEFT_ON = 'cyan';
    const COLOR_RIGHT_OFF = '#330033';
    const COLOR_RIGHT_ON = 'magenta';
    const COLOR_PADDLE_PLAYER = 'cyan';
    const COLOR_PADDLE_AI = 'magenta';
    const COLOR_BALL = 'white';

    // --- 1. Draw Backgrounds ---
    ctx.fillStyle = COLOR_BACKGROUND;
    ctx.fillRect(0, 0, W, H);
    
    // Make the center court darker to visually separate flicker zones
    ctx.fillStyle = COLOR_CENTER_COURT;
    ctx.fillRect(FLICKER_ZONE_WIDTH, 0, W - 2 * FLICKER_ZONE_WIDTH, H);

    // --- 2. Handle Flicker Logic ---
    const currentTimeS = n_intervals * interval_ms / 1000.0;
    const isLeftOn = Math.sin(2 * Math.PI * freqLeft * currentTimeS) > 0;
    const isRightOn = Math.sin(2 * Math.PI * freqRight * currentTimeS) > 0;
    const status = appStatus.status;

    let leftColor = COLOR_LEFT_OFF;
    let rightColor = COLOR_RIGHT_OFF;
    let allowLeftFlicker = false;
    let allowRightFlicker = false;

    if (status.includes('CALIBRATING_LEFT')) {
        allowLeftFlicker = true;
    } else if (status.includes('CALIBRATING_RIGHT')) {
        allowRightFlicker = true;
    } else if (status.includes('CALIBRATING_REST') || status.includes('PLAYING') || status.includes('READY')) {
        allowLeftFlicker = true;
        allowRightFlicker = true;
    }

    if (allowLeftFlicker && isLeftOn) leftColor = COLOR_LEFT_ON;
    if (allowRightFlicker && isRightOn) rightColor = COLOR_RIGHT_ON;

    ctx.fillStyle = leftColor;
    ctx.fillRect(0, 0, FLICKER_ZONE_WIDTH, H);
    ctx.fillStyle = rightColor;
    ctx.fillRect(W - FLICKER_ZONE_WIDTH, 0, FLICKER_ZONE_WIDTH, H);
    
    // --- 3. Draw Center Line ---
    ctx.strokeStyle = '#444';
    ctx.lineWidth = 4;
    ctx.setLineDash([10, 15]);
    ctx.beginPath();
    ctx.moveTo(0, H / 2);
    ctx.lineTo(W, H / 2);
    ctx.stroke();
    ctx.setLineDash([]); // Reset line dash

    // --- 4. Draw Game Objects (Player is at the bottom) ---
    const { player_x, ai_x, ball_x, ball_y, player_score, ai_score } = gameState;
    
    // AI (magenta) is at the top (y=0)
    ctx.fillStyle = COLOR_PADDLE_AI;
    ctx.fillRect(ai_x - PADDLE_WIDTH / 2, 0, PADDLE_WIDTH, PADDLE_HEIGHT);

    // Player (cyan) is at the bottom (y = H - PADDLE_HEIGHT)
    ctx.fillStyle = COLOR_PADDLE_PLAYER;
    ctx.fillRect(player_x - PADDLE_WIDTH / 2, H - PADDLE_HEIGHT, PADDLE_WIDTH, PADDLE_HEIGHT);

    // Ball
    ctx.fillStyle = COLOR_BALL;
    ctx.beginPath();
    ctx.arc(ball_x, ball_y, BALL_RADIUS, 0, 2 * Math.PI);
    ctx.fill();

    // --- 5. Draw Scores (Bigger and at the top) ---
    ctx.font = "60px monospace";
    
    // Player Score (left of center)
    ctx.fillStyle = COLOR_PADDLE_PLAYER;
    ctx.textAlign = "right";
    ctx.fillText(player_score, W/2 - 40, 60);

    // AI Score (right of center)
    ctx.fillStyle = COLOR_PADDLE_AI;
    ctx.textAlign = "left";
    ctx.fillText(ai_score, W/2 + 40, 60);
};