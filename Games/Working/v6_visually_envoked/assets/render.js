// This file contains the clientside JavaScript function for rendering the game.

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

    const PADDLE_WIDTH = 150;
    const PADDLE_HEIGHT = 20;
    const BALL_RADIUS = 10;

    // --- 1. Clear and Draw Background ---
    ctx.fillStyle = '#1a1a1a';
    ctx.fillRect(0, 0, W, H);

    // --- 2. Handle Flicker Logic ---
    const currentTimeS = n_intervals * interval_ms / 1000.0;
    const isLeftOn = Math.sin(2 * Math.PI * freqLeft * currentTimeS) > 0;
    const isRightOn = Math.sin(2 * Math.PI * freqRight * currentTimeS) > 0;
    const status = appStatus.status;

    let leftColor = '#003333';
    let rightColor = '#330033';
    let allowLeftFlicker = false;
    let allowRightFlicker = false;

    if (status.includes('CALIBRATING_LEFT')) {
        allowLeftFlicker = true;
    } else if (status.includes('CALIBRATING_RIGHT')) {
        allowRightFlicker = true;
    } else if (status.includes('CALIBRATING_REST')) {
        // --- FIX: Backgrounds now flicker during the rest phase ---
        allowLeftFlicker = true;
        allowRightFlicker = true;
    } else if (status.includes('PLAYING') || status.includes('READY')) {
        allowLeftFlicker = true;
        allowRightFlicker = true;
    }

    if (allowLeftFlicker && isLeftOn) leftColor = 'cyan';
    if (allowRightFlicker && isRightOn) rightColor = 'magenta';

    ctx.fillStyle = leftColor;
    ctx.fillRect(0, 0, W * 0.25, H);
    ctx.fillStyle = rightColor;
    ctx.fillRect(W * 0.75, 0, W * 0.25, H);

    // --- 3. Draw Game Objects (Player is at the bottom) ---
    const { player_x, ai_x, ball_x, ball_y, player_score, ai_score } = gameState;
    
    // AI (magenta) is at the top (y=0)
    ctx.fillStyle = 'magenta';
    ctx.fillRect(ai_x - PADDLE_WIDTH / 2, 0, PADDLE_WIDTH, PADDLE_HEIGHT);

    // Player (cyan) is at the bottom (y = H - PADDLE_HEIGHT)
    ctx.fillStyle = 'cyan';
    ctx.fillRect(player_x - PADDLE_WIDTH / 2, H - PADDLE_HEIGHT, PADDLE_WIDTH, PADDLE_HEIGHT);

    // Ball
    ctx.fillStyle = 'white';
    ctx.beginPath();
    ctx.arc(ball_x, ball_y, BALL_RADIUS, 0, 2 * Math.PI);
    ctx.fill();

    // --- 4. Draw Scores (Player is bottom-left) ---
    ctx.font = "40px monospace";
    ctx.textAlign = "left";

    // AI Score (top left)
    ctx.fillStyle = 'magenta';
    ctx.fillText(ai_score, 30, 50);

    // Player Score (bottom left)
    ctx.fillStyle = 'cyan';
    ctx.fillText(player_score, 30, H - 30);
};