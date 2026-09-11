package com.tencent.mobilenetssdncnn;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.View;

import java.util.Locale;

public class OverlayView extends View {

    private MobilenetSSDNcnn.Obj[] objects = null;
    private int inputWidth = 300;
    private int inputHeight = 300;
    private float inferenceTimeMs = 0f;
    private float fps = 0f;
    private boolean useGpu = false;
    private boolean isFrontCamera = false;
    private String statusMessage = "Active";

    private final Paint boxPaint = new Paint();
    private final Paint textPaint = new Paint();
    private final Paint labelBgPaint = new Paint();
    private final Paint statsBgPaint = new Paint();
    private final Paint statsTextPaint = new Paint();

    private final int[] colors = new int[]{
            Color.rgb(0, 230, 118),   // Bright Green
            Color.rgb(41, 121, 255),  // Bright Blue
            Color.rgb(255, 145, 0),   // Bright Orange
            Color.rgb(245, 0, 87),    // Bright Pink
            Color.rgb(213, 0, 249),   // Magenta
            Color.rgb(0, 229, 255),   // Bright Cyan
            Color.rgb(255, 214, 0)    // Bright Yellow
    };

    public OverlayView(Context context) {
        super(context);
        init();
    }

    public OverlayView(Context context, AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    public OverlayView(Context context, AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
        init();
    }

    private void init() {
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(6f);
        boxPaint.setAntiAlias(true);

        textPaint.setColor(Color.WHITE);
        textPaint.setTextSize(34f);
        textPaint.setFakeBoldText(true);
        textPaint.setAntiAlias(true);

        labelBgPaint.setStyle(Paint.Style.FILL);
        labelBgPaint.setAntiAlias(true);

        statsBgPaint.setColor(Color.argb(190, 20, 20, 20));
        statsBgPaint.setStyle(Paint.Style.FILL);
        statsBgPaint.setAntiAlias(true);

        statsTextPaint.setColor(Color.WHITE);
        statsTextPaint.setTextSize(36f);
        statsTextPaint.setAntiAlias(true);
        statsTextPaint.setFakeBoldText(true);
    }

    public void setResults(MobilenetSSDNcnn.Obj[] objects, int inWidth, int inHeight,
                           float inferenceTimeMs, float fps, boolean useGpu, boolean isFrontCamera, String statusMessage) {
        this.objects = objects;
        this.inputWidth = inWidth;
        this.inputHeight = inHeight;
        this.inferenceTimeMs = inferenceTimeMs;
        this.fps = fps;
        this.useGpu = useGpu;
        this.isFrontCamera = isFrontCamera;
        this.statusMessage = statusMessage;
        postInvalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);

        int viewWidth = getWidth();
        int viewHeight = getHeight();

        // 1. Draw top stats panel
        float panelHeight = 160f;
        RectF statsRect = new RectF(24, 24, viewWidth - 24, panelHeight);
        canvas.drawRoundRect(statsRect, 20f, 20f, statsBgPaint);

        String statsLine1 = String.format(Locale.US, "? %s  |  ? %.1f ms  |  ?? %.1f FPS",
                useGpu ? "GPU (Vulkan)" : "CPU (4-Thread)", inferenceTimeMs, fps);
        String statsLine2 = String.format(Locale.US, "? %s  |  ?? Detections: %d",
                statusMessage, objects == null ? 0 : objects.length);

        canvas.drawText(statsLine1, 48, 80, statsTextPaint);

        statsTextPaint.setTextSize(30f);
        statsTextPaint.setColor(Color.rgb(180, 215, 255));
        canvas.drawText(statsLine2, 48, 130, statsTextPaint);

        statsTextPaint.setTextSize(36f);
        statsTextPaint.setColor(Color.WHITE);

        if (objects == null || objects.length == 0 || inputWidth <= 0 || inputHeight <= 0) {
            return;
        }

        float scaleX = (float) viewWidth / inputWidth;
        float scaleY = (float) viewHeight / inputHeight;

        for (int i = 0; i < objects.length; i++) {
            MobilenetSSDNcnn.Obj obj = objects[i];
            if (obj == null) continue;

            int color = colors[Math.abs(obj.label != null ? obj.label.hashCode() : i) % colors.length];
            boxPaint.setColor(color);
            labelBgPaint.setColor(color);

            float left = obj.x * scaleX;
            float top = obj.y * scaleY;
            float right = (obj.x + obj.w) * scaleX;
            float bottom = (obj.y + obj.h) * scaleY;

            if (isFrontCamera) {
                float origLeft = left;
                left = viewWidth - right;
                right = viewWidth - origLeft;
            }

            left = Math.max(0, left);
            top = Math.max(0, top);
            right = Math.min(viewWidth, right);
            bottom = Math.min(viewHeight, bottom);

            // Draw bounding box
            canvas.drawRoundRect(new RectF(left, top, right, bottom), 10f, 10f, boxPaint);

            // Format label with confidence
            String label = String.format(Locale.US, "%s %.1f%%", obj.label != null ? obj.label : "object", obj.prob * 100f);
            float textWidth = textPaint.measureText(label);
            float textHeight = -textPaint.ascent() + textPaint.descent();

            float labelTop = top - textHeight - 12;
            if (labelTop < panelHeight + 10) {
                labelTop = top + 10;
            }
            float labelBottom = labelTop + textHeight + 12;
            float labelRight = Math.min(viewWidth, left + textWidth + 24);

            RectF labelBg = new RectF(left, labelTop, labelRight, labelBottom);
            canvas.drawRoundRect(labelBg, 8f, 8f, labelBgPaint);

            canvas.drawText(label, left + 12, labelBottom - textPaint.descent() - 6, textPaint);
        }
    }
}
