package com.tencent.mobilenetssdncnn;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Log;
import android.util.Size;
import android.view.Surface;
import android.view.TextureView;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.Toast;

import java.util.Collections;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

public class MainActivity extends Activity {
    private static final String TAG = "MainActivity";
    private static final int PERMISSION_REQUEST_CAMERA = 101;

    private TextureView cameraTextureView;
    private OverlayView overlayView;
    private Button btnToggleGpu;
    private Button btnSwitchCamera;
    private Button btnToggleInference;

    private final MobilenetSSDNcnn mobilenetssdncnn = new MobilenetSSDNcnn();

    private CameraDevice cameraDevice;
    private CameraCaptureSession captureSession;
    private CaptureRequest.Builder previewRequestBuilder;

    private HandlerThread backgroundThread;
    private Handler backgroundHandler;

    private HandlerThread inferenceThread;
    private Handler inferenceHandler;

    private String currentCameraId;
    private int facing = CameraCharacteristics.LENS_FACING_BACK;
    private boolean useGpu = false;
    private boolean inferenceEnabled = true;

    private final Semaphore cameraOpenCloseLock = new Semaphore(1);
    private final AtomicBoolean isInferencing = new AtomicBoolean(false);
    private Bitmap reusableBitmap = null;

    private float smoothedLatency = 0f;
    private float smoothedFps = 0f;
    private long lastFrameTimestamp = 0;

    private final TextureView.SurfaceTextureListener surfaceTextureListener = new TextureView.SurfaceTextureListener() {
        @Override
        public void onSurfaceTextureAvailable(SurfaceTexture surface, int width, int height) {
            openCamera();
        }

        @Override
        public void onSurfaceTextureSizeChanged(SurfaceTexture surface, int width, int height) {}

        @Override
        public boolean onSurfaceTextureDestroyed(SurfaceTexture surface) {
            return true;
        }

        @Override
        public void onSurfaceTextureUpdated(SurfaceTexture surface) {
            if (!inferenceEnabled || isInferencing.get()) {
                return;
            }

            if (isInferencing.compareAndSet(false, true)) {
                if (inferenceHandler != null) {
                    inferenceHandler.post(new Runnable() {
                        @Override
                        public void run() {
                            processCurrentFrame();
                        }
                    });
                } else {
                    isInferencing.set(false);
                }
            }
        }
    };

    private final CameraDevice.StateCallback stateCallback = new CameraDevice.StateCallback() {
        @Override
        public void onOpened(CameraDevice camera) {
            cameraOpenCloseLock.release();
            cameraDevice = camera;
            createCameraPreviewSession();
        }

        @Override
        public void onDisconnected(CameraDevice camera) {
            cameraOpenCloseLock.release();
            camera.close();
            cameraDevice = null;
        }

        @Override
        public void onError(CameraDevice camera, int error) {
            cameraOpenCloseLock.release();
            camera.close();
            cameraDevice = null;
            Log.e(TAG, "CameraDevice error: " + error);
        }
    };

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.main);

        // Initialize ncnn MobileNet-SSD
        boolean retInit = mobilenetssdncnn.Init(getAssets());
        if (!retInit) {
            Log.e(TAG, "mobilenetssdncnn Init failed");
            Toast.makeText(this, "ncnn model init failed", Toast.LENGTH_LONG).show();
        } else {
            Log.i(TAG, "mobilenetssdncnn Init successful");
        }

        cameraTextureView = findViewById(R.id.cameraTextureView);
        overlayView = findViewById(R.id.overlayView);
        btnToggleGpu = findViewById(R.id.btnToggleGpu);
        btnSwitchCamera = findViewById(R.id.btnSwitchCamera);
        btnToggleInference = findViewById(R.id.btnToggleInference);

        btnToggleGpu.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                useGpu = !useGpu;
                btnToggleGpu.setText(useGpu ? "Mode: GPU" : "Mode: CPU");
                btnToggleGpu.setBackgroundColor(useGpu ? 0xFF9C27B0 : 0xFF3366CC);
            }
        });

        btnSwitchCamera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                facing = (facing == CameraCharacteristics.LENS_FACING_BACK) ?
                        CameraCharacteristics.LENS_FACING_FRONT : CameraCharacteristics.LENS_FACING_BACK;
                btnSwitchCamera.setText(facing == CameraCharacteristics.LENS_FACING_BACK ? "Back Cam" : "Front Cam");
                if (backgroundHandler != null) {
                    backgroundHandler.post(new Runnable() {
                        @Override
                        public void run() {
                            closeCamera();
                            openCamera();
                        }
                    });
                }
            }
        });

        btnToggleInference.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                inferenceEnabled = !inferenceEnabled;
                btnToggleInference.setText(inferenceEnabled ? "Running" : "Paused");
                btnToggleInference.setBackgroundColor(inferenceEnabled ? 0xFF2E7D32 : 0xFFB71C1C);
                if (!inferenceEnabled) {
                    overlayView.setResults(null, 300, 300, 0f, 0f, useGpu,
                            facing == CameraCharacteristics.LENS_FACING_FRONT, "Paused");
                }
            }
        });
    }

    @Override
    protected void onResume() {
        super.onResume();
        startBackgroundThreads();

        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.CAMERA}, PERMISSION_REQUEST_CAMERA);
        } else {
            if (cameraTextureView.isAvailable()) {
                openCamera();
            } else {
                cameraTextureView.setSurfaceTextureListener(surfaceTextureListener);
            }
        }
    }

    @Override
    protected void onPause() {
        closeCamera();
        stopBackgroundThreads();
        super.onPause();
    }

    private void startBackgroundThreads() {
        backgroundThread = new HandlerThread("CameraBackground");
        backgroundThread.start();
        backgroundHandler = new Handler(backgroundThread.getLooper());

        inferenceThread = new HandlerThread("InferenceWorker");
        inferenceThread.start();
        inferenceHandler = new Handler(inferenceThread.getLooper());
    }

    private void stopBackgroundThreads() {
        if (backgroundThread != null) {
            backgroundThread.quitSafely();
            try {
                backgroundThread.join();
                backgroundThread = null;
                backgroundHandler = null;
            } catch (InterruptedException e) {
                Log.e(TAG, "stopBackgroundThreads interrupted", e);
            }
        }
        if (inferenceThread != null) {
            inferenceThread.quitSafely();
            try {
                inferenceThread.join();
                inferenceThread = null;
                inferenceHandler = null;
            } catch (InterruptedException e) {
                Log.e(TAG, "stopInferenceThreads interrupted", e);
            }
        }
    }

    private void openCamera() {
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            return;
        }

        CameraManager manager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
        try {
            if (!cameraOpenCloseLock.tryAcquire(2500, TimeUnit.MILLISECONDS)) {
                throw new RuntimeException("Time out waiting to lock camera opening.");
            }

            currentCameraId = null;
            for (String id : manager.getCameraIdList()) {
                CameraCharacteristics characteristics = manager.getCameraCharacteristics(id);
                Integer lensFacing = characteristics.get(CameraCharacteristics.LENS_FACING);
                if (lensFacing != null && lensFacing == facing) {
                    currentCameraId = id;
                    break;
                }
            }

            if (currentCameraId == null && manager.getCameraIdList().length > 0) {
                currentCameraId = manager.getCameraIdList()[0];
            }

            manager.openCamera(currentCameraId, stateCallback, backgroundHandler);
        } catch (CameraAccessException e) {
            Log.e(TAG, "Cannot access camera", e);
            cameraOpenCloseLock.release();
        } catch (InterruptedException e) {
            throw new RuntimeException("Interrupted while trying to lock camera opening.", e);
        }
    }

    private void closeCamera() {
        try {
            cameraOpenCloseLock.acquire();
            if (captureSession != null) {
                captureSession.close();
                captureSession = null;
            }
            if (cameraDevice != null) {
                cameraDevice.close();
                cameraDevice = null;
            }
        } catch (InterruptedException e) {
            throw new RuntimeException("Interrupted while trying to lock camera closing.", e);
        } finally {
            cameraOpenCloseLock.release();
        }
    }

    private void createCameraPreviewSession() {
        try {
            SurfaceTexture texture = cameraTextureView.getSurfaceTexture();
            if (texture == null) return;

            CameraManager manager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
            CameraCharacteristics characteristics = manager.getCameraCharacteristics(currentCameraId);
            StreamConfigurationMap map = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
            if (map != null) {
                Size[] sizes = map.getOutputSizes(SurfaceTexture.class);
                if (sizes != null && sizes.length > 0) {
                    Size optimalSize = sizes[0];
                    for (Size sz : sizes) {
                        if (sz.getWidth() <= 1280 && sz.getHeight() <= 720) {
                            optimalSize = sz;
                            break;
                        }
                    }
                    texture.setDefaultBufferSize(optimalSize.getWidth(), optimalSize.getHeight());
                }
            }

            Surface surface = new Surface(texture);
            previewRequestBuilder = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
            previewRequestBuilder.addTarget(surface);
            previewRequestBuilder.set(CaptureRequest.CONTROL_MODE, CaptureRequest.CONTROL_MODE_AUTO);
            previewRequestBuilder.set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE);
            previewRequestBuilder.set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_ON);

            cameraDevice.createCaptureSession(Collections.singletonList(surface), new CameraCaptureSession.StateCallback() {
                @Override
                public void onConfigured(CameraCaptureSession session) {
                    if (cameraDevice == null) return;
                    captureSession = session;
                    try {
                        captureSession.setRepeatingRequest(previewRequestBuilder.build(), null, backgroundHandler);
                    } catch (CameraAccessException e) {
                        Log.e(TAG, "Failed to start camera preview", e);
                    }
                }

                @Override
                public void onConfigureFailed(CameraCaptureSession session) {
                    Log.e(TAG, "Camera configuration failed");
                }
            }, backgroundHandler);
        } catch (CameraAccessException e) {
            Log.e(TAG, "createCameraPreviewSession error", e);
        }
    }

    private void processCurrentFrame() {
        try {
            if (reusableBitmap == null || reusableBitmap.getWidth() != 300 || reusableBitmap.getHeight() != 300) {
                reusableBitmap = Bitmap.createBitmap(300, 300, Bitmap.Config.ARGB_8888);
            }

            Bitmap frame = cameraTextureView.getBitmap(reusableBitmap);
            if (frame == null) {
                isInferencing.set(false);
                return;
            }

            long startTime = System.currentTimeMillis();
            MobilenetSSDNcnn.Obj[] objects = mobilenetssdncnn.Detect(frame, useGpu);
            long endTime = System.currentTimeMillis();

            long latency = Math.max(1, endTime - startTime);
            if (smoothedLatency == 0) {
                smoothedLatency = latency;
            } else {
                smoothedLatency = 0.8f * smoothedLatency + 0.2f * latency;
            }

            if (lastFrameTimestamp > 0) {
                long frameInterval = Math.max(1, endTime - lastFrameTimestamp);
                float instantaneousFps = 1000f / frameInterval;
                if (smoothedFps == 0) {
                    smoothedFps = instantaneousFps;
                } else {
                    smoothedFps = 0.85f * smoothedFps + 0.15f * instantaneousFps;
                }
            }
            lastFrameTimestamp = endTime;

            final MobilenetSSDNcnn.Obj[] finalObjects = objects;
            final float curLatency = smoothedLatency;
            final float curFps = smoothedFps;
            final boolean curGpu = useGpu;
            final boolean isFront = (facing == CameraCharacteristics.LENS_FACING_FRONT);

            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    overlayView.setResults(finalObjects, 300, 300, curLatency, curFps, curGpu, isFront, "Active");
                }
            });
        } catch (Exception e) {
            Log.e(TAG, "Error in processCurrentFrame", e);
        } finally {
            isInferencing.set(false);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        if (requestCode == PERMISSION_REQUEST_CAMERA) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                if (cameraTextureView.isAvailable()) {
                    openCamera();
                } else {
                    cameraTextureView.setSurfaceTextureListener(surfaceTextureListener);
                }
            } else {
                Toast.makeText(this, "Camera permission is required for live detection", Toast.LENGTH_LONG).show();
            }
        }
    }
}
