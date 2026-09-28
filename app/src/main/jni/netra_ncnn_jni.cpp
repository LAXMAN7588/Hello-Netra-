#include <jni.h>
#include <android/bitmap.h>
#include "net.h"

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_example_objectdetection_NcnnDetector_nativeInit(
        JNIEnv* env,
        jobject thiz,
        jobject assetManager) {

    ncnn::Net net;

    net.opt.num_threads = 4;
    net.opt.use_vulkan_compute = false;

    return JNI_TRUE;
}
