// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "seg.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "cpu.h"

#define MAX_STRIDE 32



SegNail::SegNail()
{
    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);
}


int SegNail::load(AAssetManager* mgr, bool use_gpu)
{
    seg.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    seg.opt = ncnn::Option();

#if NCNN_VULKAN
    seg.opt.use_vulkan_compute = use_gpu;
#endif

    seg.opt.num_threads = ncnn::get_big_cpu_count();
    seg.opt.blob_allocator = &blob_pool_allocator;
    seg.opt.workspace_allocator = &workspace_pool_allocator;

//    char parampath[256];
//    char modelpath[256];
//    sprintf(parampath, "yolov8%s-seg-sim-opt-fp16.param", modeltype);
//    sprintf(modelpath, "yolov8%s-seg-sim-opt-fp16.bin", modeltype);

    seg.load_param(mgr,"sim.param");
    seg.load_model(mgr,"sim.bin");

    mean_vals[0] = 0.37802792*255.0;
    mean_vals[1] = 0.32611448*255.0;
    mean_vals[2] = 0.29480308*255.0;
    norm_vals[0] = 1 / 0.348492 / 255.0;
    norm_vals[1] = 1 / 0.3070657 / 255.0;
    norm_vals[2] = 1 / 0.28770673 / 255.0;

    return 0;
}

int SegNail::detect(const cv::Mat& rgb)
{
    int width = rgb.cols;
    int height = rgb.rows;


    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB, width, height, input_w, input_h);

    // pad to target_size rectangle

    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = seg.create_extractor();

    ex.input("input", in);

    ncnn::Mat out;
    ex.extract("output", out);

//    const float* pCls0 = out.channel(0);
//    const float* pCls1 = out.channel(1);
//
//    for (int c = 0; c < 3; c++)
//    {
//        float* pImage = in.channel(c);
//        for (int i = 0; i < output_h*output_w; i++){
//            pImage[i] = pCls0[i] < pCls1[i]?pImage[i]*1:pImage[i]*0;
//        }
//    }

    LOGD("hhhhh h:%d,w:%d",out.h, out.w);

    cv::Mat mask(out.h, out.w, CV_8UC1);
    const float* maskmap0 = out.channel(0);
    const float* maskmap1 = out.channel(1);

    for (int i{ 0 }; i < out.h; i++) {
        for (int j{ 0 }; j < out.w; ++j) {
            mask.at<uchar>(i, j) = maskmap1[i * out.w + j] < maskmap0[i * out.w + j] ? 255 : 0;
        }
    }
    cv::resize(mask, mask, cv::Size(rgb.cols, rgb.rows), 0, 0);
//    cv::Mat segFrame;
//    cv::bitwise_and(rgb, rgb, segFrame, mask = mask);


    cv::Mat segmentedImg;
    cv::bitwise_and(rgb, rgb, segmentedImg, mask);

    segmentedImg.copyTo(rgb); // 将 segmentedImg 的内容赋值回 rgb


    return 0;
}

