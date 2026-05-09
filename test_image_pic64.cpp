#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <cfloat>

// 引入 STB 库用于读写图片
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// NCNN Headers
#include "net.h"
#include "cpu.h"

struct KeyPoint { float x, y, prob; };
struct YoloObject { 
    float x, y, w, h; 
    float prob; 
    std::vector<KeyPoint> keypoints; 
};

static const char* ACTION_NAMES[] = { "Lying", "Falling", "Sitting", "Standing" };
static const int joints[16][2] = {
    {0,1}, {1,3}, {0,2}, {2,4}, {5,6}, {5,7}, {7,9}, {6,8}, {8,10},
    {5,11}, {6,12}, {11,12}, {11,13}, {12,14}, {13,15}, {14,16}
};

static inline float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

// --- 简单的像素画图功能 ---
void draw_pixel(unsigned char* img, int w, int h, int x, int y, int r, int g, int b) {
    if (x < 0 || x >= w || y < 0 || y >= h) return;
    int idx = (y * w + x) * 3;
    img[idx] = r; img[idx+1] = g; img[idx+2] = b;
}

void draw_line(unsigned char* img, int w, int h, int x0, int y0, int x1, int y1, int r, int g, int b, int thickness = 3) {
    int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx + dy, e2;
    while (true) {
        for(int ty = -thickness/2; ty <= thickness/2; ty++)
            for(int tx = -thickness/2; tx <= thickness/2; tx++)
                draw_pixel(img, w, h, x0+tx, y0+ty, r, g, b);
        if (x0 == x1 && y0 == y1) break;
        e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
    }
}

void draw_rect(unsigned char* img, int w, int h, int x, int y, int rw, int rh, int r, int g, int b, int t = 3) {
    draw_line(img, w, h, x, y, x+rw, y, r, g, b, t);
    draw_line(img, w, h, x+rw, y, x+rw, y+rh, r, g, b, t);
    draw_line(img, w, h, x+rw, y+rh, x, y+rh, r, g, b, t);
    draw_line(img, w, h, x, y+rh, x, y, r, g, b, t);
}

static float intersection_area(const YoloObject& a, const YoloObject& b) {
    float x1 = std::max(a.x, b.x); float y1 = std::max(a.y, b.y);
    float x2 = std::min(a.x + a.w, b.x + b.w); float y2 = std::min(a.y + a.h, b.y + b.h);
    if (x1 >= x2 || y1 >= y2) return 0.f;
    return (x2 - x1) * (y2 - y1);
}

static void nms_sorted_bboxes(const std::vector<YoloObject>& objects, std::vector<int>& picked, float nms_threshold) {
    picked.clear();
    const int n = objects.size();
    std::vector<float> areas(n);
    for (int i = 0; i < n; i++) areas[i] = objects[i].w * objects[i].h;
    for (int i = 0; i < n; i++) {
        const YoloObject& a = objects[i]; int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++) {
            const YoloObject& b = objects[picked[j]];
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            if (inter_area / union_area > nms_threshold) keep = 0;
        }
        if (keep) picked.push_back(i);
    }
}

static void generate_proposals(const ncnn::Mat& feat_blob, int stride, float prob_threshold, std::vector<YoloObject>& objects) {
    const int w = feat_blob.w; const int h = feat_blob.h;
    const int reg_max_1 = 16; const int num_points = 17;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float score = sigmoid(feat_blob.channel(64).row(y)[x]);
            if (score >= prob_threshold) {
                ncnn::Mat pred_bbox(reg_max_1 * 4);
                for (int c = 0; c < reg_max_1 * 4; c++) pred_bbox[c] = feat_blob.channel(c).row(y)[x];
                pred_bbox = pred_bbox.reshape(16, 4);
                float pred_ltrb[4];
                for (int k = 0; k < 4; k++) {
                    float dis = 0.f; const float* dis_after_sm = pred_bbox.row(k);
                    float max_val = -FLT_MAX;
                    for(int l=0; l<reg_max_1; l++) if(dis_after_sm[l] > max_val) max_val = dis_after_sm[l];
                    float sum = 0.f; std::vector<float> exps(reg_max_1);
                    for(int l=0; l<reg_max_1; l++) { exps[l] = expf(dis_after_sm[l] - max_val); sum += exps[l]; }
                    for(int l=0; l<reg_max_1; l++) dis += l * (exps[l] / sum);
                    pred_ltrb[k] = dis * stride;
                }
                float pb_cx = (x + 0.5f) * stride; float pb_cy = (y + 0.5f) * stride;
                std::vector<KeyPoint> keypoints;
                for (int k = 0; k < num_points; k++) {
                    KeyPoint kp;
                    kp.x = (x + feat_blob.channel(65 + k * 3 + 0).row(y)[x] * 2) * stride;
                    kp.y = (y + feat_blob.channel(65 + k * 3 + 1).row(y)[x] * 2) * stride;
                    kp.prob = sigmoid(feat_blob.channel(65 + k * 3 + 2).row(y)[x]);
                    keypoints.push_back(kp);
                }
                YoloObject obj; 
                obj.x = pb_cx - pred_ltrb[0]; obj.y = pb_cy - pred_ltrb[1];
                obj.w = pred_ltrb[2] + pred_ltrb[0]; obj.h = pred_ltrb[3] + pred_ltrb[1];
                obj.prob = score; obj.keypoints = keypoints;
                objects.push_back(obj);
            }
        }
    }
}

int main(int argc, char** argv) {
    // 🔥 设置全局 NCNN 线程数为 4
    ncnn::set_omp_num_threads(4);
    
    ncnn::Net yolov8, classifier;
    
    // 🔥 为模型推理单独开启 4 线程
    yolov8.opt.num_threads = 4;
    classifier.opt.num_threads = 4;
    
    std::cout << "[INIT] 加载模型中 (4核全开模式)..." << std::endl;
    
    if (yolov8.load_param("yolov8n_pose_int8.param") || yolov8.load_model("yolov8n_pose_int8.bin")) {
        std::cerr << "加载 yolov8n_pose_int8 模型失败！" << std::endl;
        return -1;
    }
    if (classifier.load_param("pose_classifier.ncnn.param") || classifier.load_model("pose_classifier.ncnn.bin")) {
        std::cerr << "加载 pose_classifier 模型失败！" << std::endl;
        return -1;
    }

    // 1. 使用 STB 读取图片
    int img_w, img_h, img_channels;
    unsigned char *img_data = stbi_load("test.jpg", &img_w, &img_h, &img_channels, 3);
    if (!img_data) { std::cerr << "图片加载失败! 请检查 test.jpg 是否存在" << std::endl; return -1; }

    const int target_size = 320;
    float scale = (img_w > img_h) ? (float)target_size / img_w : (float)target_size / img_h;
    int w = img_w * scale; int h = img_h * scale;
    
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(img_data, ncnn::Mat::PIXEL_RGB, img_w, img_h, w, h);
    int wpad = target_size - w; int hpad = target_size - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad/2, hpad-hpad/2, wpad/2, wpad-wpad/2, ncnn::BORDER_CONSTANT, 114.f);
    const float norm[3] = {1/255.f, 1/255.f, 1/255.f};
    in_pad.substract_mean_normalize(0, norm);

    // 2. 开始计时 (推理 + 后处理)
    auto start = std::chrono::high_resolution_clock::now();
    
    ncnn::Extractor ex = yolov8.create_extractor();
    ex.input("in0", in_pad); 
    
    std::vector<YoloObject> proposals;
    int strides[3] = {8, 16, 32};
    for (int i = 0; i < 3; i++) {
        ncnn::Mat out; char name[16]; sprintf(name, "out%d", i); 
        ex.extract(name, out);
        generate_proposals(out, strides[i], 0.25f, proposals);
    }
    std::sort(proposals.begin(), proposals.end(), [](const YoloObject& a, const YoloObject& b){ return a.prob > b.prob; });
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, 0.45f);

    auto end = std::chrono::high_resolution_clock::now();
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    if (picked.empty()) {
        printf("\n⚠️ 未检测到任何人。推理时间: %.2f ms\n", time_ms);
    } else {
        printf("\n🏁 成功！推理总时间: %.2f ms\n", time_ms);
    }

    // 3. 映射回原图并画图
    for (int idx : picked) {
        YoloObject obj = proposals[idx];
        obj.x = (obj.x - (wpad / 2)) / scale; obj.y = (obj.y - (hpad / 2)) / scale;
        obj.w /= scale; obj.h /= scale;

        ncnn::Mat cls_in(34);
        for (int i = 0; i < 17; i++) {
            float kx = (obj.keypoints[i].x - (wpad / 2)) / scale;
            float ky = (obj.keypoints[i].y - (hpad / 2)) / scale;
            obj.keypoints[i].x = kx; obj.keypoints[i].y = ky;
            cls_in[i * 2] = std::max(0.f, std::min(1.f, (kx - obj.x) / (obj.w + 1e-6f)));
            cls_in[i * 2 + 1] = std::max(0.f, std::min(1.f, (ky - obj.y) / (obj.h + 1e-6f)));
        }
        
        ncnn::Extractor ex_cls = classifier.create_extractor();
        ex_cls.input("input", cls_in); ncnn::Mat cls_out; ex_cls.extract("output", cls_out);
        
        int max_idx = 0; float max_score = -1.f;
        for (int i = 0; i < cls_out.w; i++) { if (cls_out[i] > max_score) { max_score = cls_out[i]; max_idx = i; } }

        printf("👀 检测到动作: %s (置信度: %d%%)\n", ACTION_NAMES[max_idx], (int)(max_score * 100));

        draw_rect(img_data, img_w, img_h, (int)obj.x, (int)obj.y, (int)obj.w, (int)obj.h, 0, 255, 0, 4);
        for (int j = 0; j < 16; j++) {
            auto& p1 = obj.keypoints[joints[j][0]];
            auto& p2 = obj.keypoints[joints[j][1]];
            if (p1.prob > 0.3 && p2.prob > 0.3) {
                draw_line(img_data, img_w, img_h, (int)p1.x, (int)p1.y, (int)p2.x, (int)p2.y, 0, 255, 255, 3);
            }
        }
    }

    stbi_write_jpg("result.jpg", img_w, img_h, 3, img_data, 100);
    std::cout << "📸 图像已保存至 result.jpg" << std::endl;

    stbi_image_free(img_data);
    return 0;
}
