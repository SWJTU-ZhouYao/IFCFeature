#include <torch/script.h>
#include <torch/torch.h>

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

struct ExtractResult {
    cv::Mat kpts;          // Nx3 float: x,y,score
    cv::Mat descriptors;   // Nx256 float
    torch::Tensor prev_feat; // [1,32,H/8,W/8]
};

static inline int clamp_int(int v, int lo, int hi) {
    return std::max(lo, std::min(v, hi));
}

static cv::Mat nms_fast(const cv::Mat& in_corners_3xN, int H, int W, int dist_thresh) {
    CV_Assert(in_corners_3xN.rows == 3);
    int N = in_corners_3xN.cols;
    if (N <= 0) return cv::Mat(3, 0, CV_32F);

    std::vector<int> inds1(N);
    for (int i = 0; i < N; ++i) inds1[i] = i;
    std::sort(inds1.begin(), inds1.end(), [&](int a, int b){
        float sa = in_corners_3xN.at<float>(2, a);
        float sb = in_corners_3xN.at<float>(2, b);
        return sa > sb;
    });

    cv::Mat corners(3, N, CV_32F);
    for (int j = 0; j < N; ++j) {
        int idx = inds1[j];
        corners.at<float>(0, j) = in_corners_3xN.at<float>(0, idx);
        corners.at<float>(1, j) = in_corners_3xN.at<float>(1, idx);
        corners.at<float>(2, j) = in_corners_3xN.at<float>(2, idx);
    }

    std::vector<cv::Point> rcorners(N);
    for (int j = 0; j < N; ++j) {
        int x = (int)std::lround(corners.at<float>(0, j));
        int y = (int)std::lround(corners.at<float>(1, j));
        x = clamp_int(x, 0, W - 1);
        y = clamp_int(y, 0, H - 1);
        rcorners[j] = cv::Point(x, y);
    }

    if (N == 1) {
        cv::Mat out(3, 1, CV_32F);
        out.at<float>(0,0) = (float)rcorners[0].x;
        out.at<float>(1,0) = (float)rcorners[0].y;
        out.at<float>(2,0) = corners.at<float>(2,0);
        return out;
    }

    cv::Mat grid = cv::Mat::zeros(H, W, CV_8S);    // int8 is enough (-1/0/1)
    cv::Mat inds = cv::Mat::zeros(H, W, CV_32S);   // store i

    for (int i = 0; i < N; ++i) {
        const auto& p = rcorners[i];
        grid.at<int8_t>(p.y, p.x) = 1;
        inds.at<int>(p.y, p.x) = i;
    }

    int pad = dist_thresh;
    cv::Mat grid_pad;
    cv::copyMakeBorder(grid, grid_pad, pad, pad, pad, pad, cv::BORDER_CONSTANT, 0);

    for (int i = 0; i < N; ++i) {
        int x = rcorners[i].x + pad;
        int y = rcorners[i].y + pad;
        if (grid_pad.at<int8_t>(y, x) == 1) {
            for (int yy = y - pad; yy <= y + pad; ++yy) {
                for (int xx = x - pad; xx <= x + pad; ++xx) {
                    grid_pad.at<int8_t>(yy, xx) = 0;
                }
            }
            grid_pad.at<int8_t>(y, x) = -1; // keep
        }
    }

    std::vector<int> keep_i; keep_i.reserve(N);
    for (int y = pad; y < pad + H; ++y) {
        for (int x = pad; x < pad + W; ++x) {
            if (grid_pad.at<int8_t>(y, x) == -1) {
                int yy = y - pad;
                int xx = x - pad;
                int i = inds.at<int>(yy, xx); // index into sorted corners (0..N-1)
                keep_i.push_back(i);
            }
        }
    }
    if (keep_i.empty()) return cv::Mat(3, 0, CV_32F);

    cv::Mat out(3, (int)keep_i.size(), CV_32F);
    for (int j = 0; j < (int)keep_i.size(); ++j) {
        int i = keep_i[j];
        out.at<float>(0, j) = corners.at<float>(0, i);
        out.at<float>(1, j) = corners.at<float>(1, i);
        out.at<float>(2, j) = corners.at<float>(2, i);
    }

    std::vector<int> order(out.cols);
    for (int j = 0; j < out.cols; ++j) order[j] = j;
    std::sort(order.begin(), order.end(), [&](int a, int b){
        return out.at<float>(2, a) > out.at<float>(2, b);
    });

    cv::Mat out_sorted(3, out.cols, CV_32F);
    for (int j = 0; j < out.cols; ++j) {
        int k = order[j];
        out_sorted.at<float>(0, j) = out.at<float>(0, k);
        out_sorted.at<float>(1, j) = out.at<float>(1, k);
        out_sorted.at<float>(2, j) = out.at<float>(2, k);
    }
    return out_sorted;
}

static cv::Mat get_keypoints_from_semi_Ours(
        const torch::Tensor& semi,
        float threshold = 0.005f,
        int nms_dist = 2
) {
    torch::Tensor heat = torch::softmax(semi, 1).slice(1, 0, 64); // [1,64,Hc,Wc]
    heat = heat.squeeze(0).to(torch::kCPU).contiguous();          // [64,Hc,Wc]
    int Hc = (int)heat.size(1);
    int Wc = (int)heat.size(2);

    torch::Tensor prob_t, idx_t;
    std::tie(prob_t, idx_t) = torch::max(heat, 0); // both [Hc,Wc]
    prob_t = prob_t.contiguous();
    idx_t  = idx_t.contiguous();

    const float* prob = prob_t.data_ptr<float>();
    const int64_t* idx = idx_t.data_ptr<int64_t>();

    std::vector<float> xs, ys, ss;
    xs.reserve(Hc * Wc / 10);
    ys.reserve(Hc * Wc / 10);
    ss.reserve(Hc * Wc / 10);

    for (int y = 0; y < Hc; ++y) {
        for (int x = 0; x < Wc; ++x) {
            float p = prob[y * Wc + x];
            if (p > threshold) {
                xs.push_back((float)x);
                ys.push_back((float)y);
                ss.push_back(p);
            }
        }
    }
    if (xs.empty()) {
        return cv::Mat(0, 3, CV_32F);
    }

    int N = (int)xs.size();
    cv::Mat in_corners(3, N, CV_32F);
    for (int i = 0; i < N; ++i) {
        in_corners.at<float>(0, i) = xs[i];
        in_corners.at<float>(1, i) = ys[i];
        in_corners.at<float>(2, i) = ss[i];
    }

    cv::Mat nmsed = in_corners;
    if (nms_dist != 0 && N > 100) {
        nmsed = nms_fast(in_corners, Hc, Wc, nms_dist); // 3xM
    }

    int M = nmsed.cols;
    cv::Mat pts_out(M, 3, CV_32F);
    for (int j = 0; j < M; ++j) {
        float x = nmsed.at<float>(0, j);
        float y = nmsed.at<float>(1, j);
        float s = nmsed.at<float>(2, j);

        int xi = (int)std::lround(x);
        int yi = (int)std::lround(y);
        xi = clamp_int(xi, 0, Wc - 1);
        yi = clamp_int(yi, 0, Hc - 1);

        int c = (int)idx[yi * Wc + xi]; // 0..63

        float x_orig = x * 8.0f + float(c % 8);
        float y_orig = y * 8.0f + float(c / 8);

        pts_out.at<float>(j, 0) = x_orig;
        pts_out.at<float>(j, 1) = y_orig;
        pts_out.at<float>(j, 2) = s;
    }

    return pts_out;
}

static cv::Mat extract_descriptors(const torch::Tensor& desc, const cv::Mat& kpts) {
    CV_Assert(kpts.cols == 3);
    if (kpts.rows == 0) return cv::Mat(0, 256, CV_32F);

    torch::Tensor d = desc.squeeze(0).to(torch::kCPU).contiguous(); // [256,Hc,Wc]
    int Hc = (int)d.size(1);
    int Wc = (int)d.size(2);

    const float* dptr = d.data_ptr<float>();

    int N = kpts.rows;
    cv::Mat descriptors(N, 256, CV_32F);

    for (int i = 0; i < N; ++i) {
        float x = kpts.at<float>(i, 0);
        float y = kpts.at<float>(i, 1);
        int xc = (int)std::floor(x / 8.0f);
        int yc = (int)std::floor(y / 8.0f);
        xc = clamp_int(xc, 0, Wc - 1);
        yc = clamp_int(yc, 0, Hc - 1);

        for (int c = 0; c < 256; ++c) {
            descriptors.at<float>(i, c) = dptr[c * (Hc * Wc) + yc * Wc + xc];
        }
    }
    return descriptors;
}

static ExtractResult extract_feature_descriptors_from_Ours(
        const cv::Mat& img_gray,
        int resize_h,
        int resize_w,
        torch::jit::script::Module& module,
        const torch::Tensor& pre_feat_in,
        torch::Device device,
        int nms_dist = 2,
        float threshold = 0.005f
) {
    CV_Assert(img_gray.type() == CV_8UC1);

    int H0 = img_gray.rows, W0 = img_gray.cols;

    cv::Mat img_resized;
    bool do_resize = (resize_h > 0 && resize_w > 0);
    if (do_resize) {
        cv::resize(img_gray, img_resized, cv::Size(resize_w, resize_h));
    } else {
        img_resized = img_gray;
        resize_h = H0; resize_w = W0;
    }

    cv::Mat img_f;
    img_resized.convertTo(img_f, CV_32F, 1.0 / 255.0);

    auto img_tensor = torch::from_blob(
            img_f.data,
            {1, resize_h, resize_w, 1},
            torch::kFloat32
    ).clone(); // clone 确保内存安全

    img_tensor = img_tensor.permute({0, 3, 1, 2}).contiguous(); // [1,1,H,W]
    img_tensor = img_tensor.to(device);

    torch::Tensor pre_feat = pre_feat_in;
    if (!pre_feat.defined() || pre_feat.numel() == 0) {
        pre_feat = torch::zeros({1, 32, resize_h / 8, resize_w / 8}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    } else {
        pre_feat = pre_feat.to(device);
    }

    // forward
    torch::NoGradGuard no_grad;
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(img_tensor);
    inputs.push_back(pre_feat);

    auto out_iv = module.forward(inputs);

    auto out_tuple = out_iv.toTuple();
    auto prev_feat_out = out_tuple->elements()[0].toTensor();
    auto semi          = out_tuple->elements()[1].toTensor(); // [1,65,H/8,W/8]
    auto desc          = out_tuple->elements()[2].toTensor(); // [1,256,H/8,W/8]

    cv::Mat kpts_resized = get_keypoints_from_semi_Ours(semi, threshold, nms_dist);
    cv::Mat descriptors = extract_descriptors(desc, kpts_resized);

    cv::Mat kpts_out = kpts_resized.clone();
    if (do_resize && kpts_out.rows > 0) {
        float scale_x = float(W0) / float(resize_w);
        float scale_y = float(H0) / float(resize_h);
        for (int i = 0; i < kpts_out.rows; ++i) {
            kpts_out.at<float>(i, 0) *= scale_x;
            kpts_out.at<float>(i, 1) *= scale_y;
            // score unchanged
        }
    }

    ExtractResult res;
    res.kpts = kpts_out;
    res.descriptors = descriptors;
    res.prev_feat = prev_feat_out;
    return res;
}

static std::vector<cv::KeyPoint> to_cv2_keypoints(const cv::Mat& kptsNx3) {
    CV_Assert(kptsNx3.cols >= 2);
    std::vector<cv::KeyPoint> kps;
    kps.reserve(kptsNx3.rows);
    for (int i = 0; i < kptsNx3.rows; ++i) {
        float x = kptsNx3.at<float>(i, 0);
        float y = kptsNx3.at<float>(i, 1);
        kps.emplace_back(cv::KeyPoint(x, y, 1.0f));
    }
    return kps;
}

static cv::Mat to_desc_cv32f(const cv::Mat& desc) {
    if (desc.empty()) return desc;
    if (desc.type() == CV_32F && desc.isContinuous()) return desc;
    cv::Mat out;
    desc.convertTo(out, CV_32F);
    return out;
}


static std::vector<cv::DMatch> match_L2_ratio(
        const cv::Mat& des1, const cv::Mat& des2, float ratio_test = 0.75f
) {
    std::vector<cv::DMatch> matches;
    if (des1.empty() || des2.empty()) return matches;

    cv::BFMatcher bf(cv::NORM_L2, false);

    if (ratio_test > 0.f) {
        std::vector<std::vector<cv::DMatch>> knn;
        bf.knnMatch(des1, des2, knn, 2);
        matches.reserve(knn.size());
        for (auto& v : knn) {
            if (v.size() < 2) continue;
            const auto& m = v[0];
            const auto& n = v[1];
            if (m.distance < ratio_test * n.distance) {
                matches.push_back(m);
            }
        }
    } else {
        bf.match(des1, des2, matches);
    }
    std::sort(matches.begin(), matches.end(),
              [](const cv::DMatch& a, const cv::DMatch& b) { return a.distance < b.distance; });
    return matches;
}

static cv::Mat desc256_to_32bytes_sign(const cv::Mat& desc256) {
    if (desc256.empty()) return cv::Mat();
    CV_Assert(desc256.type() == CV_32F);
    CV_Assert(desc256.cols == 256);

    const int N = desc256.rows;
    cv::Mat out(N, 32, CV_8U);
    out.setTo(0);

    for (int i = 0; i < N; ++i) {
        const float* d = desc256.ptr<float>(i);
        uint8_t* o = out.ptr<uint8_t>(i);

        // pack 256 bits -> 32 bytes
        for (int b = 0; b < 32; ++b) {
            uint8_t byte = 0;
            // 8 bits
            for (int k = 0; k < 8; ++k) {
                int idx = b * 8 + k;       // 0..255
                uint8_t bit = (d[idx] > 0.f) ? 1u : 0u;
                byte |= (bit << k);        // LSB-first packing
            }
            o[b] = byte;
        }
    }
    return out;
}

static std::vector<cv::DMatch> match_Hamming_ratio(
        const cv::Mat& des1, const cv::Mat& des2, float ratio_test = 0.75f
) {
    std::vector<cv::DMatch> matches;
    if (des1.empty() || des2.empty()) return matches;

    CV_Assert(des1.type() == CV_8U && des2.type() == CV_8U);
    CV_Assert(des1.cols == 32 && des2.cols == 32);

    cv::BFMatcher bf(cv::NORM_HAMMING, false);

    if (ratio_test > 0.f) {
        std::vector<std::vector<cv::DMatch>> knn;
        bf.knnMatch(des1, des2, knn, 2);
        matches.reserve(knn.size());
        for (auto& v : knn) {
            if (v.size() < 2) continue;
            const auto& m = v[0];
            const auto& n = v[1];
            if (m.distance < ratio_test * n.distance) {
                matches.push_back(m);
            }
        }
    } else {
        bf.match(des1, des2, matches);
    }

    std::sort(matches.begin(), matches.end(),
              [](const cv::DMatch& a, const cv::DMatch& b) { return a.distance < b.distance; });
    return matches;
}


int main(int argc, char** argv) {
    std::string model_path = "Super_Changed_ORB_cuda.pt";
    std::string image_path1 = "1403636581063555584.png";
    std::string image_path2 = "1403636581113555456.png";

    // device
    torch::Device device(torch::kCUDA, 0);

    // load module
    torch::jit::script::Module module = torch::jit::load(model_path);
    module.eval();
    module.to(device);

    // read image (grayscale)
    cv::Mat img1 = cv::imread(image_path1, cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread(image_path2, cv::IMREAD_GRAYSCALE);

    torch::Tensor pre_feat;
    auto res1 = extract_feature_descriptors_from_Ours(
            img1, 240, 320, module, pre_feat, device,
            /*nms_dist=*/0, /*threshold=*/0.005f
    );

    auto res2 = extract_feature_descriptors_from_Ours(
            img2, 240, 320, module, res1.prev_feat, device,
            /*nms_dist=*/0, /*threshold=*/0.005f
    );


    std::cout << "img1 kpts: " << res1.kpts.rows << ", desc: " << res1.descriptors.rows << "x" << res1.descriptors.cols << "\n";
    std::cout << "img2 kpts: " << res2.kpts.rows << ", desc: " << res2.descriptors.rows << "x" << res2.descriptors.cols << "\n";

    cv::Mat img1_bgr, img2_bgr;
    cv::cvtColor(img1, img1_bgr, cv::COLOR_GRAY2BGR);
    cv::cvtColor(img2, img2_bgr, cv::COLOR_GRAY2BGR);

    auto kp1_cv2 = to_cv2_keypoints(res1.kpts);
    auto kp2_cv2 = to_cv2_keypoints(res2.kpts);

    cv::Mat des1 = desc256_to_32bytes_sign(res1.descriptors);
    cv::Mat des2 = desc256_to_32bytes_sign(res2.descriptors);

    auto matches = match_Hamming_ratio(des1, des2, 0.75f);

    std::cout << "matches: " << matches.size() << "\n";
    if (matches.empty()) {
        std::cerr << "No matches found.\n";
        return 0;
    }

    int Ndraw = std::min<int>((int)matches.size(), 200);
    std::vector<cv::DMatch> matches_draw(matches.begin(), matches.begin() + Ndraw);

    cv::Mat img_matches;
    cv::drawMatches(
            img1_bgr, kp1_cv2,
            img2_bgr, kp2_cv2,
            matches_draw,
            img_matches,
            cv::Scalar::all(-1), cv::Scalar::all(-1),
            std::vector<char>(),
            cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS
    );

    cv::imshow("match", img_matches);
    cv::waitKey(0);
    return 0;

}
