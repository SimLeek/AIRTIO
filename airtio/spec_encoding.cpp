#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <omp.h>
#include <cmath>
#include <vector>

namespace py = pybind11;

template <typename Sequence>
inline py::array_t<typename Sequence::value_type> as_pyarray(Sequence&& seq) {
    // Move entire object to heap (Ensure is moveable!). Memory handled via Python capsule
    Sequence* seq_ptr = new Sequence(std::move(seq));
    auto capsule = py::capsule(seq_ptr, [](void* p) { delete reinterpret_cast<Sequence*>(p); });
    return py::array(seq_ptr->size(),  // shape of array
                     seq_ptr->data(),  // c-style contiguous strides for Sequence
                     capsule           // numpy array references this parent
    );
}

class EdgeDetector {
public:
    static cv::Mat kernel;
    static const int SIZE_THRESHOLD = 65536; // Same as create_mask threshold*/

    static void init_kernel() {
        float s = -1.0f;
        float c = -1.0f / std::sqrt(2.0f);
        float center = -(4 * s + 4 * c);
        kernel = (cv::Mat_<float>(3, 3) << c, s, c,
                                          s, center, s,
                                          c, s, c);
        float sum_abs = cv::sum(cv::abs(kernel))[0];
        kernel /= sum_abs;
    }

    static void apply(py::array_t<float> image, py::array_t<float> output) {
        auto buf = image.request();
        auto out_buf = output.request();

        int height = buf.shape[0];
        int width = buf.shape[1];
        int channels = buf.shape[2];

        cv::Mat img(height, width, CV_32FC(channels), buf.ptr);
        cv::Mat result_mat(height, width, CV_32FC(channels), out_buf.ptr);

        cv::filter2D(img, result_mat, -1, kernel);
    }

    static std::tuple<py::array_t<int>, py::array_t<int>> create_mask(py::array_t<float> energy_array, float threshold) {
        auto buf = energy_array.request();
        int height = buf.shape[0];
        int width = buf.shape[1];
        int channels = buf.shape[2];
        float* energy_ptr = static_cast<float*>(buf.ptr);

        const int parallel_threshold = 262144; // 256K pixels, tuned for 1080p

        if (width * height > parallel_threshold) {
            std::vector<std::vector<int>> all_xs(omp_get_max_threads());
            std::vector<std::vector<int>> all_ys(omp_get_max_threads());
            std::vector<size_t> thread_sizes(omp_get_max_threads(), 0);
            #pragma omp parallel
            {
                std::vector<int> local_xs;
                std::vector<int> local_ys;
                int estimated_size = (height * width) / 50;
                local_xs.reserve(estimated_size);
                local_ys.reserve(estimated_size);

                #pragma omp for schedule(static, 64) collapse(2)
                for (int y = 0; y < height; ++y) {
                    for (int x = 0; x < width; ++x) {
                        int base_idx = (y * width + x) * channels;
                        bool above_threshold = false;
                        if (channels == 1) {
                            above_threshold = energy_ptr[base_idx] > threshold;
                        } else if (channels == 3) {
                            above_threshold = (energy_ptr[base_idx] > threshold) ||
                                             (energy_ptr[base_idx + 1] > threshold) ||
                                             (energy_ptr[base_idx + 2] > threshold);
                        } else {
                            for (int c = 0; c < channels; ++c) {
                                if (energy_ptr[base_idx + c] > threshold) {
                                    above_threshold = true;
                                    break;
                                }
                            }
                        }
                        if (above_threshold) {
                            local_xs.push_back(x);
                            local_ys.push_back(y);
                        }
                    }
                }

                int tid = omp_get_thread_num();
                thread_sizes[tid] = local_xs.size();
                all_xs[tid] = std::move(local_xs);
                all_ys[tid] = std::move(local_ys);
            }

            // Combine thread-local coordinates
            size_t total_size = 0;
            for (size_t size : thread_sizes) {
                total_size += size;
            }

            std::vector<int> xs_vec;
            std::vector<int> ys_vec;
            xs_vec.reserve(total_size);
            ys_vec.reserve(total_size);

            for (size_t tid = 0; tid < thread_sizes.size(); ++tid) {
                xs_vec.insert(xs_vec.end(), all_xs[tid].begin(), all_xs[tid].end());
                ys_vec.insert(ys_vec.end(), all_ys[tid].begin(), all_ys[tid].end());
            }

            return std::make_tuple(as_pyarray(std::move(xs_vec)), as_pyarray(std::move(ys_vec)));
        } else {
            // Serial path - optimized
            std::vector<int> xs_vec;
            std::vector<int> ys_vec;

            // Better size estimation
            int estimated_size = std::max(std::min(100,(height * width)), (height * width) / 20);
            xs_vec.reserve(estimated_size);
            ys_vec.reserve(estimated_size);

            // Use nested loops to avoid division/modulo
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    int base_idx = (y * width + x) * channels;
                    for (int c = 0; c < channels; ++c) {
                        if (energy_ptr[base_idx + c] > threshold) {
                            xs_vec.push_back(x);
                            ys_vec.push_back(y);
                            break; // Early exit
                        }
                    }
                }
            }

            return std::make_tuple(as_pyarray(std::move(xs_vec)), as_pyarray(std::move(ys_vec)));
        }
    }
};

cv::Mat EdgeDetector::kernel;

PYBIND11_MODULE(spec_encoding, m) {
    EdgeDetector::init_kernel();
    m.def("apply_edge_detector", &EdgeDetector::apply, "Apply edge detection to an image with pre-allocated output");
    m.def("create_mask", &EdgeDetector::create_mask, "Create a mask for energy values above threshold");
}