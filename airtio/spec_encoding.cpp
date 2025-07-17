#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <omp.h>
#include <cmath>
#include <vector>

namespace py = pybind11;

class EdgeDetector {
public:
    static cv::Mat kernel;

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

    static py::array_t<float> apply(py::array_t<float> image) {
        auto buf = image.request();
        int height = buf.shape[0];
        int width = buf.shape[1];
        int channels = buf.shape[2];
        cv::Mat img(height, width, CV_32FC(channels), buf.ptr);

        py::array_t<float> result = py::array_t<float>({height, width, channels});
        auto result_buf = result.request();
        cv::Mat result_mat(height, width, CV_32FC(channels), result_buf.ptr);

        int num_sections = omp_get_max_threads();
        cv::filter2D(img, result_mat, -1, kernel);

        return result;
    }

    static py::array_t<bool> create_mask(py::array_t<float> energy_array, float threshold) {
        auto buf = energy_array.request();
        int height = buf.shape[0];
        int width = buf.shape[1];
        int channels = buf.shape[2];
        cv::Mat energy(height, width, CV_32FC(channels), buf.ptr);

        py::array_t<bool> mask = py::array_t<bool>({height, width});
        auto mask_buf = mask.request();
        cv::Mat mask_mat(height, width, CV_8U, mask_buf.ptr);

        if (height * width < 65536) {
            mask_mat = cv::Mat::zeros(height, width, CV_8U);
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    for (int c = 0; c < channels; ++c) {
                        if (energy.at<cv::Vec<float, 3>>(y, x)[c] > threshold) {
                            mask_mat.at<uchar>(y, x) = 1;
                            break;
                        }
                    }
                }
            }
        } else {
            mask_mat = cv::Mat::zeros(height, width, CV_8U);
            int num_sections = omp_get_max_threads();
            int section_height = height / num_sections;

            //#pragma omp parallel for
            for (int i = 0; i < num_sections; ++i) {
                int start_y = i * section_height;
                int end_y = (i == num_sections - 1) ? height : (i + 1) * section_height;
                cv::Mat section = energy(cv::Range(start_y, end_y), cv::Range::all());
                cv::Mat local_mask = cv::Mat::zeros(end_y - start_y, width, CV_8U);

                for (int y = 0; y < local_mask.rows; ++y) {
                    for (int x = 0; x < width; ++x) {
                        for (int c = 0; c < channels; ++c) {
                            if (section.at<cv::Vec<float, 3>>(y, x)[c] > threshold) {
                                local_mask.at<uchar>(y, x) = 1;
                                break;
                            }
                        }
                    }
                }

                //#pragma omp critical
                //{
                local_mask.copyTo(mask_mat(cv::Range(start_y, end_y), cv::Range::all()));
                //}
            }
        }

        return mask;
    }
};

cv::Mat EdgeDetector::kernel;

PYBIND11_MODULE(spec_encoding, m) {
    EdgeDetector::init_kernel();
    m.def("apply_edge_detector", &EdgeDetector::apply, "Apply edge detection to an image");
    m.def("create_mask", &EdgeDetector::create_mask, "Create a mask for energy values above threshold");
}
