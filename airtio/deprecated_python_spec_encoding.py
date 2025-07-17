import cv2
import numpy as np
from joblib import Parallel, delayed
import multiprocessing

class EdgeDetector:
    """Static class to hold the Gaussian-derived edge detection kernel."""
    s = -1.0  # Side weight
    c = -1.0 / np.sqrt(2)  # Corner weight
    center = -(4 * s + 4 * c)  # Center to make sum 0
    kernel = np.array([[c, s, c],
                       [s, center, s],
                       [c, s, c]], dtype=np.float32)
    kernel = kernel / np.sum(np.abs(kernel))  # Normalize by sum of absolute values

    @staticmethod
    def apply(image):
        """Apply edge detection to the entire multi-channel image, parallelizing for large images."""
        height, width = image.shape[:2]
        num_sections = multiprocessing.cpu_count()  # Use 8 cores
        if height * width < 65536:  # Avoid parallelization for small images
            edges = cv2.filter2D(image, -1, EdgeDetector.kernel)
            return np.abs(edges)

        # Split image into horizontal sections
        section_height = height // num_sections
        sections = []
        for i in range(num_sections):
            start_y = i * section_height
            end_y = (i + 1) * section_height if i < num_sections - 1 else height
            section = image[start_y:end_y, :, :]
            sections.append((section, start_y, end_y))

        def process_section(local_section, local_start_y, local_end_y):
            # Apply filter2D to section
            local_filtered = cv2.filter2D(local_section, -1, EdgeDetector.kernel)
            return np.abs(local_filtered), local_start_y, local_end_y

        # Parallel processing
        results = Parallel(n_jobs=num_sections, backend='threading')(
            delayed(process_section)(section, start_y, end_y) for section, start_y, end_y in sections
        )

        # Combine results
        edges = np.zeros_like(image)
        for filtered, start_y, end_y in results:
            edges[start_y:end_y, :, :] = filtered

        return edges


def create_gaussian_pyramid(frame, min_size=1):
    """Create a pyramid using manual downscaling with INTER_AREA."""
    pyramid = [frame]
    current = frame
    while min(current.shape[:2]) > min_size:
        new_height = max(min_size, current.shape[0] // 2)
        new_width = max(min_size, current.shape[1] // 2)
        current = cv2.resize(current, (new_width, new_height), interpolation=cv2.INTER_AREA)
        pyramid.append(current)
    return pyramid


def compute_positional_encodings(xs, ys, level, width, height, max_width, max_height, levels):
    """Compute positional encodings for multiple pixels at a given level."""
    rw_values = np.array([2 ** i for i in range(int(np.ceil(np.log2(max_width))) + 1)], dtype=np.float32)
    rh_values = np.array([2 ** i for i in range(int(np.ceil(np.log2(max_height))) + 1)], dtype=np.float32)
    rl_values = np.array([2 ** i for i in range(int(np.ceil(np.log2(levels))) + 1)], dtype=np.float32)

    center_x, center_y, center_l = width / 2, height / 2, levels / 2
    rel_x = xs - center_x
    rel_y = ys - center_y
    rel_l = level - center_l

    num_pixels = len(xs)
    encoding_len = len(rw_values) + len(rh_values) + len(rl_values)
    encodings = np.zeros((encoding_len, num_pixels), dtype=np.float32)

    # Vectorized computation for x, y, l encodings
    for i, r in enumerate(rw_values):
        encodings[i] = np.sin((rel_x * np.pi / 2) / r)
    for i, r in enumerate(rh_values):
        encodings[i + len(rw_values)] = np.sin((rel_y * np.pi / 2) / r)
    for i, r in enumerate(rl_values):
        encodings[i + len(rw_values) + len(rh_values)] = np.sin((rel_l * np.pi / 2) / r)

    return encodings


def create_mask(energy_array, threshold=1.0, small_size=65536):
    """Create a mask for energy values > threshold, parallelizing for large arrays."""
    height, width, channels = energy_array.shape
    if height * width < small_size:
        mask = np.zeros((height, width), dtype=bool)
        for y in range(height):
            for x in range(width):
                for c in range(channels):
                    if energy_array[y, x, c] > threshold:
                        mask[y, x] = True
                        break
        return mask

    # Split array into horizontal sections
    num_sections = multiprocessing.cpu_count()  # Use 8 cores
    section_height = height // num_sections
    sections = []
    for i in range(num_sections):
        start_y = i * section_height
        end_y = (i + 1) * section_height if i < num_sections - 1 else height
        section = energy_array[start_y:end_y, :, :]
        sections.append((section, start_y, end_y))

    def process_section(local_section, local_start_y, local_end_y):
        # Apply np.any to section
        local_mask = np.any(local_section > threshold, axis=2)
        return local_mask, local_start_y, local_end_y

    # Parallel processing
    results = Parallel(n_jobs=num_sections, backend='threading')(
        delayed(process_section)(section, start_y, end_y) for section, start_y, end_y in sections
    )

    # Combine results
    mask = np.zeros((height, width), dtype=bool)
    for local_mask, start_y, end_y in results:
        mask[start_y:end_y, :] = local_mask

    return mask


class SPEC(object):
    """Sparse Pyramid Edge CSRs"""
    def __init__(self, energy_per_frame=None, target_nnz=None, target_bits_per_frame=None, alpha=0.05):
        if energy_per_frame is not None and (target_nnz is None and target_bits_per_frame is None):
            self.energy_per_frame = energy_per_frame
            self.target_nnz = None
            self.target_bytes_per_frame = None
        elif target_nnz is not None and (energy_per_frame is None and target_bits_per_frame is None):
            self.energy_per_frame = None
            self.target_nnz = target_nnz
            self.target_bytes_per_frame = None
        elif target_bits_per_frame is not None and (target_nnz is None and energy_per_frame is None):
            self.energy_per_frame = None
            self.target_nnz = None
            self.target_bytes_per_frame = target_bits_per_frame
        elif target_bits_per_frame is None and target_nnz is None and energy_per_frame is None:
            self.energy_per_frame = 0.5
            self.target_nnz = None
            self.target_bytes_per_frame = None
        else:
            raise ValueError("parameters are mutually exclusive")

        self.alpha = alpha
        self.energy_arrays = None

    def __call__(self, frame):
        pyramid = create_gaussian_pyramid(frame)

        if (
                self.energy_arrays is None or
                len(self.energy_arrays) != len(pyramid) or
                any([pyramid[i].shape != self.energy_arrays[i].shape for i in range(len(pyramid))])
        ):
            self.energy_arrays = [np.zeros_like(p) for p in pyramid]

        levels = len(pyramid)
        y_indices = []
        x_indices = []
        l_indices = []
        values = []
        ptrs = [0]
        heights = []
        widths = []
        max_height, max_width = pyramid[0].shape[:2]

        for level, img in enumerate(pyramid):
            height, width, channels = img.shape[:3]
            heights.append(height)
            widths.append(width)
            if min(img.shape[:2]) > 1:
                edges = EdgeDetector.apply(img)
                self.energy_arrays[level] += edges * self.energy_per_frame

                # Calculate number of encoding values
                encoding_len = int(np.ceil(np.log2(max_width))) + 1 + int(np.ceil(np.log2(max_height))) + 1 + int(np.ceil(np.log2(levels))) + 1
                value_dim = channels + encoding_len  # 3 energies (R, G, B) + encodings

                # Use parallelized mask creation
                mask = create_mask(self.energy_arrays[level])
                ys, xs = np.nonzero(mask)

                idx = len(ys)

                if idx > 0:
                    row_indices_y = ys.astype(np.int32)
                    row_indices_x = xs.astype(np.int32)
                    row_indices_l = np.full(idx, level, dtype=np.int32)
                    row_values = np.zeros((value_dim, idx), dtype=np.float32)

                    # Vectorized positional encodings
                    encodings = compute_positional_encodings(xs, ys, level, width, height, max_width, max_height, levels)
                    row_values[:channels] = self.energy_arrays[level][ys, xs, :channels].T
                    row_values[channels:] = encodings
                    self.energy_arrays[level][ys, xs, :channels] = 0

                    y_indices.append(row_indices_y)
                    x_indices.append(row_indices_x)
                    l_indices.append(row_indices_l)
                    values.append(row_values)
                ptrs.append(ptrs[-1] + idx)
            else:
                self.energy_arrays[level] += img * self.energy_per_frame

                # Calculate number of encoding values
                encoding_len = int(np.ceil(np.log2(max_width))) + 1 + int(np.ceil(np.log2(max_height))) + 1 + int(
                    np.ceil(np.log2(levels))) + 1
                value_dim = channels + encoding_len  # channels energies (R, G, B) + encodings

                if any([self.energy_arrays[level][0, 0, c] > 1 for c in range(channels)]):
                    row_values = np.zeros((value_dim, 1), dtype=np.float32)
                    row_indices_y = np.zeros(1, dtype=np.int32)
                    row_indices_x = np.zeros(1, dtype=np.int32)
                    row_indices_l = np.ones(1, dtype=np.int32) * level
                    encodings = compute_positional_encodings(np.array([0]), np.array([0]), level, width, height, max_width, max_height, levels)
                    row_values[:channels, 0] = self.energy_arrays[level][0, 0, :channels]
                    self.energy_arrays[level][0, 0, :channels] = 0
                    row_values[channels:, 0] = encodings[:, 0]
                    y_indices.append(row_indices_y)
                    x_indices.append(row_indices_x)
                    l_indices.append(row_indices_l)
                    values.append(row_values)
                    ptrs.append(ptrs[-1] + 1)

        # Build output
        y_indices = np.concatenate(y_indices) if len(y_indices) > 0 else None
        x_indices = np.concatenate(x_indices) if len(x_indices) > 0 else None
        l_indices = np.concatenate(l_indices) if len(l_indices) > 0 else None
        values = np.concatenate(values, axis=1) if len(values) > 0 else None
        ptrs = np.array(ptrs, dtype=np.int32)

        if self.target_nnz is not None:
            current_nnz = values.shape[1]
            error = current_nnz - self.target_nnz
            self.energy_per_frame *= (1.0 - self.alpha * np.sign(error))
        elif self.target_bytes_per_frame is not None:
            current_bytes = values.nbytes
            error = current_bytes - self.target_bytes_per_frame
            self.energy_per_frame *= (1.0 - self.alpha * np.sign(error))

        return y_indices, x_indices, l_indices, values, ptrs, [heights, widths, levels]


def process_video(input_path, energy_factor=0.5):
    """Process video to create pyramid, apply edge detection, and generate custom sparse output."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file")

    all_outputs = []
    spec = SPEC(energy_factor)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        float_frame = frame.astype(np.float32) / 255.0
        y_indices, x_indices, l_indices, values, ptrs, whl = spec(float_frame)
        heights, widths, levels = whl
        print(values)

    cap.release()
    return all_outputs


def main():
    try:
        input_video = "input_video.mp4"  # Replace with your video path
        outputs = process_video(input_video, energy_factor=0.5)
        print(f"Generated {len(outputs)} sparse outputs.")
        for i, out in enumerate(outputs):
            print(f"Output {i} shape: {out['shape']}, non-zero elements: {len(out['indices'])}, "
                  f"values shape: {out['values'].shape}")
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    import cProfile
    import pstats
    cProfile.run('main()', 'prof_data.prof')
    p = pstats.Stats('prof_data.prof')
    p.sort_stats('cumtime').print_stats()
    p.sort_stats('tottime').print_stats()