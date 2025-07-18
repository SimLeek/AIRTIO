import cv2
import numpy as np
import spec_encoding  # C++ module

class EdgeDetector:
    @staticmethod
    def apply(image, output):
        """Apply edge detection using C++ implementation with pre-allocated output."""
        return spec_encoding.apply_edge_detector(image, output)

def create_gaussian_pyramid(frame, min_size=1):
    """Create a pyramid using manual downscaling with INTER_AREA."""
    SIZE_THRESHOLD = 65536  # Same as C++ threshold
    pyramid = [frame]
    current = frame
    while min(current.shape[:2]) > min_size:
        new_height = max(min_size, current.shape[0] // 2)
        new_width = max(min_size, current.shape[1] // 2)
        current = cv2.resize(current, (new_width, new_height), interpolation=cv2.INTER_AREA)
        pyramid.append(current)
    return pyramid

def compute_positional_encodings(xs, ys, level, width, height, max_width, max_height, levels):
    """Compute positional encodings for multiple pixels at a given level using vectorized operations."""
    # Pre-compute all r values
    rw_len = int(np.ceil(np.log2(max_width))) + 1
    rh_len = int(np.ceil(np.log2(max_height))) + 1
    rl_len = int(np.ceil(np.log2(levels))) + 1

    # Create r values arrays
    rw_values = 2.0 ** np.arange(rw_len, dtype=np.float32)
    rh_values = 2.0 ** np.arange(rh_len, dtype=np.float32)
    rl_values = 2.0 ** np.arange(rl_len, dtype=np.float32)

    # Compute relative positions
    center_x, center_y, center_l = width / 2, height / 2, levels / 2
    rel_x = xs - center_x
    rel_y = ys - center_y
    rel_l = level - center_l

    # Vectorized computation using broadcasting
    x_encodings = np.sin((rel_x[None, :] * np.pi / 2) / rw_values[:, None])
    y_encodings = np.sin((rel_y[None, :] * np.pi / 2) / rh_values[:, None])
    l_encodings = np.sin((rel_l * np.pi / 2) / rl_values[:, None])

    # Replicate l_encodings to match n pixels
    n = xs.shape[0]
    l_encodings = np.repeat(l_encodings, n, axis=1)  # Shape: (rl_len, n)

    # Concatenate all encodings
    encodings = np.vstack([x_encodings, y_encodings, l_encodings])

    return encodings

class SPEC:
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
        self.input_frame_shape = None
        self.edges_buffer = None  # Pre-allocated buffer for edge detection

    def __call__(self, frame):
        pyramid = create_gaussian_pyramid(frame)

        if (
                self.energy_arrays is None or
                self.input_frame_shape is None or frame.shape != self.input_frame_shape
        ):
            self.energy_arrays = [np.zeros_like(p) for p in pyramid]
            self.input_frame_shape = frame.shape
            self.edges_buffer = [np.zeros_like(p) for p in pyramid]
            #self.mask_buffer = [np.zeros(p.shape[:2], dtype=bool) for p in pyramid]

            self.max_total_pixels = frame.shape[0]*frame.shape[1]*2
            self.y_indices_buffer = np.empty(self.max_total_pixels, dtype=np.int32)
            self.x_indices_buffer = np.empty(self.max_total_pixels, dtype=np.int32)
            self.l_indices_buffer = np.empty(self.max_total_pixels, dtype=np.int32)
            self.values_buffer = None

        levels = len(pyramid)
        total_count = 0
        ptrs = [0]
        heights = []
        widths = []
        max_height, max_width = pyramid[0].shape[:2]

        for level, img in enumerate(pyramid):
            height, width, channels = img.shape[:3]
            heights.append(height)
            widths.append(width)
            encoding_len = int(np.ceil(np.log2(max_width))) + 1 + int(np.ceil(np.log2(max_height))) + 1 + int(np.ceil(np.log2(levels))) + 1
            value_dim = channels + encoding_len

            if self.values_buffer is None:
                self.values_buffer = np.empty((value_dim, self.max_total_pixels), dtype=np.float32)


            if min(img.shape[:2]) > 1:
                EdgeDetector.apply(img, self.edges_buffer[level])
                self.energy_arrays[level] += np.abs(self.edges_buffer[level]) * self.energy_per_frame

                xs, ys =  spec_encoding.create_mask(self.energy_arrays[level], 1.0)
                #mask = self.mask_buffer[level]
                # ys, xs = np.nonzero(self.mask_buffer[level])
                idx = len(ys)

                if idx > 0:
                    self.y_indices_buffer[total_count:total_count + idx] = ys
                    self.x_indices_buffer[total_count:total_count + idx] = xs
                    self.l_indices_buffer[total_count:total_count + idx] = level

                    encodings = compute_positional_encodings(xs, ys, level, width, height, max_width, max_height, levels)
                    self.values_buffer[:channels, total_count:total_count + idx] = self.energy_arrays[level][ys, xs, :channels].T
                    self.values_buffer[channels:, total_count:total_count + idx] = encodings
                    self.energy_arrays[level][ys, xs, :channels] = 0

                    total_count += idx
                ptrs.append(total_count)
            else:
                self.energy_arrays[level] += img * self.energy_per_frame
                pixel = self.energy_arrays[level][0, 0, :channels]
                if any(pixel[c] > 1 for c in range(channels)):
                    self.y_indices_buffer[total_count] = 0
                    self.x_indices_buffer[total_count] = 0
                    self.l_indices_buffer[total_count] = level

                    encodings = compute_positional_encodings(np.array([0]), np.array([0]), level, width, height,
                                                             max_width, max_height, levels)
                    self.values_buffer[:channels, total_count] = pixel
                    self.values_buffer[channels:, total_count] = encodings[:, 0]
                    self.energy_arrays[level][0, 0, :channels] = 0

                    total_count += 1
                ptrs.append(total_count)

        y_indices = self.y_indices_buffer[:total_count]
        x_indices = self.x_indices_buffer[:total_count]
        l_indices = self.l_indices_buffer[:total_count]
        values = self.values_buffer[:, :total_count]
        ptrs = np.array(ptrs, dtype=np.int32)

        if self.target_nnz is not None:
            current_nnz = values.shape[1]
            error = current_nnz - self.target_nnz
            self.energy_per_frame *= (1.0 - self.alpha * np.sign(error))
        elif self.target_bytes_per_frame is not None:
            current_bytes = values.nbytes
            error = current_bytes - self.target_bytes_per_frame
            self.energy_per_frame *= (1.0 - self.alpha * np.sign(error))

        print(total_count, values)

        return y_indices, x_indices, l_indices, values, ptrs, [heights, widths, levels]

def proc_vid_sec(cap, max_frames, frame_count, spec):
    ret, frame = cap.read()
    if not ret or (max_frames is not None and frame_count >= max_frames):
        return None
    float_frame = frame.astype(np.float32) / 255.0
    #cv2.convertScaleAbs(frame, dst=float_frame, alpha=1.0 / 255.0)
    spec(float_frame)
    frame_count += 1
    return frame_count

def process_video(input_path, energy_factor=0.5, max_frames=None):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file")

    spec = SPEC(energy_factor)
    frame_count = 0
    while cap.isOpened():
        frame_count = proc_vid_sec(cap, max_frames, frame_count, spec)
        if frame_count is None:
            break
    cap.release()
    return frame_count

def main():
    import cProfile, pstats

    class Stats(pstats.Stats):
        sort_arg_dict_default = {
            "calls": (((1, -1),), "call count"),
            "ncalls": (((1, -1),), "call count"),
            "cumtime": (((4, -1),), "cumulative time"),
            "cumulative": (((4, -1),), "cumulative time"),
            "file": (((6, 1),), "file name"),
            "filename": (((6, 1),), "file name"),
            "line": (((7, 1),), "line number"),
            "module": (((6, 1),), "file name"),
            "name": (((8, 1),), "function name"),
            "nfl": (((8, 1), (6, 1), (7, 1),), "name/file/line"),
            "pcalls": (((0, -1),), "primitive call count"),
            "stdname": (((9, 1),), "standard name"),
            "time": (((2, -1),), "internal time"),
            "tottime": (((2, -1),), "internal time"),
            "cumulativepercall": (((5, -1),), "cumulative time per call"),
            "totalpercall": (((3, -1),), "total time per call"),
        }

        def sort_stats(self, *field):
            if not field:
                self.fcn_list = 0
                return self
            if len(field) == 1 and isinstance(field[0], int):
                field = [{-1: "stdname",
                          0: "calls",
                          1: "time",
                          2: "cumulative"}[field[0]]]
            elif len(field) >= 2:
                for arg in field[1:]:
                    if type(arg) != type(field[0]):
                        raise TypeError("Can't have mixed argument type")

            sort_arg_defs = self.get_sort_arg_defs()

            sort_tuple = ()
            self.sort_type = ""
            connector = ""
            for word in field:
                if isinstance(word, pstats.SortKey):
                    word = word.value
                sort_tuple = sort_tuple + sort_arg_defs[word][0]
                self.sort_type += connector + sort_arg_defs[word][1]
                connector = ", "

            stats_list = []
            for func, (cc, nc, tt, ct, callers) in self.stats.items():
                if nc == 0:
                    npc = 0
                else:
                    npc = float(tt) / nc

                if cc == 0:
                    cpc = 0
                else:
                    cpc = float(ct) / cc

                stats_list.append((cc, nc, tt, npc, ct, cpc) + func +
                                  (pstats.func_std_string(func), func))

            stats_list.sort(key=pstats.cmp_to_key(pstats.TupleComp(sort_tuple).compare))

            self.fcn_list = fcn_list = []
            for tuple in stats_list:
                fcn_list.append(tuple[-1])
            return self

    cProfile.run('process_video("input_video.mp4", max_frames=1000)', 'prof_data.prof')
    p = Stats('prof_data.prof')
    p.sort_stats('cumtime').print_stats()

if __name__ == "__main__":
    main()
