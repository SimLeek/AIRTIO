import cv2
import numpy as np
import spec_encoding  # C++ module

print(dir(spec_encoding))

class EdgeDetector:
    @staticmethod
    def apply(image):
        """Apply edge detection using C++ implementation."""
        return spec_encoding.apply_edge_detector(image)

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

    for i, r in enumerate(rw_values):
        encodings[i] = np.sin((rel_x * np.pi / 2) / r)
    for i, r in enumerate(rh_values):
        encodings[i + len(rw_values)] = np.sin((rel_y * np.pi / 2) / r)
    for i, r in enumerate(rl_values):
        encodings[i + len(rw_values) + len(rh_values)] = np.sin((rel_l * np.pi / 2) / r)

    return encodings

def create_mask(energy_array, threshold=1.0):
    """Create a mask using C++ implementation."""
    return spec_encoding.create_mask(energy_array, threshold)

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

    def __call__(self, frame):
        pyramid = create_gaussian_pyramid(frame)

        if (
            self.energy_arrays is None or
            len(self.energy_arrays) != len(pyramid)# or
            #any(pyramid[i].shape != self.energy_arrays[i].shape for i in range(len(pyramid)))
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
            encoding_len = int(np.ceil(np.log2(max_width))) + 1 + int(np.ceil(np.log2(max_height))) + 1 + int(np.ceil(np.log2(levels))) + 1
            value_dim = channels + encoding_len

            if min(img.shape[:2]) > 1:
                edges = EdgeDetector.apply(img)
                self.energy_arrays[level] += edges * self.energy_per_frame

                mask = create_mask(self.energy_arrays[level])
                ys, xs = np.nonzero(mask)
                idx = len(ys)

                if idx > 0:
                    row_indices_y = ys  # Already int32 from np.nonzero
                    row_indices_x = xs  # Already int32 from np.nonzero
                    row_indices_l = np.full(idx, level, dtype=np.int32)
                    row_values = np.zeros((value_dim, idx), dtype=np.float32)

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
                pixel = self.energy_arrays[level][0, 0, :channels]
                if any(pixel[c] > 1 for c in range(channels)):
                    row_values = np.zeros((value_dim, 1), dtype=np.float32)
                    row_indices_y = np.zeros(1, dtype=np.int32)
                    row_indices_x = np.zeros(1, dtype=np.int32)
                    row_indices_l = np.ones(1, dtype=np.int32) * level
                    encodings = compute_positional_encodings(np.array([0]), np.array([0]), level, width, height, max_width, max_height, levels)
                    row_values[:channels, 0] = pixel
                    self.energy_arrays[level][0, 0, :channels] = 0
                    row_values[channels:, 0] = encodings[:, 0]
                    y_indices.append(row_indices_y)
                    x_indices.append(row_indices_x)
                    l_indices.append(row_indices_l)
                    values.append(row_values)
                    ptrs.append(ptrs[-1] + 1)

        y_indices = np.concatenate(y_indices) if y_indices else np.array([], dtype=np.int32)
        x_indices = np.concatenate(x_indices) if x_indices else np.array([], dtype=np.int32)
        l_indices = np.concatenate(l_indices) if l_indices else np.array([], dtype=np.int32)
        values = np.concatenate(values, axis=1) if values else np.array([], dtype=np.float32).reshape(value_dim, 0)
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

def proc_vid_sec(cap, max_frames, frame_count, spec):
    ret, frame = cap.read()
    if not ret or (max_frames is not None and frame_count >= max_frames):
        return None
    #float_frame = frame.astype(np.float32) / 255.0
    float_frame = np.empty_like(frame, dtype=np.float32)
    cv2.convertScaleAbs(frame, dst=float_frame, alpha=1.0 / 255.0)
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
        # https://stackoverflow.com/a/78404643/782170
        # list the tuple indices and directions for sorting,
        # along with some printable description
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
                # Be compatible with old profiler
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

    cProfile.run('process_video("input_video_1080p.mp4", max_frames=1000)', 'prof_data.prof')
    p = Stats('prof_data.prof')
    p.sort_stats('cumulativepercall').print_stats()

if __name__ == "__main__":
    main()
