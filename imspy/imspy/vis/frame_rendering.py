import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import imageio.v2 as imageio
from matplotlib import colors as mcolors
from tqdm import tqdm
from imspy.timstof import TimsDatasetDDA
from imspy.timstof.dia import TimsDatasetDIA

# Distinct colors for overlays
COLORS = list(mcolors.TABLEAU_COLORS.values()) + [
    mcolors.CSS4_COLORS[c] for c in [
        "crimson", "mediumseagreen", "darkorange", "slateblue", "turquoise",
        "goldenrod", "orchid", "dodgerblue", "tomato", "limegreen"
    ]
]

def configure_gpu_memory(limit_gb: float = 4.0):
    import tensorflow as tf

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if not gpus:
        return
    cfg = tf.config.experimental.VirtualDeviceConfiguration(memory_limit=int(limit_gb * 1024))
    for gpu in gpus:
        tf.config.experimental.set_virtual_device_configuration(gpu, [cfg])


def get_frame_matrix(handle, frame_id: int,
                     index_max: int = 1700, scan_max: int = 972) -> np.ndarray:
    frame = handle[frame_id]

    frame = frame.filter(mz_max=float(index_max))

    return frame.vectorized(0).get_tensor_repr(
        dense=True, zero_indexed=False, re_index=False,
        index_max=index_max, scan_max=scan_max
    )


def choose_color(id_val: int) -> str:
    return COLORS[int(id_val) % len(COLORS)]


def overlay_precursor(ax, meta_df, image_width: int):
    """
    Overlay precursor isolation boxes (DDA): m/z-based windows per precursor.
    """
    for _, row in meta_df.iterrows():
        color = choose_color(row['Frame'])
        mz_mid = row['IsolationMz']
        iso_w = row['IsolationWidth']
        # calculate m/z-based window
        x = int(np.floor(mz_mid - iso_w/2)) - 2
        y = row['ScanNumBegin']
        w = int(np.ceil(iso_w)) + 4
        h = row['ScanNumEnd'] - y
        rect = patches.Rectangle((x, y), w, h,
                                 linewidth=1.0, edgecolor=color,
                                 facecolor='none', linestyle='-', alpha=0.6)
        ax.add_patch(rect)
        lbl = (
            f"ID:{int(row['Precursor'])}"
            f"CE:{row['CollisionEnergy']:.1f}eV"
            f"m/z:{mz_mid:.2f}±{iso_w:.2f}"
        )
        ax.text(x + 25, y + 10, lbl,
                fontsize=8, color='white', backgroundcolor='black',
                verticalalignment='top', horizontalalignment='left', alpha=0.9)


def overlay_fragment(ax, meta_df, image_width: int):
    """
    Overlay fragment selection windows across full m/z axis (DDA).
    """
    for _, row in meta_df.iterrows():
        color = choose_color(row['Frame'])
        y = row['ScanNumBegin']
        h = row['ScanNumEnd'] - y
        rect = patches.Rectangle((0, y), image_width, h,
                                 linewidth=1.0, edgecolor=color,
                                 facecolor='none', linestyle='--', alpha=0.6)
        ax.add_patch(rect)
        lbl = (
            f"ID:{int(row['Precursor'])}, "
            f"CE:{row['CollisionEnergy']:.1f}eV,"
            f"m/z:{row['IsolationMz']:.2f}±{row['IsolationWidth']:.2f}"
        )
        ax.text(2, y + 2, lbl,
                fontsize=8, color='white', backgroundcolor='black',
                verticalalignment='bottom', horizontalalignment='left', alpha=0.9)


def overlay_windows(ax, df_windows, image_width: int,
                    text_dx: int = 2, text_dy: int = 2,
                    ha: str = 'left', va: str = 'top'):
    """
    Overlay DIA selection windows (m/z-based boxes).
    """
    for _, row in df_windows.iterrows():
        group = int(row['WindowGroup'])
        color = choose_color(group)
        mz_mid = row['IsolationMz']
        iso_w = row['IsolationWidth']
        x = int(np.floor(mz_mid - iso_w/2)) - 2
        y = row['ScanNumBegin']
        w = int(np.ceil(iso_w)) + 4
        h = row['ScanNumEnd'] - y
        rect = patches.Rectangle((x, y), w, h,
                                 linewidth=1.2, edgecolor=color,
                                 facecolor='none', linestyle='-', alpha=0.8)
        ax.add_patch(rect)
        lbl = f"WG:{group} CE:{row['CollisionEnergy']:.1f}eV m/z:{mz_mid:.2f}±{iso_w:.2f}"
        ax.text(x + text_dx, y + text_dy, lbl,
                fontsize=8, color='white', backgroundcolor='black',
                horizontalalignment=ha, verticalalignment=va, alpha=0.9)


class BaseFrameRenderer:
    def __init__(self, memory_limit_gb: float = 4.0):
        configure_gpu_memory(memory_limit_gb)

    def batch_render(self, out_dir: str, frame_ids=None, **render_kwargs):
        if frame_ids is None:
            frame_ids = self._all_frame_ids
        os.makedirs(out_dir, exist_ok=True)
        for fid in tqdm(frame_ids, desc='Rendering frames', ncols=80):
            out_path = os.path.join(out_dir, f"frame_{fid:04d}.png")
            self._render_frame(fid, save_path=out_path, **render_kwargs)

    def create_video(self, frames_dir: str, output_path: str,
                     fps: int = 10, ext: str = 'png'):
        files = sorted(f for f in os.listdir(frames_dir) if f.endswith(f'.{ext}'))
        if not files:
            raise RuntimeError(f"No .{ext} frames in {frames_dir}")
        writer = imageio.get_writer(output_path, fps=fps)
        for fname in tqdm(files, desc='Creating video', ncols=80):
            img = imageio.imread(os.path.join(frames_dir, fname))
            writer.append_data(img)
        writer.close()


class DDAFrameRenderer(BaseFrameRenderer):
    def __init__(self, data_path: str,
                 memory_limit_gb: float = 4.0,
                 use_bruker_sdk: bool = True,
                 in_memory: bool = False):
        super().__init__(memory_limit_gb)
        self.handle = TimsDatasetDDA(data_path, use_bruker_sdk=use_bruker_sdk, in_memory=in_memory)
        self.meta = self.handle._load_pasef_meta_data()
        diffs = np.diff(self.handle.precursor_frames)
        self.fragments_map = {f: set(range(f+1, f+d)) for f, d in zip(self.handle.precursor_frames[:-1], diffs)}
        self.precursor_frames = set(self.handle.precursor_frames)
        self.fragment_frames = set(self.handle.fragment_frames)
        self._all_frame_ids = list(self.handle.meta_data.frame_id)

    def _render_frame(self, frame_id: int, save_path=None,
                      dpi: int = 150, cmap: str = 'inferno',
                      annotate: bool = True):
        F = get_frame_matrix(self.handle, frame_id, scan_max=self.handle.num_scans)
        fig, ax = plt.subplots(figsize=(10 * F.shape[1]/F.shape[0], 10), dpi=dpi)
        ax.imshow(np.cbrt(F), cmap=cmap, origin='upper')
        ax.set(xlabel='m/z (1 Th bins)', ylabel='Scan Num',
               title=(f"Frame {frame_id}"
                      f"{' [Precursor]' if frame_id in self.precursor_frames else ''}"
                      f"{' [+Fragment]' if frame_id in self.fragment_frames else ''}"))
        if annotate:
            if frame_id in self.precursor_frames:
                fragments = self.fragments_map.get(frame_id, set())
                dfp = self.meta[self.meta['Frame'].isin(fragments)]
                overlay_precursor(ax, dfp, F.shape[1])
            if frame_id in self.fragment_frames:
                dff = self.meta[self.meta['Frame'] == frame_id]
                overlay_fragment(ax, dff, F.shape[1])
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path)
            plt.close(fig)
        else:
            plt.show()


class DIAFrameRenderer(BaseFrameRenderer):
    def __init__(self, data_path: str,
                 memory_limit_gb: float = 4.0,
                 use_bruker_sdk: bool = True,
                 in_memory: bool = False):
        super().__init__(memory_limit_gb)
        self.handle = TimsDatasetDIA(data_path, use_bruker_sdk=use_bruker_sdk, in_memory=in_memory)
        self.windows = self.handle.dia_ms_ms_windows.copy()
        self.frame_to_group = dict(zip(self.handle.dia_ms_ms_info.Frame, self.handle.dia_ms_ms_info.WindowGroup))
        self.precursor_to_fragments = {}
        current = None
        for _, row in self.handle.meta_data.iterrows():
            idx, typ = row.Id, row.MsMsType
            if typ == 0:
                current = idx
                self.precursor_to_fragments[current] = set()
            else:
                self.precursor_to_fragments[current].add(idx)
        self.precursor_frames = set(self.precursor_to_fragments.keys())
        self.fragment_frames = set(self.handle.meta_data[self.handle.meta_data.MsMsType == 1].Id)
        self._all_frame_ids = list(self.handle.meta_data.Id)

    def _render_frame(self, frame_id: int, save_path=None,
                      dpi: int = 150, cmap: str = 'inferno',
                      annotate: bool = True):
        F = get_frame_matrix(self.handle, frame_id, scan_max=self.handle.num_scans)
        fig, ax = plt.subplots(figsize=(10 * F.shape[1]/F.shape[0], 10), dpi=dpi)
        ax.imshow(np.sqrt(F), cmap=cmap, origin='upper')
        ax.set(xlabel='m/z (1 Th bins)', ylabel='Scan Num',
               title=(f"Frame {frame_id}"
                      f"{' [Precursor]' if frame_id in self.precursor_frames else ''}"
                      f"{' [+Fragment]' if frame_id in self.fragment_frames else ''}"))
        if annotate:
            if frame_id in self.precursor_frames:
                df_w = self.windows.copy()
                df_w['Frame'] = frame_id
            else:
                grp = self.frame_to_group.get(frame_id)
                df_w = self.windows[self.windows.WindowGroup == grp].copy()
                df_w['Frame'] = frame_id
            overlay_windows(ax, df_w, F.shape[1], text_dx=-20, text_dy=-20, ha='right', va='bottom')
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path)
            plt.close(fig)
        else:
            plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Render TIMS frames (DDA or DIA) & build video'
    )
    parser.add_argument('mode', choices=['dda', 'dia'],
                        help='Acquisition mode')
    parser.add_argument('data_path', help='Path to .d folder')
    parser.add_argument('--out_dir', default='./frames',
                        help='Output directory for PNGs')
    parser.add_argument('--frames', type=int, nargs='+',
                        help='List of frame IDs to render')
    parser.add_argument('--video_out', help='Path for output video file')
    parser.add_argument('--fps', type=int, default=10,
                        help='Frames per second')
    parser.add_argument('--mem', type=float, default=4.0,
                        help='GPU memory limit (GB)')
    parser.add_argument('--no-annotate', dest='annotate', action='store_false',
                        help='Disable drawing of window annotations')
    args = parser.parse_args()

    common = {'annotate': args.annotate}
    if args.mode == 'dda':
        renderer = DDAFrameRenderer(args.data_path, memory_limit_gb=args.mem)
    else:
        renderer = DIAFrameRenderer(args.data_path, memory_limit_gb=args.mem)

    renderer.batch_render(args.out_dir, frame_ids=args.frames, **common)
    if args.video_out:
        renderer.create_video(args.out_dir, args.video_out, fps=args.fps)
