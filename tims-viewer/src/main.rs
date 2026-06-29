//! tims-viewer — native GPU viewer for Bruker timsTOF raw data.
//!
//! Usage:
//!   tims-viewer <path/to/run.d>      open a real Bruker .d dataset
//!   tims-viewer DEMO                 synthetic point cloud (no Bruker data needed)
//!   tims-viewer DEMO --budget 25000000

use std::path::{Path, PathBuf};

use anyhow::Result;
use clap::Parser;
use winit::event_loop::{ControlFlow, EventLoop};

use tims_viewer::app::{App, Plan};
use tims_viewer::data::meta::MetaIndex;
use tims_viewer::offscreen;

/// Command-line arguments.
#[derive(Parser, Debug)]
#[command(name = "tims-viewer", about = "Native GPU viewer for Bruker timsTOF raw data")]
struct Args {
    /// Path to a Bruker `.d` folder, or the literal `DEMO` for a synthetic cloud.
    input: String,
    /// On-GPU point budget (resident capacity). Larger = more detail, more VRAM.
    #[arg(long)]
    budget: Option<usize>,
    /// Number of synthetic frames for DEMO.
    #[arg(long, default_value_t = 800)]
    demo_frames: usize,
    /// Total synthetic points for DEMO (before downsampling to the budget).
    #[arg(long, default_value_t = 30_000_000)]
    demo_points: u64,
    /// Headless: render one frame to this PNG path and exit (no window needed).
    #[arg(long)]
    render_png: Option<String>,
    /// Serve the loaded slice's packed points over HTTP on this port (for the web shell), then
    /// block. No window/GPU needed; works with a `.d` path or `DEMO`.
    #[arg(long)]
    serve: Option<u16>,
    /// With --render-png: render the volume raycaster instead of the point cloud.
    #[arg(long)]
    volume: bool,
    /// With --render-png --volume: use maximum-intensity projection instead of composite.
    #[arg(long)]
    mip: bool,
    /// MS level to render: all | ms1 | ms2.
    #[arg(long, default_value = "all")]
    ms: String,
    /// With --render-png: overlay DDA precursor / DIA isolation-window annotations.
    #[arg(long)]
    annotations: bool,
    /// Output image width / height for --render-png.
    #[arg(long, default_value_t = 1280)]
    width: u32,
    #[arg(long, default_value_t = 800)]
    height: u32,
    /// Transfer-function overrides for --render-png (default from data percentiles).
    #[arg(long)]
    i_min: Option<f32>,
    #[arg(long)]
    i_max: Option<f32>,
    #[arg(long)]
    exposure: Option<f32>,
}

/// Is `p` a Bruker `.d` dataset folder (named `*.d`, or containing `analysis.tdf`)?
fn is_dataset_dir(p: &Path) -> bool {
    p.is_dir()
        && (p.extension().is_some_and(|e| e.eq_ignore_ascii_case("d"))
            || p.join("analysis.tdf").exists())
}

/// Recursively collect `.d` datasets under `dir` (pruning at each `.d` — it holds files, not nested
/// datasets — and bounded in depth so a deep tree can't blow up the scan).
fn collect_datasets(dir: &Path, depth: usize, out: &mut Vec<PathBuf>) {
    if depth > 4 {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for p in entries.filter_map(|e| e.ok().map(|e| e.path())) {
        if is_dataset_dir(&p) {
            out.push(p); // a .d — collect it, don't descend
        } else if p.is_dir() {
            collect_datasets(&p, depth + 1, out);
        }
    }
}

/// Resolve the server's selectable datasets from the launch input: a single `.d` → just it; a
/// directory → every `.d` beneath it (flat `runs/a.d` or nested `runs/a/a.d` both work). The launch
/// directory is the confinement root.
fn discover_datasets(input: &str) -> Result<Vec<PathBuf>> {
    let p = Path::new(input);
    if is_dataset_dir(p) {
        return Ok(vec![p.to_path_buf()]);
    }
    anyhow::ensure!(p.is_dir(), "{input} is not a .d folder or a directory of them");
    let mut v = Vec::new();
    collect_datasets(p, 0, &mut v);
    v.sort();
    anyhow::ensure!(!v.is_empty(), "no .d datasets found under {input}");
    Ok(v)
}

fn main() -> Result<()> {
    // Keep our own INFO logs, but silence the very chatty wgpu/naga internals (e.g.
    // "Device::maintain: waiting for submission index ..." every frame). RUST_LOG overrides.
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(
        "info,wgpu_core=warn,wgpu_hal=warn,wgpu=warn,naga=warn",
    ))
    .init();
    let args = Args::parse();

    // Point-streaming server mode (no window/GPU): serve packed points to the web shell. Multi-dataset
    // + lazy — discover the selectable `.d`s (a single `.d`, or every `.d` under a directory) and let
    // the frontend pick; don't eagerly load any one's metadata here.
    if let Some(port) = args.serve {
        let source = if args.input.eq_ignore_ascii_case("DEMO") {
            log::info!("DEMO mode: {} frames, {} points", args.demo_frames, args.demo_points);
            tims_viewer::serve::ServeSource::Demo(MetaIndex::demo(args.demo_frames, args.demo_points))
        } else {
            tims_viewer::serve::ServeSource::Datasets(discover_datasets(&args.input)?)
        };
        return tims_viewer::serve::serve(source, args.budget, port);
    }

    let (meta, is_demo) = if args.input.eq_ignore_ascii_case("DEMO") {
        log::info!(
            "DEMO mode: {} frames, {} synthetic points",
            args.demo_frames,
            args.demo_points
        );
        (MetaIndex::demo(args.demo_frames, args.demo_points), true)
    } else {
        log::info!("loading metadata from {}", args.input);
        let meta = MetaIndex::load(&args.input)?;
        log::info!(
            "{} frames, ~{} points, RT [{:.1}, {:.1}] s, m/z [{:.1}, {:.1}], 1/K0 [{:.3}, {:.3}]",
            meta.frames.len(),
            meta.total_points_estimate,
            meta.bounds.rt.min,
            meta.bounds.rt.max,
            meta.bounds.mz.min,
            meta.bounds.mz.max,
            meta.bounds.im.min,
            meta.bounds.im.max,
        );
        (meta, false)
    };

    let plan = Plan::new(meta, is_demo, args.budget);

    // Headless one-frame render mode (no window).
    if let Some(path) = args.render_png {
        let ms = match args.ms.to_ascii_lowercase().as_str() {
            "ms1" | "1" => offscreen::MsFilter::Ms1,
            "ms2" | "2" => offscreen::MsFilter::Ms2,
            "all" => offscreen::MsFilter::All,
            other => anyhow::bail!("--ms must be all|ms1|ms2, got '{other}'"),
        };
        let opts = offscreen::Options {
            width: args.width,
            height: args.height,
            volume: args.volume,
            mip: args.mip,
            ms,
            annotations: args.annotations,
            i_min: args.i_min,
            i_max: args.i_max,
            exposure: args.exposure,
        };
        offscreen::render_png(plan, std::path::Path::new(&path), &opts)?;
        log::info!("wrote {}", path);
        return Ok(());
    }

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new(plan);
    event_loop.run_app(&mut app)?;
    Ok(())
}
