//! tims-viewer — native GPU viewer for Bruker timsTOF raw data.
//!
//! Usage:
//!   tims-viewer <path/to/run.d>      open a real Bruker .d dataset
//!   tims-viewer DEMO                 synthetic point cloud (no Bruker data needed)
//!   tims-viewer DEMO --budget 25000000

mod app;
mod camera;
mod data;
mod offscreen;
mod render;
mod state;
mod ui;

use anyhow::Result;
use clap::Parser;
use winit::event_loop::{ControlFlow, EventLoop};

use app::{App, Plan};
use data::meta::MetaIndex;

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

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args = Args::parse();

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
