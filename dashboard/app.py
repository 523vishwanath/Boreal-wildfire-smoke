import tempfile
import gradio as gr
from smoke_pipeline import SmokeDetector

detector = SmokeDetector()

def analyze(video, threshold, min_dur, progress=gr.Progress()):
    if video is None:
        raise gr.Error("Upload a video first.")
    out_dir = tempfile.mkdtemp(prefix="smoke_")
    res = detector.run(video, out_dir, threshold=threshold, min_dur=min_dur,
                       progress=lambda f, d: progress(f, desc=d))
    return res["video"], res["verdict"], res["alerts_df"], res["gallery"], res["figure"], res["tracks_df"]

with gr.Blocks(title="Wildfire smoke alerts") as demo:
    gr.Markdown("# Wildfire smoke alerts from drone video\n"
                "Upload a video. A YOLO26m-seg model finds smoke in each frame, BoT-SORT follows each region "
                "over time, and an alert is raised when a region stays visible and confidently detected for "
                "long enough. Videos are analysed at 4 frames per second, up to 3 minutes.")
    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Video(label="Upload drone video", sources=["upload"])
            thr = gr.Slider(0.3, 0.9, value=0.6, step=0.05, label="Alert threshold (track confidence)")
            dur = gr.Slider(1, 10, value=3, step=0.5, label="Minimum track duration (seconds)")
            btn = gr.Button("Analyze", variant="primary")
        with gr.Column(scale=2):
            verdict = gr.HTML()
            out_video = gr.Video(label="Annotated video")
    with gr.Row():
        alerts = gr.Dataframe(label="Alerts", interactive=False)
        gallery = gr.Gallery(label="Alert snapshots", columns=3, height=240)
    plot = gr.Plot(label="Smoke score over time")
    tracks = gr.Dataframe(label="All tracked regions", interactive=False)
    btn.click(analyze, [inp, thr, dur], [out_video, verdict, alerts, gallery, plot, tracks])

demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=True)
