import sys
import io
import json
import os

# Set stdout encoding to utf-8 for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import qai_hub as hub

jobs = {
    'Encoder Compile': ('jgol70jxg', 'compile'),
    'Decoder Compile': ('jgjr6mjxp', 'compile'),
    'Encoder Profile': ('jpe701j15', 'profile'),
    'Decoder Profile': ('j5wl0vo6p', 'profile'),
    'Encoder Inference': ('jg9z71vlp', 'inference'),
    'Decoder Inference': ('jp1nkl02g', 'inference'),
}

print("=== Polling Qualcomm AI Hub Jobs ===")
for name, (jid, jtype) in jobs.items():
    job = hub.get_job(jid)
    print(f"\n--- {name} ({jid}) ---")
    print(f"URL: {job.url}")
    print(f"Device: {getattr(job, 'device', 'Snapdragon X Elite CRD')}")
    
    # Wait for job to finish if it's running/measuring
    status = job.get_status()
    print(f"Current Status: {status.code} - {status.message}")
    
    if not status.finished:
        print(f"Waiting for {name} ({jid}) to finish...")
        job.wait()
        status = job.get_status()
        print(f"Finished Status: {status.code}")
        
    if jtype in ['profile', 'inference']:
        try:
            profile = job.download_profile()
            print("\nProfile Summary:")
            if isinstance(profile, dict):
                perf_summary = profile.get("execution_summary", {})
                print(f"Estimated Inference Time: {perf_summary.get('estimated_inference_time', 'N/A')}")
                print(f"Peak Memory Range: {perf_summary.get('inference_memory_peak_range', 'N/A')}")
                print(f"Compute Units: {perf_summary.get('compute_units', 'N/A')}")
                print("Execution Summary Details:", json.dumps(perf_summary, indent=2))
        except Exception as e:
            print("Could not download/parse profile:", e)

    if jtype == 'compile':
        try:
            model = job.get_target_model()
            out_dir = os.path.join(os.path.dirname(__file__), "downloaded_models")
            os.makedirs(out_dir, exist_ok=True)
            model_path = model.download(out_dir)
            print(f"Downloaded compiled model artifact to: {model_path}")
        except Exception as e:
            print("Could not download compiled model artifact:", e)
