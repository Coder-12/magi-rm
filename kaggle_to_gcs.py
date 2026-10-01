
from google.cloud import storage
import os, hashlib

bucket_name = "laptops_backups"  # your bucket
source_file = "output/rm_outputs_paper_bundle.tar.gz"
destination_blob = "Magicore_Models/rm_outputs_paper_bundle.tar.gz"

client = storage.Client.from_service_account_json("../input/drive-connect/moonlit-state-444018-u5-54041e05edec.json")
bucket = client.bucket(bucket_name)
blob = bucket.blob(destination_blob)

print("⏳ Uploading...")
blob.upload_from_filename(source_file)
print(f"✅ Uploaded to gs://{bucket_name}/{destination_blob}")


def verify_gcs_md5(local_path, bucket_name, blob_path, key_path):
    import base64, hashlib
    from google.cloud import storage
    client = storage.Client.from_service_account_json(key_path)
    blob = client.bucket(bucket_name).get_blob(blob_path)
    local_md5 = hashlib.md5(open(local_path, "rb").read()).hexdigest()
    remote_md5 = base64.b64decode(blob.md5_hash).hex()
    print("Local MD5 :", local_md5)
    print("Remote MD5:", remote_md5)
    print("✅ Match!" if local_md5 == remote_md5 else "❌ Mismatch!")

verify_gcs_md5(
    "output/rm_outputs_paper_bundle.tar.gz",
    "laptops_backups",
    "Magicore_Models/rm_outputs_paper_bundle.tar.gz",
    "../input/drive-connect/moonlit-state-444018-u5-54041e05edec.json"
)



# === CONFIG ===
key_path = "../input/drive-connect/moonlit-state-444018-u5-54041e05edec.json"
bucket_name = "laptops_backups"
target_prefix = "Magicore_Models"

# === FILES TO UPLOAD ===
files_to_upload = [
    "magicore/runs/checkpoints/best_model.pt",
    "magicore/runs/checkpoints/epoch_5.pt",
    "magicore/runs_compressed_logs.tar.gz",
    "magicore/runs/metrics.csv",
    "magicore/runs/metrics.jsonl",
    "magicore/runs/train.log",
]

file_in_gcs = [
    "Magicore_Models/best_model.pt",
    "Magicore_Models/epoch_5.pt",
    "Magicore_Models/runs_compressed_logs.tar.gz",
    "Magicore_Models/metrics.csv",
    "Magicore_Models/metrics.jsonl",
    "Magicore_Models/train.log"
]

# === INIT STORAGE CLIENT ===
client = storage.Client.from_service_account_json(key_path)
bucket = client.bucket(bucket_name)

def md5sum(filename):
    """Compute local file's MD5 checksum (for post-upload verification)."""
    with open(filename, "rb") as f:
        m = hashlib.md5()
        for chunk in iter(lambda: f.read(8192), b""):
            m.update(chunk)
        return m.hexdigest()

uploaded = []

for local_path in files_to_upload:
    if not os.path.exists(local_path):
        print(f"⚠️ Skipping missing file: {local_path}")
        continue

    blob_name = f"{target_prefix}/{os.path.basename(local_path)}"
    blob = bucket.blob(blob_name)

    print(f"⏳ Uploading {local_path} → gs://{bucket_name}/{blob_name}")
    blob.upload_from_filename(local_path)

    uploaded.append(blob_name)

print("\n🎯 All uploads finished successfully!")
for name in uploaded:
    print(f"🔗 gs://{bucket_name}/{name}")


for i in range(len(files_to_upload)):
    verify_gcs_md5(
        files_to_upload[i],
        "laptops_backups",
        file_in_gcs[i],
        "../input/drive-connect/moonlit-state-444018-u5-54041e05edec.json"
    )
