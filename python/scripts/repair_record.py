import base64
import glob
import hashlib
import sys

filename = sys.argv[1]


# Compute the new hash and size
def urlsafe_b64encode(data: bytes) -> bytes:
    return base64.urlsafe_b64encode(data).rstrip(b"=")


hasher = hashlib.sha256()
with open(filename, "rb") as f:
    data = f.read()
    hasher.update(data)
hash_str = urlsafe_b64encode(hasher.digest()).decode("ascii")
size = len(data)

# Update the record file
record_file = glob.glob("*/RECORD")[0]
with open(record_file, "r") as f:
    lines = [l.split(",") for l in f.readlines()]

for l in lines:
    if filename == l[0]:
        l[1] = hash_str
        l[2] = f"{size}\n"

with open(record_file, "w") as f:
    for l in lines:
        f.write(",".join(l))
