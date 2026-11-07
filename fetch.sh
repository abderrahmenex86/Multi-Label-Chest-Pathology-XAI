#!/bin/bash

REMOTE=""
PORT="22"
PROJECT_NAME=$(basename "$(pwd)")
REMOTE_DIR="/workspace/$PROJECT_NAME"

while [[ $# -gt 0 ]]; do
   case $1 in
   -p)
      PORT="$2"
      shift 2
      ;;
   *)
      REMOTE="$1"
      shift
      ;;
   esac
done

if [ -z "$REMOTE" ]; then
   echo "[ERROR] Usage: ./fetch.sh user@host [-p port]"
   exit 1
fi

echo "[INFO] Verifying artifacts exist on the remote server..."
if ! ssh -p "$PORT" "$REMOTE" "[ -d $REMOTE_DIR/artifacts ]"; then
   echo "[ERROR] Sanity Check Failed: No 'artifacts' directory found in $REMOTE_DIR."
   exit 1
fi

echo "[INFO] Packaging artifacts on the remote server (using 200MB chunks)..."
ssh -p "$PORT" "$REMOTE" "cd $REMOTE_DIR && rm -f artifacts.tar chunk_* && tar -cf artifacts.tar artifacts/ && split -b 200M artifacts.tar chunk_"

echo "[INFO] Fetching chunk list..."
CHUNKS=$(ssh -p "$PORT" "$REMOTE" "cd $REMOTE_DIR && ls chunk_* 2>/dev/null")

if [ -z "$CHUNKS" ]; then
   echo "[ERROR] Failed to generate chunks on the remote server."
   exit 1
fi

mkdir -p artifacts_temp
cd artifacts_temp || exit

echo "[INFO] Opening parallel connections (Batched to prevent server drop)..."

MAX_CONCURRENT=5
for CHUNK in $CHUNKS; do
   echo "  -> Spawning download stream for $CHUNK..."
   scp -P "$PORT" -q "$REMOTE:$REMOTE_DIR/$CHUNK" . &

   # Simple job control: wait if we hit the concurrency limit
   while (($(jobs -r -p | wc -l) >= MAX_CONCURRENT)); do
      sleep 1
   done
done

# Wait for the final batch to finish
wait

echo "[INFO] All chunks received. Reassembling..."
cat chunk_* >artifacts.tar

echo "[INFO] Extracting artifacts..."
tar -xf artifacts.tar -C ../

echo "[INFO] Cleaning up remote and local temporary files..."
cd ../
rm -rf artifacts_temp
ssh -p "$PORT" "$REMOTE" "cd $REMOTE_DIR && rm -f artifacts.tar chunk_*"

echo "[SUCCESS] High-speed parallel fetch complete!"
